"""Qwen3.5 MoE block: numeric parity against transformers, and config shapes.

    CUDA_VISIBLE_DEVICES=5 pytest models/tests/test_moe_qwen35.py -q

The port is torchtitan b21f7d43e (`models/common/moe.py` + `moe_sharding.py`), so
the thing worth checking is not that it matches torchtitan line for line -- it does
-- but that the *config* built from an HF `config.json` wires it up to compute what
HF's `Qwen3_5MoeSparseMoeBlock` computes. Four things could silently disagree and
all four change the output rather than raising:

  1. `gate_up_proj` is CONCATENATED [gate; up] in HF, while `FeedForward.w13` is
     INTERLEAVED -- so routed experts keep w1/w3 separate and the shared expert
     goes through the w1/w3 state-dict hooks.
  2. scores are applied AFTER the experts (`score_before_experts` is false for
     Qwen3.5): combining in the wrong order rescales every token.
  3. `route_norm` divides the top-k scores by their own sum; without it the
     output is off by a per-token factor.
  4. the shared expert is sigmoid-GATED (`mlp.shared_expert_gate`, a Linear(D,1)).

Everything runs in bf16 because `torch._grouped_mm` takes bf16 only, which is also
how training runs (FSDP param_dtype=bf16). Tolerances are bf16 tolerances.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[2]

DIM = 128
MOE_HIDDEN = 64
SHARED_HIDDEN = 96
NUM_EXPERTS = 8
TOP_K = 2
TOKENS = 64

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="_grouped_mm needs CUDA"
)


def _our_moe(device, dtype):
    from models.qwen3_5.configs import _qwen35_moe_config

    config = _qwen35_moe_config(
        dim=DIM,
        moe_hidden_dim=MOE_HIDDEN,
        num_experts=NUM_EXPERTS,
        top_k=TOP_K,
        shared_expert_hidden_dim=SHARED_HIDDEN,
        layer_id=0,
        load_balance_coeff=None,
    )
    moe = config.build().to(device=device, dtype=dtype)
    moe.init_states(buffer_device=device)
    return moe.to(device=device, dtype=dtype)


def _hf_block(device, dtype):
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
        Qwen3_5MoeTextConfig,
    )
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
        Qwen3_5MoeSparseMoeBlock,
    )

    config = Qwen3_5MoeTextConfig(
        hidden_size=DIM,
        moe_intermediate_size=MOE_HIDDEN,
        shared_expert_intermediate_size=SHARED_HIDDEN,
        num_experts=NUM_EXPERTS,
        num_experts_per_tok=TOP_K,
        num_hidden_layers=4,
    )
    block = Qwen3_5MoeSparseMoeBlock(config).to(device=device, dtype=dtype)
    for p in block.parameters():
        torch.nn.init.normal_(p, std=0.05)
    return block


def _copy_hf_into(ours, hf):
    """HF weights -> our module, through the real key layout.

    `gate_up_proj` is [E, 2F, D] = cat([gate, up]) on dim 0, so the gate half is
    w1 and the up half is w3. The shared expert is loaded via w1/w3 so
    `FeedForward._merge_w13_on_load` does the interleave -- the same hook the
    checkpoint path uses.
    """
    with torch.no_grad():
        experts = ours.routed_experts.inner_experts
        gate_up = hf.experts.gate_up_proj
        experts.w1_EFD.copy_(gate_up[:, :MOE_HIDDEN, :])
        experts.w3_EFD.copy_(gate_up[:, MOE_HIDDEN:, :])
        experts.w2_EDF.copy_(hf.experts.down_proj)
        ours.router.gate.weight.copy_(hf.gate.weight)
        ours.shared_experts.load_state_dict(
            {
                "w1.weight": hf.shared_expert.gate_proj.weight,
                "w3.weight": hf.shared_expert.up_proj.weight,
                "w2.weight": hf.shared_expert.down_proj.weight,
                "gate.weight": hf.shared_expert_gate.weight,
            }
        )


@requires_cuda
def test_moe_matches_hf_sparse_block():
    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    hf = _hf_block(device, dtype)
    ours = _our_moe(device, dtype)
    _copy_hf_into(ours, hf)

    x = torch.randn(TOKENS, DIM, device=device, dtype=dtype)
    ours.eval()
    with torch.no_grad():
        mine = ours(x)
        theirs = hf(x.unsqueeze(0)).squeeze(0)

    scale = theirs.abs().mean().item()
    mean_err = (mine - theirs).abs().mean().item()
    assert mean_err < 0.02 * scale, f"mean |d| {mean_err:.3e} vs scale {scale:.3e}"
    assert torch.isfinite(mine).all()


@requires_cuda
def test_routing_picks_the_same_experts_as_hf():
    """Parity of the whole block can hide a router that disagrees on a few tokens
    and is saved by averaging. Compare the expert choice directly."""
    torch.manual_seed(1)
    device, dtype = "cuda", torch.bfloat16
    hf = _hf_block(device, dtype)
    ours = _our_moe(device, dtype)
    _copy_hf_into(ours, hf)

    x = torch.randn(TOKENS, DIM, device=device, dtype=dtype)
    with torch.no_grad():
        _, their_scores, their_ids = hf.gate(x)
        our_scores, our_ids, routing_map = ours.router(x, None)

    assert sorted_rows(our_ids).equal(sorted_rows(their_ids))
    # route_norm: both sides divide the top-k scores by their own sum, so each
    # row sums to 1. Without it ours would sum to the raw softmax mass.
    assert torch.allclose(
        our_scores.float().sum(-1), torch.ones(TOKENS, device=device), atol=1e-2
    )
    assert torch.allclose(our_scores.float(), their_scores.float(), atol=1e-2)
    assert routing_map.sum().item() == TOKENS * TOP_K


def sorted_rows(t):
    return t.sort(dim=-1).values


@requires_cuda
def test_grouped_experts_match_a_per_expert_loop():
    """`_grouped_mm` with uneven group sizes against the obvious loop. This is the
    one place a wrong `offs` (cumsum vs exclusive-scan) would go unnoticed: the
    shapes come out right either way."""
    from models.common.moe import GroupedExperts

    torch.manual_seed(2)
    device, dtype = "cuda", torch.bfloat16
    experts = GroupedExperts.Config(
        dim=DIM, hidden_dim=MOE_HIDDEN, num_experts=4
    ).build().to(device=device, dtype=dtype)
    for p in experts.parameters():
        torch.nn.init.normal_(p, std=0.05)

    counts = torch.tensor([5, 0, 11, 3], device=device)
    x = torch.randn(int(counts.sum()), DIM, device=device, dtype=dtype)
    with torch.no_grad():
        got = experts(x, counts)

    want = []
    start = 0
    for e, n in enumerate(counts.tolist()):
        chunk = x[start : start + n]
        start += n
        h = torch.nn.functional.silu(chunk @ experts.w1_EFD[e].T) * (
            chunk @ experts.w3_EFD[e].T
        )
        want.append(h @ experts.w2_EDF[e].T)
    want = torch.cat(want)

    assert got.shape == want.shape
    assert (got.float() - want.float()).abs().max() < 0.05


def test_35b_config_matches_the_torchtitan_reference():
    """Shapes from torchtitan b21f7d43e `_35b_a3b`: dim 2048, 40 layers, 256
    experts of width 512, top-8, a 512-wide shared expert, ViT depth 27/1152.
    A typo in the JSON would still build, just not the right model."""
    from models.qwen3_5.configs import qwen35_config_from_hf

    config = qwen35_config_from_hf(
        REPO / "configs/models/qwen3_5_35b_a3b.json", seq_len=4096
    )
    assert config.dim == 2048
    assert len(config.layers) == 40
    assert config.enable_weight_tying is False

    for layer in config.layers:
        assert layer.feed_forward is None, "MoE layers carry no dense FFN"
        moe = layer.moe
        assert moe.num_experts == 256
        assert moe.router.top_k == 8
        assert moe.router.score_func == "softmax"
        assert moe.router.route_norm is True
        inner = moe.routed_experts.inner_experts
        assert (inner.dim, inner.hidden_dim, inner.num_experts) == (2048, 512, 256)
        # shared expert: w13 is the fused gate+up, hence 2 * 512
        assert moe.shared_experts.w13.out_features == 2 * 512
        assert moe.shared_experts.gate.out_features == 1

    ve = config.vision_encoder
    assert (ve.num_layers, ve.dim) == (27, 1152)
    # the projector has to land in the decoder's width or the scatter fails
    assert ve.merger.fc2.out_features == config.dim

    full = sum(1 for layer in config.layers if layer.attention is not None)
    assert full == 10, f"every 4th layer is full attention, got {full}"


def test_state_dict_round_trips_the_moe_keys():
    """`Qwen35StateDictAdapter` carried the MoE mapping from the day it was
    vendored but nothing ever built an MoE to exercise it. The fusion is the part
    that can be silently backwards: HF's `gate_up_proj` is `cat([w1, w3])` on the
    out-features dim, so a swap loads `up` where `gate` belongs and training just
    converges worse.
    """
    import torch

    from models.qwen3_5.configs import qwen35_config_from_hf
    from models.qwen3_5.state_dict_adapter import (
        is_local_only_state,
        Qwen35StateDictAdapter,
    )

    config = qwen35_config_from_hf(
        REPO / "configs/models/qwen3_5_35b_a3b.json", seq_len=4096
    )
    with torch.device("meta"):
        model = config.build()
    state = model.state_dict()
    adapter = Qwen35StateDictAdapter(config, None)

    hf = adapter.to_hf(state)
    prefix = "model.language_model.layers.0.mlp"
    assert hf[f"{prefix}.experts.gate_up_proj"].shape == (256, 2 * 512, 2048)
    assert hf[f"{prefix}.experts.down_proj"].shape == (256, 2048, 512)
    assert hf[f"{prefix}.gate.weight"].shape == (256, 2048)
    assert hf[f"{prefix}.shared_expert_gate.weight"].shape == (1, 2048)
    # the dense FFN keys must not appear for a sparse model
    assert f"{prefix}.gate_proj.weight" not in hf

    back = adapter.from_hf(hf)
    ours = [k for k in state if ".moe." in k and not is_local_only_state(k)]
    # 8 HF-backed tensors per layer: 3 grouped experts, the router gate, and the
    # shared expert's w1/w2/w3 + gate. `expert_bias_E` is ours alone.
    assert len(ours) == 40 * 8, f"8 MoE tensors per layer, got {len(ours) / 40}"
    assert not [k for k in ours if k not in back]
    assert not [k for k in ours if back[k].shape != state[k].shape]


def test_gate_and_up_are_not_swapped_by_the_fusion():
    """The half of `gate_up_proj` that HF chunks off first is the GATE. Checked on
    real tensors, because a meta round trip compares shapes only -- and w1 and w3
    have identical shapes."""
    from models.common.moe import make_routed_experts_config
    from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter
    from models.qwen3_5.configs import qwen35_config_from_hf

    config = qwen35_config_from_hf(
        REPO / "configs/models/qwen3_5_35b_a3b.json", seq_len=4096
    )
    adapter = Qwen35StateDictAdapter(config, None)
    experts = "layers.0.moe.routed_experts.inner_experts"
    e, f, d = 2, 3, 4
    state = {
        f"{experts}.w1_EFD": torch.full((e, f, d), 1.0),
        f"{experts}.w3_EFD": torch.full((e, f, d), 3.0),
        f"{experts}.w2_EDF": torch.full((e, d, f), 2.0),
    }
    fused = adapter.to_hf(state)["model.language_model.layers.0.mlp.experts.gate_up_proj"]
    assert fused.shape == (e, 2 * f, d)
    assert (fused[:, :f] == 1.0).all(), "first half of gate_up_proj must be w1 (gate)"
    assert (fused[:, f:] == 3.0).all(), "second half must be w3 (up)"

    back = adapter.from_hf(
        {
            "model.language_model.layers.0.mlp.experts.gate_up_proj": fused,
            # from_hf refuses a dict with neither lm_head nor embed_tokens
            "lm_head.weight": torch.zeros(1, 1),
        }
    )
    assert torch.equal(back[f"{experts}.w1_EFD"], state[f"{experts}.w1_EFD"])
    assert torch.equal(back[f"{experts}.w3_EFD"], state[f"{experts}.w3_EFD"])


def test_flops_bill_topk_experts_not_all_of_them():
    """`_moe_flops` must scale with top_k, not num_experts: billing all 256 would
    overstate the model by ~32x and make every MFU number meaningless."""
    from train.flops_estimation import build_flops_model

    hf_config = json.loads(
        (REPO / "configs/models/qwen3_5_35b_a3b.json").read_text()
    )
    model = build_flops_model(hf_config)

    tc = hf_config["text_config"]
    doubled = json.loads(json.dumps(hf_config))
    doubled["text_config"]["num_experts_per_tok"] = 2 * tc["num_experts_per_tok"]
    more_experts = json.loads(json.dumps(hf_config))
    more_experts["text_config"]["num_experts"] = 2 * tc["num_experts"]

    base = model.dense_per_token
    with_double_topk = build_flops_model(doubled).dense_per_token
    with_double_experts = build_flops_model(more_experts).dense_per_token

    from train.flops_estimation import FMA, FWD_BWD, _mlp_flops

    layers, dim = tc["num_hidden_layers"], tc["hidden_size"]
    # Going from top-8 to top-16 adds exactly one more set of 8 routed FFNs.
    routed = layers * tc["num_experts_per_tok"] * _mlp_flops(
        dim, tc["moe_intermediate_size"], swiglu=True
    )
    assert with_double_topk - base == routed
    # Twice the experts only widens the router GEMM (dim -> num_experts): the
    # experts a token does not visit cost nothing.
    assert with_double_experts - base == layers * FWD_BWD * FMA * dim * tc["num_experts"]
    assert with_double_experts < 1.01 * base


# --------------------------------------------------------------------------
# aux-loss-free load balancing (train.utils.register_moe_load_balancing_hook)


def _tiny_moe_model(load_balance_coeff=1e-3):
    """Two MoE blocks' worth of module tree, enough for the hook to walk."""
    import torch.nn as nn

    from models.common.module import ModuleDict
    from models.qwen3_5.configs import _qwen35_moe_config

    class FakeBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.moe_enabled = True
            self.moe = _qwen35_moe_config(
                dim=DIM,
                moe_hidden_dim=MOE_HIDDEN,
                num_experts=NUM_EXPERTS,
                top_k=TOP_K,
                shared_expert_hidden_dim=SHARED_HIDDEN,
                layer_id=0,
                load_balance_coeff=load_balance_coeff,
            ).build()

    class FakeModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = ModuleDict({"0": FakeBlock(), "1": FakeBlock()})

    return FakeModel()


@pytest.fixture(scope="module")
def _parallel_dims():
    """Single-rank ParallelDims with a real mesh.

    The hook calls `get_optional_mesh("loss")`, which builds the device mesh and
    so needs a process group even at world_size 1. gloo, because this fixture
    has no business taking a GPU.
    """
    import os

    from train.parallel.parallel_dims import ParallelDims

    created = not torch.distributed.is_initialized()
    if created:
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29577")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        torch.distributed.init_process_group("gloo")
    pd = ParallelDims(
        dp_replicate=1, dp_shard=1, cp=1, tp=1, pp=1, ep=1, world_size=1
    )
    pd.build_mesh()
    yield pd
    if created:
        torch.distributed.destroy_process_group()


def test_hook_moves_bias_toward_the_underused_experts(_parallel_dims):
    from train.utils import _update_expert_bias

    model = _tiny_moe_model()
    moe = model.layers["0"].moe
    assert moe.expert_bias_E is not None, "load_balance_coeff set => buffer exists"
    assert torch.equal(moe.expert_bias_E, torch.zeros(NUM_EXPERTS))

    # expert 0 starved, expert 7 hammered
    counts = torch.tensor([0.0, 10, 10, 10, 10, 10, 10, 100])
    for block in model.layers.children():
        block.moe.tokens_per_expert_E.copy_(counts)

    _update_expert_bias(model, _parallel_dims)

    bias = model.layers["0"].moe.expert_bias_E
    assert bias[0] > 0, "starved expert must be nudged up"
    assert bias[7] < 0, "hammered expert must be nudged down"
    # sign() * coeff, then mean-centred, so the magnitude is bounded by coeff
    assert bias.abs().max() <= 2e-3 + 1e-9
    # mean-free: the hook subtracts the mean so the bias never drifts as a whole
    assert abs(bias.mean().item()) < 1e-9
    # counters reset for the next step's window
    assert torch.equal(
        model.layers["0"].moe.tokens_per_expert_E, torch.zeros(NUM_EXPERTS)
    )
    # both layers updated, not just the first
    assert torch.equal(bias, model.layers["1"].moe.expert_bias_E)


def test_hook_is_not_registered_without_a_coefficient(_parallel_dims):
    from train.utils import register_moe_load_balancing_hook

    model = _tiny_moe_model(load_balance_coeff=None)
    assert model.layers["0"].moe.expert_bias_E is None
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
    assert register_moe_load_balancing_hook(optimizer, model, _parallel_dims) is False


def test_hook_is_a_noop_on_a_dense_model(_parallel_dims):
    """Every dense config goes through the same call site in train_qwen."""
    import torch.nn as nn

    from models.common.module import ModuleDict
    from train.utils import register_moe_load_balancing_hook

    class DenseBlock(nn.Module):
        pass

    model = nn.Module()
    model.layers = ModuleDict({"0": DenseBlock()})
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=0.0)
    assert register_moe_load_balancing_hook(optimizer, model, _parallel_dims) is False


def test_hook_fires_on_optimizer_step(_parallel_dims):
    """Registered as a step PRE-hook, so it must run once per optimizer.step(),
    not once per micro-batch."""
    from train.utils import register_moe_load_balancing_hook

    model = _tiny_moe_model()
    param = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.SGD([param], lr=0.0)
    assert register_moe_load_balancing_hook(optimizer, model, _parallel_dims) is True

    for block in model.layers.children():
        block.moe.tokens_per_expert_E.copy_(
            torch.tensor([0.0, 1, 1, 1, 1, 1, 1, 50])
        )
    param.grad = torch.zeros(1)
    optimizer.step()

    bias = model.layers["0"].moe.expert_bias_E
    assert bias.abs().sum() > 0, "step() must have run the pre-hook"


def test_expert_bias_does_not_break_an_hf_load():
    """`expert_bias_E` is persistent, so it is in `state_dict()`, and no HF
    checkpoint has it. Both load paths reconcile missing keys and would raise."""
    from models.qwen3_5.state_dict_adapter import is_local_only_state

    assert is_local_only_state("layers.0.moe.expert_bias_E")
    assert not is_local_only_state("layers.0.moe.router.gate.weight")
    assert not is_local_only_state("layers.0.moe.routed_experts.inner_experts.w1_EFD")

    from models.qwen3_5.configs import qwen35_config_from_hf

    config = qwen35_config_from_hf(
        REPO / "configs/models/qwen3_5_35b_a3b.json", seq_len=4096
    )
    with torch.device("meta"):
        model = config.build()
    state = model.state_dict()
    bias_keys = [k for k in state if k.endswith(".moe.expert_bias_E")]
    assert len(bias_keys) == 40, "one persistent bias per MoE layer"
    # the non-persistent counter must stay out of the checkpoint
    assert not [k for k in state if k.endswith(".moe.tokens_per_expert_E")]
    # and they are exactly the keys `to_hf` drops
    from models.qwen3_5.state_dict_adapter import Qwen35StateDictAdapter

    hf = Qwen35StateDictAdapter(config, None).to_hf(state)
    assert not [k for k in hf if "expert_bias" in k]


def test_load_balance_stats_measure_imbalance():
    """The numbers that reach wandb. Layer 0 is maximally lopsided, layer 1 is
    perfectly flat, so the aggregate must sit between them and the worst-layer
    index must point at 0."""
    from train.utils import _moe_load_balance_stats, MOE_STAT_NAMES

    counts = torch.tensor(
        [
            [0.0, 10, 10, 10, 10, 10, 10, 100],  # one expert hogging, one dead
            [5.0, 5, 5, 5, 5, 5, 5, 5],  # perfect
        ]
    )
    bias = torch.zeros(2, 8)
    bias[0, 0] = 0.04
    stats = dict(
        zip(MOE_STAT_NAMES, _moe_load_balance_stats(counts, bias).tolist(), strict=True)
    )

    # layer 0: mean 20, max 100 -> 5.0; layer 1: flat -> 1.0
    assert stats["max_over_mean_worst"] == pytest.approx(5.0)
    assert stats["max_over_mean"] == pytest.approx(3.0)
    assert stats["worst_layer"] == 0
    assert stats["dead_expert_frac"] == pytest.approx(1 / 16)
    assert stats["dead_expert_frac_worst"] == pytest.approx(1 / 8)
    assert stats["expert_bias_absmax"] == pytest.approx(0.04)
    assert stats["cv"] > 0

    # perfect balance everywhere is exactly 1.0, with no dead experts
    flat = _moe_load_balance_stats(torch.ones(4, 8), torch.zeros(4, 8)).tolist()
    flat = dict(zip(MOE_STAT_NAMES, flat, strict=True))
    assert flat["max_over_mean"] == pytest.approx(1.0)
    assert flat["max_over_mean_worst"] == pytest.approx(1.0)
    assert flat["cv"] == pytest.approx(0.0)
    assert flat["dead_expert_frac"] == 0.0


def test_a_skipped_step_does_not_divide_by_zero():
    """`train_step` skips optimizer.step() on a non-finite loss, but the hook can
    still see an all-zero counter (e.g. the very first step after a resume)."""
    from train.utils import _moe_load_balance_stats

    stats = _moe_load_balance_stats(torch.zeros(2, 8), torch.zeros(2, 8))
    assert torch.isfinite(stats).all(), stats


def test_stats_are_captured_before_the_counter_is_zeroed(_parallel_dims):
    """The ordering that makes this work at all: the hook zeroes
    `tokens_per_expert_E`, so the histogram has to be read inside it."""
    from train.utils import _update_expert_bias

    model = _tiny_moe_model()
    for block in model.layers.children():
        block.moe.tokens_per_expert_E.copy_(
            torch.tensor([0.0, 1, 1, 1, 1, 1, 1, 50])
        )

    stats: dict = {}
    _update_expert_bias(model, _parallel_dims, stats=stats)

    assert "vec" in stats
    assert torch.isfinite(stats["vec"]).all()
    # read from the pre-zero counts, not the zeroed buffer
    from train.utils import MOE_STAT_NAMES

    got = dict(zip(MOE_STAT_NAMES, stats["vec"].tolist(), strict=True))
    assert got["max_over_mean"] > 1.0, "zeroed counts would have given 1.0 or nan"
    assert got["dead_expert_frac"] == pytest.approx(1 / 8)
    assert torch.equal(
        model.layers["0"].moe.tokens_per_expert_E, torch.zeros(NUM_EXPERTS)
    )
