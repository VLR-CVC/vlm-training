"""Layer-wise parity of `models/qwen4` against `transformers`' Qwen4-Exp.

The released checkpoint (`Qwen/Qwen3.8-Flash-Next`) is ~335 GiB, far past what
fits on a dev box, so parity is checked module by module on a tiny random
config instead. Every module here is instantiated twice from the same config,
given the same weights, and fed the same input.

Run:
    pytest models/tests/test_qwen4_parity.py -v
"""
from __future__ import annotations

import dataclasses
import json
import math

import pytest
import torch
import torch.nn.functional as F

from transformers.models.qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

from models.qwen4.config import Qwen4TextConfig
from models.qwen4 import model as ours

TINY = {
    "vocab_size": 512,
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 32,
    "hidden_act": "silu",
    "max_position_embeddings": 4096,
    "rms_norm_eps": 1e-6,
    "tie_word_embeddings": False,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "eos_token_id": 1,
    "bos_token_id": 1,
    "layer_types": [
        "linear_attention", "linear_attention",
        "linear_attention", "qwen_sparse_attention",
    ],
    "linear_conv_kernel_dim": 4,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "output_gate_type": "sigmoid",
    "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32,
    "num_experts": 8,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "router_aux_loss_coef": 0.001,
    "hc_count": 2,
    "hc_lowrank": 16,
    "indexer_n_heads": 2,
    "indexer_kv_heads": 1,
    "indexer_head_dim": 16,
    "indexer_budget": 8,
    "indexer_compress_ratio": 2,
    "ple_layer_ids": [2],
    "ple_embed_dim": 32,
    "ple_conv_kernel_size": 4,
    "ngram_size": 3,
    "heads_per_ngram": 2,
    "ngram_vocab_size_base": 1000,
    "make_ngram_vocab_size_divisible_by": 128,
    "seed": 1234,
    "split_ngram_parts": 4,
    "rope_parameters": {
        "rope_type": "default",
        "rope_theta": 10000.0,
        "partial_rotary_factor": 0.5,
        "mrope_section": [3, 3, 2],
        "mrope_interleaved": True,
    },
}

SEQ = 24
ATOL = 2e-5
RTOL = 2e-5
# bf16 paths run through Triton kernels (fla, causal_conv1d) that do not
# accumulate in the same order as HF's; the repo's existing Qwen3.5 parity
# suite uses atol=0.5 / rtol=0.1 for whole-model logits. These are tighter,
# and still ~20x below the cross-document leak they are there to catch.
ATOL_BF16 = 0.15
RTOL_BF16 = 0.1


@pytest.fixture(scope="module")
def cfgs():
    return Qwen4ExpTextConfig(**TINY), Qwen4TextConfig.from_dict(TINY)


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


def randomize(module: torch.nn.Module) -> torch.nn.Module:
    """Give every float parameter/buffer a non-degenerate value.

    Qwen4 norms init to zeros and the router to zeros, so parity on the default
    init would pass even with the weights wired up wrong.
    """
    with torch.no_grad():
        for p in module.parameters():
            p.normal_(0.0, 0.05)
        for sub in module.modules():
            if hasattr(sub, "packed_table"):
                # Our n-gram table pads each head's row block out to a common
                # size (so it can be head-sharded); those pad rows are never
                # read. Zero them, or a round trip through the packed layout
                # is not bit-exact.
                sub.load_packed_table(sub.packed_table())
    return module


# Config-derived on both sides, so a mirror does not have to carry them. Ours
# are non-persistent because they differ per rank once the table is sharded.
_DERIVED_PLE_BUFFERS = frozenset(
    {"ngram_heads_vocab_sizes", "ngram_heads_offsets", "ngram_head_orders"}
)


def _is_derived(key: str) -> bool:
    return key.rsplit(".", 1)[-1] in _DERIVED_PLE_BUFFERS
_NGRAM_TABLE = "ngram_embedding.weight"


def _canonical(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """State dict with the n-gram table in the packed (HF/checkpoint) layout."""
    state = {
        k: v for k, v in module.state_dict().items()
        if not _is_derived(k)
    }
    for name, sub in module.named_modules():
        if hasattr(sub, "packed_table"):
            state[f"{name}.{_NGRAM_TABLE}".lstrip(".")] = sub.packed_table()
    return state


def mirror(src: torch.nn.Module, dst: torch.nn.Module) -> None:
    """Copy `src`'s state into `dst`, requiring identical keys and shapes."""
    ss, ds = _canonical(src), _canonical(dst)
    assert set(ss) == set(ds), (
        f"key mismatch\n  only in src: {sorted(set(ss) - set(ds))}"
        f"\n  only in dst: {sorted(set(ds) - set(ss))}"
    )
    for k in ss:
        assert ss[k].shape == ds[k].shape, f"{k}: {ss[k].shape} vs {ds[k].shape}"

    tables = {k: v for k, v in ss.items() if k.endswith(_NGRAM_TABLE)}
    missing, unexpected = dst.load_state_dict(
        {k: v for k, v in ss.items() if k not in tables}, strict=False
    )
    assert not unexpected, unexpected
    assert all(_is_derived(m) or m in tables for m in missing), missing

    with torch.no_grad():
        for key, packed in tables.items():
            owner = dst.get_submodule(key[: -len(_NGRAM_TABLE) - 1] or "")
            if hasattr(owner, "load_packed_table"):
                owner.load_packed_table(packed)
            else:
                owner.ngram_embedding.weight.copy_(packed)


def close(a: torch.Tensor, b: torch.Tensor, what: str, atol=ATOL, rtol=RTOL) -> None:
    a, b = a.float(), b.float()
    diff = (a - b).abs().max().item()
    assert torch.allclose(a, b, atol=atol, rtol=rtol), f"{what}: max abs diff {diff:.3e}"


def rope_cos_sin(hf_cfg, seq_len: int, batch: int = 1):
    """HF position embeddings plus the (cos, sin) our modules expect."""
    rotary = hf.Qwen4ExpTextRotaryEmbedding(hf_cfg)
    position_ids = torch.arange(seq_len).view(1, 1, -1).expand(3, batch, -1)
    cos, sin = rotary(torch.zeros(batch, seq_len, hf_cfg.hidden_size), position_ids)
    return (cos, sin)


# ---------------------------------------------------------------- norms

def test_rmsnorm_plain(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps))
    b = ours.OffsetRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
    mirror(a, b)
    x = torch.randn(1, SEQ, cfg.hidden_size)
    close(a(x), b(x), "RMSNorm")


def test_rmsnorm_grouped(cfgs):
    hf_cfg, cfg = cfgs
    dim = cfg.hc_count * cfg.hidden_size
    a = randomize(hf.Qwen4ExpTextRMSNorm(dim, group_size=cfg.hidden_size, eps=cfg.rms_norm_eps))
    b = ours.OffsetRMSNorm(dim, group_size=cfg.hidden_size, eps=cfg.rms_norm_eps)
    mirror(a, b)
    x = torch.randn(1, SEQ, dim)
    close(a(x), b(x), "grouped RMSNorm")


@pytest.mark.parametrize("activation", ["sigmoid", "silu"])
def test_rmsnorm_gated(cfgs, activation):
    _, cfg = cfgs
    dim = cfg.linear_value_head_dim
    a = randomize(hf.Qwen4ExpTextRMSNormGated(dim, eps=cfg.rms_norm_eps, activation=activation))
    b = ours.RMSNormGated(dim, eps=cfg.rms_norm_eps, activation=activation)
    mirror(a, b)
    x = torch.randn(1, SEQ, cfg.linear_num_value_heads, dim)
    g = torch.randn_like(x)
    close(a(x, g), b(x, g), f"RMSNormGated({activation})")


# ---------------------------------------------------- hyper-connections

def test_gated_residual_combine(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=True))
    b = ours.GatedResidual(cfg, use_combine=True)
    mirror(a, b)
    x = torch.randn(1, SEQ, cfg.hc_count * cfg.hidden_size)
    ma, ha, ia = a(x)
    mb, hb, ib = b(x)
    close(ma, mb, "GatedResidual mixed")
    close(ha, hb, "GatedResidual hyper_input")
    close(ia, ib, "GatedResidual inject")


def test_gated_residual_mixer(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=False))
    b = ours.GatedResidual(cfg, use_combine=False)
    mirror(a, b)
    x = torch.randn(1, SEQ, cfg.hc_count * cfg.hidden_size)
    close(a(x), b(x), "hyper_connection_mixer")


def test_gated_residual_width_guard(cfgs):
    _, cfg = cfgs
    b = ours.GatedResidual(cfg)
    with pytest.raises(ValueError):
        b(torch.randn(1, SEQ, cfg.hidden_size))


# ------------------------------------------------------------------ MoE

def test_router(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextTopKRouter(hf_cfg))
    b = ours.TopKRouter(cfg)
    mirror(a, b)
    x = torch.randn(SEQ, cfg.hidden_size)
    la, wa, ia = a(x)
    lb, wb, ib = b(x)
    close(la, lb, "router logits")
    close(wa, wb, "router weights")
    assert torch.equal(ia, ib), "router picked different experts"


def test_experts_loop(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextExperts(hf_cfg))
    b = ours.Experts(cfg)
    mirror(a, b)
    x = torch.randn(SEQ, cfg.hidden_size)
    idx = torch.randint(0, cfg.num_experts, (SEQ, cfg.num_experts_per_tok))
    w = torch.rand(SEQ, cfg.num_experts_per_tok)
    w = w / w.sum(-1, keepdim=True)
    close(a(x, idx, w), b._forward_loop(x, idx, w), "experts")


def test_moe_block(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextSparseMoeBlock(hf_cfg))
    b = ours.SparseMoeBlock(cfg)
    mirror(a, b)
    x = torch.randn(1, SEQ, cfg.hidden_size)
    close(a(x), b(x), "SparseMoeBlock")


@pytest.mark.cuda_only
@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_experts_grouped_matches_loop(cfgs):
    """The training path (`torch._grouped_mm`) must match the reference loop."""
    hf_cfg, cfg = cfgs
    b = randomize(ours.Experts(cfg)).cuda().to(torch.bfloat16)
    x = torch.randn(SEQ, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    idx = torch.randint(0, cfg.num_experts, (SEQ, cfg.num_experts_per_tok), device="cuda")
    w = torch.rand(SEQ, cfg.num_experts_per_tok, device="cuda", dtype=torch.bfloat16)
    w = w / w.sum(-1, keepdim=True)
    close(b._forward_loop(x, idx, w), b._forward_grouped(x, idx, w),
          "grouped vs loop experts", atol=2e-2, rtol=2e-2)


# ------------------------------------------------------------------ PLE

def test_ngram_embedding(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextNGramEmbedding(hf_cfg, cfg.ple_embed_dim, layer_idx=1,
                                                ple_layer_index=0))
    b = ours.NGramEmbedding(cfg, cfg.ple_embed_dim, ple_layer_index=0)
    mirror(a, b)
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    ids[0, SEQ // 2] = cfg.eos_token_id  # exercise the segment reset
    close(a(ids, None), b(ids), "NGramEmbedding")


@pytest.mark.parametrize("tp_size", [2, 4])
def test_ngram_head_shard_reassembles(cfgs, tp_size):
    """The head shards must concatenate into the unsharded table's output.

    The n-gram table is block-diagonal in the head: head `h` owns its own
    vocabulary rows and lands in exactly its own `head_dim` columns of the
    flattened output. That makes a shard on the flat head index row- and
    column-parallel at once, so `cat(shards, dim=-1)` is the unsharded result
    with no permutation and no collective in the gather. This test is what
    pins that; `_shard_ple` only declares the layout.
    """
    _, cfg = cfgs
    heads = (cfg.ngram_size - 1) * cfg.heads_per_ngram
    if heads % tp_size:
        pytest.skip(f"{heads} n-gram heads is not divisible by tp={tp_size}")

    whole = ours.NGramEmbedding(cfg, cfg.ple_embed_dim, ple_layer_index=0)
    packed = torch.randn(whole.packed_rows, whole.ngram_embedding.embedding_dim)
    whole.load_packed_table(packed)

    ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    ids[0, SEQ // 2] = cfg.eos_token_id
    reference = whole(ids)

    shards = []
    for rank in range(tp_size):
        sharded = dataclasses.replace(cfg, ple_tp_size=tp_size, ple_tp_rank=rank)
        part = ours.NGramEmbedding(sharded, cfg.ple_embed_dim, ple_layer_index=0)
        # the global row count must not depend on the TP size, or a checkpoint
        # could not be reloaded at a different TP width
        assert part.rows_per_head == whole.rows_per_head
        assert part.ngram_embedding.num_embeddings * tp_size == (
            whole.ngram_embedding.num_embeddings
        )
        part.load_packed_table(packed)
        shards.append(part(ids))

    assert torch.equal(reference, torch.cat(shards, dim=-1))


def test_ple_layer(cfgs):
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextPLELayer(hf_cfg, layer_idx=1, ple_layer_index=0))
    b = ours.PLELayer(cfg, ple_layer_index=0)
    mirror(a, b)
    x = torch.randn(1, SEQ, cfg.hc_count * cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    ids[0, SEQ // 2] = cfg.eos_token_id
    close(a(x, ids, None, conv_mask=None), b(x, ids), "PLELayer")


def test_ple_conv_is_document_local():
    """The dilated PLE conv must not read across a packed-document boundary."""
    cfg = Qwen4TextConfig.from_dict(TINY)
    b = randomize(ours.PLELayer(cfg, ple_layer_index=0))
    lens = [17, 23]
    total = sum(lens)

    ids = torch.randint(0, cfg.vocab_size, (1, total))
    ids[0, lens[0] - 1] = cfg.eos_token_id
    ids[0, total - 1] = cfg.eos_token_id
    x = torch.randn(1, total, cfg.hc_count * cfg.hidden_size)
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32)

    packed = b(x, ids, cu)
    lo = 0
    for n in lens:
        solo = b(x[:, lo : lo + n], ids[:, lo : lo + n])
        close(packed[:, lo : lo + n], solo, f"PLE packed doc @{lo}")
        lo += n


def test_ple_conv_single_document_unchanged():
    """With one document the fix-up must be a no-op (matches HF exactly)."""
    cfg = Qwen4TextConfig.from_dict(TINY)
    b = randomize(ours.PLELayer(cfg, ple_layer_index=0))
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ))
    x = torch.randn(1, SEQ, cfg.hc_count * cfg.hidden_size)
    cu = torch.tensor([0, SEQ], dtype=torch.int32)
    close(b(x, ids, cu), b(x, ids, None), "PLE single document")


# ------------------------------------------------------------------ QSA

def _hf_causal_mask(seq_len: int) -> torch.Tensor:
    """The 4D bool mask HF's indexer consumes (sdpa flavour)."""
    idx = torch.arange(seq_len)
    return (idx[None, :] <= idx[:, None]).view(1, 1, seq_len, seq_len)


def _reference_qsa(indexer, cfg, x, cos, sin, seq_len):
    """Direct port of HF's per-(batch, query) loop, kept as the test's own oracle.

    Returns ``(attend, scores)``: the boolean (seq, seq) attend matrix and the
    per-query block scores with ``-inf`` on ineligible blocks.
    """
    ratio = cfg.indexer_compress_ratio
    dim = cfg.indexer_head_dim
    block_topk = cfg.indexer_budget // ratio
    qk = indexer.index_qk_proj(x)
    q, token_k = torch.split(
        qk, [cfg.indexer_n_heads * dim, cfg.indexer_kv_heads * dim], dim=-1
    )
    q = indexer.q_layernorm(q.reshape(1, seq_len, -1, dim))
    q = hf.apply_rotary_pos_emb(q, cos=cos, sin=sin, unsqueeze_dim=2)[0]  # (S, H, D)
    raw = token_k.reshape(seq_len, dim)

    max_blocks = seq_len // ratio
    attend = torch.zeros(seq_len, seq_len, dtype=torch.bool)
    scores = torch.full((seq_len, max_blocks), float("-inf"))

    for t in range(seq_len):
        n_complete = (t + 1) // ratio
        if n_complete:
            bti = torch.arange(n_complete * ratio).view(n_complete, ratio)
            groups = raw.index_select(0, bti.flatten()).view(n_complete, ratio, dim)
            pooled = indexer.k_layernorm(groups.float().mean(1).to(raw.dtype))
            starts = bti[:, 0]
            keys = hf.apply_rotary_pos_emb(
                pooled.unsqueeze(1),
                cos=cos[0].index_select(0, starts),
                sin=sin[0].index_select(0, starts),
            ).squeeze(1)
            row = (
                torch.matmul(q[t].float(), keys.float().transpose(-1, -2))
                .transpose(-1, -2)
                .relu()
                .sum(dim=-1)
                / math.sqrt(dim)
            )
            scores[t, :n_complete] = row
            chosen = row.topk(min(block_topk, n_complete)).indices
            attend[t, bti.index_select(0, chosen).flatten()] = True
        attend[t, n_complete * ratio : t + 1] = True
    return attend, scores


def _selected_blocks(indexer, x, cos, sin, seq_len):
    cu = torch.tensor([0, seq_len], dtype=torch.int32)
    seg = torch.zeros(seq_len, dtype=torch.long)
    selected, block_of_token, tail_start = indexer(x, cos, sin, cu, seg)
    q_idx = torch.arange(seq_len).unsqueeze(1)
    kv_idx = torch.arange(seq_len).unsqueeze(0)
    attend = (
        selected[q_idx, block_of_token[kv_idx]] | (kv_idx >= tail_start[q_idx])
    ) & (kv_idx <= q_idx)
    return attend, selected


@pytest.mark.parametrize("seq_len", [24, 256])
def test_qsa_selection_is_a_valid_topk(cfgs, seq_len):
    """Our vectorized selection must be *a* top-k under HF's own scores.

    Exact index equality is not a well-posed assertion: candidate blocks
    routinely tie at a score of 0 (every head's dot product is negative, so
    relu zeroes them), and neither `topk` implementation defines which of the
    tied blocks wins. What both must agree on is the score multiset — that is
    what makes a selection a valid top-k.
    """
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextQSAIndexer(hf_cfg, layer_idx=3))
    b = ours.QSAIndexer(cfg)
    mirror(a, b)

    x = torch.randn(1, seq_len, cfg.hidden_size)
    cos, sin = rope_cos_sin(hf_cfg, seq_len)

    with torch.no_grad():
        ref_attend, scores = _reference_qsa(a, cfg, x, cos, sin, seq_len)
        got_attend, selected = _selected_blocks(b, x, cos, sin, seq_len)

    assert torch.equal(
        ref_attend.sum(dim=1), got_attend.sum(dim=1)
    ), "different number of attended tokens per query"

    ratio = cfg.indexer_compress_ratio
    for t in range(seq_len):
        n_complete = (t + 1) // ratio
        if n_complete == 0:
            continue
        ref_blocks = ref_attend[t, : n_complete * ratio].nonzero().flatten() // ratio
        ref_blocks = torch.unique(ref_blocks)
        got_blocks = selected[t].nonzero().flatten()
        assert ref_blocks.numel() == got_blocks.numel(), f"query {t}: block count"
        ref_s = scores[t, ref_blocks].sort(descending=True).values
        got_s = scores[t, got_blocks].sort(descending=True).values
        close(ref_s, got_s, f"query {t} selected block scores")


@pytest.mark.parametrize("seq_len", [24, 256])
def test_qsa_selection_beats_every_rejected_block(cfgs, seq_len):
    """No rejected eligible block may outscore a selected one."""
    hf_cfg, cfg = cfgs
    a = randomize(hf.Qwen4ExpTextQSAIndexer(hf_cfg, layer_idx=3))
    b = ours.QSAIndexer(cfg)
    mirror(a, b)
    x = torch.randn(1, seq_len, cfg.hidden_size)
    cos, sin = rope_cos_sin(hf_cfg, seq_len)
    with torch.no_grad():
        _, scores = _reference_qsa(a, cfg, x, cos, sin, seq_len)
        _, selected = _selected_blocks(b, x, cos, sin, seq_len)

    for t in range(seq_len):
        picked = selected[t]
        if not picked.any():
            continue
        row = scores[t, : picked.numel()]
        eligible = torch.isfinite(row)
        rejected = eligible & ~picked
        if not rejected.any():
            continue
        assert row[picked].min() >= row[rejected].max() - ATOL, (
            f"query {t}: rejected block scores above a selected one"
        )


def test_qsa_selection_packed_matches_unpacked(cfgs):
    """Packing two documents into one row must not change either's selection."""
    _, cfg = cfgs
    b = randomize(ours.QSAIndexer(cfg))
    hf_cfg = Qwen4ExpTextConfig(**TINY)

    lens = [10, 14]
    total = sum(lens)
    x = torch.randn(1, total, cfg.hidden_size)
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32)
    seg = torch.cat([torch.zeros(lens[0], dtype=torch.long),
                     torch.ones(lens[1], dtype=torch.long)])

    # positions restart per document, exactly as the model builds them
    pos = torch.cat([torch.arange(lens[0]), torch.arange(lens[1])])
    from models.qwen4.utils import mrope_cos_sin
    inv = 1.0 / (10000.0 ** (torch.arange(0, 16, 2, dtype=torch.float32) / 16))
    cos, sin = mrope_cos_sin(inv, pos.view(1, 1, -1).expand(3, 1, -1), cfg.rope_parameters["mrope_section"])

    sel_p, blk_p, tail_p = b(x, cos, sin, cu, seg)

    for d, (lo, hi) in enumerate(zip([0, lens[0]], [lens[0], total])):
        n = hi - lo
        cos_d, sin_d = cos[:, lo:hi], sin[:, lo:hi]
        cu_d = torch.tensor([0, n], dtype=torch.int32)
        sel_d, blk_d, tail_d = b(x[:, lo:hi], cos_d, sin_d, cu_d, torch.zeros(n, dtype=torch.long))

        qi = torch.arange(n).unsqueeze(1)
        ki = torch.arange(n).unsqueeze(0)
        alone = (sel_d[qi, blk_d[ki]] | (ki >= tail_d[qi])) & (ki <= qi)

        qp = torch.arange(lo, hi).unsqueeze(1)
        kp = torch.arange(lo, hi).unsqueeze(0)
        packed = (
            (sel_p[qp, blk_p[kp]] | (kp >= tail_p[qp]))
            & (kp <= qp)
            & (seg[qp] == seg[kp])
        )
        assert torch.equal(alone, packed), f"document {d} selection changed under packing"


def test_qsa_attention_output(cfgs):
    """Attention math parity, holding the selected mask fixed.

    The mask itself is covered by the selection tests above; here both sides
    are handed the *same* mask so any difference is pure attention arithmetic
    (flex-attention with a block mask vs HF's eager softmax).
    """
    hf_cfg, cfg = cfgs
    hf_cfg = Qwen4ExpTextConfig(**{**TINY})
    hf_cfg._attn_implementation = "eager"
    seq_len = 256

    a = randomize(hf.Qwen4ExpTextAttention(hf_cfg, layer_idx=3))
    b = ours.SelfAttention(cfg)
    mirror(a, b)

    x = torch.randn(1, seq_len, cfg.hidden_size)
    cos, sin = rope_cos_sin(hf_cfg, seq_len)
    cu = torch.tensor([0, seq_len], dtype=torch.int32)
    seg = torch.zeros(seq_len, dtype=torch.long)

    with torch.no_grad():
        attend, _ = _selected_blocks(b.indexer, x, cos, sin, seq_len)
        float_mask = torch.zeros(1, 1, seq_len, seq_len).masked_fill(
            ~attend, torch.finfo(torch.float32).min
        )
        # HF's attention re-derives the mask through its own indexer; feed it a
        # mask that already encodes our selection so the indexer is a no-op.
        a.indexer = _FrozenIndexer(attend)
        out_hf, _ = a(x, (cos, sin), attention_mask=float_mask)
        out_ours = b(x, cos, sin, cu, seq_len, seg)

    close(out_hf, out_ours, "QSA attention output", atol=1e-4, rtol=1e-4)


class _FrozenIndexer(torch.nn.Module):
    """Stand-in that returns a pre-computed selection mask."""

    def __init__(self, attend: torch.Tensor):
        super().__init__()
        self.register_buffer("attend", attend, persistent=False)

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values):
        return torch.zeros_like(attention_mask).masked_fill(
            ~self.attend, torch.finfo(attention_mask.dtype).min
        )


# ------------------------------------------------- linear attention (CUDA)

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fla / causal_conv1d are Triton kernels"
)


@pytest.mark.cuda_only
@cuda_only
def test_gated_delta_net(cfgs):
    hf_cfg, cfg = cfgs
    seq_len = 256
    a = randomize(hf.Qwen4ExpTextGatedDeltaNet(hf_cfg, layer_idx=0))
    b = ours.GatedDeltaNet(cfg)
    mirror(a, b)
    a, b = a.cuda().to(torch.bfloat16), b.cuda().to(torch.bfloat16)

    x = torch.randn(1, seq_len, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        close(a(x), b(x, cu_seqlens=cu), "GatedDeltaNet", atol=2e-3, rtol=2e-2)


@pytest.mark.cuda_only
@cuda_only
def test_gated_delta_net_respects_packing(cfgs):
    """Packed documents must not leak state across the cu_seqlens boundary."""
    _, cfg = cfgs
    lens = [96, 160]
    total = sum(lens)
    b = randomize(ours.GatedDeltaNet(cfg)).cuda().to(torch.bfloat16)
    x = torch.randn(1, total, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        packed = b(x, cu_seqlens=cu)
        lo = 0
        for n in lens:
            solo = b(
                x[:, lo : lo + n],
                cu_seqlens=torch.tensor([0, n], dtype=torch.int32, device="cuda"),
            )
            close(packed[:, lo : lo + n], solo, f"GDN packed doc @{lo}",
                  atol=2e-3, rtol=2e-2)
            lo += n


# ------------------------------------------------------- whole decoder stack

def _capture_selections(monkeypatch):
    """Record every QSAIndexer output produced during the next forward."""
    captured: list[tuple] = []
    original = ours.QSAIndexer.forward

    def spy(self, x, cos, sin, cu_seqlens, seg_id, query_chunk=1024):
        out = original(self, x, cos, sin, cu_seqlens, seg_id, query_chunk)
        captured.append(out)
        return out

    monkeypatch.setattr(ours.QSAIndexer, "forward", spy)
    return captured


def _attend_from(selection, seq_len):
    selected, block_of_token, tail_start = selection
    q_idx = torch.arange(seq_len, device=selected.device).unsqueeze(1)
    kv_idx = torch.arange(seq_len, device=selected.device).unsqueeze(0)
    return (
        selected[q_idx, block_of_token[kv_idx]] | (kv_idx >= tail_start[q_idx])
    ) & (kv_idx <= q_idx)


@pytest.mark.cuda_only
@cuda_only
def test_text_stack(cfgs, monkeypatch):
    """Full 4-layer text stack: linear attention, PLE, MoE, hyper-connections, QSA.

    The sparse layer's block choice is pinned to ours on both sides (selection
    itself is covered by the QSA tests) so this compares only the arithmetic.
    """
    hf_cfg, cfg = cfgs
    hf_cfg = Qwen4ExpTextConfig(**TINY)
    hf_cfg._attn_implementation = "eager"
    seq_len = 256

    a = randomize(hf.Qwen4ExpTextModel(hf_cfg))
    b = ours.LanguageModel(cfg)
    mirror(a, b)
    a, b = a.cuda().to(torch.bfloat16).eval(), b.cuda().to(torch.bfloat16).eval()

    ids = torch.randint(0, cfg.vocab_size, (1, seq_len), device="cuda")
    ids[0, seq_len // 3] = cfg.eos_token_id
    cos, sin = rope_cos_sin(hf_cfg, seq_len)
    cos, sin = cos.cuda().to(torch.bfloat16), sin.cuda().to(torch.bfloat16)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")

    captured = _capture_selections(monkeypatch)
    with torch.no_grad():
        embeds = b.embed_tokens(ids)
        out_ours = b(embeds, cos, sin, cu, seq_len, ple_input_ids=ids)

    assert len(captured) == sum(t == "qwen_sparse_attention" for t in cfg.layer_types)
    it = iter(captured)
    for layer in a.layers:
        if hasattr(layer, "self_attn"):
            layer.self_attn.indexer = _FrozenIndexer(_attend_from(next(it), seq_len))

    with torch.no_grad():
        out_hf = a(input_ids=ids, use_cache=False).last_hidden_state

    close(out_hf, out_ours, "text stack", atol=ATOL_BF16, rtol=RTOL_BF16)


@pytest.mark.cuda_only
@cuda_only
def test_text_stack_packing_is_isolated(cfgs):
    """Two documents packed into one row must match two separate forwards."""
    _, cfg = cfgs
    # fp32, not bf16: packing isolation is an *exact* property, and in fp32 it
    # holds to 0. In bf16 the randomly-initialized PLE injection amplifies
    # rounding into the 0.3 range on some draws, which turns the test into a
    # tolerance-tuning exercise rather than a check that documents are isolated.
    lens = [96, 160]
    total = sum(lens)
    b = randomize(ours.LanguageModel(cfg)).cuda().float().eval()

    ids = torch.randint(0, cfg.vocab_size, (1, total), device="cuda")
    ids[0, lens[0] - 1] = cfg.eos_token_id  # documents end with EOS when packed
    ids[0, total - 1] = cfg.eos_token_id

    from models.qwen4.utils import mrope_cos_sin

    rope_dim = int(cfg.head_dim * cfg.rope_parameters["partial_rotary_factor"])
    inv = 1.0 / (
        cfg.rope_parameters["rope_theta"]
        ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device="cuda") / rope_dim)
    )
    pos = torch.cat([torch.arange(n, device="cuda") for n in lens])
    cos, sin = mrope_cos_sin(
        inv, pos.view(1, 1, -1).expand(3, 1, -1), cfg.rope_parameters["mrope_section"]
    )
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32, device="cuda")

    with torch.no_grad():
        packed = b(b.embed_tokens(ids), cos, sin, cu, max(lens), ple_input_ids=ids)
        lo = 0
        for n in lens:
            ids_d = ids[:, lo : lo + n]
            solo = b(
                b.embed_tokens(ids_d),
                cos[:, lo : lo + n],
                sin[:, lo : lo + n],
                torch.tensor([0, n], dtype=torch.int32, device="cuda"),
                n,
                ple_input_ids=ids_d,
            )
            close(packed[:, lo : lo + n], solo, f"text stack packed doc @{lo}",
                  atol=1e-5, rtol=0.0)
            lo += n


TINY_VISION = {
    "depth": 2,
    "hidden_size": 32,
    "intermediate_size": 64,
    "num_heads": 2,
    "in_channels": 3,
    "patch_size": 4,
    "temporal_patch_size": 2,
    "spatial_merge_size": 2,
    "num_position_embeddings": 64,
    "out_hidden_size": 64,
    "hidden_act": "gelu_pytorch_tanh",
}


def _tiny_full_config(tmp_path):
    from models.qwen4.config import Qwen4Config

    raw = {
        "text_config": TINY,
        "vision_config": TINY_VISION,
        "image_token_id": 500,
        "video_token_id": 501,
        "vision_start_token_id": 502,
        "vision_end_token_id": 503,
        "tie_word_embeddings": False,
        "torch_dtype": "bfloat16",
    }
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw))
    return Qwen4Config.from_json(path)


@pytest.mark.cuda_only
@cuda_only
def test_end_to_end_forward_backward(tmp_path):
    """Whole model: loss backward, gradients where we expect them and nowhere else."""
    cfg = _tiny_full_config(tmp_path)
    model = randomize(ours.Qwen4ForCausalLM(cfg)).cuda().to(torch.bfloat16)

    lens = [64, 64]
    total = sum(lens)
    ids = torch.randint(0, cfg.text.vocab_size, (1, total), device="cuda")
    ids[0, lens[0] - 1] = cfg.text.eos_token_id
    ids[0, total - 1] = cfg.text.eos_token_id
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32, device="cuda")
    labels = ids.clone()

    out = model(ids, attention_mask=cu, labels=labels)
    assert torch.isfinite(out.loss), "loss is not finite"
    out.loss.backward()

    no_grad, bad = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.startswith("model.visual."):
            continue  # text-only input, the vision tower never runs
        if p.grad is None:
            no_grad.append(name)
        elif not torch.isfinite(p.grad).all():
            bad.append(name)

    assert not bad, f"non-finite grads: {bad[:5]}"
    # the QSA indexer is the one part that legitimately gets nothing
    assert all(".indexer." in n for n in no_grad), (
        f"unexpected parameters without gradients: "
        f"{[n for n in no_grad if '.indexer.' not in n][:8]}"
    )
    assert any(".indexer." in n for n in no_grad), "indexer unexpectedly received gradients"


@pytest.mark.cuda_only
@cuda_only
def test_end_to_end_with_images(tmp_path):
    """Image tokens are replaced by merged vision features and reach the loss."""
    cfg = _tiny_full_config(tmp_path)
    model = randomize(ours.Qwen4ForCausalLM(cfg)).cuda().to(torch.bfloat16)

    v = cfg.vision
    grid = torch.tensor([[1, 4, 4]], device="cuda")           # t, h, w
    n_patches = int(grid.prod(dim=1).sum())
    n_tokens = n_patches // (v.spatial_merge_size ** 2)
    patch_dim = v.in_channels * v.temporal_patch_size * v.patch_size ** 2
    pixels = torch.randn(n_patches, patch_dim, device="cuda", dtype=torch.bfloat16)

    total = 64
    ids = torch.randint(4, cfg.text.vocab_size, (1, total), device="cuda")
    ids[0, 8 : 8 + n_tokens] = cfg.image_token_id
    ids[0, total - 1] = cfg.text.eos_token_id
    cu = torch.tensor([0, total], dtype=torch.int32, device="cuda")

    out = model(ids, pixel_values=pixels, image_grid_thw=grid, labels=ids.clone())
    assert torch.isfinite(out.loss)
    out.loss.backward()

    vision_grads = [
        n for n, p in model.model.visual.named_parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert vision_grads, "vision tower received no gradients"


# ------------------------------------------- checkpoint round-trip

def test_checkpoint_roundtrip(tmp_path):
    """Save a model the way `utils/make_qwen4.py` does, load it back.

    Two things this pins that a forward test would not. The PLE hashing
    buffers are integers -- `layer_multipliers` holds 64-bit splitmix
    constants -- and casting them to the model dtype on load silently
    destroys them, after which the n-gram gather indexes out of bounds. And
    the n-gram table is written as `split_ngram_parts` shards, so the load
    goes through the concatenating branch rather than a 1:1 key match.
    """
    from safetensors.torch import save_file

    from utils.make_qwen4 import shard_ple_tables

    cfg = _tiny_full_config(tmp_path)      # also writes tmp_path/config.json
    model = randomize(ours.Qwen4ForCausalLM(cfg)).to(torch.bfloat16)
    before = {k: v.clone() for k, v in model.state_dict().items()}

    ints = [k for k, v in before.items() if not v.is_floating_point()]
    assert ints, "expected the PLE hashing buffers in the state dict"

    state = shard_ple_tables(before, cfg.text.split_ngram_parts, model)
    shards = [k for k in state if ".ngram_embedding.shard_" in k]
    assert len(shards) == cfg.text.split_ngram_parts

    save_file(
        {k: v.contiguous() for k, v in state.items()},
        str(tmp_path / "model.safetensors"),
        metadata={"format": "pt"},
    )
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in state}})
    )

    loaded, _ = ours.Qwen4ForCausalLM.from_pretrained(
        tmp_path, dtype=torch.bfloat16, device="cpu"
    )
    after = loaded.state_dict()

    assert set(after) == set(before)
    for k in before:
        assert after[k].dtype == before[k].dtype, f"{k} changed dtype"
        assert torch.equal(after[k], before[k]), f"{k} did not survive the round trip"
