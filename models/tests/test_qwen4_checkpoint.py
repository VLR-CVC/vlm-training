"""Parity against the *real* Qwen4-Exp weights, one decoder layer at a time.

`Qwen/Qwen3.8-Flash-Next` is ~335 GiB, so it never gets instantiated whole.
Instead each test builds a one-layer config, materializes a single
`DecoderLayer` (~5 GiB for the 512 stacked experts), loads that layer's tensors
straight out of the safetensors shards, and compares against the same layer
built from `transformers`.

Set `QWEN4_SNAPSHOT` to point elsewhere. Tests skip cleanly when the snapshot,
or the particular shards a layer lives in, have not been downloaded yet.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

import pytest
import torch
from safetensors.torch import safe_open

from transformers.models.qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp import modeling_qwen4_exp as hf

from models.qwen4.config import Qwen4Config, Qwen4TextConfig
from models.qwen4 import model as ours
from models.tests.test_qwen4_parity import (
    close, mirror, rope_cos_sin, _FrozenIndexer, _attend_from,
)

SNAPSHOT = Path(os.environ.get("QWEN4_SNAPSHOT", "/data/151-2/users/tockier/models/qwen4"))
SEQ = 256
ATOL_BF16 = 0.15
RTOL_BF16 = 0.1

pytestmark = pytest.mark.skipif(
    not (SNAPSHOT / "model.safetensors.index.json").exists(),
    reason=f"no Qwen4 snapshot (or still downloading) at {SNAPSHOT}",
)


def _index() -> dict[str, str]:
    with open(SNAPSHOT / "model.safetensors.index.json") as f:
        return json.load(f)["weight_map"]


def _tensors_with_prefix(
    prefix: str, dtype=torch.bfloat16, rename=None, skip=None
) -> dict[str, torch.Tensor]:
    """Read every checkpoint tensor under `prefix`, keyed relative to it.

    `rename` maps the relative key to the key the module expects (return
    `None` to drop it); `skip` is a predicate on the relative key applied
    before anything is read off disk.
    """
    weight_map = _index()
    wanted = {k: v for k, v in weight_map.items() if k.startswith(prefix)}
    assert wanted, f"nothing under {prefix!r} in the checkpoint index"
    if skip is not None:
        wanted = {k: v for k, v in wanted.items() if not skip(k[len(prefix):])}

    missing = sorted({s for s in wanted.values() if not (SNAPSHOT / s).exists()})
    if missing:
        pytest.skip(f"{prefix} shards not downloaded yet ({len(missing)} missing)")

    by_shard: dict[str, list[str]] = {}
    for k, shard in wanted.items():
        by_shard.setdefault(shard, []).append(k)

    out = {}
    for shard, keys in by_shard.items():
        with safe_open(str(SNAPSHOT / shard), framework="pt", device="cpu") as f:
            for k in keys:
                rel = k[len(prefix):]
                if rename is not None:
                    rel = rename(rel)
                    if rel is None:
                        continue
                t = f.get_tensor(k)
                out[rel] = t.to(dtype) if t.is_floating_point() else t
    return out


def _layer_tensors(layer_idx: int, dtype=torch.bfloat16) -> dict[str, torch.Tensor]:
    """Read one decoder layer's tensors, keyed relative to the layer."""
    return _tensors_with_prefix(f"model.language_model.layers.{layer_idx}.", dtype)


def _one_layer_configs(layer_idx: int):
    """A real config narrowed to a single layer, in both dialects."""
    with open(SNAPSHOT / "config.json") as f:
        raw = json.load(f)
    tc = copy.deepcopy(raw["text_config"])
    tc["layer_types"] = [tc["layer_types"][layer_idx]]
    tc["num_hidden_layers"] = 1
    tc["ple_layer_ids"] = []          # the PLE table alone is ~90 GiB
    tc.pop("mtp", None)
    hf_cfg = Qwen4ExpTextConfig(**tc)
    hf_cfg._attn_implementation = "eager"
    cfg = Qwen4TextConfig.from_dict(tc)
    # released checkpoints label the indexed layers "full_attention"; both
    # config classes normalize that to "qwen_sparse_attention"
    return hf_cfg, cfg, cfg.layer_types[0]


def _layers_of_type(layer_type: str) -> list[int]:
    cfg = Qwen4Config.from_json(SNAPSHOT / "config.json")
    return [i for i, t in enumerate(cfg.text.layer_types) if t == layer_type]


# -------------------------------------------------------------- config only

def test_config_matches_checkpoint_state_dict():
    """Every parameter our model declares must exist in the checkpoint."""
    cfg = Qwen4Config.from_json(SNAPSHOT / "config.json")
    with torch.device("meta"):
        model = ours.Qwen4ForCausalLM(cfg)

    import re

    shard_re = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")
    ckpt = set()
    for k in _index():
        m = shard_re.match(k)
        ckpt.add(f"{m.group(1)}.weight" if m else k)

    ours_keys = set(model.state_dict())
    assert not (ours_keys - ckpt), f"not in checkpoint: {sorted(ours_keys - ckpt)[:10]}"

    # The checkpoint tensors we do not declare are the MTP head, which
    # `transformers` does not implement either, and the PLE per-head sizes and
    # offsets, which we derive from the config instead of loading (they differ
    # per rank once the table is head-sharded, and a plain buffer that differs
    # per rank is not safely checkpointable).
    extra = ckpt - ours_keys
    allowed = (".ngram_heads_vocab_sizes", ".ngram_heads_offsets")
    extra = {k for k in extra if not k.endswith(allowed)}
    assert all(k.startswith("mtp.") for k in extra), (
        f"unexpected checkpoint tensors: {sorted(k for k in extra if not k.startswith('mtp.'))[:10]}"
    )


def test_config_matches_checkpoint_shapes():
    """Shapes too, read from the safetensors headers without loading tensors."""
    cfg = Qwen4Config.from_json(SNAPSHOT / "config.json")
    with torch.device("meta"):
        model = ours.Qwen4ForCausalLM(cfg)
    state = {k: tuple(v.shape) for k, v in model.state_dict().items()}

    weight_map = _index()
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        if k in state and (SNAPSHOT / shard).exists():
            by_shard.setdefault(shard, []).append(k)
    if not by_shard:
        pytest.skip("no shards downloaded yet")

    checked = 0
    for shard, keys in by_shard.items():
        with safe_open(str(SNAPSHOT / shard), framework="pt", device="cpu") as f:
            for k in keys:
                got = tuple(f.get_slice(k).get_shape())
                assert got == state[k], f"{k}: checkpoint {got} vs model {state[k]}"
                checked += 1
    assert checked > 0


# ------------------------------------------------------- real-weight layers

cuda_only = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fla / causal_conv1d are Triton kernels"
)


def _run_layer_parity(layer_idx: int):
    hf_cfg, cfg, layer_type = _one_layer_configs(layer_idx)
    tensors = _layer_tensors(layer_idx)

    # build directly in bf16: the 512 stacked experts are ~5 GiB per layer and
    # an fp32 materialization would double that for no benefit
    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        a = hf.Qwen4ExpTextDecoderLayer(hf_cfg, layer_idx=0)
        b = ours.DecoderLayer(cfg, layer_idx=0, ple_layer_index=None)
    finally:
        torch.set_default_dtype(default)
    for module in (a, b):
        missing, unexpected = module.load_state_dict(tensors, strict=False)
        assert not unexpected, f"{type(module).__name__}: unexpected {unexpected[:5]}"
        assert not missing, f"{type(module).__name__}: missing {missing[:5]}"
    a, b = a.cuda().eval(), b.cuda().eval()

    torch.manual_seed(0)
    x = torch.randn(
        1, SEQ, cfg.hc_count * cfg.hidden_size, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    cos, sin = rope_cos_sin(hf_cfg, SEQ)
    cos, sin = cos.cuda().to(torch.bfloat16), sin.cuda().to(torch.bfloat16)
    cu = torch.tensor([0, SEQ], dtype=torch.int32, device="cuda")
    seg = torch.zeros(SEQ, dtype=torch.long, device="cuda")

    with torch.no_grad():
        out_ours = b(x, cos, sin, cu, SEQ, seg, None)

        if layer_type == "qwen_sparse_attention":
            # pin HF to the block selection we made; the selection itself is
            # covered by models/tests/test_qwen4_parity.py
            attend = _attend_from(
                b.self_attn.indexer(b.attn_hyper_connection(x)[0], cos, sin, cu, seg), SEQ
            )
            a.self_attn.indexer = _FrozenIndexer(attend)
            mask = torch.zeros(1, 1, SEQ, SEQ, device="cuda", dtype=torch.bfloat16)
            mask = mask.masked_fill(~attend, torch.finfo(torch.bfloat16).min)
        else:
            mask = None

        out_hf = a(x, position_embeddings=(cos, sin), attention_mask=mask)

    close(out_hf, out_ours, f"layer {layer_idx} ({layer_type})",
          atol=ATOL_BF16, rtol=RTOL_BF16)


@pytest.mark.cuda_only
@cuda_only
def test_linear_attention_layer_real_weights():
    layers = _layers_of_type("linear_attention")
    _run_layer_parity(layers[0])


@pytest.mark.cuda_only
@cuda_only
def test_sparse_attention_layer_real_weights():
    layers = _layers_of_type("qwen_sparse_attention")
    _run_layer_parity(layers[0])


# ------------------------------------------------------- multi-layer stack

def _stack_configs(layer_indices: list[int]):
    """The real config narrowed to a contiguous run of layers, both dialects."""
    with open(SNAPSHOT / "config.json") as f:
        raw = json.load(f)
    tc = copy.deepcopy(raw["text_config"])
    tc["layer_types"] = [tc["layer_types"][i] for i in layer_indices]
    tc["num_hidden_layers"] = len(layer_indices)
    tc["ple_layer_ids"] = []          # the PLE table alone is ~95 GiB
    tc.pop("mtp", None)
    hf_cfg = Qwen4ExpTextConfig(**tc)
    hf_cfg._attn_implementation = "eager"
    return hf_cfg, Qwen4TextConfig.from_dict(tc)


def _stack_inputs(cfg, hf_cfg, seq: int, device="cuda"):
    torch.manual_seed(0)
    x = torch.randn(
        1, seq, cfg.hc_count * cfg.hidden_size, device=device, dtype=torch.bfloat16
    ) * 0.05
    cos, sin = rope_cos_sin(hf_cfg, seq)
    return x, cos.to(device, torch.bfloat16), sin.to(device, torch.bfloat16)


@pytest.mark.cuda_only
@cuda_only
def test_layer_stack_real_weights():
    """A full `layer_types` cycle, so the hyper-connection stream is carried
    across a layer boundary rather than just entering and leaving one layer.

    The two stacks are built one after the other and the first is freed before
    the second is allocated: four layers are ~21 GiB of bf16 weights each, and
    both at once does not fit on a 46 GiB card.
    """
    cfg_full = Qwen4Config.from_json(SNAPSHOT / "config.json")
    types = cfg_full.text.layer_types
    period = types.index("qwen_sparse_attention") + 1
    indices = list(range(period))
    assert len(set(types[:period])) > 1, "expected a mixed layer-type cycle"

    hf_cfg, cfg = _stack_configs(indices)
    # `_stack_configs` disables PLE, so drop those tensors rather than reading
    # a ~95 GiB n-gram table off disk. PLE has its own tests below.
    tensors = [
        _tensors_with_prefix(
            f"model.language_model.layers.{i}.", skip=lambda r: r.startswith("ple.")
        )
        for i in indices
    ]

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)

    # --- ours, keeping the QSA selections so HF can be pinned to them
    try:
        ours_layers = torch.nn.ModuleList(
            ours.DecoderLayer(cfg, layer_idx=i, ple_layer_index=None)
            for i in range(len(indices))
        )
    finally:
        torch.set_default_dtype(default)
    for layer, t in zip(ours_layers, tensors):
        assert not layer.load_state_dict(t, strict=True)[1]
    ours_layers = ours_layers.cuda().eval()

    x, cos, sin = _stack_inputs(cfg, hf_cfg, SEQ)
    cu = torch.tensor([0, SEQ], dtype=torch.int32, device="cuda")
    seg = torch.zeros(SEQ, dtype=torch.long, device="cuda")

    attends: dict[int, torch.Tensor] = {}
    with torch.no_grad():
        h = x
        for i, layer in enumerate(ours_layers):
            if cfg.layer_types[i] == "qwen_sparse_attention":
                sel = layer.self_attn.indexer(
                    layer.attn_hyper_connection(h)[0], cos, sin, cu, seg
                )
                attends[i] = _attend_from(sel, SEQ)
            h = layer(h, cos, sin, cu, SEQ, seg, None)
        out_ours = h.clone()

    del ours_layers
    torch.cuda.empty_cache()

    # --- HF
    torch.set_default_dtype(torch.bfloat16)
    try:
        hf_layers = torch.nn.ModuleList(
            hf.Qwen4ExpTextDecoderLayer(hf_cfg, layer_idx=i) for i in range(len(indices))
        )
    finally:
        torch.set_default_dtype(default)
    for layer, t in zip(hf_layers, tensors):
        assert not layer.load_state_dict(t, strict=True)[1]
    hf_layers = hf_layers.cuda().eval()

    with torch.no_grad():
        h = x
        for i, layer in enumerate(hf_layers):
            mask = None
            if i in attends:
                layer.self_attn.indexer = _FrozenIndexer(attends[i])
                mask = torch.zeros(1, 1, SEQ, SEQ, device="cuda", dtype=torch.bfloat16)
                mask = mask.masked_fill(~attends[i], torch.finfo(torch.bfloat16).min)
            h = layer(h, position_embeddings=(cos, sin), attention_mask=mask)
        out_hf = h

    close(out_hf, out_ours, f"layer stack {indices} ({cfg.layer_types})",
          atol=ATOL_BF16, rtol=RTOL_BF16)


# --------------------------------------------------------- embeddings, head

@pytest.mark.cuda_only
@cuda_only
def test_embedding_mixer_head_real_weights():
    """`embed_tokens` -> hc streams -> `hyper_connection_mixer` -> `lm_head`.

    Qwen4 has no final RMSNorm; the mixer is the last op before the head, so a
    mis-wired mixer would otherwise only show up in a whole-model run.
    """
    hf_cfg, cfg = _stack_configs([0])
    embed_w = _tensors_with_prefix("model.language_model.embed_tokens.")["weight"]
    lm_head_w = _tensors_with_prefix("lm_head.")["weight"]
    mixer_t = _tensors_with_prefix("model.language_model.hyper_connection_mixer.")

    assert not cfg.tie_word_embeddings if hasattr(cfg, "tie_word_embeddings") else True
    assert not torch.equal(embed_w, lm_head_w), "embed_tokens and lm_head must be untied"

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        a = hf.Qwen4ExpTextGatedResidual(hf_cfg, use_combine=False)
        b = ours.GatedResidual(cfg, use_combine=False)
    finally:
        torch.set_default_dtype(default)
    for m in (a, b):
        assert not m.load_state_dict(mixer_t, strict=True)[1]
    a, b = a.cuda().eval(), b.cuda().eval()

    embed = torch.nn.Embedding.from_pretrained(embed_w.cuda(), freeze=True)
    head = torch.nn.Linear(
        cfg.hidden_size, cfg.vocab_size, bias=False, dtype=torch.bfloat16
    ).cuda()
    with torch.no_grad():
        head.weight.copy_(lm_head_w)

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ), device="cuda")
    with torch.no_grad():
        x = embed(ids).repeat(1, 1, cfg.hc_count)
        logits_hf = head(a(x))
        logits_ours = head(b(x))

    close(logits_hf, logits_ours, "mixer + lm_head", atol=ATOL_BF16, rtol=RTOL_BF16)
    # a real embedding table plus a real head should not produce a flat
    # distribution; catches an all-zero or transposed load
    assert logits_ours.float().std().item() > 0.5


# --------------------------------------------------------- vision tower

@pytest.mark.cuda_only
@cuda_only
def test_vision_tower_real_weights():
    """The whole vision tower (449M params, all in shard 1) against HF."""
    with open(SNAPSHOT / "config.json") as f:
        raw = json.load(f)
    from transformers.models.qwen4_exp import Qwen4ExpVisionConfig

    hf_vcfg = Qwen4ExpVisionConfig(**copy.deepcopy(raw["vision_config"]))
    hf_vcfg._attn_implementation = "eager"
    cfg = Qwen4Config.from_json(SNAPSHOT / "config.json")

    tensors = _tensors_with_prefix("model.visual.")

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        a = hf.Qwen4ExpVisionModel(hf_vcfg)
        b = ours.VisionModel(cfg.vision)
    finally:
        torch.set_default_dtype(default)
    for m in (a, b):
        assert not m.load_state_dict(tensors, strict=True)[1]
    a, b = a.cuda().eval(), b.cuda().eval()

    v = cfg.vision
    grid = torch.tensor([[1, 8, 8], [1, 4, 6]], dtype=torch.long, device="cuda")
    n_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum())
    patch_dim = v.in_channels * v.temporal_patch_size * v.patch_size ** 2
    torch.manual_seed(0)
    pixels = torch.randn(n_patches, patch_dim, device="cuda", dtype=torch.bfloat16) * 0.5

    with torch.no_grad():
        out_hf = a(pixels, grid).pooler_output
        out_ours = b(pixels, grid)

    merge_sq = v.spatial_merge_size ** 2
    assert out_ours.shape == (n_patches // merge_sq, v.out_hidden_size)
    close(out_hf, out_ours, "vision tower", atol=ATOL_BF16, rtol=RTOL_BF16)


# ------------------------------------------------------------------- PLE

PLE_PREFIX_FMT = "model.language_model.layers.{}.ple."


def _ple_layer_index() -> tuple[int, int]:
    """(zero-based decoder layer, position within `ple_layer_ids`)."""
    cfg = Qwen4Config.from_json(SNAPSHOT / "config.json").text
    if not cfg.ple_layer_ids:
        pytest.skip("checkpoint has no PLE layers")
    return cfg.ple_layer_ids[0] - 1, 0


def _ple_shard_headers() -> list[tuple[int, tuple[int, int]]]:
    """(shard index, shape) for every n-gram table shard, from the headers."""
    import struct

    layer_idx, _ = _ple_layer_index()
    prefix = PLE_PREFIX_FMT.format(layer_idx) + "ple_embedding.ngram_embedding."
    weight_map = _index()
    parts: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        if k.startswith(prefix) and k.endswith(".weight"):
            parts.setdefault(shard, []).append(k)
    if not parts:
        pytest.skip("no PLE shards in the index")

    out = []
    for shard, keys in parts.items():
        path = SNAPSHOT / shard
        if not path.exists():
            pytest.skip(f"PLE shard file {shard} not downloaded")
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for k in keys:
            idx = int(k[len(prefix):].removeprefix("shard_").removesuffix(".weight"))
            out.append((idx, tuple(header[k]["shape"])))
    out.sort()
    return out


def test_ple_shard_layout():
    """The n-gram shards must tile our single runtime table exactly.

    Read from the safetensors headers only: the real table is ~95 GiB.
    """
    cfg = Qwen4Config.from_json(SNAPSHOT / "config.json").text
    layer_idx, ple_index = _ple_layer_index()
    with torch.device("meta"):
        ngram = ours.NGramEmbedding(cfg, cfg.ple_embed_dim, ple_index)
    table = ngram.ngram_embedding

    shards = _ple_shard_headers()
    assert [i for i, _ in shards] == list(range(len(shards))), "shard ids are not contiguous"
    assert len(shards) == cfg.split_ngram_parts, (
        f"{len(shards)} shards vs split_ngram_parts={cfg.split_ngram_parts}"
    )
    widths = {shape[1] for _, shape in shards}
    assert widths == {table.embedding_dim}, f"shard widths {widths} vs {table.embedding_dim}"

    # The shards tile the *packed* table -- heads back to back at their true
    # prime sizes. The runtime table pads each head out to a common block so it
    # can be head-sharded across TP ranks, so it is a few rows taller.
    total_rows = sum(shape[0] for _, shape in shards)
    assert total_rows == ngram.packed_rows, (
        f"shards hold {total_rows} rows, packed table has {ngram.packed_rows}"
    )
    assert ngram.rows_per_head * ngram.ngram_heads == table.num_embeddings
    assert table.num_embeddings >= ngram.total_vocab_size
    # our loader concatenates in *numeric* order; a lexicographic sort would
    # place shard_10 third. Uniform shard heights are what make the offset
    # arithmetic in the loader checkable at all.
    heights = {shape[0] for _, shape in shards}
    assert len(heights) == 1, f"non-uniform shard heights {sorted(heights)}"


@pytest.mark.cuda_only
@cuda_only
def test_ple_layer_real_weights():
    """The PLE layer with its real conv / projection / norm weights.

    The n-gram table itself is narrowed to a toy vocabulary and mirrored
    between the two implementations: at the real `ngram_vocab_size_base` it is
    ~95 GiB, and nothing in this layer's arithmetic depends on its height.
    """
    with open(SNAPSHOT / "config.json") as f:
        raw = json.load(f)
    layer_idx, ple_index = _ple_layer_index()

    tc = copy.deepcopy(raw["text_config"])
    tc["layer_types"] = [tc["layer_types"][layer_idx]]
    tc["num_hidden_layers"] = 1
    tc["ple_layer_ids"] = [1]
    tc["ngram_vocab_size_base"] = 4096
    tc.pop("mtp", None)
    hf_cfg = Qwen4ExpTextConfig(**tc)
    hf_cfg._attn_implementation = "eager"
    cfg = Qwen4TextConfig.from_dict(tc)

    # everything except the n-gram table and its hashing buffers: those are
    # derived from `ngram_vocab_size_base`, which the toy config narrows
    real = _tensors_with_prefix(
        PLE_PREFIX_FMT.format(layer_idx), skip=lambda r: r.startswith("ple_embedding.")
    )

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        a = hf.Qwen4ExpTextPLELayer(hf_cfg, layer_idx=0, ple_layer_index=ple_index)
        b = ours.PLELayer(cfg, ple_index)
    finally:
        torch.set_default_dtype(default)

    missing, unexpected = b.load_state_dict(real, strict=False)
    assert not unexpected, unexpected
    assert all(m.startswith("ple_embedding.") for m in missing), missing
    with torch.no_grad():
        b.ple_embedding.ngram_embedding.weight.normal_(0.0, 0.05)
    mirror(b, a)

    a, b = a.cuda().eval(), b.cuda().eval()

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.vocab_size, (1, SEQ), device="cuda")
    x = torch.randn(
        1, SEQ, cfg.hc_count * cfg.hidden_size, device="cuda", dtype=torch.bfloat16
    ) * 0.05
    with torch.no_grad():
        out_hf = a(x, ids, None)
        out_ours = b(x, ids, None)
    close(out_hf, out_ours, "PLE layer (real projections)", atol=ATOL_BF16, rtol=RTOL_BF16)


@pytest.mark.skipif(
    os.environ.get("QWEN4_TEST_PLE_TABLE") != "1",
    reason="set QWEN4_TEST_PLE_TABLE=1: ~95 GiB of host RAM and ~17 min of I/O",
)
def test_ple_table_load_real_weights():
    """Exercise `load_safetensors_into`'s PLE shard-concatenation branch.

    The loader is handed a stub whose only parameter is the n-gram table, so
    the rest of the 335 GiB checkpoint is skipped; every other key falls
    through the `k not in state` branch.
    """
    from models.qwen4.utils import load_safetensors_into

    cfg_full = Qwen4Config.from_json(SNAPSHOT / "config.json")
    cfg = cfg_full.text
    layer_idx, ple_index = _ple_layer_index()

    class _Stub(torch.nn.Module):
        def __init__(self):
            super().__init__()
            with torch.device("meta"):
                template = ours.NGramEmbedding(cfg, cfg.ple_embed_dim, ple_index)
            self.ngram_embedding = torch.nn.Embedding(
                template.ngram_embedding.num_embeddings,
                template.ngram_embedding.embedding_dim,
                dtype=torch.bfloat16,
            )

    prefix = PLE_PREFIX_FMT.format(layer_idx) + "ple_embedding."
    holder = torch.nn.Module()
    node = holder
    for part in prefix.rstrip(".").split("."):
        child = _Stub() if part == "ple_embedding" else torch.nn.Module()
        node.add_module(part, child)
        node = child
    holder.cfg = cfg_full

    load_safetensors_into(holder, SNAPSHOT, device="cpu", dtype=torch.bfloat16, load_vision=False)

    table = node.ngram_embedding.weight
    shards = _ple_shard_headers()
    height = shards[0][1][0]

    # spot-check the placement of shards that a lexicographic sort would move
    src_prefix = prefix + "ngram_embedding."
    weight_map = _index()
    for shard_idx in (0, 1, 10, len(shards) - 1):
        key = f"{src_prefix}shard_{shard_idx}.weight"
        with safe_open(str(SNAPSHOT / weight_map[key]), framework="pt", device="cpu") as f:
            sl = f.get_slice(key)
            first, last = sl[0:1], sl[height - 1 : height]
        base = shard_idx * height
        assert torch.equal(table[base : base + 1], first.to(torch.bfloat16)), (
            f"shard {shard_idx} first row is not at offset {base}"
        )
        assert torch.equal(table[base + height - 1 : base + height], last.to(torch.bfloat16)), (
            f"shard {shard_idx} last row is not at offset {base + height - 1}"
        )


# ------------------------------------------------ QSA indexer, real weights

def test_qsa_indexer_selection_real_weights():
    """The real indexer's selection, against HF's own per-query reference.

    The sparse-layer test above pins HF to *our* selection so that it measures
    attention arithmetic alone; nothing there checks that the real
    `index_qk_proj` / `q_layernorm` / `k_layernorm` weights produce the same
    blocks. This does, and it runs on CPU: the indexer is ~1.6M parameters.

    The sequence has to be longer than `indexer_budget` or the top-k is
    vacuous — every complete block fits in the budget and both sides trivially
    select all of them.
    """
    from models.tests.test_qwen4_parity import _reference_qsa, _selected_blocks

    layers = _layers_of_type("qwen_sparse_attention")
    layer_idx = layers[0]
    hf_cfg, cfg, _ = _one_layer_configs(layer_idx)

    ratio = cfg.indexer_compress_ratio
    block_topk = cfg.indexer_budget // ratio
    seq_len = int((block_topk + block_topk // 4) * ratio)   # 25% of blocks rejected
    assert seq_len // ratio > block_topk, "sequence too short to exercise the top-k"

    tensors = _tensors_with_prefix(
        f"model.language_model.layers.{layer_idx}.self_attn.indexer.", dtype=torch.float32
    )

    a = hf.Qwen4ExpTextQSAIndexer(hf_cfg, layer_idx=layer_idx)
    b = ours.QSAIndexer(cfg)
    for m in (a, b):
        assert not m.load_state_dict(tensors, strict=True)[1]
    a, b = a.eval(), b.eval()

    torch.manual_seed(0)
    x = torch.randn(1, seq_len, cfg.hidden_size) * 0.05
    cos, sin = rope_cos_sin(hf_cfg, seq_len)

    with torch.no_grad():
        ref_attend, scores = _reference_qsa(a, cfg, x, cos, sin, seq_len)
        got_attend, selected = _selected_blocks(b, x, cos, sin, seq_len)

    assert torch.equal(ref_attend.sum(dim=1), got_attend.sum(dim=1)), (
        "different number of attended tokens per query"
    )

    # ties are common, so compare the selected *score multiset* rather than the
    # indices; see models/tests/test_qwen4_parity.py for the full argument
    rejected_any = False
    for t in range(0, seq_len, 7):     # every query is O(seq) work; sample
        n_complete = (t + 1) // ratio
        if n_complete == 0:
            continue
        ref_blocks = torch.unique(
            ref_attend[t, : n_complete * ratio].nonzero().flatten() // ratio
        )
        got_blocks = selected[t].nonzero().flatten()
        assert ref_blocks.numel() == got_blocks.numel(), f"query {t}: block count"
        close(
            scores[t, ref_blocks].sort(descending=True).values,
            scores[t, got_blocks].sort(descending=True).values,
            f"query {t} selected block scores",
            atol=1e-4,
            rtol=1e-4,
        )
        if n_complete > block_topk:
            rejected_any = True
    assert rejected_any, "no sampled query actually had to reject a block"


# ------------------------------------------ packing isolation, real weights

@pytest.mark.cuda_only
@cuda_only
def test_linear_attention_packing_real_weights():
    """Two documents packed into one row must give each document's own output.

    Real dims matter here: `linear_attn` runs a `conv_kernel_size=4` causal
    conv and carries a recurrent state, both of which bleed across a document
    boundary if `cu_seqlens` is not threaded all the way into the kernels.
    """
    layer_idx = _layers_of_type("linear_attention")[0]
    _, cfg, _ = _one_layer_configs(layer_idx)
    tensors = _tensors_with_prefix(
        f"model.language_model.layers.{layer_idx}.linear_attn."
    )

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        layer = ours.GatedDeltaNet(cfg)
    finally:
        torch.set_default_dtype(default)
    assert not layer.load_state_dict(tensors, strict=True)[1]
    layer = layer.cuda().eval()

    lens = [96, 160]
    total = sum(lens)
    torch.manual_seed(0)
    x = torch.randn(1, total, cfg.hidden_size, device="cuda", dtype=torch.bfloat16) * 0.05
    cu = torch.tensor([0, lens[0], total], dtype=torch.int32, device="cuda")
    seg = torch.repeat_interleave(
        torch.arange(2, device="cuda"), torch.tensor(lens, device="cuda")
    )

    with torch.no_grad():
        packed = layer(x, cu_seqlens=cu, seg_id=seg)
        lo = 0
        for i, n in enumerate(lens):
            alone = layer(
                x[:, lo : lo + n],
                cu_seqlens=torch.tensor([0, n], dtype=torch.int32, device="cuda"),
                seg_id=torch.zeros(n, dtype=torch.long, device="cuda"),
            )
            close(alone, packed[:, lo : lo + n], f"document {i} under packing",
                  atol=ATOL_BF16, rtol=RTOL_BF16)
            lo += n


# --------------------------------------------------------- whole model

def _full_configs():
    """Both config dialects for the whole model, with PLE disabled.

    The n-gram table is ~95 GiB per copy and two copies of it do not fit
    alongside two copies of the decoder. PLE is covered by
    `test_ple_layer_real_weights` and `test_ple_table_load_real_weights`.
    """
    from transformers.models.qwen4_exp import Qwen4ExpConfig

    with open(SNAPSHOT / "config.json") as f:
        raw = json.load(f)
    r = copy.deepcopy(raw)
    r.pop("architectures", None)
    r.pop("_name_or_path", None)
    r["text_config"]["ple_layer_ids"] = []
    r["text_config"].pop("mtp", None)

    hf_cfg = Qwen4ExpConfig(**r)
    hf_cfg.text_config._attn_implementation = "eager"
    hf_cfg._attn_implementation = "eager"

    cfg = Qwen4Config.from_json(SNAPSHOT / "config.json")
    cfg.text.ple_layer_ids = []
    return hf_cfg, cfg


def _load_both_from_checkpoint(models: list[torch.nn.Module], dtype=torch.bfloat16) -> None:
    """Fill several models from one pass over the shards.

    Each shard is opened once and every tensor copied into all the models that
    declare it, so the ~235 GiB of decoder weights is read from disk once
    rather than once per model.
    """
    weight_map = _index()
    states = [dict(m.state_dict()) for m in models]
    by_shard: dict[str, list[str]] = {}
    for k, shard in weight_map.items():
        if k.startswith("mtp.") or ".ple." in k:
            continue
        by_shard.setdefault(shard, []).append(k)

    seen = [set() for _ in models]
    for shard, keys in sorted(by_shard.items()):
        with safe_open(str(SNAPSHOT / shard), framework="pt", device="cpu") as f:
            for k in keys:
                tensor = None
                for state, got in zip(states, seen):
                    if k not in state:
                        continue
                    if tensor is None:
                        t = f.get_tensor(k)
                        tensor = t.to(dtype) if t.is_floating_point() else t
                    assert state[k].shape == tensor.shape, (
                        f"{k}: checkpoint {tuple(tensor.shape)} vs model {tuple(state[k].shape)}"
                    )
                    state[k].copy_(tensor)
                    got.add(k)

    for model, state, got in zip(models, states, seen):
        missing = set(state) - got
        assert not missing, (
            f"{type(model).__name__}: {len(missing)} unloaded params, "
            f"e.g. {sorted(missing)[:5]}"
        )


@pytest.mark.skipif(
    os.environ.get("QWEN4_TEST_FULL_MODEL") != "1",
    reason="set QWEN4_TEST_FULL_MODEL=1: holds two ~235 GiB copies of the model in host RAM",
)
@cuda_only
def test_full_model_parity_streamed():
    """All 48 layers, real weights, ours against `transformers`.

    Both models live in host RAM; each decoder layer is moved to the GPU for
    the duration of its own forward and moved back, so the GPU only ever holds
    two layers (~10 GiB). This is what makes the whole-model comparison
    possible on a single card, and it is also required rather than merely
    convenient: 36 of the 48 layers are `linear_attention`, whose gated
    delta rule is a Triton kernel with no CPU implementation.

    The two hidden states are *not* re-synchronized between layers, so the
    printed table shows both where a divergence starts and how bf16 rounding
    accumulates through the stack.
    """
    from transformers.models.qwen4_exp import Qwen4ExpForConditionalGeneration

    seq = int(os.environ.get("QWEN4_FULL_SEQ", "2560"))
    hf_cfg, cfg = _full_configs()

    default = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            a = Qwen4ExpForConditionalGeneration(hf_cfg)
            b = ours.Qwen4ForCausalLM(cfg)
    finally:
        torch.set_default_dtype(default)
    a = a.to_empty(device="cpu").eval()
    b = b.to_empty(device="cpu").eval()

    _load_both_from_checkpoint([a, b])

    a_lm, b_lm = a.model.language_model, b.model.language_model
    layer_types = cfg.text.layer_types
    assert len(layer_types) == len(b_lm.layers) == len(a_lm.layers)

    torch.manual_seed(0)
    ids = torch.randint(0, cfg.text.vocab_size, (1, seq))
    cos, sin = rope_cos_sin(hf_cfg.text_config, seq)
    cos, sin = cos.cuda().to(torch.bfloat16), sin.cuda().to(torch.bfloat16)
    cu = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    seg = torch.zeros(seq, dtype=torch.long, device="cuda")

    with torch.no_grad():
        b_lm.embed_tokens.cuda()
        embeds = b_lm.embed_tokens(ids.cuda())
        b_lm.embed_tokens.cpu()
        close(
            a_lm.embed_tokens(ids).cuda(), embeds, "embed_tokens",
            atol=0.0, rtol=0.0,
        )

        h_ours = embeds.repeat(1, 1, cfg.text.hc_count)
        h_hf = h_ours.clone()

        report = []
        for i, (lo, lh) in enumerate(zip(b_lm.layers, a_lm.layers)):
            lo.cuda()
            lh.cuda()
            real_indexer = None
            mask = None
            if layer_types[i] == "qwen_sparse_attention":
                sel = lo.self_attn.indexer(lo.attn_hyper_connection(h_ours)[0], cos, sin, cu, seg)
                attend = _attend_from(sel, seq)
                real_indexer = lh.self_attn.indexer
                lh.self_attn.indexer = _FrozenIndexer(attend)
                mask = torch.zeros(1, 1, seq, seq, device="cuda", dtype=torch.bfloat16)
                mask = mask.masked_fill(~attend, torch.finfo(torch.bfloat16).min)

            h_ours = lo(h_ours, cos, sin, cu, seq, seg, None)
            h_hf = lh(h_hf, position_embeddings=(cos, sin), attention_mask=mask)

            if real_indexer is not None:
                lh.self_attn.indexer = real_indexer
            lo.cpu()
            lh.cpu()
            torch.cuda.empty_cache()

            diff = (h_ours.float() - h_hf.float()).abs().max().item()
            scale = h_hf.float().abs().max().item()
            report.append((i, layer_types[i], diff, scale))

        b_lm.hyper_connection_mixer.cuda()
        a_lm.hyper_connection_mixer.cuda()
        out_ours = b_lm.hyper_connection_mixer(h_ours)
        out_hf = a_lm.hyper_connection_mixer(h_hf)
        b_lm.hyper_connection_mixer.cpu()
        a_lm.hyper_connection_mixer.cpu()

        b.lm_head.cuda()
        logits_ours = b.lm_head(out_ours).float()
        b.lm_head.cpu()
        a.lm_head.cuda()
        logits_hf = a.lm_head(out_hf).float()
        a.lm_head.cpu()

    print(f"\nwhole-model parity, seq={seq}")
    print(f"{'layer':>5} {'type':>22} {'max |diff|':>12} {'max |hf|':>12} {'rel':>10}")
    for i, t, d, s in report:
        print(f"{i:>5} {t:>22} {d:>12.4e} {s:>12.4e} {d / max(s, 1e-9):>10.2e}")

    rel = [d / max(s, 1e-9) for _, _, d, s in report]
    top1 = (logits_ours.argmax(-1) == logits_hf.argmax(-1)).float().mean().item()
    lo_f, hf_f = logits_ours.flatten(), logits_hf.flatten()
    corr = torch.corrcoef(torch.stack([lo_f, hf_f]))[0, 1].item()
    print(f"final logits: top-1 agreement {top1:.4f}  corr {corr:.6f}  "
          f"max |diff| {(lo_f - hf_f).abs().max().item():.4e}")

    # What this can and cannot assert. Every layer is individually exact --
    # `test_layer_wiring_exact_in_fp32` pins that at ~5e-7 -- but the 36
    # `linear_attention` layers run FLA's Triton chunked kernel against HF's
    # `torch_chunk_gated_delta_rule`, and those two accumulate differently.
    # In bf16 that seeds ~3e-2 of relative disagreement per linear layer, which
    # compounds through a residual stream that has no final norm to rescale it,
    # reaching ~3e-1 by layer 47. Measured on this checkpoint: top-1 0.950,
    # corr 0.989. So the assertions here are about the shape of the growth and
    # the decision the logits encode, not an elementwise tolerance.
    assert corr > 0.98, f"logit correlation {corr:.6f}"
    assert top1 > 0.90, f"top-1 token agreement {top1:.4f}"

    # A wiring bug in one layer shows up as a *discontinuous* jump rather than
    # the smooth compounding above; the largest observed step is ~2.5x.
    diffs = [d for _, _, d, _ in report]
    steps = [(i + 1, diffs[i + 1] / diffs[i]) for i in range(len(diffs) - 1) if diffs[i] > 0]
    worst_layer, worst = max(steps, key=lambda t: t[1])
    assert worst < 4.0, (
        f"layer {worst_layer} multiplied the divergence by {worst:.2f}x, "
        "which is a step change rather than accumulation"
    )


def _gdn_torch_reference(q, k, v, g, beta, cu_seqlens):
    """HF's own chunked gated delta rule, as a stand-in for the FLA kernel."""
    out, _ = hf.torch_chunk_gated_delta_rule(
        q, k, v, g=g, beta=beta, initial_state=None, output_final_state=False,
        use_qk_l2norm_in_kernel=True, cu_seqlens=cu_seqlens.to(torch.int64),
    )
    return out


@pytest.mark.cuda_only
@cuda_only
@pytest.mark.parametrize("layer_type", ["linear_attention", "qwen_sparse_attention"])
def test_layer_wiring_exact_in_fp32(layer_type):
    """In fp32, with the same kernels on both sides, a layer must match to ~1e-6.

    The bf16 tests above run at `atol=0.15`, which is loose enough to hide a
    genuine wiring error. Everything except the gated delta rule is exact in
    fp32 already; for `linear_attention` we additionally swap our FLA Triton
    kernel for the `torch_chunk_gated_delta_rule` that HF itself calls, so the
    two sides differ only in how *we* wire the projections, the causal conv,
    the l2 norm, `g`/`beta`, head grouping, the gated norm and `out_proj`.

    What is left over when that swap is *not* made is the kernel gap alone:
    ~5e-3 to 4e-2 relative in fp32, ~3e-2 in bf16. That is the dominant term in
    `test_full_model_parity_streamed`, and it is a property of the kernel
    choice, not of this implementation.
    """
    layer_idx = _layers_of_type(layer_type)[0]
    hf_cfg, cfg, parsed = _one_layer_configs(layer_idx)
    assert parsed == layer_type
    seq = 512

    tensors = _tensors_with_prefix(
        f"model.language_model.layers.{layer_idx}.",
        dtype=torch.float32,
        skip=lambda r: r.startswith("ple."),
    )
    a = hf.Qwen4ExpTextDecoderLayer(hf_cfg, layer_idx=0)
    b = ours.DecoderLayer(cfg, layer_idx=0, ple_layer_index=None)
    for m in (a, b):
        assert not m.load_state_dict(tensors, strict=True)[1]
    a, b = a.cuda().eval(), b.cuda().eval()

    torch.manual_seed(0)
    x = torch.randn(1, seq, cfg.hc_count * cfg.hidden_size, device="cuda") * 0.05
    cos, sin = rope_cos_sin(hf_cfg, seq)
    cos, sin = cos.cuda(), sin.cuda()
    cu = torch.tensor([0, seq], dtype=torch.int32, device="cuda")
    seg = torch.zeros(seq, dtype=torch.long, device="cuda")

    saved = ours.GatedDeltaNet._run_gated_delta_rule
    if layer_type == "linear_attention":
        ours.GatedDeltaNet._run_gated_delta_rule = staticmethod(_gdn_torch_reference)
    try:
        with torch.no_grad():
            out_ours = b(x, cos, sin, cu, seq, seg, None)
            mask = None
            if layer_type == "qwen_sparse_attention":
                attend = _attend_from(
                    b.self_attn.indexer(b.attn_hyper_connection(x)[0], cos, sin, cu, seg), seq
                )
                a.self_attn.indexer = _FrozenIndexer(attend)
                mask = torch.zeros(1, 1, seq, seq, device="cuda").masked_fill(
                    ~attend, torch.finfo(torch.float32).min
                )
            out_hf = a(x, position_embeddings=(cos, sin), attention_mask=mask)
    finally:
        ours.GatedDeltaNet._run_gated_delta_rule = saved

    rel = (out_hf - out_ours).abs().max().item() / out_hf.abs().max().item()
    assert rel < 1e-5, f"{layer_type}: relative diff {rel:.3e} is not fp32 rounding"
