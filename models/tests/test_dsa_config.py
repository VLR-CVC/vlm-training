"""`attn_backend = "dsa"` wiring: the toml reaches the inner attention.

No CUDA and no kernels here -- this is the config tree only. The numerics live
in `test_dsa_equivalence.py`.
"""

import dataclasses
import json
from pathlib import Path

import pytest

from models.common.attention import VarlenInnerAttention
from models.common.dsa.inner_attention import DSAIndexer, DSAInnerAttention
from models.qwen3_5.configs import qwen35_config_from_hf
from train.config import DSA

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs/models/qwen3_5_9b.json"


def _inner_attentions(model_config):
    return [
        layer.attention.inner_attention
        for layer in model_config.layers
        if layer.attention is not None
    ]


def test_dsa_config_defaults_agree():
    """`[dsa]` documents the defaults; the module configs hold them.

    `_qwen35_attention_config` splits the section by field name and passes it
    through without restating a single default, so the two sides agree or a user
    reading `[dsa]` is being told the wrong number.
    """
    module_defaults = {
        f.name: f.default
        for cls in (DSAIndexer.Config, DSAInnerAttention.Config)
        for f in dataclasses.fields(cls)
        if f.default is not dataclasses.MISSING
    }
    for field in dataclasses.fields(DSA):
        assert field.name in module_defaults, (
            f"[dsa].{field.name} matches no field of DSAIndexer.Config or "
            "DSAInnerAttention.Config, so it would be rejected at build time"
        )
        assert module_defaults[field.name] == field.default, (
            f"[dsa].{field.name} documents {field.default!r} but the module "
            f"defaults to {module_defaults[field.name]!r}"
        )


def test_dsa_backend_builds_indexers_on_full_attention_layers_only():
    hidden = json.loads(CONFIG.read_text())["text_config"]["hidden_size"]
    config = qwen35_config_from_hf(
        CONFIG, seq_len=4096, attn_backend="dsa",
        dsa=dataclasses.asdict(DSA(index_topk=4096, min_sparse_len=99)),
    )
    inners = _inner_attentions(config)
    assert inners, "no full-attention layer in the schedule"
    for inner in inners:
        assert isinstance(inner, DSAInnerAttention.Config), type(inner)
        assert inner.indexer.index_topk == 4096
        assert inner.min_sparse_len == 99
        # The indexer projects from the block input, so its width is the model's,
        # not anything the toml could have said.
        assert inner.indexer.dim == hidden
    # GatedDeltaNet layers have no attention at all, so nothing was given one.
    assert any(layer.attention is None for layer in config.layers)


def test_varlen_is_untouched_and_ignores_the_section():
    config = qwen35_config_from_hf(
        CONFIG, seq_len=4096, dsa=dataclasses.asdict(DSA(index_topk=128))
    )
    for inner in _inner_attentions(config):
        assert isinstance(inner, VarlenInnerAttention.Config), type(inner)


def test_unknown_key_is_rejected():
    """A typo in `[dsa]` must name itself rather than be silently dropped."""
    with pytest.raises(TypeError, match="index_topkk"):
        qwen35_config_from_hf(
            CONFIG, seq_len=4096, attn_backend="dsa", dsa={"index_topkk": 1024}
        )


def test_qwen3_vl_rejects_dsa():
    from models.qwen3_vl.configs import qwen3_vl_config_from_hf

    with pytest.raises(ValueError, match="not wired for qwen3_vl"):
        qwen3_vl_config_from_hf(CONFIG, seq_len=4096, attn_backend="dsa")


@pytest.mark.parametrize("tp,sp", [(2, False), (1, True)])
def test_dsa_refuses_tensor_and_sequence_parallel(tp, sp):
    """Better a named error than a teacher reduced over local heads only."""
    from models.qwen3_5.checkpoint import build_meta

    with pytest.raises(ValueError, match="wired for tp=1"):
        build_meta(CONFIG, seq_len=4096, attn_backend="dsa", tp=tp, enable_sp=sp)


def test_dsa_blocks_compile_without_fullgraph():
    """`apply_compile` has to break at the DSA kernel, not demand a full graph.

    The flag lives on the kernel so `compile.py` never imports the Triton
    module, and it is read off the built module, so a hybrid model compiles its
    GatedDeltaNet blocks whole either way.
    """
    import torch.nn as nn

    from train.parallel.compile import block_supports_fullgraph

    dense = nn.Module()
    dense.inner_attention = VarlenInnerAttention.Config().build()
    assert block_supports_fullgraph(dense)

    sparse = nn.Module()
    sparse.inner_attention = DSAInnerAttention.Config(
        indexer=DSAIndexer.Config(dim=64, index_topk=32)
    ).build()
    assert not block_supports_fullgraph(sparse)

    # No attention at all (GatedDeltaNet) is not a reason to break the graph.
    assert block_supports_fullgraph(nn.Module())
