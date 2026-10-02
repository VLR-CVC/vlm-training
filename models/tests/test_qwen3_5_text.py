"""`models/qwen3_5_text`: text-only, and the attention schedule is a knob.

Everything builds on the meta device -- these assert structure, not numerics, so
no weights and no GPU are needed.
"""

import json

import pytest
import torch

from models.qwen3_5.configs import LAYER_TYPES
from models.qwen3_5.gdn import GatedDeltaNet
from models.qwen3_5.model import Qwen35Attention
from models.qwen3_5_text.configs import (
    layer_schedule,
    qwen35_text_config_from_hf,
    resolve_attention_kind,
)
from train.flops_estimation import build_flops_model

CONFIG = "configs/models/qwen3_5_2b.json"
N_LAYERS = json.loads(open(CONFIG).read())["text_config"]["num_hidden_layers"]


def build(kind: str):
    cfg = qwen35_text_config_from_hf(CONFIG, seq_len=4096, attention_kind=kind)
    with torch.device("meta"):
        return cfg.build()


def kinds_of(model) -> list[str]:
    """Which operator each layer actually ended up with, read off the modules."""
    out = []
    for layer in model.layers.values():
        assert isinstance(layer.attn, (Qwen35Attention, GatedDeltaNet)), type(layer.attn)
        out.append(
            "full_attention" if isinstance(layer.attn, Qwen35Attention)
            else "linear_attention"
        )
    return out


def test_no_vision_tower():
    model = build("hybrid")
    assert model.vision_encoder is None
    assert not any("vision" in name for name, _ in model.named_parameters())


def test_hybrid_matches_the_snapshot_schedule():
    """Default must reproduce stock Qwen3.5, or the A/B has no baseline."""
    expected = json.loads(open(CONFIG).read())["text_config"]["layer_types"]
    assert kinds_of(build("hybrid")) == expected
    # and that is not a uniform schedule, so the next two tests are real changes
    assert len(set(expected)) == 2


@pytest.mark.parametrize("kind", LAYER_TYPES)
def test_uniform_schedule_replaces_every_layer(kind):
    assert kinds_of(build(kind)) == [kind] * N_LAYERS


def test_aliases():
    assert resolve_attention_kind("softmax") == "full_attention"
    assert resolve_attention_kind("gdn") == "linear_attention"
    assert resolve_attention_kind("") == "hybrid"
    assert layer_schedule("hybrid", N_LAYERS) is None
    with pytest.raises(ValueError, match="attention_kind"):
        resolve_attention_kind("quadratic")


def test_with_vision_is_rejected():
    with pytest.raises(ValueError, match="text-only"):
        qwen35_text_config_from_hf(CONFIG, seq_len=4096, with_vision=True)


def test_flops_follow_the_built_schedule():
    """The bug this guards: billing the snapshot's schedule for an overridden model.

    All-softmax must cost more per token than the hybrid it replaced -- softmax
    layers carry a quadratic term that GatedDeltaNet has none of, so the pair
    coefficient has to rise with the number of full-attention layers.
    """
    hf = json.loads(open(CONFIG).read())
    hybrid = build_flops_model(hf)
    softmax = build_flops_model(hf, layer_types=layer_schedule("softmax", N_LAYERS))
    gdn = build_flops_model(hf, layer_types=layer_schedule("gdn", N_LAYERS))

    assert gdn.text_attn_pair == 0.0
    assert softmax.text_attn_pair > hybrid.text_attn_pair > 0.0
    # every layer quadratic vs every fourth
    assert softmax.text_attn_pair == pytest.approx(
        hybrid.text_attn_pair * N_LAYERS / (N_LAYERS // 4)
    )


def test_flops_without_a_vision_config():
    """A text-only JSON has no `vision_config`; that must not KeyError."""
    hf = json.loads(open(CONFIG).read())
    del hf["vision_config"]
    hf["model_type"] = "qwen3_5_text"
    model = build_flops_model(hf)
    assert model.vision_per_patch == 0.0 and model.vision_attn_pair == 0.0
    assert model.dense_per_token > 0.0


def test_tp_config_survives_a_uniform_schedule():
    """`apply_parallelism_config` used to walk off the end of an empty generator
    when a schedule left it no GatedDeltaNet layer (or no attention layer)."""
    from models.qwen3_5_text.configs import apply_parallelism_config

    for kind in LAYER_TYPES:
        cfg = qwen35_text_config_from_hf(CONFIG, seq_len=4096, attention_kind=kind)
        apply_parallelism_config(cfg, tp=2, enable_sp=True)  # must not raise


def test_shipped_toml_builds_what_it_claims():
    """`configs/text/qwen3_5_2b.toml` must build, and its header quotes param
    counts per schedule -- a stale header is a lie nobody would notice."""
    import tomllib

    cfg = tomllib.loads(open("configs/text/qwen3_5_2b.toml", "rb").read().decode())
    assert cfg["training"]["random_init"], "a non-hybrid schedule cannot load weights"
    assert cfg["model"]["train_vit"] is False, "there is no tower to train"

    model = build(cfg["model"]["attention_kind"])
    assert model.vision_encoder is None
    params = sum(p.numel() for p in model.parameters())
    assert params == pytest.approx(1.767e9, rel=0.01), params
