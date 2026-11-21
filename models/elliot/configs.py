"""ElliotVL config from its HF `config.json` (`model_type: elliot_vlm`).

- SigLIP2 so400m dims, LayerNorm eps 1e-6 (post_layernorm 1e-4, as trained), exact GELU in the block MLPs (`nn.GELU()`),
  2D RoPE (`rope_theta`) on q/k, attention scale head_dim**-0.5;
- the 32x32 learned position table, resized bicubically (`ElliotVisionEncoder`);
- `post_layernorm` before the 2x2 merge = our merger's pre-merge norm;
- projector `fc1` (4*1152 -> llm dim), exact GELU, `fc2` (llm dim -> llm dim).
"""

import dataclasses
import json
from pathlib import Path

from models.common.nn_modules import GELU
from models.qwen3_5.configs import (
    _linear,
    _partial_bias_rowwise_linear,
    _qwen35_vision_encoder_config,
    qwen35_config_from_hf,
)
from models.qwen3_5.configs import apply_parallelism_config as apply_parallelism_config

from .model import ElliotModel
from .vision_encoder import ElliotVisionEncoder

SPATIAL_MERGE = 2  # Elliot's projector concatenates 2x2 patches
TEMPORAL_PATCH = 2  # input layout of our data pipeline (Qwen processor)
POST_LAYERNORM_EPS = 1e-4 # for last post layernorm on vision tower

def elliot_vision_config(vc: dict, *, out_hidden_size: int) -> ElliotVisionEncoder.Config:
    if vc.get("model_type") != "elliot_siglip2_navit":
        raise NotImplementedError(f"vision tower {vc.get('model_type')!r} is not ported")
    if not vc.get("preserve_original_pe", True) or not vc.get("use_rope", True):
        raise NotImplementedError("only the tower with learned PE and 2D RoPE is ported")
    base = _qwen35_vision_encoder_config(
        dim=vc["hidden_size"],
        ffn_dim=vc["intermediate_size"],
        num_layers=vc["num_hidden_layers"],
        num_heads=vc["num_attention_heads"],
        patch_size=vc["patch_size"],
        temporal_patch_size=TEMPORAL_PATCH,
        spatial_merge_size=SPATIAL_MERGE,
        out_hidden_size=out_hidden_size,
        num_position_embeddings=(vc["image_size"] // vc["patch_size"]) ** 2,
        layer_norm_eps=vc.get("layer_norm_eps", 1e-6),
        rope_theta=float(vc.get("rope_theta", 10000.0)),
        in_channels=vc.get("num_channels", 3),
    )
    exact = GELU.Config(approximate="none")
    block = dataclasses.replace(base.block, mlp=dataclasses.replace(base.block.mlp, act_fn=exact))
    merged = base.merger.merged_hidden_size
    merger = dataclasses.replace(
        base.merger,
        norm=dataclasses.replace(base.merger.norm, eps=POST_LAYERNORM_EPS),
        fc1=_linear(merged, out_hidden_size),
        act_fn=exact,
        fc2=_partial_bias_rowwise_linear(out_hidden_size, out_hidden_size),
    )
    fields = {f.name: getattr(base, f.name) for f in dataclasses.fields(base) if not f.name.startswith("_")}
    return ElliotVisionEncoder.Config(**{**fields, "block": block, "merger": merger})

def elliot_config_from_hf(
    config_path: str | Path,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    decoder_mask: str = "causal_doc",
    with_vision: bool = True,
    image_positions: str = "mrope",
) -> ElliotModel.Config:
    path = Path(config_path)
    raw = json.loads((path / "config.json" if path.is_dir() else path).read_text())
    if raw.get("model_type") != "elliot_vlm":
        raise ValueError(f"{path} is not an elliot_vlm config")
    if raw.get("projector_type") != "mlp" or raw.get("use_naflex", False):
        raise NotImplementedError("only the 2-layer MLP projector without NaFlex is ported")
    # "causal" is Elliot-only (ElliotModel rewrites positions); the decoder is built as usual
    text = qwen35_config_from_hf(config_path, seq_len=seq_len, attn_backend=attn_backend,
                                 with_vision=False)
    fields = {f.name: getattr(text, f.name) for f in dataclasses.fields(text) if not f.name.startswith("_")}
    fields["vision_encoder"] = (
        elliot_vision_config(raw["vision_config"], out_hidden_size=raw["llm_hidden_size"])
        if with_vision else None
    )
    return ElliotModel.Config(**fields, image_positions=image_positions,
                              decoder_mask=decoder_mask, row_len=seq_len)

def qwen_style_hf_config(raw: dict) -> dict:
    vc = raw["vision_config"]
    return {
        "model_type": "qwen3_5",
        "tie_word_embeddings": raw["text_config"].get("tie_word_embeddings", True),
        "text_config": raw["text_config"],
        "image_token_id": raw["image_pad_token_id"],
        "video_token_id": raw.get("video_token_id", 248057),  # <|video_pad|>, unused
        "vision_start_token_id": raw["vision_start_token_id"],
        "vision_end_token_id": raw["vision_end_token_id"],
        "vision_config": {
            "depth": vc["num_hidden_layers"],
            "hidden_size": vc["hidden_size"],
            "intermediate_size": vc["intermediate_size"],
            "num_heads": vc["num_attention_heads"],
            "patch_size": vc["patch_size"],
            "in_channels": vc.get("num_channels", 3),
            "temporal_patch_size": TEMPORAL_PATCH,
            "spatial_merge_size": SPATIAL_MERGE,
            "out_hidden_size": raw["llm_hidden_size"],
            "num_position_embeddings": (vc["image_size"] // vc["patch_size"]) ** 2,
        },
    }
