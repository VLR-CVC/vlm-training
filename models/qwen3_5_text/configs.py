import json
from pathlib import Path

from models.qwen3_5.configs import (
    LAYER_TYPES,
    qwen35_config_from_hf,
)
from models.qwen3_5.model import Qwen35Model

ATTENTION_KINDS = ("hybrid", *LAYER_TYPES)
_ALIASES = {
    "softmax": "full_attention",
    "full": "full_attention",
    "gdn": "linear_attention",
    "linear": "linear_attention",
    "": "hybrid",
}

def resolve_attention_kind(kind: str) -> str:
    resolved = _ALIASES.get(kind, kind)
    if resolved not in ATTENTION_KINDS:
        raise ValueError(f"attention_kind {kind!r} not in {ATTENTION_KINDS} (aliases: {sorted(_ALIASES)})")
    return resolved

def layer_schedule(kind: str, n_layers: int) -> list[str] | None:
    kind = resolve_attention_kind(kind)
    return None if kind == "hybrid" else [kind] * n_layers

def qwen35_text_config_from_hf(
    config_path: str | Path,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    decoder_mask: str = "causal_doc",
    attention_kind: str = "hybrid",
    with_vision: bool = False,
) -> Qwen35Model.Config:
    """Text-only Qwen3.5 from an HF-format `config.json` or snapshot directory"""
    if with_vision:
        raise ValueError("models/qwen3_5_text is text-only; use models/qwen3_5 for vision")
    path = Path(config_path)
    raw = json.loads((path / "config.json" if path.is_dir() else path).read_text())
    n_layers = raw["text_config"]["num_hidden_layers"]
    return qwen35_config_from_hf(
        config_path,
        seq_len=seq_len,
        attn_backend=attn_backend,
        decoder_mask=decoder_mask,
        with_vision=False,
        layer_types=layer_schedule(attention_kind, n_layers),
    )
