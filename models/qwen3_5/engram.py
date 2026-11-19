"""Engram for Qwen3.5: the options, and the configs `qwen35_config_from_hf` builds.

The module itself is generic (`models/common/engram.py`). This file holds what is
Qwen3.5-specific: the init rules and norm type of the rest of the model, the
fill token for missing n-gram context (the snapshot's EOS), and the vision
placeholder ids that must stay out of every n-gram.
"""

from dataclasses import dataclass, field
from functools import partial

import torch.nn as nn

from models.common.engram import Engram, EngramTable
from models.common.engram_hash import build_ngram_hash_spec
from models.common.nn_modules import Linear

__all__ = ["EngramArgs", "qwen35_engram_configs"]


@dataclass(frozen=True)
class EngramArgs:
    """What `train.config.Model`'s `engram_*` fields turn into. See the field
    docstrings there for what each one does and costs."""

    layers: tuple[int, ...] = (2,)
    orders: tuple[int, ...] = (2, 3)
    heads_per_order: int = 8
    rows_per_head: int = 262_144
    dim: int = 0
    """Concatenated width of all heads; 0 means the model's hidden size."""
    seed: int = 0
    gate_sqrt: bool = True
    table_init_std: float = 0.02
    pad_id: int | None = None
    """Fill for missing n-gram context; None reads `eos_token_id` from the config."""
    blocked_token_ids: tuple[int, ...] = field(default_factory=tuple)
    """Added to the vision placeholder ids read from the config."""


_ZERO_INIT = {"weight": nn.init.zeros_}


def qwen35_engram_configs(
    args: EngramArgs | None,
    *,
    dim: int,
    vocab_size: int,
    n_layers: int,
    raw_config: dict,
) -> list[Engram.Config]:
    """One `Engram.Config` per entry of ``args.layers`` (sorted). Module ``i``
    runs after decoder layer ``layers[i]`` (0-indexed), on its output."""
    if args is None:
        return []
    from .configs import _LINEAR_INIT, _offset_norm

    layers = sorted(set(args.layers))
    if not layers:
        raise ValueError("engram is on but engram_layers is empty")
    bad = [i for i in layers if not 0 <= i < n_layers]
    if bad:
        raise ValueError(f"engram_layers {bad} outside the model's layers [0, {n_layers})")
    num_heads = len(set(args.orders)) * args.heads_per_order
    width = args.dim or dim
    if width % num_heads:
        raise ValueError(
            f"engram_dim ({width}) must be divisible by the number of hash heads "
            f"({num_heads} = {len(set(args.orders))} orders x {args.heads_per_order})"
        )

    tc = raw_config.get("text_config", raw_config)
    pad_id = args.pad_id
    if pad_id is None:
        eos = tc.get("eos_token_id", raw_config.get("eos_token_id"))
        if isinstance(eos, list):
            eos = eos[0] if eos else None
        if eos is None:
            raise ValueError("the config has no eos_token_id; set model.engram_pad_id")
        pad_id = int(eos)
    blocked = {
        raw_config[k] for k in ("image_token_id", "video_token_id") if k in raw_config
    }
    blocked |= set(args.blocked_token_ids)

    configs = []
    for index, _ in enumerate(layers):
        spec = build_ngram_hash_spec(
            orders=args.orders,
            heads_per_order=args.heads_per_order,
            rows_per_head=args.rows_per_head,
            vocab_size=vocab_size,
            pad_id=pad_id,
            seed=args.seed,
            layer_index=index,
        )
        configs.append(
            Engram.Config(
                orders=spec.orders,
                heads_per_order=args.heads_per_order,
                rows_per_head=args.rows_per_head,
                vocab_size=vocab_size,
                pad_id=pad_id,
                seed=args.seed,
                layer_index=index,
                blocked_token_ids=tuple(sorted(blocked)),
                gate_sqrt=args.gate_sqrt,
                table=EngramTable.Config(
                    num_embeddings=spec.num_rows,
                    embedding_dim=width // num_heads,
                    param_init={
                        "weight": partial(nn.init.normal_, std=args.table_init_std)
                    },
                ),
                key_proj=Linear.Config(
                    in_features=width, out_features=dim, bias=False, param_init=_LINEAR_INIT
                ),
                # zero: the model is exactly the base model until the first update.
                # DeepSeek instead zero-inits the output conv and keeps W_V random
                # (D021, see the value-scale note in models/common/engram.py)
                value_proj=Linear.Config(
                    in_features=width, out_features=dim, bias=False, param_init=_ZERO_INIT
                ),
                key_norm=_offset_norm(dim),
                query_norm=_offset_norm(dim),
            )
        )
    return configs


def engram_layer_ids(args: EngramArgs | None) -> list[int]:
    return [] if args is None else sorted(set(args.layers))
