# Vendored from torchtitan b21f7d43e: torchtitan/models/common/vision_encoder.py
# (no remat regions).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
"""Shared ViT building blocks: per-image varlen metadata, attention, MLP, block.

RoPE differs per model, so encoders pass ``rope_cache`` (a tensor) and
``rope_apply`` (``(q, k, rope_cache) -> (q, k)``) through the block.

Shape suffixes: T = packed visual tokens, D = vision dim, H = heads, Dh = head dim.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch

from models.common.attention import (
    create_varlen_metadata_for_document,
    local_head_split,
    VarlenInnerAttention,
    VarlenMetadata,
)
from models.common.module import Module
from models.common.nn_modules import GELU, LayerNorm, Linear

RopeApply = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]

def create_block_diagonal_mask(
    segment_lengths: torch.Tensor, total_tokens: int, device: torch.device
) -> VarlenMetadata:
    """Varlen offsets over contiguous packed segments; attention is bidirectional
    within a segment (`window_size=(-1, -1)`). Same bucketing as the decoder's."""
    lengths = segment_lengths.to(device=device, dtype=torch.long)
    starts = torch.cumsum(lengths, 0) - lengths
    positions = torch.ones(total_tokens, dtype=torch.int32, device=device)
    positions[starts] = 0
    return create_varlen_metadata_for_document(positions)

class VisionMLP(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        fc1: Linear.Config
        fc2: Linear.Config
        act_fn: GELU.Config = field(default_factory=lambda: GELU.Config(approximate="tanh"))

    def __init__(self, config: Config):
        super().__init__()
        self.linear_fc1 = config.fc1.build()
        self.linear_fc2 = config.fc2.build()
        self.act_fn = config.act_fn.build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

class VisionAttention(Module):
    """Multi-head self-attention over visual patches, bidirectional varlen kernel, injected RoPE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        proj: Linear.Config
        inner_attention: Module.Config = field(
            default_factory=lambda: VarlenInnerAttention.Config(window_size=(-1, -1))
        )

    def __init__(self, config: Config):
        super().__init__()
        if config.dim % config.num_heads != 0:
            raise ValueError(
                f"VisionAttention dim ({config.dim}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.head_dim = config.dim // config.num_heads
        self.attn_scale = float(config.dim // config.num_heads) ** -0.5
        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()
        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: VarlenMetadata,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        # -1 heads: under TP the colwise projections hold num_heads / TP heads
        q_THDh = local_head_split(self.wq(x), self.head_dim)
        k_THDh = local_head_split(self.wk(x), self.head_dim)
        v_THDh = local_head_split(self.wv(x), self.head_dim)
        q_THDh, k_THDh = rope_apply(q_THDh, k_THDh, rope_cache)
        out_THDh = self.inner_attention(
            q_THDh, k_THDh, v_THDh,
            attention_masks=attention_mask,
            scale=self.attn_scale,
        )
        return self.proj(out_THDh.reshape(num_tokens, -1))

class VisionTransformerBlock(Module):
    """Pre-norm block: norm -> attn -> residual -> norm -> mlp -> residual."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        norm1: LayerNorm.Config
        norm2: LayerNorm.Config
        attn: VisionAttention.Config
        mlp: VisionMLP.Config

    def __init__(self, config: Config):
        super().__init__()
        self.norm1 = config.norm1.build()
        self.norm2 = config.norm2.build()
        self.attn = config.attn.build()
        self.mlp = config.mlp.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: VarlenMetadata,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=rope_cache,
            rope_apply=rope_apply,
            attention_mask=attention_mask,
        )
        return x + self.mlp(self.norm2(x))
