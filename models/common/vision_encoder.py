# Vendored from torchtitan b21f7d43e: torchtitan/models/common/vision_encoder.py
# (no remat regions).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
"""Shared ViT building blocks: block-diagonal flex mask, attention, MLP, block.

RoPE differs per model, so encoders pass ``rope_cache`` (a tensor) and
``rope_apply`` (``(q, k, rope_cache) -> (q, k)``) through the block.

Shape suffixes: T = packed visual tokens, D = vision dim, H = heads, Dh = head dim.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
from torch.nn.attention.flex_attention import BlockMask, create_block_mask

from models.common.attention import FlexInnerAttention, local_head_split
from models.common.module import Module
from models.common.nn_modules import GELU, LayerNorm, Linear

compiled_create_block_mask = torch.compile(create_block_mask)

RopeApply = Callable[
    [torch.Tensor, torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
]

def create_block_diagonal_mask(
    segment_lengths: torch.Tensor, total_tokens: int, device: torch.device
) -> BlockMask:
    """Flex mask over contiguous packed segments (bidirectional within a segment)."""
    segment_ids = torch.repeat_interleave(
        torch.arange(segment_lengths.shape[0], device=device, dtype=torch.int32),
        segment_lengths.to(device=device, dtype=torch.int32),
        output_size=total_tokens,
    )

    def mask_mod(b, h, q_idx, kv_idx):
        return segment_ids[q_idx] == segment_ids[kv_idx]

    return compiled_create_block_mask(
        mask_mod, 1, None, total_tokens, total_tokens, device=device
    )

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
    """Multi-head self-attention over visual patches, flex kernel, injected RoPE."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        num_heads: int
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        proj: Linear.Config
        inner_attention: Module.Config = field(default_factory=FlexInnerAttention.Config)

    def __init__(self, config: Config):
        super().__init__()
        if config.dim % config.num_heads != 0:
            raise ValueError(
                f"VisionAttention dim ({config.dim}) must be divisible by "
                f"num_heads ({config.num_heads})."
            )
        self.head_dim = config.dim // config.num_heads
        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.proj = config.proj.build()
        self.flex_attention = config.inner_attention.build()

    def forward(
        self,
        x: torch.Tensor,
        *,
        rope_cache: torch.Tensor,
        rope_apply: RopeApply,
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        num_tokens = x.shape[0]
        # -1 heads: under TP the colwise projections hold num_heads / TP heads
        q_THDh = local_head_split(self.wq(x), self.head_dim)
        k_THDh = local_head_split(self.wk(x), self.head_dim)
        v_THDh = local_head_split(self.wv(x), self.head_dim)
        q_THDh, k_THDh = rope_apply(q_THDh, k_THDh, rope_cache)
        out_THDh = self.flex_attention(q_THDh, k_THDh, v_THDh, attention_masks=attention_mask)
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
        attention_mask: BlockMask,
    ) -> torch.Tensor:
        x = x + self.attn(
            self.norm1(x),
            rope_cache=rope_cache,
            rope_apply=rope_apply,
            attention_mask=attention_mask,
        )
        return x + self.mlp(self.norm2(x))
