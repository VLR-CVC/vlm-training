# Vendored from torchtitan b21f7d43e: torchtitan/models/common/attention.py, trimmed
# to varlen + flex inner attention (no SDPA, CP, batch-invariant mode or GQAttention).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

# Shape suffixes: T = packed tokens, H = heads, K = q/k head dim, V = value head dim.

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import ClassVar, NamedTuple

import spmd_types as spmd
import torch
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    BlockMask,
    flex_attention,
)
from torch.nn.attention.varlen import varlen_attn as _varlen_attn

from models.common.module import Module

__all__ = [
    "AttentionMasksType",
    "BaseAttention",
    "FlexInnerAttention",
    "InnerAttention",
    "VarlenInnerAttention",
    "VarlenMetadata",
    "create_varlen_metadata_for_document",
    "local_head_split",
]

class VarlenMetadata(NamedTuple):
    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: int
    max_k: int

AttentionMasksType = (
    Mapping[str, BlockMask | VarlenMetadata | None] | BlockMask | VarlenMetadata
)

@spmd.no_typecheck(out_types=spmd.PartitionSpec(("dp", "cp"), "tp", None))
def varlen_attn(*args, **kwargs):
    return _varlen_attn(*args, **kwargs)

def local_head_split(t: torch.Tensor, head_dim: int, *, dp_shard_dim: int = 0) -> torch.Tensor:
    input_type = {"dp": spmd.S(dp_shard_dim), "tp": spmd.S(t.ndim - 1)}
    with spmd.local():
        if spmd.is_type_checking():
            spmd.assert_type(t, input_type)
        out = t.view(*t.shape[:-1], -1, head_dim)
        if spmd.is_type_checking():
            spmd.assert_type(out, input_type)
    return out

class InnerAttention(Module):
    """Base class for attention kernels used by outer attention modules."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        pass

class VarlenInnerAttention(InnerAttention):
    """``torch.nn.attention.varlen.varlen_attn`` over packed documents.

    Runs in bf16 (the flash kernels take fp16/bf16 only) and casts back.
    ``window_size=(-1, 0)`` is causal.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        # causal attention by default
        window_size: tuple[int, int] = (-1, 0)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.window_size = config.window_size

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        assert isinstance(attention_masks, VarlenMetadata), type(attention_masks)
        kwargs = {"enable_gqa": True} if enable_gqa else {}
        out = varlen_attn(
            q_THK.to(torch.bfloat16),
            k_THK.to(torch.bfloat16),
            v_THV.to(torch.bfloat16),
            attention_masks.cu_seq_q,
            attention_masks.cu_seq_k,
            attention_masks.max_q,
            attention_masks.max_k,
            scale=scale,
            window_size=self.window_size,
            **kwargs,
        )
        return out.to(q_THK.dtype)

class FlexInnerAttention(InnerAttention):
    """``flex_attention`` with a class-level compiled kernel.
    only used in the vision encoder.

    Inputs are ``[T, H, K]``; the kernel wants ``[1, H, T, K]``, adapted here.

    TODO: add support for index causal attn in the decoder
    for bidirectional attention between image features
    """

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        block_size: int | tuple[int, int] = _DEFAULT_SPARSE_BLOCK_SIZE
        kernel_options: dict = field(default_factory=dict)

    # One compiled instance per process; per-instance compiles are slow.
    _compiled_flex_attn: ClassVar = torch.compile(flex_attention)

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.kernel_options = config.kernel_options

    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: BlockMask,
        scale: float | None = None,
        enable_gqa: bool = False,
    ) -> torch.Tensor:
        assert isinstance(attention_masks, BlockMask), type(attention_masks)
        q = q_THK.transpose(0, 1).unsqueeze(0)
        k = k_THK.transpose(0, 1).unsqueeze(0)
        v = v_THV.transpose(0, 1).unsqueeze(0)
        with spmd.no_typecheck():
            out = FlexInnerAttention._compiled_flex_attn(
                q,
                k,
                v,
                block_mask=attention_masks,
                scale=scale,
                enable_gqa=enable_gqa,
                kernel_options=self.kernel_options,
            )
        if spmd.is_type_checking():
            spmd.assert_type(out, spmd.get_local_type(q), spmd.get_partition_spec(q))
        return out.squeeze(0).transpose(0, 1)

def create_varlen_metadata_for_document(positions: torch.Tensor) -> VarlenMetadata:
    num_tokens = positions.shape[0]
    doc_starts = (positions == 0).nonzero(as_tuple=True)[0].to(torch.int32)
    cu_seqlens = torch.cat(
        [doc_starts, torch.tensor([num_tokens], dtype=torch.int32, device=positions.device)]
    )
    seq_lengths = torch.diff(cu_seqlens)
    max_seqlen = int(seq_lengths.max().item()) if seq_lengths.numel() > 0 else 0
    # DEVIATION from torchtitan b21f7d43e: round up to a power of two
    max_seqlen = 1 if max_seqlen <= 1 else 1 << (max_seqlen - 1).bit_length()
    if spmd.is_type_checking():
        spmd.mutate_type(cu_seqlens, "dp", src=spmd.R, dst=spmd.V)
    return VarlenMetadata(cu_seqlens, cu_seqlens, max_seqlen, max_seqlen)

class BaseAttention(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        n_heads: int
        inner_attention: Module.Config

        def __post_init__(self):
            assert self.n_heads > 0, "n_heads must be > 0"
