# Vendored from torchtitan b21f7d43e: torchtitan/models/common/{nn_modules,linear,
# embedding,param_init}.py, trimmed to the modules Qwen3.5 uses.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
"""Configurable wrappers around ``torch.nn`` modules.

Each class uses diamond inheritance (``nn.X`` + ``Module``) so the module tree
stays flat and all ``nn.X`` behaviour (forward, state_dict) is reused as-is.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# our own module definition
from models.common.module import Module
from train.parallel.spmd import spmd_mesh_group

def skip_param_init(param: nn.Parameter) -> None:
    """No-op initializer, for parameters tied to another one."""

def depth_scaled_std(base_std: float, layer_id: int) -> float:
    """``base_std / sqrt(2 * (layer_id + 1))``."""
    return base_std / (2 * (layer_id + 1)) ** 0.5

class Linear(nn.Linear, Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_features: int
        out_features: int
        bias: bool = False

    def __init__(self, config: Config):
        super().__init__(config.in_features, config.out_features, bias=config.bias)

class PartialBiasRowwiseLinear(Linear):
    """Rowwise linear whose invariant bias becomes TP-partial in forward.

    Under TP the rowwise output is summed across ranks, so each rank adds the
    bias as a partial (``P``) value to get it exactly once.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def __init__(self, config: Config):
        if not config.bias:
            raise ValueError("PartialBiasRowwiseLinear requires bias=True")
        super().__init__(config)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        bias = self.bias
        tp_group = spmd_mesh_group("tp")
        if tp_group is not None:
            bias = spmd.convert(bias, tp_group, src=spmd.I, dst=spmd.P, expert_mode=True)
        return F.linear(input, self.weight, bias)

class Embedding(nn.Embedding, Module):
    """Embedding with local vocab-parallel execution when a TP mesh is active."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        num_embeddings: int
        embedding_dim: int

    def __init__(self, config: Config):
        super().__init__(config.num_embeddings, config.embedding_dim)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        tp_group = spmd_mesh_group("tp")
        if tp_group is None:
            return F.embedding(input, self.weight)
        tp_size = dist.get_world_size(tp_group)
        chunk_size = (self.num_embeddings + tp_size - 1) // tp_size
        offset = dist.get_rank(tp_group) * chunk_size
        mask = (input >= offset) & (input < offset + self.weight.shape[0])
        local_input = (input - offset).clamp(0, self.weight.shape[0] - 1)
        out = F.embedding(local_input, self.weight)
        return out * mask.unsqueeze(-1).to(out.dtype)

class Conv1d(nn.Conv1d, Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        in_channels: int
        out_channels: int
        kernel_size: int
        stride: int = 1
        padding: int = 0
        groups: int = 1
        bias: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.in_channels,
            config.out_channels,
            config.kernel_size,
            stride=config.stride,
            padding=config.padding,
            groups=config.groups,
            bias=config.bias,
        )

class GELU(nn.GELU, Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        approximate: str = "none"

    def __init__(self, config: Config):
        super().__init__(approximate=config.approximate)

class LayerNorm(nn.LayerNorm, Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        normalized_shape: int
        eps: float = 1e-5
        elementwise_affine: bool = True

    def __init__(self, config: Config):
        super().__init__(
            config.normalized_shape,
            eps=config.eps,
            elementwise_affine=config.elementwise_affine,
        )
