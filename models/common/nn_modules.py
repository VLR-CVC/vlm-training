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
from torch.autograd.function import once_differentiable

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

@spmd.register_local_autograd_function
class _RouterGateLinearFunction(torch.autograd.Function):
    """Router projection with FP32 output and backward GEMMs."""

    @staticmethod
    def forward(  # pyrefly: ignore[bad-override]
        ctx, input_TD: torch.Tensor, weight_ED: torch.Tensor
    ) -> torch.Tensor:
        use_cuda_bf16_forward = (
            input_TD.device.type == "cuda"
            and input_TD.dtype is torch.bfloat16
            and weight_ED.dtype is torch.bfloat16
        )
        if use_cuda_bf16_forward:
            input_forward_TD = input_TD
            weight_forward_ED = weight_ED
            # CUDA supports BF16 matmul with FP32 accumulation and output via
            # out_dtype. The portable path below promotes the operands because
            # this mixed input/output dtype is not supported by all devices.
            output_TE = torch.mm(
                input_forward_TD, weight_forward_ED.T, out_dtype=torch.float32
            )
        else:
            input_forward_TD = input_TD.float()
            weight_forward_ED = weight_ED.float()
            output_TE = torch.mm(input_forward_TD, weight_forward_ED.T)

        ctx.save_for_backward(input_forward_TD, weight_forward_ED)
        ctx.input_dtype = input_TD.dtype
        ctx.weight_dtype = weight_ED.dtype
        return output_TE

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_output_TE: torch.Tensor):  # pyrefly: ignore[bad-override]
        input_forward_TD, weight_forward_ED = ctx.saved_tensors
        grad_output_fp32_TE = grad_output_TE.float()

        grad_input_TD = None
        if ctx.needs_input_grad[0]:
            grad_input_TD = torch.mm(grad_output_fp32_TE, weight_forward_ED.float()).to(
                ctx.input_dtype
            )

        grad_weight_ED = None
        if ctx.needs_input_grad[1]:
            grad_weight_ED = torch.mm(
                grad_output_fp32_TE.T, input_forward_TD.float()
            ).to(ctx.weight_dtype)

        return grad_input_TD, grad_weight_ED

class RouterGateLinear(Linear):
    """Router projection with FP32 output and backward compute.

    CUDA uses BF16 forward compute when both operands are BF16. All other
    forward paths use FP32 compute.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Linear.Config):
        pass

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        output_TE = _RouterGateLinearFunction.apply(input, self.weight)
        if self.bias is not None:
            output_TE = output_TE + self.bias.float()
        return output_TE

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
