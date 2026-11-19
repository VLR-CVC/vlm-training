# Order and structure from torchtitan b21f7d43e: torchtitan/models/qwen3_5/parallelize.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
"""TP (``Module.parallelize``) -> compile -> FSDP/replicate, on a meta model."""

import torch
import torch.nn as nn

from train.parallel.ac import apply_ac
from train.parallel.compile import apply_compile
from train.parallel.engram import (
    ENGRAM_SHARDING_MODES,
    apply_engram_row_sharding,
    engram_row_mesh,
)
from train.parallel.fsdp import (
    apply_data_parallel,
    resolve_fsdp_mesh,
    resolve_sparse_fsdp_mesh,
)
from train.parallel.parallel_dims import ParallelDims

def parallelize_qwen3_5(
    model: nn.Module,
    parallel_dims: ParallelDims,
    *,
    mode: str,
    compile: bool,
    async_tp: bool = False,
    selective_ac_freq: int = 0,
    selective_ac_op: bool = False,
    selective_ac_vision: bool = False,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward: bool = False,
    engram_sharding: str = "fsdp",
    engram_table_device: str = "gpu",
) -> nn.Module:
    """``model`` must be built from a config that went through
    ``apply_parallelism_config``; with TP=1 the sharding configs still wrap each
    module but every redistribution is over a size-1 axis."""
    if parallel_dims.cp_enabled:
        raise NotImplementedError(
            "Context Parallel is not supported for Qwen3.5: GatedDeltaNet needs the "
            "full sequence."
        )
    model.parallelize(parallel_dims)
    # AC before compile and before FSDP, the order torchtitan uses.
    apply_ac(
        model,
        selective_ac_freq,
        op_level=selective_ac_op,
        include_vision=selective_ac_vision,
    )
    if compile:
        apply_compile(model, parallel_dims=parallel_dims, enable_async_tp=async_tp)
    ignored_params = set()
    if engram_sharding == "rows":
        # before FSDP: the tables become DTensors FSDP leaves alone
        ignored_params = apply_engram_row_sharding(
            model, engram_row_mesh(parallel_dims),
            compute_dtype=param_dtype, grad_dtype=reduce_dtype,
            host=engram_table_device == "host",
        )
    elif engram_table_device != "gpu":
        raise ValueError("engram_table_device='host' needs engram_sharding='rows'")
    elif engram_sharding != "fsdp":
        raise ValueError(f"engram_sharding must be one of {ENGRAM_SHARDING_MODES}, got {engram_sharding!r}")
    mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims, mode)
    # (None, None) unless EP is on: the routed experts are already sharded over
    # the `ep` axis, so FSDP has to shard them over `efsdp` and not over the
    # dense DP axis.
    edp_mesh, edp_mesh_dims = resolve_sparse_fsdp_mesh(parallel_dims)
    apply_data_parallel(
        model,
        mesh,
        mode=mode,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        reshard_after_forward=reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
        edp_mesh=edp_mesh,
        edp_mesh_dims=edp_mesh_dims,
        ignored_params=ignored_params,
    )
    return model
