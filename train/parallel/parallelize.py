# Order and structure from torchtitan b21f7d43e: torchtitan/models/qwen3_5/parallelize.py.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
"""TP (``Module.parallelize``) -> compile -> FSDP/replicate, on a meta model."""

import torch
import torch.nn as nn

from train.parallel.compile import apply_compile
from train.parallel.fsdp import apply_data_parallel, resolve_fsdp_mesh
from train.parallel.parallel_dims import ParallelDims

def parallelize_qwen3_5(
    model: nn.Module,
    parallel_dims: ParallelDims,
    *,
    mode: str,
    compile: bool,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward: bool = False,
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
    if compile:
        apply_compile(model)
    mesh, dp_mesh_dims = resolve_fsdp_mesh(parallel_dims, mode)
    apply_data_parallel(
        model,
        mesh,
        mode=mode,
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        reshard_after_forward=reshard_after_forward,
        dp_mesh_dims=dp_mesh_dims,
    )
    return model
