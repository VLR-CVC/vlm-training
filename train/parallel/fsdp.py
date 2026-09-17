# Vendored from torchtitan b21f7d43e: torchtitan/distributed/fsdp.py (dense path:
# apply_fsdp_to_vision_encoder, apply_fsdp_to_decoder, disable_fsdp_gradient_division),
# extended so the same wrapping units drive either FSDP2 or FSDP2-style replicate.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

import logging

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import FSDPModule
from torch.distributed._composable.replicate_with_fsdp import replicate
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import DataParallelMeshDims, fully_shard, MixedPrecisionPolicy

from train.parallel.parallel_dims import ParallelDims

logger = logging.getLogger(__name__)


_DENSE_STORAGE_AXES = ["dp_replicate", "dp_shard", "cp", "tp"]

def resolve_fsdp_mesh(
    parallel_dims: ParallelDims,
) -> tuple[DeviceMesh, DataParallelMeshDims | None]:
    """Dense storage mesh and the DP axes FSDP shards over (torchtitan
    ``distributed/fsdp.py``). ``dp_shard`` is always kept alive so FSDP can find the
    DP submesh inside the (dp_replicate, dp_shard, cp, tp) storage mesh."""
    storage_mesh = parallel_dims.get_activated_mesh(_DENSE_STORAGE_AXES)
    assert storage_mesh is not None
    if storage_mesh.size() == 1:
        return storage_mesh, None
    shard_axes = ["dp_shard"] + (["cp"] if parallel_dims.cp_enabled else [])
    shard = tuple(shard_axes) if len(shard_axes) > 1 else shard_axes[0]
    replicate_axis = "dp_replicate" if parallel_dims.dp_replicate_enabled else None
    return storage_mesh, DataParallelMeshDims(shard=shard, replicate=replicate_axis)

def disable_fsdp_gradient_division(model: nn.Module) -> None:
    """Reduce gradients as a SUM. The loss is already divided by the global
    valid-token count, so FSDP's default mean over DP would divide twice.
    ``ReplicateModule`` inherits ``FSDPModule``, so this covers both modes."""
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.set_gradient_divide_factor(1.0)

def apply_data_parallel(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    mode: str,
    param_dtype: torch.dtype,
    reduce_dtype: torch.dtype,
    reshard_after_forward: bool = False,
    dp_mesh_dims: DataParallelMeshDims | None = None,
) -> None:
    """Wrap a Qwen3.5-TT model for data parallelism, torchtitan's units:

    - vision encoder as ONE unit (one all-gather for the whole tower)
    - ``tok_embeddings``; ``[norm, lm_head]`` together (or all three when tied)
    - every decoder block
    - the root

    ``mode="fsdp"`` shards, ``mode="ddp"`` replicates (FSDP2 replicate, which
    takes the same MixedPrecisionPolicy and gradient-sync controls).

    Under TP pass the storage mesh and ``dp_mesh_dims`` from ``resolve_fsdp_mesh``:
    parameters are then TP-local plain tensors with SPMD annotations, which
    ``fully_shard`` turns into DTensors over the whole mesh.
    """
    if mode not in ("fsdp", "ddp"):
        raise ValueError(f"data_parallel must be 'fsdp' or 'ddp', got {mode!r}")
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype, reduce_dtype=reduce_dtype, cast_forward_inputs=False
    )
    # upstream apply_fsdp_to_vision_encoder keeps the default cast_forward_inputs=True:
    # the encoder casts pixel_values to its (sharded, master-dtype) weight dtype, and
    # FSDP must bring them back to param_dtype. Without it fp32 master weights fail
    # with "mat1 and mat2 must have the same dtype, but got Float and BFloat16".
    vision_mp_policy = MixedPrecisionPolicy(param_dtype=param_dtype, reduce_dtype=reduce_dtype)
    if mode == "ddp" and dp_mesh_dims is not None:
        raise NotImplementedError("replicate on a multi-axis (TP) mesh is not ported; use fsdp")
    extra = {"dp_mesh_dims": dp_mesh_dims} if dp_mesh_dims is not None else {}
    if mode == "fsdp":
        def wrap(module, *, reshard, policy=mp_policy):
            fully_shard(module, mesh=mesh, mp_policy=policy, reshard_after_forward=reshard, **extra)
    else:
        def wrap(module, *, reshard, policy=mp_policy):
            replicate(module, mesh=mesh, mp_policy=policy)

    if model.vision_encoder is not None:
        wrap(model.vision_encoder, reshard=reshard_after_forward, policy=vision_mp_policy)

    if model.enable_weight_tying:
        # tok_embeddings and lm_head share one parameter: one unit, one all-gather
        wrap([model.tok_embeddings, model.norm, model.lm_head], reshard=False)
    else:
        wrap(model.tok_embeddings, reshard=reshard_after_forward)
        # not resharded after forward: FSDP would prefetch them immediately
        wrap([model.norm, model.lm_head], reshard=False)

    for block in model.layers.values():
        wrap(block, reshard=reshard_after_forward)
    wrap(model, reshard=reshard_after_forward)

    disable_fsdp_gradient_division(model)
    logger.info(
        f"applied {'FSDP' if mode == 'fsdp' else 'replicate'} over {mesh.size()} ranks "
        f"(param {param_dtype}, reduce {reduce_dtype}, reshard_after_forward={reshard_after_forward})"
    )
