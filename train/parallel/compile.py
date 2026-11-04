# Vendored from torchtitan b21f7d43e: torchtitan/distributed/compile.py (apply_compile),
# without async TP and the regional-inductor backend (not used here).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

import warnings

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.distributed.device_mesh import DeviceMesh

from models.common.attention import InnerAttention
from train.logger import logger
from train.parallel.parallel_dims import ParallelDims

# TODO: remove once FakeTensorMode.__init__ is decorated with
# @torch.compiler.disable(recursive=True) upstream (pytorch/pytorch#178887).
FakeTensorMode.__init__ = torch.compiler.disable(  # type: ignore[method-assign]
    FakeTensorMode.__init__, recursive=True
)

def apply_compile(model: nn.Module, *, parallel_dims: ParallelDims | None = None,
                  enable_async_tp: bool = False, backend: str = "inductor") -> None:
    """Compile every decoder block (and ViT block, if present) with
    ``fullgraph=True``: a graph break is an error, not a silent slowdown.

    A block whose attention kernel declares ``supports_fullgraph = False``
    (sparse attention) drops to ``fullgraph=False`` instead, so Dynamo breaks
    at the kernel and still compiles the norms, MLP and RoPE around it. Without
    this the block raises rather than compiling, and demanding a full graph
    would mean rewriting a vendored kernel.

    ``parallel_dims`` is only needed to find the TP mesh for async TP, so the
    single-GPU parity tests call this with neither.
    """
    if enable_async_tp and parallel_dims is None:
        raise ValueError("enable_async_tp needs parallel_dims to find the TP mesh")
    # handles data-dependent dynamic shapes for MoE
    torch._dynamo.config.capture_scalar_outputs = True
    # Does not replay forward "side effects" (e.g. RoPE cache updates)
    # during AC recompute in backward
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    _maybe_enable_async_tp(
        enable_async_tp=enable_async_tp,
        tp_mesh=(
            parallel_dims.get_dense_tp_mesh()
            if parallel_dims is not None and parallel_dims.tp_enabled
            else None
        ),
    )

    blocks = list(model.layers.values())
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is not None:
        blocks += list(vision_encoder.layers.values())

    partial = 0
    for block in blocks:
        fullgraph = block_supports_fullgraph(block)
        partial += not fullgraph
        block.compile(backend=backend, fullgraph=fullgraph)

    if partial:
        logger.info(
            "compiled %d transformer blocks with fullgraph=True and %d with "
            "fullgraph=False (sparse attention kernel)",
            len(blocks) - partial,
            partial,
        )
    else:
        logger.info("compiled each transformer block with fullgraph=True")

def block_supports_fullgraph(block: nn.Module) -> bool:
    """False if any attention kernel in ``block`` cannot be traced whole."""
    return all(
        module.supports_fullgraph
        for module in block.modules()
        if isinstance(module, InnerAttention)
    )

def _maybe_enable_async_tp(
    enable_async_tp: bool,
    tp_mesh: DeviceMesh | None,
) -> None:
    """Configure Inductor's async TP pass for the provided TP mesh."""
    if not enable_async_tp or tp_mesh is None:
        return

    group_name = tp_mesh.get_group().group_name
    # TODO: Remove this call once PyTorch automatically registers symmetric
    # memory for process groups used by async TP:
    # https://github.com/pytorch/pytorch/issues/193027
    from torch.distributed._symmetric_memory import (
        enable_symm_mem_for_group,  # pyrefly: ignore [deprecated]
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        enable_symm_mem_for_group(group_name)  # pyrefly: ignore [deprecated]

    torch._inductor.config._micro_pipeline_tp = True
    logger.info("Async TP is enabled")
