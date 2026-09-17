# Vendored from torchtitan b21f7d43e: torchtitan/distributed/compile.py (apply_compile),
# without async TP and the regional-inductor backend (not used here).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

import logging

import torch
import torch.nn as nn
from torch._subclasses.fake_tensor import FakeTensorMode

logger = logging.getLogger(__name__)

# TODO: remove once FakeTensorMode.__init__ is decorated with
# @torch.compiler.disable(recursive=True) upstream (pytorch/pytorch#178887).
FakeTensorMode.__init__ = torch.compiler.disable(  # type: ignore[method-assign]
    FakeTensorMode.__init__, recursive=True
)


def apply_compile(model: nn.Module, *, backend: str = "inductor") -> None:
    """Compile every decoder block (and ViT block, if present) with
    ``fullgraph=True``: a graph break is an error, not a silent slowdown."""
    torch._dynamo.config.capture_scalar_outputs = True
    torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True

    for block in model.layers.values():
        block.compile(backend=backend, fullgraph=True)
    vision_encoder = getattr(model, "vision_encoder", None)
    if vision_encoder is not None:
        for block in vision_encoder.layers.values():
            block.compile(backend=backend, fullgraph=True)
    logger.info("compiled each transformer block with fullgraph=True")
