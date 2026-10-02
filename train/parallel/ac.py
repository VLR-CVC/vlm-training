# Layer-selective activation checkpointing. Ported from torchtitan 2b9565ed:
# torchtitan/distributed/activation_checkpoint.py (`_apply_layer_sac`), which is
# what upstream runs by default (`mode="selective"`, `selective_ac_option="2"`).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/main/LICENSE

from collections import defaultdict
from collections.abc import Iterator

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)
from torch.utils.checkpoint import (
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)

from train.logger import logger

def _op_save_list() -> set:
    save = {
        torch.ops.aten.mm.default,
        torch.ops.aten.max.default,
        torch.ops._c10d_functional.reduce_scatter_tensor.default,
        torch._higher_order_ops.flex_attention,
    }
    # `torch_attn` only exists once torch.nn.attention.varlen has been imported.
    varlen = getattr(torch.ops.torch_attn, "_varlen_attn", None)
    if varlen is not None:
        save.add(varlen.default)
    # Every attn_gym kernel, is added to save list
    gym = getattr(torch.ops, "attn_gym", None)
    if gym is not None:
        for name in dir(gym):
            if name.startswith("_") or name == "name":
                continue
            overload = getattr(getattr(gym, name), "default", None)
            if overload is not None:
                save.add(overload)
    return save

def _apply_op_sac(module: nn.Module) -> nn.Module:
    """Per-op selective checkpointing: save the expensive ops, recompute the rest"""
    save_list = _op_save_list()

    def _policy_fn(meta):
        def _policy(ctx, func, *args, **kwargs):
            key = "recompute_mm" if ctx.is_recompute else "forward_mm"
            if func == torch.ops.aten.mm.default:
                meta[key] += 1
                # save odd-numbered mms, recompute even-numbered ones
                if meta[key] % 2 == 0:
                    return CheckpointPolicy.PREFER_RECOMPUTE
            return (
                CheckpointPolicy.MUST_SAVE
                if func in save_list
                else CheckpointPolicy.PREFER_RECOMPUTE
            )

        return _policy

    def _context_fn():
        return create_selective_checkpoint_contexts(_policy_fn(defaultdict(int)))

    return ptd_checkpoint_wrapper(
        module, preserve_rng_state=False, context_fn=_context_fn
    )

def _block_owners(model: nn.Module, include_vision: bool) -> Iterator[nn.Module]:
    yield model.layers
    vision_encoder = getattr(model, "vision_encoder", None)
    if include_vision and vision_encoder is not None:
        yield vision_encoder.layers

def apply_ac(
    model: nn.Module, freq: int, *, op_level: bool = False, include_vision: bool = True
) -> None:
    """Wrap every ``freq``-th block in an activation checkpoint, in place"""
    if freq <= 0:
        return

    seen = wrapped = 0
    for owner in _block_owners(model, include_vision):
        for name, block in list(owner.named_children()):
            seen += 1
            if op_level:
                owner.register_module(name, _apply_op_sac(block))
                wrapped += 1
            elif seen % freq == 0:
                owner.register_module(
                    name, ptd_checkpoint_wrapper(block, preserve_rng_state=False)
                )
                wrapped += 1

    if op_level:
        logger.info(
            f"activation checkpointing: op-level SAC on {wrapped}/{seen} blocks "
            f"({len(_op_save_list())} ops in the save list)"
        )
        return

    logger.info(
        f"activation checkpointing: wrapped {wrapped}/{seen} blocks (every {freq})"
    )
