# Vendored from torchtitan b21f7d43e: torchtitan/models/common/feed_forward.py and the
# FFN helpers of torchtitan/models/common/config_utils.py (no remat regions).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

# Shape suffixes: T = tokens, D = model dim, F = feed-forward hidden dim.

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from models.common.module import Module
from models.common.nn_modules import Linear

__all__ = ["FeedForward", "make_ffn_config"]

class FeedForward(Module):
    """SwiGLU FFN with one fused, interleaved gate/up projection ``w13``.

    Checkpoints still see ``w1`` (gate) and ``w3`` (up): the state-dict hooks
    split on save and merge on load.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w13: Linear.Config
        w2: Linear.Config

    def __init__(self, config: Config):
        super().__init__()
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.register_state_dict_post_hook(self._split_w13_on_save)
        self.register_load_state_dict_pre_hook(self._merge_w13_on_load)

    @staticmethod
    def _split_w13_on_save(module, state_dict, prefix, local_metadata) -> None:
        for param_name in ("weight", "bias"):
            fused_key = f"{prefix}w13.{param_name}"
            if fused_key not in state_dict:
                continue
            gate_up = state_dict.pop(fused_key).unflatten(0, (-1, 2))
            state_dict[f"{prefix}w1.{param_name}"] = gate_up[:, 0].contiguous()
            state_dict[f"{prefix}w3.{param_name}"] = gate_up[:, 1].contiguous()

    @staticmethod
    def _merge_w13_on_load(module, state_dict, prefix, *args) -> None:
        for param_name in ("weight", "bias"):
            gate_key = f"{prefix}w1.{param_name}"
            up_key = f"{prefix}w3.{param_name}"
            if gate_key not in state_dict or up_key not in state_dict:
                continue
            state_dict[f"{prefix}w13.{param_name}"] = torch.stack(
                [state_dict.pop(gate_key), state_dict.pop(up_key)], dim=1
            ).flatten(0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_TF, up_TF = self.w13(x).unflatten(-1, (-1, 2)).unbind(-1)
        return self.w2(F.silu(gate_TF) * up_TF)

def _make_fused_linear_init(gate_init: Callable, up_init: Callable) -> Callable:
    def _init(t: torch.Tensor) -> None:
        gate_up = t.unflatten(0, (-1, 2))
        gate_init(gate_up[:, 0])
        up_init(gate_up[:, 1])

    return _init

def make_ffn_config(
    *,
    dim: int,
    hidden_dim: int,
    w1_param_init: dict[str, Callable],
    w2w3_param_init: dict[str, Callable],
) -> FeedForward.Config:
    return FeedForward.Config(
        w13=Linear.Config(
            in_features=dim,
            out_features=2 * hidden_dim,
            param_init={
                "weight": _make_fused_linear_init(
                    w1_param_init["weight"], w2w3_param_init["weight"]
                )
            },
        ),
        w2=Linear.Config(in_features=hidden_dim, out_features=dim, param_init=w2w3_param_init),
    )
