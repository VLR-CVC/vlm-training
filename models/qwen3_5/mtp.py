# Adapted from torchtitan b21f7d43e: `MTPTransformerBlock` and `MTPDecoder` of
# torchtitan/models/deepseek_v3/mtp.py, with the design note at
# torchtitan/models/deepseek_v3/MTP.md.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

import copy
from dataclasses import dataclass, field

import torch

from models.common.mtp import roll_mtp_sequence
from models.common.nn_modules import Linear
from train.loss import IGNORE_INDEX

from .model import OffsetRMSNorm, Qwen35AttentionMaskDict, Qwen35TransformerBlock

__all__ = [
    "Qwen35MTPBlock",
    "mtp_depth_batch",
    "qwen35_mtp_layer_configs",
]

class Qwen35MTPBlock(Qwen35TransformerBlock):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35TransformerBlock.Config):
        enorm: OffsetRMSNorm.Config
        hnorm: OffsetRMSNorm.Config
        eh_proj: Linear.Config
        mtp_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__(config)
        self.enorm = config.enorm.build()
        self.hnorm = config.hnorm.build()
        self.eh_proj = config.eh_proj.build()
        self.mtp_norm = config.mtp_norm.build()

    def forward(  # pyrefly: ignore [bad-override]
        self,
        mtp_input_embed_TD: torch.Tensor,
        prev_embed_TD: torch.Tensor,
        mtp_input_valid_mask_T: torch.Tensor,
        attention_masks: Qwen35AttentionMaskDict | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            mtp_input_embed_TD: embeddings of this depth's shifted tokens.
            prev_embed_TD: hidden state from the previous depth (the main
                decoder's pre-norm output at depth 1), at the SAME positions.
            mtp_input_valid_mask_T: false where the shift ran off the end of the
                row or crossed a packed-document boundary. The previous hidden
                state is zeroed there so an invalid position cannot carry
                another document's state into this one.
            positions: the ORIGINAL positions, never rolled -- each output stays
                anchored at its own position for attention and RoPE, and only
                the embedding branch moves. Rolling them would move the output.
        """
        mask_T1 = mtp_input_valid_mask_T.unsqueeze(-1).to(dtype=prev_embed_TD.dtype)
        prev_embed_TD = prev_embed_TD * mask_T1
        h_TD = self.eh_proj(
            torch.cat(
                [self.enorm(mtp_input_embed_TD), self.hnorm(prev_embed_TD)], dim=-1
            )
        )
        h_TD = super().forward(h_TD, attention_masks, positions)
        return self.mtp_norm(h_TD)

def qwen35_mtp_layer_configs(
    last_layer: Qwen35TransformerBlock.Config,
    *,
    dim: int,
    num_mtp_layers: int,
    eps: float = 1e-6,
    param_init: dict | None = None,
    norm_param_init: dict | None = None,
) -> list["Qwen35MTPBlock.Config"]:
    if num_mtp_layers < 0:
        raise ValueError(f"num_mtp_layers must be >= 0, got {num_mtp_layers}")
    norm_init = norm_param_init or {}

    def _norm() -> OffsetRMSNorm.Config:
        return OffsetRMSNorm.Config(dim=dim, eps=eps, param_init=norm_init)

    return [
        Qwen35MTPBlock.Config(
            attention=copy.deepcopy(last_layer.attention),
            delta_net=copy.deepcopy(last_layer.delta_net),
            feed_forward=copy.deepcopy(last_layer.feed_forward),
            moe=copy.deepcopy(last_layer.moe),
            attention_norm=copy.deepcopy(last_layer.attention_norm),
            ffn_norm=copy.deepcopy(last_layer.ffn_norm),
            enorm=_norm(),
            hnorm=_norm(),
            eh_proj=Linear.Config(
                in_features=dim * 2,
                out_features=dim,
                bias=False,
                param_init=param_init or {},
            ),
            mtp_norm=_norm(),
        )
        for _ in range(num_mtp_layers)
    ]

def mtp_depth_batch(
    tokens: torch.Tensor,
    labels: torch.Tensor,
    positions: torch.Tensor,
    *,
    num_depths: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    depth_tokens, depth_labels, valid_masks = [], [], []
    for depth in range(1, num_depths + 1):
        shifted, valid = roll_mtp_sequence(
            tokens,
            shift=depth,
            positions=positions,
            fill_value=0,
            return_valid_mask=True,
        )
        depth_tokens.append(shifted)
        depth_labels.append(
            roll_mtp_sequence(
                labels, shift=depth, positions=positions, fill_value=IGNORE_INDEX
            )
        )
        valid_masks.append(valid)
    return tuple(depth_tokens), tuple(depth_labels), tuple(valid_masks)
