# Vendored from torchtitan b21f7d43e: torchtitan/models/common/decoder.py and
# torchtitan/protocols/model.py, trimmed (no MoE, CP, PP, flex masks for the decoder).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

from dataclasses import dataclass

import torch

from models.common.module import Module, ModuleDict
from models.common.nn_modules import Embedding, Linear

__all__ = ["Decoder"]

class Decoder(Module):
    """Decoder-only language model base: embeddings, layers, final norm, head.

    ``_skip_lm_head`` is set by a chunked loss that applies ``lm_head`` per
    chunk itself, so ``forward`` returns hidden states.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        vocab_size: int
        lm_head: Linear.Config
        tok_embeddings: Embedding.Config
        norm: Module.Config
        layers: list
        # Share one weight between ``tok_embeddings`` and ``lm_head``.
        enable_weight_tying: bool = False

        @property
        def first_attention(self):
            """Attention config of the first layer that has one (hybrid models
            do not carry one on every layer), else None."""
            return next(
                (layer.attention for layer in self.layers if layer.attention is not None),
                None,
            )

    _skip_lm_head: bool = False

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.tok_embeddings = config.tok_embeddings.build()
        self.layers = ModuleDict()
        for i, layer_config in enumerate(config.layers):
            self.layers[str(i)] = layer_config.build()
        self.norm = config.norm.build()
        self.lm_head = config.lm_head.build()
        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def init_states(self, *, buffer_device: torch.device | None = None) -> None:
        if self.enable_weight_tying:
            # `to_empty` re-materialises parameters and breaks the tie; restore it
            # before init so the embedding is not initialised on its own.
            self.tok_embeddings.weight = self.lm_head.weight
        super().init_states(buffer_device=buffer_device)

    def retie_weights(self) -> None:
        """Restore the embedding/head tie after ``to_empty``."""
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight
