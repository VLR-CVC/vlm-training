# Vendored from torchtitan b21f7d43e: torchtitan/models/common/rope.py (CosSinRoPE
# only) and torchtitan/models/qwen3_5/rope.py (MRoPE).
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

from dataclasses import dataclass, field

import spmd_types as spmd
import torch

from models.common.module import Module

__all__ = ["CosSinRoPE", "MRoPE"]

@spmd.no_typecheck()
def _maybe_check_max_pos(positions: torch.Tensor, *, max_valid_pos: int) -> None:
    """Async bounds check on position ids; skipped under torch.compile."""
    if torch.compiler.is_compiling():
        return
    torch._assert_async(
        torch.all(positions <= max_valid_pos),
        f"position_ids exceed {max_valid_pos=}",
    )

@spmd.local_map(
    out_types=(
        {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
        spmd.PartitionSpec(("dp", "cp"), None, None),
    )
)
def _reshape_for_broadcast(
    rope_cache: torch.Tensor,
    query_shape: torch.Size | tuple[int, ...],
    positions: torch.Tensor | None = None,
) -> torch.Tensor:
    cache_width = rope_cache.shape[-1]
    num_tokens = query_shape[0]
    if positions is None:
        rope_cache = rope_cache[:num_tokens]
    else:
        rope_cache = rope_cache[positions]
    return rope_cache.view(num_tokens, 1, cache_width)

class CosSinRoPE(Module):
    """Rotate-half RoPE with a concatenated ``[cos, sin]`` cache of width ``2 * dim``."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        max_context_length: int
        theta: float = 10000.0

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.register_buffer("cache", self._precompute_cache(), persistent=False)

    def _precompute_cache(self) -> torch.Tensor:
        cfg = self.config
        inv_freq = 1.0 / (
            cfg.theta ** (torch.arange(0, cfg.dim, 2)[: (cfg.dim // 2)].float() / cfg.dim)
        )
        t = torch.arange(cfg.max_context_length, dtype=inv_freq.dtype, device=inv_freq.device)
        freqs = torch.outer(t, inv_freq).float()
        theta = torch.cat([freqs, freqs], dim=-1)
        return torch.cat([theta.cos(), theta.sin()], dim=-1)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        if buffer_device is None:
            buffer_device = self.cache.device
        with torch.device(buffer_device):
            self.cache = self._precompute_cache()

    def _reshape_cache(
        self, query: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if positions is not None:
            _maybe_check_max_pos(positions, max_valid_pos=self.cache.shape[0] - 1)
        return _reshape_for_broadcast(self.cache, query.shape, positions)

    @staticmethod
    def apply_rotary_emb(
        query: torch.Tensor, key: torch.Tensor | None, rope_cache: torch.Tensor
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        head_dim = query.shape[-1]
        cos = rope_cache[..., :head_dim]
        sin = rope_cache[..., head_dim:]
        query_f = query.float()
        xq_out = (query_f * cos) + (CosSinRoPE._rotate_half(query_f) * sin)
        if key is None:
            return xq_out.type_as(query)
        key_f = key.float()
        xk_out = (key_f * cos) + (CosSinRoPE._rotate_half(key_f) * sin)
        return xq_out.type_as(query), xk_out.type_as(key)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        return self.apply_rotary_emb(query, key, self._reshape_cache(query, positions))

class MRoPE(CosSinRoPE):
    """Interleaved multi-dimensional RoPE for temporal/height/width positions.

    2D ``(num_tokens, 3)`` positions build the interleaved cache; 1D text
    positions fall back to the plain ``CosSinRoPE`` lookup.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(CosSinRoPE.Config):
        mrope_section: list[int] = field(default_factory=lambda: [24, 20, 20])

    def __init__(self, config: Config):
        if len(config.mrope_section) != 3 or any(s < 0 for s in config.mrope_section):
            raise ValueError(f"bad mrope_section {config.mrope_section}")
        if sum(config.mrope_section) != config.dim // 2:
            raise ValueError(
                f"mrope_section must sum to dim // 2 ({config.dim // 2}), "
                f"got {config.mrope_section}."
            )
        super().__init__(config)

    def _reshape_cache(
        self, query: torch.Tensor, positions: torch.Tensor | None = None
    ) -> torch.Tensor:
        if positions is not None and positions.ndim == 2:
            if positions.shape[-1] != 3:
                raise ValueError(
                    "2D MRoPE positions must have shape (num_tokens, 3), "
                    f"got {tuple(positions.shape)}."
                )
            return self._compute_mrope_cache(positions)
        return super()._reshape_cache(query, positions)

    def _compute_mrope_cache(self, pos: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        rope_cache = self.cache
        _maybe_check_max_pos(pos, max_valid_pos=rope_cache.shape[0] - 1)
        head_dim = rope_cache.shape[-1] // 2
        cos_cache = rope_cache[:, :head_dim]
        sin_cache = rope_cache[:, head_dim:]

        # Temporal positions everywhere, then overwrite the interleaved
        # height/width columns with their own position ids.
        t_pos = pos[..., 0].long()
        mrope_cos = cos_cache[t_pos]
        mrope_sin = sin_cache[t_pos]
        half = head_dim // 2
        for dim, offset in enumerate((1, 2), start=1):
            length = cfg.mrope_section[dim] * 3
            low = torch.arange(offset, length, 3, device=rope_cache.device)
            col_indices = torch.cat([low, low + half])
            dim_pos = pos[..., dim].long()
            mrope_cos[..., col_indices] = cos_cache[:, col_indices][dim_pos]
            mrope_sin[..., col_indices] = sin_cache[:, col_indices][dim_pos]
        return torch.cat([mrope_cos, mrope_sin], dim=-1).unsqueeze(1)
