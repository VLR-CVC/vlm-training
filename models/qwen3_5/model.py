# Vendored from torchtitan b21f7d43e: torchtitan/models/qwen3_5/model.py
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
from torch import nn
from torch.nn.attention.flex_attention import BlockMask

from models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    VarlenInnerAttention,
    VarlenMetadata,
)
from models.common.decoder import Decoder
from models.common.module import Module
from models.common.multimodal import get_vision_positions, scatter_vision_embeds
from models.common.nn_modules import Linear
from models.common.rope import MRoPE
from models.common.decoder_sharding import decoder_input_sharding
from models.common.vision_encoder_sharding import multimodal_input_sharding
from train.parallel.parallel_dims import MeshAxisName, ParallelDims
from train.parallel.spmd import (
    annotate_input_spmd_types,
    set_current_spmd_mesh,
    spmd_local_context,
)

from .gdn import GatedDeltaNet
from .vision_encoder import Qwen35VisionEncoder

# Shape suffixes:
# T = packed tokens, D = model dimension, C = projection channels,
# H = attention heads,
# K = query/key head dimension, V = value head dimension,
# R = rotary dimension, P = non-rotary dimension.

Qwen35AttentionMaskDict = dict[str, BlockMask | VarlenMetadata | None]

class OffsetRMSNorm(Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``.

    Weight is zero-initialized so the norm starts as identity-scaled.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Upcast to float32 for numerical stability in pow/rsqrt
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return ((1.0 + self.weight.float()) * x).to(input_dtype)

class Qwen35Attention(BaseAttention):
    """Full attention with output gating and partial RoPE for Qwen3.5.

    Differences from GQAttention:
    - wq is 2x wider: produces both query and sigmoid gate
    - Partial RoPE: only first ``rotary_dim`` elements get RoPE
    - Output gating: ``attn_output * sigmoid(gate)`` before ``wo``
    - QK norm uses OffsetRMSNorm

    Uses separate ``wq``/``wk``/``wv`` instead of the common fused ``qkv_linear``
    (so this subclasses ``BaseAttention``, not ``GQAttention``): the 2x-wide,
    gated ``wq`` doesn't fit a fused QKV projection that TP-shards by head.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rotary_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: OffsetRMSNorm.Config
        k_norm: OffsetRMSNorm.Config
        inner_attention: Module.Config

    def __init__(self, config: Config):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.rotary_dim = config.rotary_dim
        self.enable_gqa = self.n_heads > self.n_kv_heads

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

        self.rope = config.rope.build()

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.scaling = self.head_dim**-0.5

        self.inner_attention = config.inner_attention.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]

        # wq is 2x wider: produces query + gate
        xq_gate_THC = self.wq(x_TD).view(num_tokens, -1, self.head_dim * 2)
        xq_THK, gate_THV = xq_gate_THC.chunk(2, dim=-1)
        xk_THK = self.wk(x_TD).view(num_tokens, -1, self.head_dim)
        xv_THV = self.wv(x_TD).view(num_tokens, -1, self.head_dim)

        # QK norm (before RoPE)
        xq_THK = self.q_norm(xq_THK)
        xk_THK = self.k_norm(xk_THK)

        # Partial RoPE: only first rotary_dim elements get positional encoding
        assert self.rotary_dim <= self.head_dim
        xq_THR, xq_THP = (
            xq_THK[..., : self.rotary_dim],
            xq_THK[..., self.rotary_dim :],
        )
        xk_THR, xk_THP = (
            xk_THK[..., : self.rotary_dim],
            xk_THK[..., self.rotary_dim :],
        )
        xq_THR, xk_THR = self.rope(xq_THR, xk_THR, positions)
        xq_THK = torch.cat([xq_THR, xq_THP], dim=-1)
        xk_THK = torch.cat([xk_THR, xk_THP], dim=-1)

        out_THV = self.inner_attention(
            xq_THK,
            xk_THK,
            xv_THV,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()

        # Output gating
        out_THV = out_THV * torch.sigmoid(gate_THV)
        out_TD = out_THV.view(num_tokens, -1)
        return self.wo(out_TD)

class Qwen35TransformerBlock(Module):
    """Hybrid transformer block for Qwen3.5.

    Each layer uses either full attention (Qwen35Attention) or linear
    attention (GatedDeltaNet), determined by which config is provided.
    Both types share the same FFN/MoE structure.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Qwen35Attention.Config | None = None
        delta_net: GatedDeltaNet.Config | None = None
        feed_forward: Module.Config | None = None
        moe: Module.Config | None = None
        attention_norm: OffsetRMSNorm.Config
        ffn_norm: OffsetRMSNorm.Config

    def __init__(self, config: Config):
        super().__init__()
        self.full_attn = config.attention is not None
        self.attn_mask_key = "quadratic_attention" if self.full_attn else "deltanet"

        if self.full_attn:
            self.attn = config.attention.build()  # pyrefly: ignore [missing-attribute]
        else:
            assert config.delta_net is not None
            self.attn = config.delta_net.build()

        self.moe_enabled = config.moe is not None
        if self.moe_enabled:
            # pyrefly: ignore [missing-attribute]
            self.moe = config.moe.build()
        else:
            assert config.feed_forward is not None
            self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: Qwen35AttentionMaskDict | None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        layer_mask = (
            attention_masks[self.attn_mask_key] if attention_masks is not None else None
        )
        h_TD = self.attention_norm(x_TD)
        if self.full_attn:
            h_TD = self.attn(h_TD, layer_mask, positions)
        else:
            h_TD = self.attn(h_TD, layer_mask)
        x_TD = x_TD + h_TD

        h_TD = self.ffn_norm(x_TD)
        if self.moe_enabled:
            x_TD = x_TD + self.moe(h_TD)
        else:
            x_TD = x_TD + self.feed_forward(h_TD)
        return x_TD

class Qwen35Model(Decoder):
    """Qwen3.5: Multimodal model with hybrid attention.

    Combines a hybrid decoder (GatedDeltaNet linear attention + full
    attention with output gating and partial RoPE) with a Vision
    Transformer encoder for multimodal understanding.

    Key architectural features:
    - Hybrid attention: 75% GatedDeltaNet (linear) + 25% full attention
    - Output gating on full attention: ``attn_out * sigmoid(gate)``
    - Partial RoPE: only first ``rotary_dim`` elements get positional encoding
    - OffsetRMSNorm: ``(1 + weight) * norm(x)`` with zero-init weight
    - MRoPE: 3D (temporal/height/width) position IDs for multimodal batches;
      text batches use the plain 1D positions
    - MoE variant: routed experts + shared expert with sigmoid gate

    MRoPE positions (shape ``(num_tokens, 3)``) are built by the dataloader as
    ``mrope_positions``. After building the attention masks from the 2D
    ``positions``, ``preprocess_inputs`` picks which one the RoPE layers see:
    ``mrope_positions`` when present (multimodal), else the 2D ``positions`` --
    the chosen tensor overwrites the single ``positions`` input. This keeps RoPE
    consistent across every pipeline stage even though the raw vision inputs
    (``pixel_values``/``grid_thw``) only reach the first stage. The per-layer
    MRoPE dispatches on the position rank.

    Forward pass flow::

        forward(tokens, pixel_values, grid_thw, positions, ...)
          │
          ├─ _prepare_multimodal_embeds
          │    ├─ tok_embeddings(tokens)              → text embeddings
          │    ├─ _get_vision_embeds(pixel_values)     → vision embeddings
          │    │    └─ vision_encoder(pixel_values)     → merge patches
          │    ├─ get_vision_positions              → locate vision regions
          │    └─ _scatter_vision_embeds                → scatter into text sequence
          │
          └─ transformer layers (hybrid), each given ``positions`` (3D or 2D)
               └─ for each layer:
                    ├─ full attention (every Nth):  QK-norm → partial RoPE → SDPA → gate
                    │    (the layer's MRoPE builds the cos/sin cache from positions)
                    └─ GatedDeltaNet (others):      Conv1d → gated delta rule → gated norm
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)

        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )
        self.spatial_merge_size = (
            config.vision_encoder.spatial_merge_size
            if config.vision_encoder is not None
            else None
        )

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build masks from 1D ``positions``, pick the RoPE positions, SPMD-annotate
        the inputs (when ``parallel_dims`` is given), split ``input``/``labels`` out.

        Upstream also CP-shards here; CP is not supported for Qwen3.5.
        """
        batch: dict[str, Any] = dict(input_dict)
        positions = batch.get("positions")
        if positions is not None:
            batch["attention_masks"] = self.get_attention_masks(positions)
        mrope_positions = batch.pop("mrope_positions", None)
        rope_positions = positions if mrope_positions is None else mrope_positions
        assert rope_positions is not None, (
            "Qwen3.5 needs RoPE positions: the batch must provide "
            "'positions' or 'mrope_positions'."
        )
        batch["positions"] = rope_positions
        if parallel_dims is not None:
            from .sharding import annotate_deltanet_cu_seqlens

            input_sharding = {**decoder_input_sharding(), **multimodal_input_sharding()}
            if mrope_positions is not None:
                # MRoPE positions are (tokens, 3): replicate the trailing component
                # axis instead of the 1D token layout.
                input_sharding["positions"] = spmd.SpmdType(
                    {MeshAxisName.DP: spmd.V, MeshAxisName.CP: spmd.V, MeshAxisName.TP: spmd.R},
                    partition_spec=spmd.PartitionSpec((MeshAxisName.DP, MeshAxisName.CP), None),
                )
            batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)
            # the GatedDeltaNet cu_seq_q sits inside attention_masks; annotate it there
            attention_masks = batch.get("attention_masks")
            if attention_masks is not None:
                with set_current_spmd_mesh(parallel_dims.spmd_dense_mesh()):
                    annotate_deltanet_cu_seqlens(attention_masks)
        inputs = batch.pop("input")
        labels = batch.pop("labels")
        return inputs, labels, batch

    def get_attention_masks(self, positions: torch.Tensor) -> Qwen35AttentionMaskDict:
        """Document offsets shared by GatedDeltaNet and varlen attention.

        Multimodal padding uses position 0 for every padded token. A real
        document start is position 0 followed by position 1; keep index 0 as
        the first start, so padded runs are not split into 1-token documents.
        """
        attn_config = self.config.first_attention
        if attn_config is not None and not isinstance(
            attn_config.inner_attention, VarlenInnerAttention.Config
        ):
            raise NotImplementedError("only varlen attention is ported")
        followed_by_one = torch.cat(
            [positions[1:] == 1, torch.zeros(1, dtype=torch.bool, device=positions.device)]
        )
        first_token = torch.arange(positions.shape[0], device=positions.device) == 0
        sequence_starts = ((positions == 0) & followed_by_one) | first_token
        metadata = create_varlen_metadata_for_document(torch.where(sequence_starts, 0, 1))
        return {"quadratic_attention": metadata, "deltanet": metadata}

    def _get_vision_embeds(
        self,
        pixel_values: torch.Tensor,
        *,
        grid_thw: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the vision encoder and return packed embeddings with token counts.

        Args:
            pixel_values: Packed patches ``(total_num_patches, patch_dim)``.
            grid_thw: Grid dimensions (num_items, 3) for [t, h, w]

        Returns:
            vision_embeds: Packed vision embeddings ``(total_tokens, dim)``.
            num_tokens_per_item: (num_items,) actual token count per item
        """
        if self.vision_encoder is None:
            raise ValueError("Vision inputs were provided without a vision encoder.")
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        vision_embeds = self.vision_encoder(pixel_values, grid_thw=grid_thw)

        merge_unit = self.vision_encoder.spatial_merge_unit
        num_tokens_per_item = grid_thw.prod(-1) // merge_unit

        return vision_embeds, num_tokens_per_item

    def _prepare_multimodal_embeds(
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None,
        pixel_values_videos: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        grid_thw_videos: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> torch.Tensor:
        """Embed tokens, run vision encoder, scatter vision into text.

        Args:
            tokens: Input token IDs ``(num_tokens,)``.
            pixel_values: Image patches or None
            pixel_values_videos: Video patches or None
            grid_thw: Grid dimensions for images or None
            grid_thw_videos: Grid dimensions for videos or None
            special_tokens: Special token definitions

        Returns:
            ``(num_tokens, dim)`` embeddings with vision tokens scattered in.
        """
        inputs_embeds = (
            self.tok_embeddings(tokens) if self.tok_embeddings is not None else tokens
        )

        if pixel_values is not None and grid_thw is not None:
            if special_tokens is None:
                raise ValueError("special_tokens is required for image inputs")
            vision_embeds, num_tokens = self._get_vision_embeds(
                pixel_values, grid_thw=grid_thw
            )
            image_positions = get_vision_positions(
                tokens, num_tokens, special_tokens["image_id"]
            )
            if image_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
                    vision_positions=image_positions,
                )

        if pixel_values_videos is not None and grid_thw_videos is not None:
            if special_tokens is None:
                raise ValueError("special_tokens is required for video inputs")
            vision_embeds, num_tokens = self._get_vision_embeds(
                pixel_values_videos, grid_thw=grid_thw_videos
            )
            video_positions = get_vision_positions(
                tokens, num_tokens, special_tokens["video_id"]
            )
            if video_positions:
                inputs_embeds = scatter_vision_embeds(
                    inputs_embeds,
                    vision_embeds=vision_embeds,
                    vision_positions=video_positions,
                )

        return inputs_embeds

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: Qwen35AttentionMaskDict | None = None,
        positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
    ):
        with spmd_local_context("dp"):
            if self.tok_embeddings is not None:
                x = self._prepare_multimodal_embeds(
                    tokens,
                    pixel_values=pixel_values,
                    pixel_values_videos=pixel_values_videos,
                    grid_thw=grid_thw,
                    grid_thw_videos=grid_thw_videos,
                    special_tokens=special_tokens,
                )
            else:
                x = tokens

        if spmd.is_type_checking():
            spmd.assert_type(
                x,
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                spmd.PartitionSpec(("dp", "cp"), None),
            )

        # ``positions`` is 3D MRoPE (batch, seq, 3) for multimodal batches and
        # 2D (batch, seq) for text; ``preprocess_inputs`` resolved which one to
        # forward. The per-layer MRoPE dispatches on rank.
        for layer in self.layers.values():
            x = layer(x, attention_masks, positions)

        x = self.norm(x) if self.norm is not None else x
        if self._skip_lm_head:
            return x
        return self.lm_head(x) if self.lm_head is not None else x
