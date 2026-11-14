from dataclasses import dataclass
from typing import Any

import spmd_types as spmd
import torch
from torch import nn

from models.common.attention import (
    AttentionMasksType,
    BaseAttention,
    create_varlen_metadata_for_document,
    VarlenInnerAttention,
    VarlenMetadata,
)
from models.common.decoder import Decoder
from models.common.decoder_sharding import decoder_input_sharding
from models.common.module import Module
from models.common.multimodal import (
    get_vision_positions,
    scatter_vision_embeds,
    vision_token_mask,
)
from models.common.nn_modules import Linear
from models.common.rope import MRoPE
from models.common.vision_encoder_sharding import multimodal_input_sharding
from models.qwen3_5.vision_encoder import Qwen35VisionEncoder
from train.parallel.parallel_dims import MeshAxisName, ParallelDims
from train.parallel.spmd import annotate_input_spmd_types, spmd_local_context

class RMSNorm(Module):
    """HF ``Qwen3VLTextRMSNorm``: normalise in fp32, scale in the input dtype."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        eps: float = 1e-6

    def __init__(self, config: Config):
        super().__init__()
        self.eps = config.eps
        self.weight = nn.Parameter(torch.empty(config.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        h = x.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * h.to(input_dtype)

class Qwen3VLAttention(BaseAttention):
    """GQA with per-head QK RMSNorm before full-width MRoPE. No output gate."""

    @dataclass(kw_only=True, slots=True)
    class Config(BaseAttention.Config):
        n_heads: int
        n_kv_heads: int
        head_dim: int
        rope: MRoPE.Config
        wq: Linear.Config
        wk: Linear.Config
        wv: Linear.Config
        wo: Linear.Config
        q_norm: RMSNorm.Config
        k_norm: RMSNorm.Config
        inner_attention: Module.Config
        # In-fusion vision capacity (arXiv 2611.17334 Eq. 2). All four together or
        # none. No output gate here, so these are plain head-shaped twins.
        vision_wq: Linear.Config | None = None
        vision_wk: Linear.Config | None = None
        vision_wv: Linear.Config | None = None
        vision_wo: Linear.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        self.head_dim = config.head_dim
        self.enable_gqa = config.n_heads > config.n_kv_heads
        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()
        self.rope = config.rope.build()
        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()
        self.scaling = self.head_dim**-0.5
        self.inner_attention = config.inner_attention.build()

        vision = (config.vision_wq, config.vision_wk, config.vision_wv, config.vision_wo)
        if any(v is not None for v in vision) and not all(v is not None for v in vision):
            raise ValueError("vision_w{q,k,v,o} must all be set or all be None")
        self.infusion = config.vision_wq is not None
        if self.infusion:
            self.vision_wq = config.vision_wq.build()  # pyrefly: ignore [missing-attribute]
            self.vision_wk = config.vision_wk.build()  # pyrefly: ignore [missing-attribute]
            self.vision_wv = config.vision_wv.build()  # pyrefly: ignore [missing-attribute]
            self.vision_wo = config.vision_wo.build()  # pyrefly: ignore [missing-attribute]

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: AttentionMasksType | None,
        positions: torch.Tensor | None = None,
        vision_mask_T1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        num_tokens = x_TD.shape[0]
        infusion = self.infusion and vision_mask_T1 is not None
        q_TC, k_TC, v_TC = self.wq(x_TD), self.wk(x_TD), self.wv(x_TD)
        if infusion:
            q_TC = q_TC + self.vision_wq(x_TD) * vision_mask_T1
            k_TC = k_TC + self.vision_wk(x_TD) * vision_mask_T1
            v_TC = v_TC + self.vision_wv(x_TD) * vision_mask_T1
        # -1 heads: under TP the colwise projections hold n_heads / TP heads
        xq_THK = self.q_norm(q_TC.view(num_tokens, -1, self.head_dim))
        xk_THK = self.k_norm(k_TC.view(num_tokens, -1, self.head_dim))
        xv_THK = v_TC.view(num_tokens, -1, self.head_dim)
        xq_THK, xk_THK = self.rope(xq_THK, xk_THK, positions)
        out_THK = self.inner_attention(
            xq_THK,
            xk_THK,
            xv_THK,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        out_TD = out_THK.view(num_tokens, -1)
        if infusion:
            return self.wo(out_TD) + self.vision_wo(out_TD) * vision_mask_T1
        return self.wo(out_TD)

class Qwen3VLTransformerBlock(Module):
    """Pre-norm block. ``deepstack_TD`` is the DeepStack feature HF adds to the
    output of the previous layer (zero outside image tokens); adding it to this
    layer's input is the same sum, and lets the Module protocol shard it like
    ``x_TD``."""

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        attention: Qwen3VLAttention.Config
        feed_forward: Module.Config
        attention_norm: RMSNorm.Config
        ffn_norm: RMSNorm.Config
        # In-fusion vision capacity (arXiv 2611.17334 Eq. 3).
        vision_feed_forward: Module.Config | None = None

    def __init__(self, config: Config):
        super().__init__()
        # `attn` and the norm names match the Qwen3.5 blocks, so the state-dict
        # adapter maps both models' HF keys the same way
        self.attn = config.attention.build()
        self.feed_forward = config.feed_forward.build()
        self.attention_norm = config.attention_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.vision_feed_forward = (
            config.vision_feed_forward.build()
            if config.vision_feed_forward is not None
            else None
        )

    def forward(
        self,
        x_TD: torch.Tensor,
        attention_masks: VarlenMetadata | None,
        positions: torch.Tensor | None = None,
        deepstack_TD: torch.Tensor | None = None,
        vision_mask_T1: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if deepstack_TD is not None:
            x_TD = x_TD + deepstack_TD
        x_TD = x_TD + self.attn(
            self.attention_norm(x_TD), attention_masks, positions, vision_mask_T1
        )
        h_TD = self.ffn_norm(x_TD)
        x_TD = x_TD + self.feed_forward(h_TD)
        if self.vision_feed_forward is not None and vision_mask_T1 is not None:
            x_TD = x_TD + self.vision_feed_forward(h_TD) * vision_mask_T1
        return x_TD

class Qwen3VLModel(Decoder):
    """Qwen3-VL: dense decoder + vision tower with DeepStack.

    Inputs follow `models/qwen3_5` exactly (``input``, ``positions``,
    ``mrope_positions``, ``pixel_values``, ``grid_thw``), so `data/model_batch.py`,
    `train/step.py` and the FSDP/compile wrappers serve both models.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        vision_encoder: Qwen35VisionEncoder.Config | None = None

    def __init__(self, config: Config):
        super().__init__(config)
        # Only build the modality mask when some layer can use it.
        self._has_infusion = any(
            layer.vision_feed_forward is not None or layer.attn.infusion
            for layer in self.layers.values()
        )
        self.vision_encoder = (
            config.vision_encoder.build() if config.vision_encoder is not None else None
        )

    def _vision_mask(
        self, tokens: torch.Tensor, special_tokens: dict[str, int] | None
    ) -> torch.Tensor | None:
        """The in-fusion modality mask, or None when no layer can use it."""
        if not self._has_infusion:
            return None
        return vision_token_mask(tokens, special_tokens)

    def preprocess_inputs(
        self,
        input_dict: dict[str, torch.Tensor],
        *,
        parallel_dims: ParallelDims | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """As `Qwen35Model.preprocess_inputs`, without the GatedDeltaNet offsets."""
        batch: dict[str, Any] = dict(input_dict)
        positions = batch["positions"]
        batch["attention_masks"] = self.get_attention_masks(positions)
        mrope_positions = batch.pop("mrope_positions", None)
        batch["positions"] = positions if mrope_positions is None else mrope_positions
        if parallel_dims is not None:
            input_sharding = {**decoder_input_sharding(), **multimodal_input_sharding()}
            if mrope_positions is not None:
                input_sharding["positions"] = spmd.SpmdType(
                    {MeshAxisName.DP: spmd.V, MeshAxisName.CP: spmd.V, MeshAxisName.TP: spmd.R},
                    partition_spec=spmd.PartitionSpec((MeshAxisName.DP, MeshAxisName.CP), None),
                )
            batch = annotate_input_spmd_types(parallel_dims, batch, input_sharding)
        return batch.pop("input"), batch.pop("labels"), batch

    def get_attention_masks(self, positions: torch.Tensor) -> VarlenMetadata:
        """Document offsets for varlen attention; same start rule as Qwen3.5."""
        attn_config = self.config.first_attention
        if not isinstance(attn_config.inner_attention, VarlenInnerAttention.Config):
            raise NotImplementedError("only varlen attention is ported")
        followed_by_one = torch.cat(
            [positions[1:] == 1, torch.zeros(1, dtype=torch.bool, device=positions.device)]
        )
        first_token = torch.arange(positions.shape[0], device=positions.device) == 0
        sequence_starts = ((positions == 0) & followed_by_one) | first_token
        return create_varlen_metadata_for_document(torch.where(sequence_starts, 0, 1))

    def _zero_deepstack(self, x: torch.Tensor) -> list[torch.Tensor]:
        """One all-zero DeepStack tensor per DeepStack layer.

        Constant length so `forward`'s `1 <= i <= len(deepstack)` test takes the
        same branch on every rank.
        """
        n = len(self.vision_encoder.deepstack_visual_indexes)
        return [torch.zeros_like(x) for _ in range(n)]

    def _zero_vision_contribution(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Run the tower on a minimal dummy image and add nothing to `x`.

        Keeps the vision parameters in the autograd graph so FSDP issues the same
        collectives as every other rank; the gradients come out as exact zeros.
        """
        dummy_pixels, dummy_grid = self.vision_encoder.dummy_inputs(
            device=x.device, dtype=self.vision_encoder.patch_embed.weight.dtype
        )
        out = self.vision_encoder(dummy_pixels, grid_thw=dummy_grid)
        return x + (out * 0.0).sum(), self._zero_deepstack(x)

    def _embed(
        self,
        tokens: torch.Tensor,
        pixel_values: torch.Tensor | None,
        grid_thw: torch.Tensor | None,
        special_tokens: dict[str, int] | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Token embeddings with image features scattered in, and one full-length
        DeepStack tensor per DeepStack layer (zero outside image tokens)."""
        x = self.tok_embeddings(tokens)

        if pixel_values is None or grid_thw is None:
            if self.vision_encoder is None:
                return x, []
            return self._zero_vision_contribution(x)
        if self.vision_encoder is None:
            raise ValueError("Vision inputs were provided without a vision encoder.")
        if special_tokens is None:
            raise ValueError("special_tokens is required for image inputs")
        pixel_values = pixel_values.to(self.vision_encoder.patch_embed.weight.dtype)
        out = self.vision_encoder(pixel_values, grid_thw=grid_thw)
        num_tokens = grid_thw.prod(-1) // self.vision_encoder.spatial_merge_unit
        image_positions = get_vision_positions(tokens, num_tokens, special_tokens["image_id"])
        if not image_positions:
            # forward already ran, so only the BACKWARD would desync here
            return x + (out * 0.0).sum(), self._zero_deepstack(x)
        n = sum(count for _, _, count in image_positions)
        features = out.split(n)  # merged, then one per DeepStack layer
        x = scatter_vision_embeds(x, vision_embeds=features[0], vision_positions=image_positions)
        deepstack = [
            scatter_vision_embeds(
                torch.zeros_like(x), vision_embeds=f, vision_positions=image_positions
            )
            for f in features[1:]
        ]
        return x, deepstack

    def forward(  # pyrefly: ignore [bad-override]
        self,
        tokens: torch.Tensor,
        *,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        grid_thw: torch.Tensor | None = None,
        grid_thw_videos: torch.Tensor | None = None,
        attention_masks: VarlenMetadata | None = None,
        positions: torch.Tensor | None = None,
        special_tokens: dict[str, int] | None = None,
    ):
        if pixel_values_videos is not None:
            raise NotImplementedError("video inputs are not ported for Qwen3-VL")
        with spmd_local_context("dp"):
            x, deepstack = self._embed(tokens, pixel_values, grid_thw, special_tokens)

        if spmd.is_type_checking():
            spmd.assert_type(
                x,
                {"dp": spmd.V, "cp": spmd.V, "tp": spmd.R},
                spmd.PartitionSpec(("dp", "cp"), None),
            )

        vision_mask_T1 = self._vision_mask(tokens, special_tokens)
        for i, layer in enumerate(self.layers.values()):
            ds = deepstack[i - 1] if 1 <= i <= len(deepstack) else None
            x = layer(x, attention_masks, positions, deepstack_TD=ds,
                      vision_mask_T1=vision_mask_T1)

        x = self.norm(x)
        if self._skip_lm_head:
            return x
        return self.lm_head(x)
