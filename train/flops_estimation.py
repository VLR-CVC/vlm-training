from dataclasses import dataclass
from types import SimpleNamespace

import torch

# - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
#       backward wgrad [weight gradient], backward dgrad [data gradient]).
FWD_BWD = 3
# - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
FMA = 2

@dataclass(frozen=True)
class FlopsModel:
    """Coefficients of `batch_flops`, all for the whole (un-sharded) model.

    `*_attn_pair` is the cost of one (query, key) pair summed over every layer that
    does quadratic attention: QK^T and (QK^T)V are each `query_projection_size`
    multiply-adds per pair.
    """

    dense_per_token: float  # decoder: attention projections + MLP + logits
    text_attn_pair: float
    vision_per_patch: float  # ViT: attention projections + MLP
    vision_attn_pair: float
    infusion_per_vision_token: float = 0.0  # in-fusion branches, vision tokens only
    vision_merge_unit: int = 1  # patches per backbone vision token

    def batch_flops(
        self, positions: torch.Tensor, grid_thw: torch.Tensor | None
    ) -> torch.Tensor:
        """FLOPs for one micro-batch, as a device scalar"""
        total = positions.numel() * self.dense_per_token
        total = total + self.text_attn_pair * (positions.to(torch.float64) + 1).sum()
        if grid_thw is not None:
            patches = grid_thw.prod(-1).to(torch.float64)
            total = total + self.vision_per_patch * patches.sum()
            total = total + self.vision_attn_pair * (patches * patches).sum()
            if self.infusion_per_vision_token:
                # Billed per vision token, not per token. The masked-dense
                # implementation does run these GEMMs on text tokens too, but that
                # arithmetic is multiplied away -- counting it would report work
                # the model does not use and inflate MFU. A gather implementation
                # changes what it costs, not what it is worth.
                vision_tokens = patches.sum() / self.vision_merge_unit
                total = total + self.infusion_per_vision_token * vision_tokens
        return total

def build_flops_model(hf_config: dict, layer_types: list[str] | None = None) -> FlopsModel:
    """We use the HF config, which defines the model to derive the FLOPS usage"""
    model_config = SimpleNamespace(
        # older snapshots keep tie_word_embeddings only at the top level
        text=SimpleNamespace(**{"tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
                                **hf_config["text_config"],
                                **({"layer_types": layer_types} if layer_types else {})}),
        # text-only configs carry no vision tower
        vision=SimpleNamespace(**hf_config["vision_config"])
        if "vision_config" in hf_config
        else None,
    )
    model_type = hf_config["model_type"]
    if model_type == "qwen3_vl":
        return _qwen3_vl_flops(model_config)
    if model_type in ("qwen3_5", "qwen3_5_text", "qwen3_5_moe"):
        return _qwen3_5_flops(model_config)
    raise NotImplementedError(f"no FLOPs formula for model_type {model_type!r}")

def _attn_proj_flops(kv_channels, num_heads, num_kv_heads, hidden_size, output_gate=False):
    """qkv projections + out projection, per token. Independent of sequence length."""
    query_projection_size = kv_channels * num_heads
    kv_projection_size = kv_channels * num_kv_heads
    gate_projection_size = query_projection_size if output_gate else 0
    return FWD_BWD * FMA * (
        hidden_size * (query_projection_size + 2 * kv_projection_size + gate_projection_size)
        + query_projection_size * hidden_size
    )

def _attn_pair_flops(kv_channels, num_heads):
    """QK^T and (QK^T)V for one (query, key) pair, one layer."""
    return FWD_BWD * FMA * 2 * kv_channels * num_heads

def _mlp_flops(hidden_size, intermediate_size, swiglu):
    # - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
    # - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
    ffn_expansion_factor = 3 if swiglu else 2
    return intermediate_size * ffn_expansion_factor * hidden_size * FWD_BWD * FMA

def _logits_flops(hidden_size, vocab_size):
    return FWD_BWD * FMA * hidden_size * vocab_size

def _moe_flops(hidden_size, moe_intermediate_size, shared_intermediate_size, num_experts,
               top_k):
    """Per-token FLOPs of one Qwen3.5 MoE block"""
    return (
        top_k * _mlp_flops(hidden_size, moe_intermediate_size, swiglu=True)
        + _mlp_flops(hidden_size, shared_intermediate_size, swiglu=True)
        + FWD_BWD * FMA * hidden_size * (num_experts + 1)
    )

def _gdn_layer_flops(hidden_size, qk_head_dim, v_head_dim, num_qk_heads, num_v_heads,
                     conv_kernel_dim):
    """Approximate FLOPs for a Gated DeltaNet (linear-attention) layer"""
    qk_dim = qk_head_dim * num_qk_heads
    v_dim = v_head_dim * num_v_heads
    return FWD_BWD * FMA * (
        ## in_proj: qkv (2*qk_dim + v_dim) + z (v_dim) + a/b (num_v_heads each)
        hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
        ## depthwise causal conv1d over the conv channels (2*qk_dim + v_dim)
        + conv_kernel_dim * (2 * qk_dim + v_dim)
        ## gated delta-rule recurrence (approximation)
        + num_v_heads * (v_head_dim ** 2) * 4
        ## out_proj: v_dim -> hidden_size
        + hidden_size * v_dim
    )

def _vision_terms(vision) -> tuple[float, float]:
    """Vision tower (shared by Qwen3-VL and Qwen3.5): non-gated attention, non-gated MLP"""
    if vision is None:
        return 0.0, 0.0
    kv_channels = vision.hidden_size // vision.num_heads
    per_patch = vision.depth * (
        _attn_proj_flops(kv_channels, vision.num_heads, vision.num_heads, vision.hidden_size)
        + _mlp_flops(vision.hidden_size, vision.intermediate_size, swiglu=False)
    )
    per_pair = vision.depth * _attn_pair_flops(kv_channels, vision.num_heads)
    return per_patch, per_pair

def _qwen3_vl_flops(model_config) -> FlopsModel:
    text = model_config.text
    num_layers = text.num_hidden_layers
    # Qwen3-VL attention has no output gate; the text MLP is SwiGLU (gate/up/down).
    dense = num_layers * (
        _attn_proj_flops(text.head_dim, text.num_attention_heads, text.num_key_value_heads,
                         text.hidden_size)
        + _mlp_flops(text.hidden_size, text.intermediate_size, swiglu=True)
    ) + _logits_flops(text.hidden_size, text.vocab_size)
    vision_per_patch, vision_attn_pair = _vision_terms(model_config.vision)
    # Every Qwen3-VL layer is full attention, so an augmented layer always carries
    # the Eq. 2 twins -- unlike the hybrid Qwen3.5 stack, where only some do.
    infusion = getattr(text, "infusion", None)
    infusion_per_vision_token = (
        _infusion_flops(text, infusion, len(infusion["layers"]), output_gate=False)
        if infusion
        else 0.0
    )
    merge = getattr(model_config.vision, "spatial_merge_size", 1) if model_config.vision else 1
    return FlopsModel(
        dense_per_token=dense,
        text_attn_pair=num_layers * _attn_pair_flops(text.head_dim, text.num_attention_heads),
        vision_per_patch=vision_per_patch,
        vision_attn_pair=vision_attn_pair,
        infusion_per_vision_token=infusion_per_vision_token,
        vision_merge_unit=merge * merge,
    )

def _infusion_flops(text, infusion, num_full_attn_in_layers: int, output_gate=True) -> float:
    """In-fusion vision capacity per vision token (arXiv 2611.17334 Eqs. 2 and 3).

    Read from the config's ``infusion`` block, so a model built by passing
    ``InfusionConfig`` straight to ``qwen35_config_from_hf`` -- which tests do --
    is billed nothing. Training runs declare it in the config JSON.
    """
    hidden = text.hidden_size
    per_layer_ffn = _mlp_flops(hidden, infusion["ffn_hidden_dim"], swiglu=True)
    total = len(infusion["layers"]) * per_layer_ffn
    if infusion.get("attention_projections", True):
        total += num_full_attn_in_layers * _attn_proj_flops(
            text.head_dim,
            text.num_attention_heads,
            text.num_key_value_heads,
            hidden,
            output_gate=output_gate,
        )
    return total

def _qwen3_5_flops(model_config) -> FlopsModel:
    text = model_config.text
    num_layers = text.num_hidden_layers
    layer_types = getattr(text, "layer_types", None)
    if layer_types:
        num_full_attn_layers = sum(1 for t in layer_types if t == "full_attention")
    else:
        num_full_attn_layers = sum(
            1 for i in range(num_layers) if (i + 1) % text.full_attention_interval == 0
        )
    num_linear_attn_layers = num_layers - num_full_attn_layers
    num_experts = getattr(text, "num_experts", 0)
    if num_experts:
        per_layer_ffn = _moe_flops(
            text.hidden_size,
            text.moe_intermediate_size,
            text.shared_expert_intermediate_size,
            num_experts,
            text.num_experts_per_tok,
        )
    else:
        per_layer_ffn = _mlp_flops(text.hidden_size, text.intermediate_size, swiglu=True)
    # Qwen3.5 full attention has an output gate (q_proj emits q + gate).
    dense = (
        num_full_attn_layers * _attn_proj_flops(
            text.head_dim, text.num_attention_heads, text.num_key_value_heads,
            text.hidden_size, output_gate=True)
        + num_linear_attn_layers * _gdn_layer_flops(
            text.hidden_size,
            qk_head_dim=text.linear_key_head_dim,
            v_head_dim=text.linear_value_head_dim,
            num_qk_heads=text.linear_num_key_heads,
            num_v_heads=text.linear_num_value_heads,
            conv_kernel_dim=text.linear_conv_kernel_dim,
        )
        # Every decoder layer (full attention or GDN) carries a SwiGLU MLP, or an
        # MoE block on the sparse variants.
        + num_layers * per_layer_ffn
        + _logits_flops(text.hidden_size, text.vocab_size)
    )
    vision_per_patch, vision_attn_pair = _vision_terms(model_config.vision)
    infusion = getattr(text, "infusion", None)
    infusion_per_vision_token = 0.0
    if infusion:
        interval = text.full_attention_interval
        full_in_layers = sum(
            1
            for i in infusion["layers"]
            if (layer_types[i] == "full_attention" if layer_types else (i + 1) % interval == 0)
        )
        infusion_per_vision_token = _infusion_flops(text, infusion, full_in_layers)
    merge = getattr(model_config.vision, "spatial_merge_size", 1) if model_config.vision else 1
    return FlopsModel(
        dense_per_token=dense,
        text_attn_pair=num_full_attn_layers * _attn_pair_flops(
            text.head_dim, text.num_attention_heads),
        vision_per_patch=vision_per_patch,
        vision_attn_pair=vision_attn_pair,
        infusion_per_vision_token=infusion_per_vision_token,
        vision_merge_unit=merge * merge,
    )