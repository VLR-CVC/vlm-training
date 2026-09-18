"""FLOPs of one packed micro-batch.

The Megatron formula this started from (see `flops_estimation` below) bills every
token for attending to `seq_len / 2` predecessors and bills the vision tower at the
decoder's sequence length. Neither holds here:

  * training is varlen-packed, so attention is causal *within a document*. A 32768
    row of 2k documents does ~1/16 of the attention FLOPs the row length implies.
  * the ViT attends within one image (a few hundred patches), and runs over
    `spatial_merge_size**2` patches per image token, not once per decoder token.

Both terms are therefore data-dependent, and `FlopsModel` splits them out: the dense
part stays a per-token constant, the two attention cores are counted from the batch's
own `positions` and `grid_thw`. Everything is a device tensor, so no host sync.
"""

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

    def batch_flops(
        self, positions: torch.Tensor, grid_thw: torch.Tensor | None
    ) -> torch.Tensor:
        """FLOPs for one micro-batch, as a device scalar.

        `positions` restarts at 0 per document, so a token at position p attends to
        p + 1 keys and `sum(positions + 1)` is the exact causal pair count -- no
        document-boundary scan needed. The ViT is bidirectional within an image, so
        an image of P patches contributes P**2 pairs.
        """
        total = positions.numel() * self.dense_per_token
        total = total + self.text_attn_pair * (positions.to(torch.float64) + 1).sum()
        if grid_thw is not None:
            patches = grid_thw.prod(-1).to(torch.float64)
            total = total + self.vision_per_patch * patches.sum()
            total = total + self.vision_attn_pair * (patches * patches).sum()
        return total


def build_flops_model(hf_config: dict) -> FlopsModel:
    """``hf_config`` is the parsed HF-format ``config.json`` (``model.model_config``);
    its ``model_type`` picks the formula."""
    model_config = SimpleNamespace(
        # older snapshots keep tie_word_embeddings only at the top level
        text=SimpleNamespace(**{"tie_word_embeddings": hf_config.get("tie_word_embeddings", False),
                                **hf_config["text_config"]}),
        vision=SimpleNamespace(**hf_config["vision_config"]),
    )
    model_type = hf_config["model_type"]
    if model_type == "qwen3_vl":
        return _qwen3_vl_flops(model_config)
    if model_type == "qwen3_5":
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


def _gdn_layer_flops(hidden_size, qk_head_dim, v_head_dim, num_qk_heads, num_v_heads,
                     conv_kernel_dim):
    """Approximate FLOPs for a Gated DeltaNet (linear-attention) layer.

    Megatron only approximates the GDN block: the in/out projections and the
    depthwise causal conv are counted exactly, while the gated delta-rule
    recurrence is approximated by `num_v_heads * v_head_dim**2 * 4` (no
    quadratic sequence term, unlike full attention).
    """
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
    """Vision tower (shared by Qwen3-VL and Qwen3.5): non-gated attention, non-gated MLP.

    Attention is bidirectional within an image, so a pair costs the same as a causal
    one -- there are just P**2 of them per image rather than P*(P+1)/2.
    """
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
    return FlopsModel(
        dense_per_token=dense,
        text_attn_pair=num_layers * _attn_pair_flops(text.head_dim, text.num_attention_heads),
        vision_per_patch=vision_per_patch,
        vision_attn_pair=vision_attn_pair,
    )


def _qwen3_5_flops(model_config) -> FlopsModel:
    text = model_config.text
    num_layers = text.num_hidden_layers
    # Hybrid split: every `full_attention_interval`-th layer is full attention,
    # the rest are Gated DeltaNet. Mirrors LanguageModel.__init__ in the model.
    num_full_attn_layers = sum(
        1 for i in range(num_layers) if (i + 1) % text.full_attention_interval == 0
    )
    num_linear_attn_layers = num_layers - num_full_attn_layers
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
        # Every decoder layer (full attention or GDN) carries a SwiGLU MLP.
        + num_layers * _mlp_flops(text.hidden_size, text.intermediate_size, swiglu=True)
        + _logits_flops(text.hidden_size, text.vocab_size)
    )
    vision_per_patch, vision_attn_pair = _vision_terms(model_config.vision)
    return FlopsModel(
        dense_per_token=dense,
        text_attn_pair=num_full_attn_layers * _attn_pair_flops(
            text.head_dim, text.num_attention_heads),
        vision_per_patch=vision_per_patch,
        vision_attn_pair=vision_attn_pair,
    )