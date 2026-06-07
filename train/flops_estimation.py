import torch
from train.config import ModelType

from models.qwen3_vl.model import Qwen3VLConfig
from models.qwen3_5.model import Qwen3_5Config

def get_dense_model_nparams_and_flops(
    model_type: ModelType,
    model_config: Qwen3_5Config | Qwen3VLConfig,
    model: torch.nn.Module,
    seq_len: int,
) -> tuple[int, int]:
    nparams = sum(p.numel() for p in model.parameters())

    num_flops_per_token = flops_estimation(model_type, model_config, seq_len)

    return int(nparams), int(num_flops_per_token)

# - 3x: Each GEMM in the model needs to be performed 3 times (forward pass,
#       backward wgrad [weight gradient], backward dgrad [data gradient]).
global forward_backward_expansion_factor
forward_backward_expansion_factor = 3
# - 2x: A GEMM of a m*n tensor with a n*k tensor requires 2mnk floating-point operations.
global fma_expansion_factor
fma_expansion_factor = 2

def flops_estimation(model_type: ModelType, model_config: Qwen3VLConfig | Qwen3_5Config, seq_len: int):
    """
    Taken from Megatron-Energon

    see: https://github.com/NVIDIA/Megatron-LM/blob/ad58411ddb396aeb196f6a08bd9c4000a0f10361/megatron/training/training.py#L290
    """

    def mlp_layer_flops(hidden_size, intermediate_size, swiglu=False):
        """Calculate FLOPs for an MLP layer."""
        # - 3x (SwiGLU enabled): h->2*ffn_h GEMM and ffn_h->h GEMM are stacked.
        # - 2x (SwiGLU disabled): h->ffn_h GEMM and ffn_h->h GEMM are stacked.
        ffn_expansion_factor = 3 if swiglu else 2
        return intermediate_size * ffn_expansion_factor * hidden_size * forward_backward_expansion_factor * fma_expansion_factor

    def logits_layer_flops(hidden_size, vocab_size):
        return forward_backward_expansion_factor * fma_expansion_factor * hidden_size * vocab_size

    def self_attn_flops(
            kv_channels,
            num_heads,
            num_kv_heads,
            hidden_size,
            seq_len,
            attention_output_gate=False,
    ):
        query_projection_size = kv_channels * num_heads
        key_projection_size = kv_channels * num_kv_heads
        value_projection_size = kv_channels * num_kv_heads
        gate_projection_size = query_projection_size if attention_output_gate else 0

        standard_self_attn_term = (
            forward_backward_expansion_factor
            * fma_expansion_factor
            * (
                ## qkv proj
                hidden_size
                * (
                    query_projection_size
                    + key_projection_size
                    + value_projection_size
                    + gate_projection_size
                )
                ## core attention
                + query_projection_size
                * seq_len
                / 2  # causal mask (only half of the mask is non-zero)
                * 2  # QK^T and (QK^T)V
                ## out proj
                + query_projection_size
                * hidden_size
            )
        )

        return standard_self_attn_term

    def gdn_layer_flops(
            hidden_size,
            qk_head_dim,
            v_head_dim,
            num_qk_heads,
            num_v_heads,
            conv_kernel_dim,
    ):
        """Approximate FLOPs for a Gated DeltaNet (linear-attention) layer.

        Megatron only approximates the GDN block: the in/out projections and the
        depthwise causal conv are counted exactly, while the gated delta-rule
        recurrence is approximated by `num_v_heads * v_head_dim**2 * 4` (no
        quadratic sequence term, unlike full attention).
        """
        qk_dim = qk_head_dim * num_qk_heads
        v_dim = v_head_dim * num_v_heads

        return (
            forward_backward_expansion_factor
            * fma_expansion_factor
            * (
                ## in_proj: qkv (2*qk_dim + v_dim) + z (v_dim) + a/b (num_v_heads each)
                hidden_size * (2 * qk_dim + 2 * v_dim + 2 * num_v_heads)
                ## depthwise causal conv1d over the conv channels (2*qk_dim + v_dim)
                + conv_kernel_dim * (2 * qk_dim + v_dim)
                ## gated delta-rule recurrence (approximation)
                + num_v_heads * (v_head_dim ** 2) * 4
                ## out_proj: v_dim -> hidden_size
                + hidden_size * v_dim
            )
        )

    def vision_flops(model_config: Qwen3VLConfig | Qwen3_5Config):
        """Vision tower (shared by Qwen3-VL and Qwen3.5): non-gated attention + non-gated MLP."""
        vision_kv_channels = model_config.vision.hidden_size // model_config.vision.num_heads
        vision_hidden_size = model_config.vision.hidden_size
        vision_intermediate_size = model_config.vision.intermediate_size
        vision_num_heads = model_config.vision.num_heads
        vision_num_kv_heads = model_config.vision.num_heads
        vision_num_layers = model_config.vision.depth

        vision_self_attn_term = self_attn_flops(vision_kv_channels, vision_num_heads, vision_num_kv_heads, vision_hidden_size, seq_len)
        vision_mlp_term = mlp_layer_flops(vision_hidden_size, vision_intermediate_size, swiglu=False)

        return (
            vision_self_attn_term * vision_num_layers
            + vision_mlp_term * vision_num_layers
        )

    def qwen3_vl_flops(model_config: Qwen3VLConfig | Qwen3_5Config):
        is_tied = model_config.text.tie_word_embeddings
        # Qwen3 carries an explicit head_dim that may differ from hidden_size // num_heads.
        kv_channels = model_config.text.head_dim
        hidden_size = model_config.text.hidden_size
        intermediate_size = model_config.text.intermediate_size
        num_heads = model_config.text.num_attention_heads
        num_kv_heads = model_config.text.num_key_value_heads

        vocab_size = model_config.text.vocab_size
        num_layers = model_config.text.num_hidden_layers

        # Qwen3-VL attention has no output gate; the text MLP is SwiGLU (gate/up/down).
        self_attn_term = self_attn_flops(kv_channels, num_heads, num_kv_heads, hidden_size, seq_len)
        mlp_term = mlp_layer_flops(hidden_size, intermediate_size, swiglu=True)
        logits_term = logits_layer_flops(hidden_size, vocab_size)

        text_total_flops = (
            self_attn_term * num_layers
            + mlp_term * num_layers
            + logits_term
        )

        return text_total_flops + vision_flops(model_config)

    def qwen3_5_flops(model_config: Qwen3_5Config):
        text = model_config.text
        # Qwen3 carries an explicit head_dim that may differ from hidden_size // num_heads.
        kv_channels = text.head_dim
        hidden_size = text.hidden_size
        intermediate_size = text.intermediate_size
        num_heads = text.num_attention_heads
        num_kv_heads = text.num_key_value_heads

        vocab_size = text.vocab_size
        num_layers = text.num_hidden_layers

        # Hybrid split: every `full_attention_interval`-th layer is full attention,
        # the rest are Gated DeltaNet. Mirrors LanguageModel.__init__ in the model.
        num_full_attn_layers = sum(
            1 for i in range(num_layers) if (i + 1) % text.full_attention_interval == 0
        )
        num_linear_attn_layers = num_layers - num_full_attn_layers

        # Qwen3.5 full attention has an output gate (q_proj emits q + gate).
        full_attn_term = self_attn_flops(
            kv_channels, num_heads, num_kv_heads, hidden_size, seq_len,
            attention_output_gate=True,
        )
        gdn_term = gdn_layer_flops(
            hidden_size,
            qk_head_dim=text.linear_key_head_dim,
            v_head_dim=text.linear_value_head_dim,
            num_qk_heads=text.linear_num_key_heads,
            num_v_heads=text.linear_num_value_heads,
            conv_kernel_dim=text.linear_conv_kernel_dim,
        )
        # Every decoder layer (full attention or GDN) carries a SwiGLU MLP.
        mlp_term = mlp_layer_flops(hidden_size, intermediate_size, swiglu=True)
        logits_term = logits_layer_flops(hidden_size, vocab_size)

        text_total_flops = (
            full_attn_term * num_full_attn_layers
            + gdn_term * num_linear_attn_layers
            + mlp_term * num_layers
            + logits_term
        )

        return text_total_flops + vision_flops(model_config)

    if model_type is ModelType.Qwen3_vl:
        return qwen3_vl_flops(model_config)
    elif model_type is ModelType.Qwen3_5:
        return qwen3_5_flops(model_config)
    else:
        raise NotImplementedError()
