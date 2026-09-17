# Vendored from torchtitan b21f7d43e: the config builders of
# torchtitan/models/qwen3_5/__init__.py (dense only). The per-size functions
# (_2b, _9b, ...) are replaced by `qwen35_config_from_hf`, which reads the same
# numbers from the checkpoint's config.json instead of restating them.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE

import json
from collections.abc import Callable
from functools import partial
from pathlib import Path

import torch.nn as nn

from models.common.attention import VarlenInnerAttention
from models.common.feed_forward import make_ffn_config
from models.common.nn_modules import (
    Conv1d,
    depth_scaled_std,
    Embedding,
    GELU,
    LayerNorm,
    Linear,
    PartialBiasRowwiseLinear,
)
from models.common.rope import MRoPE
from models.common.vision_encoder import VisionAttention, VisionMLP, VisionTransformerBlock

from .gdn import GatedDeltaKernel, GatedDeltaNet, InnerGatedDeltaNet, RMSNormGated
from .model import OffsetRMSNorm, Qwen35Attention, Qwen35Model, Qwen35TransformerBlock
from .vision_encoder import PatchMerger, Qwen35VisionEncoder, VisionRotaryEmbedding

QWEN3_5_SPECIAL_TOKENS = {
    "image_token": "<|image_pad|>",
    "video_token": "<|video_pad|>",
    "vision_start_token": "<|vision_start|>",
    "vision_end_token": "<|vision_end|>",
    "pad_token": "<|endoftext|>",
}

_LINEAR_INIT = {
    "weight": partial(nn.init.trunc_normal_, std=0.02),
    "bias": nn.init.zeros_,
}
_OFFSET_NORM_INIT = {"weight": nn.init.zeros_}
_EMBEDDING_INIT = {"weight": partial(nn.init.normal_, std=1.0)}
_POS_EMBED_INIT = {"pos_embed": partial(nn.init.trunc_normal_, mean=0.0, std=0.02)}

_EPS = 1e-6

def _output_linear_init(dim: int) -> dict[str, Callable]:
    s = dim**-0.5
    return {
        "weight": partial(nn.init.trunc_normal_, std=s, a=-3 * s, b=3 * s),
        "bias": nn.init.zeros_,
    }

def _depth_init(layer_id: int) -> dict[str, Callable]:
    return {
        "weight": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "bias": nn.init.zeros_,
    }

def _depth_experts_init(layer_id: int) -> dict[str, Callable]:
    return {
        "w1_EFD": partial(nn.init.trunc_normal_, std=0.02),
        "w2_EDF": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
        "w3_EFD": partial(nn.init.trunc_normal_, std=depth_scaled_std(0.02, layer_id)),
    }

def _a_log_init(param: nn.Parameter) -> None:
    # Match https://github.com/huggingface/transformers/pull/47944 to avoid
    # near-zero decay heads under bf16 initialization.
    param.data.uniform_(0.01, 16.0).log_()

def _linear(in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )

def _partial_bias_rowwise_linear(
    in_features: int, out_features: int
) -> PartialBiasRowwiseLinear.Config:
    return PartialBiasRowwiseLinear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=True,
        param_init=_LINEAR_INIT,
    )

def _offset_norm(dim: int) -> OffsetRMSNorm.Config:
    return OffsetRMSNorm.Config(dim=dim, eps=_EPS, param_init=_OFFSET_NORM_INIT)

def _qwen35_vision_encoder_config(
    *,
    dim: int,
    ffn_dim: int,
    num_layers: int,
    num_heads: int,
    patch_size: int,
    temporal_patch_size: int,
    spatial_merge_size: int,
    out_hidden_size: int,
    num_position_embeddings: int,
    layer_norm_eps: float = 1e-6,
    rope_theta: float = 10000.0,
    in_channels: int = 3,
    deepstack_visual_indexes: list[int] = (),
) -> Qwen35VisionEncoder.Config:
    """Build a fully-specified Qwen35VisionEncoder.Config. ``deepstack_visual_indexes``
    adds Qwen3-VL's DeepStack mergers (empty for Qwen3.5)."""
    patch_dim = in_channels * temporal_patch_size * patch_size * patch_size
    merged_hidden_size = dim * (spatial_merge_size**2)
    head_dim = dim // num_heads
    _norm = LayerNorm.Config(normalized_shape=dim, eps=layer_norm_eps)
    return Qwen35VisionEncoder.Config(
        dim=dim,
        num_layers=num_layers,
        num_heads=num_heads,
        patch_size=patch_size,
        temporal_patch_size=temporal_patch_size,
        in_channels=in_channels,
        spatial_merge_size=spatial_merge_size,
        num_position_embeddings=num_position_embeddings,
        patch_embed_proj=_linear(patch_dim, dim),
        block=VisionTransformerBlock.Config(
            norm1=_norm,
            norm2=_norm,
            attn=VisionAttention.Config(
                dim=dim,
                num_heads=num_heads,
                wq=_linear(dim, dim),
                wk=_linear(dim, dim),
                wv=_linear(dim, dim),
                proj=_partial_bias_rowwise_linear(dim, dim),
            ),
            mlp=VisionMLP.Config(
                fc1=_linear(dim, ffn_dim),
                fc2=_partial_bias_rowwise_linear(ffn_dim, dim),
            ),
        ),
        rotary_pos_emb=VisionRotaryEmbedding.Config(
            dim=head_dim // 2, theta=rope_theta
        ),
        merger=PatchMerger.Config(
            spatial_merge_size=spatial_merge_size,
            merged_hidden_size=merged_hidden_size,
            norm=LayerNorm.Config(normalized_shape=dim, eps=layer_norm_eps),
            fc1=_linear(merged_hidden_size, merged_hidden_size),
            # DEVIATION from torchtitan b21f7d43e (GELU approximate="tanh"): HF's
            # Qwen3_5VisionPatchMerger uses exact nn.GELU(). The blocks' MLP is
            # gelu_pytorch_tanh in HF, so VisionMLP keeps tanh.
            act_fn=GELU.Config(approximate="none"),
            fc2=_partial_bias_rowwise_linear(merged_hidden_size, out_hidden_size),
        ),
        deepstack_visual_indexes=list(deepstack_visual_indexes),
        deepstack_merger=PatchMerger.Config(
            spatial_merge_size=spatial_merge_size,
            merged_hidden_size=merged_hidden_size,
            norm=LayerNorm.Config(normalized_shape=merged_hidden_size, eps=layer_norm_eps),
            fc1=_linear(merged_hidden_size, merged_hidden_size),
            act_fn=GELU.Config(approximate="none"),
            fc2=_partial_bias_rowwise_linear(merged_hidden_size, out_hidden_size),
            postshuffle_norm=True,
        )
        if deepstack_visual_indexes
        else None,
        param_init=_POS_EMBED_INIT,
    )

def _qwen35_attention_config(
    *,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    attn_backend: str,
    layer_id: int,
) -> Qwen35Attention.Config:
    """Build a fully-specified Qwen35Attention.Config."""
    assert attn_backend == "varlen", attn_backend  # validated in qwen35_config_from_hf
    inner_attention = VarlenInnerAttention.Config()
    return Qwen35Attention.Config(
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        head_dim=head_dim,
        rotary_dim=rotary_dim,
        rope=rope,
        wq=Linear.Config(
            in_features=dim,
            out_features=n_heads * head_dim * 2,
            param_init=_LINEAR_INIT,
        ),
        wk=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        wv=Linear.Config(
            in_features=dim,
            out_features=n_kv_heads * head_dim,
            param_init=_LINEAR_INIT,
        ),
        wo=Linear.Config(
            in_features=n_heads * head_dim,
            out_features=dim,
            param_init=_depth_init(layer_id),
        ),
        q_norm=_offset_norm(head_dim),
        k_norm=_offset_norm(head_dim),
        inner_attention=inner_attention,
    )

def _qwen35_deltanet_config(
    *,
    dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    layer_id: int,
    conv_kernel_size: int = 4,
) -> GatedDeltaNet.Config:
    """Build a fully-specified GatedDeltaNet.Config."""
    key_dim = n_key_heads * key_head_dim
    value_dim = n_value_heads * value_head_dim

    def _proj(in_f: int, out_f: int, init: dict) -> Linear.Config:
        return Linear.Config(
            in_features=in_f, out_features=out_f, bias=False, param_init=init
        )

    def _conv(channels: int) -> Conv1d.Config:
        # Depthwise causal conv (groups == channels). Causal left-padding is
        # applied in the forward, so padding=0 here.
        return Conv1d.Config(
            in_channels=channels,
            out_channels=channels,
            kernel_size=conv_kernel_size,
            groups=channels,
            padding=0,
            bias=False,
        )

    return GatedDeltaNet.Config(
        key_head_dim=key_head_dim,
        value_head_dim=value_head_dim,
        conv_kernel_size=conv_kernel_size,
        in_proj_q=_proj(dim, key_dim, _LINEAR_INIT),
        in_proj_k=_proj(dim, key_dim, _LINEAR_INIT),
        in_proj_v=_proj(dim, value_dim, _LINEAR_INIT),
        in_proj_z=_proj(dim, value_dim, _LINEAR_INIT),
        in_proj_a=_proj(dim, n_value_heads, _LINEAR_INIT),
        in_proj_b=_proj(dim, n_value_heads, _LINEAR_INIT),
        conv_q=_conv(key_dim),
        conv_k=_conv(key_dim),
        conv_v=_conv(value_dim),
        inner_gated_delta_net=InnerGatedDeltaNet.Config(
            kernel=GatedDeltaKernel.Config(),
        ),
        norm=RMSNormGated.Config(
            dim=value_head_dim,
            eps=1e-6,
            param_init={"weight": nn.init.ones_},
        ),
        out_proj=_proj(value_dim, dim, _depth_init(layer_id)),
        param_init={
            "A_log": _a_log_init,
            "dt_bias": nn.init.ones_,
        },
    )

def _build_qwen35_layers(
    *,
    n_layers: int,
    dim: int,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rotary_dim: int,
    rope: MRoPE.Config,
    hidden_dim: int,
    n_key_heads: int,
    n_value_heads: int,
    key_head_dim: int,
    value_head_dim: int,
    full_attention_interval: int = 4,
    attn_backend: str,
) -> list[Qwen35TransformerBlock.Config]:
    """Build per-layer configs for dense Qwen3.5 models."""
    layers = []
    for layer_id in range(n_layers):
        is_full = (layer_id + 1) % full_attention_interval == 0

        attention = (
            _qwen35_attention_config(
                dim=dim,
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rotary_dim=rotary_dim,
                rope=rope,
                attn_backend=attn_backend,
                layer_id=layer_id,
            )
            if is_full
            else None
        )
        deltanet = (
            _qwen35_deltanet_config(
                dim=dim,
                n_key_heads=n_key_heads,
                n_value_heads=n_value_heads,
                key_head_dim=key_head_dim,
                value_head_dim=value_head_dim,
                layer_id=layer_id,
            )
            if not is_full
            else None
        )

        layers.append(
            Qwen35TransformerBlock.Config(
                attention=attention,
                delta_net=deltanet,
                feed_forward=make_ffn_config(
                    dim=dim,
                    hidden_dim=hidden_dim,
                    w1_param_init=_LINEAR_INIT,
                    w2w3_param_init=_depth_init(layer_id),
                ),
                attention_norm=_offset_norm(dim),
                ffn_norm=_offset_norm(dim),
            )
        )
    return layers

ATTN_BACKENDS = ("varlen",)
DECODER_MASKS = ("causal_doc",)

def resolve_model_config(model_config: str, model_dir: str, use_model_dir_config: bool) -> Path:
    """The HF-format ``config.json`` that defines the architecture: ``model_config``
    by default, ``<model_dir>/config.json`` with ``use_model_dir_config``."""
    unset = model_config in ("", "NULL")
    if use_model_dir_config:
        if not unset:
            raise ValueError("set either model.model_config or model.use_model_dir_config, not both")
        return Path(model_dir) / "config.json"
    if unset:
        raise ValueError(
            "impl = 'titan' needs model.model_config (e.g. configs/models/qwen3_5_9b.json), "
            "or model.use_model_dir_config = true to read training.model_dir/config.json"
        )
    return Path(model_config)

def qwen35_config_from_hf(
    config_path: str | Path,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    decoder_mask: str = "causal_doc",
    with_vision: bool = True,
) -> Qwen35Model.Config:
    """Dense Qwen3.5 config from an HF-format ``config.json``: a file, or an HF
    snapshot directory that holds one.

    Same construction as upstream's ``_9b`` etc., with the numbers read from
    the JSON. ``tie_word_embeddings`` maps to ``enable_weight_tying``
    (upstream's 2B builder leaves it off and so does not match the checkpoint).
    """
    if attn_backend not in ATTN_BACKENDS:
        raise ValueError(f"attn_backend {attn_backend!r} not in {ATTN_BACKENDS}")
    if decoder_mask not in DECODER_MASKS:
        raise ValueError(f"decoder_mask {decoder_mask!r} not in {DECODER_MASKS}")
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.json"
    raw = json.loads(path.read_text())
    tc, vc = raw["text_config"], raw["vision_config"]
    rope = tc["rope_parameters"]
    if not rope.get("mrope_interleaved", False):
        raise ValueError("MRoPE here is the interleaved layout; config says otherwise")
    head_dim = tc["head_dim"]
    rotary_dim = int(head_dim * rope.get("partial_rotary_factor", 1.0))
    dim = tc["hidden_size"]
    vocab_size = tc["vocab_size"]
    layer_types = tc["layer_types"]
    interval = tc["full_attention_interval"]
    expected = [
        "full_attention" if (i + 1) % interval == 0 else "linear_attention"
        for i in range(tc["num_hidden_layers"])
    ]
    if layer_types != expected:
        raise ValueError(f"layer_types is not every-{interval}th full attention")

    return Qwen35Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_offset_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        enable_weight_tying=bool(tc.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))),
        layers=_build_qwen35_layers(
            rope=MRoPE.Config(
                dim=rotary_dim,
                max_context_length=seq_len,
                theta=float(rope["rope_theta"]),
                mrope_section=list(rope["mrope_section"]),
            ),
            attn_backend=attn_backend,
            n_layers=tc["num_hidden_layers"],
            dim=dim,
            n_heads=tc["num_attention_heads"],
            n_kv_heads=tc["num_key_value_heads"],
            head_dim=head_dim,
            rotary_dim=rotary_dim,
            hidden_dim=tc["intermediate_size"],
            n_key_heads=tc["linear_num_key_heads"],
            n_value_heads=tc["linear_num_value_heads"],
            key_head_dim=tc["linear_key_head_dim"],
            value_head_dim=tc["linear_value_head_dim"],
            full_attention_interval=interval,
        ),
        vision_encoder=_qwen35_vision_encoder_config(
            dim=vc["hidden_size"],
            ffn_dim=vc["intermediate_size"],
            num_layers=vc["depth"],
            num_heads=vc["num_heads"],
            patch_size=vc["patch_size"],
            temporal_patch_size=vc["temporal_patch_size"],
            spatial_merge_size=vc["spatial_merge_size"],
            out_hidden_size=vc["out_hidden_size"],
            num_position_embeddings=vc["num_position_embeddings"],
            in_channels=vc.get("in_channels", 3),
        )
        if with_vision
        else None,
    )

def apply_parallelism_config(config: Qwen35Model.Config, *, tp: int, enable_sp: bool) -> None:
    """torchtitan's ``Qwen35Model.Config.update_from_config`` (TP part): validate
    that TP divides every head count, then fill ``sharding_config`` on every
    sub-config. ``Module.parallelize`` reads them; without this call it is a no-op.
    Call on the config, before ``build``."""
    from .sharding import set_qwen35_sharding_config

    if tp > 1:
        attention = config.first_attention
        for name, n in (("n_heads", attention.n_heads), ("n_kv_heads", attention.n_kv_heads)):
            if n % tp:
                raise ValueError(f"tensor_parallel_degree ({tp}) must divide {name} ({n})")
        dn = next(layer.delta_net for layer in config.layers if layer.delta_net is not None)
        n_key_heads = dn.in_proj_q.out_features // dn.key_head_dim
        n_value_heads = dn.in_proj_v.out_features // dn.value_head_dim
        if n_key_heads % tp or n_value_heads % tp:
            raise ValueError(
                f"tensor_parallel_degree ({tp}) must divide n_key_heads ({n_key_heads}) "
                f"and n_value_heads ({n_value_heads})."
            )
    set_qwen35_sharding_config(config, enable_sp=enable_sp)
