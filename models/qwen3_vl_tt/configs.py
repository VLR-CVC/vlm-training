"""Qwen3-VL config from an HF-format ``config.json``, and its TP sharding.

Built the way `models/qwen3_5_tt/configs.py` builds Qwen3.5 (same init rules,
same vision tower builder); the decoder follows torchtitan's dense Qwen3.
"""

import json
from pathlib import Path

import spmd_types as spmd
import torch.nn as nn

from models.common.attention import VarlenInnerAttention
from models.common.feed_forward import make_ffn_config
from models.common.nn_modules import Embedding, Linear
from models.common.rope import MRoPE
from models.qwen3_5_tt.configs import (
    _depth_init,
    _EMBEDDING_INIT,
    _LINEAR_INIT,
    _output_linear_init,
    _qwen35_vision_encoder_config,
    ATTN_BACKENDS,
    DECODER_MASKS,
)

from .model import Qwen3VLAttention, Qwen3VLModel, Qwen3VLTransformerBlock, RMSNorm

_NORM_INIT = {"weight": nn.init.ones_}

def _norm(dim: int, eps: float) -> RMSNorm.Config:
    return RMSNorm.Config(dim=dim, eps=eps, param_init=_NORM_INIT)

def qwen3_vl_config_from_hf(
    config_path: str | Path,
    *,
    seq_len: int,
    attn_backend: str = "varlen",
    decoder_mask: str = "causal_doc",
    with_vision: bool = True,
) -> Qwen3VLModel.Config:
    if attn_backend not in ATTN_BACKENDS:
        raise ValueError(f"attn_backend {attn_backend!r} not in {ATTN_BACKENDS}")
    if decoder_mask not in DECODER_MASKS:
        raise ValueError(f"decoder_mask {decoder_mask!r} not in {DECODER_MASKS}")
    path = Path(config_path)
    if path.is_dir():
        path = path / "config.json"
    raw = json.loads(path.read_text())
    tc, vc = raw["text_config"], raw["vision_config"]
    # transformers 4.x snapshots keep these under rope_scaling, 5.x under rope_parameters
    rope = tc.get("rope_parameters") or tc.get("rope_scaling") or {}
    if not rope.get("mrope_interleaved", False):
        raise ValueError("MRoPE here is the interleaved layout; config says otherwise")
    theta = rope.get("rope_theta", tc.get("rope_theta"))
    if theta is None:
        raise ValueError(f"{path}: no rope_theta")
    if tc.get("attention_bias", False):
        raise NotImplementedError("attention_bias is not ported")

    dim, head_dim, eps = tc["hidden_size"], tc["head_dim"], tc["rms_norm_eps"]
    vocab_size = tc["vocab_size"]
    n_heads, n_kv_heads = tc["num_attention_heads"], tc["num_key_value_heads"]
    mrope = MRoPE.Config(
        dim=head_dim,
        max_context_length=seq_len,
        theta=float(theta),
        mrope_section=list(rope["mrope_section"]),
    )
    layers = [
        Qwen3VLTransformerBlock.Config(
            attention=Qwen3VLAttention.Config(
                n_heads=n_heads,
                n_kv_heads=n_kv_heads,
                head_dim=head_dim,
                rope=mrope,
                wq=Linear.Config(in_features=dim, out_features=n_heads * head_dim, param_init=_LINEAR_INIT),
                wk=Linear.Config(in_features=dim, out_features=n_kv_heads * head_dim, param_init=_LINEAR_INIT),
                wv=Linear.Config(in_features=dim, out_features=n_kv_heads * head_dim, param_init=_LINEAR_INIT),
                wo=Linear.Config(in_features=n_heads * head_dim, out_features=dim, param_init=_depth_init(i)),
                q_norm=_norm(head_dim, eps),
                k_norm=_norm(head_dim, eps),
                inner_attention=VarlenInnerAttention.Config(),
            ),
            feed_forward=make_ffn_config(
                dim=dim,
                hidden_dim=tc["intermediate_size"],
                w1_param_init=_LINEAR_INIT,
                w2w3_param_init=_depth_init(i),
            ),
            attention_norm=_norm(dim, eps),
            ffn_norm=_norm(dim, eps),
        )
        for i in range(tc["num_hidden_layers"])
    ]
    return Qwen3VLModel.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_norm(dim, eps),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size, embedding_dim=dim, param_init=_EMBEDDING_INIT
        ),
        lm_head=Linear.Config(
            in_features=dim, out_features=vocab_size, param_init=_output_linear_init(dim)
        ),
        enable_weight_tying=bool(tc.get("tie_word_embeddings", raw.get("tie_word_embeddings", False))),
        layers=layers,
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
            deepstack_visual_indexes=vc.get("deepstack_visual_indexes", []),
        )
        if with_vision
        else None,
    )

def apply_parallelism_config(config: Qwen3VLModel.Config, *, tp: int, enable_sp: bool) -> None:
    """Validate TP against the head counts, then fill ``sharding_config``: the
    Qwen3.5 rules (a Qwen3-VL block is a Qwen3.5 full-attention block without
    the output gate) plus the DeepStack input of every block."""
    from models.common.decoder_sharding import (
        dense_activation_placement,
        dense_sequence_parallel_placement,
    )
    from models.qwen3_5_tt.sharding import set_qwen35_sharding_config

    attention = config.first_attention
    for name, n in (("n_heads", attention.n_heads), ("n_kv_heads", attention.n_kv_heads)):
        if n % tp:
            raise ValueError(f"tensor_parallel_degree ({tp}) must divide {name} ({n})")
    set_qwen35_sharding_config(config, enable_sp=enable_sp)

    # DeepStack features are built next to the scattered embeddings (replicated on
    # TP) and enter the block in the block's own input layout, as `x_TD` does.
    layer_input_layout = (
        dense_sequence_parallel_placement()
        if enable_sp
        else dense_activation_placement(tp=spmd.I, cp=spmd.S(0))
    )
    for layer_cfg in config.layers:
        sc = layer_cfg.sharding_config
        sc.in_src_shardings["deepstack_TD"] = dense_activation_placement(tp=spmd.R, cp=spmd.S(0))
        sc.in_dst_shardings["deepstack_TD"] = layer_input_layout
