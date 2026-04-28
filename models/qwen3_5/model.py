from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn

from fla.ops.gated_delta_rule import chunk_gated_delta_rule as _fla_chunk_gated_delta_rule
from fla.modules.fused_norm_gate import rms_norm_gated as _fla_rms_norm_gated
from causal_conv1d import causal_conv1d_fn as _causal_conv1d_fn

from models.qwen3_5.config import (
    Qwen3VLConfig, Qwen3_5TextConfig, Qwen3_5VisionConfig
)
from models.qwen3_5.utils import (
    _dtensor_unwrap,
    _dtensor_rewrap,
    _local,
    CausalLMOutput,
    causal_lm_loss,
    apply_rope,
    mrope_cos_sin,
    apply_rope_vision,
    load_safetensors_into,
)

class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        D = orig_shape[-1]
        (hs_local, gate_local), wrap = _dtensor_unwrap(hidden_states, gate)
        out = _fla_rms_norm_gated(
            hs_local.reshape(-1, D),
            gate_local.reshape(-1, D),
            _local(self.weight),
            None,
            "swish",
            residual=None,
            eps=self.eps,
            prenorm=False,
            residual_in_fp32=False,
        )
        return _dtensor_rewrap(out.reshape(orig_shape), wrap)

class OffsetRMSNorm(nn.Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``, weight init to zeros.
    Taken from Torchtitan - shares impl w/ transformers
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)

        # Llama does x.to(float16) * w whilst Qwen3_5 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        return ((1.0 + self.weight.float()) * x).to(input_dtype)

class SelfAttention(nn.Module):
    def __init__(self, cfg: Qwen3_5TextConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    @torch.compiler.disable(recursive=True)
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        assert x.dim() == 3 and x.shape[0] == 1, f"expected (1, total, hidden), got {tuple(x.shape)}"
        total = x.shape[1]

        input_shape = x.shape[:-1]
        q, gate = torch.chunk(self.q_proj(x).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        q = q.view(1, total, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(1, total, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(1, total, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        q = q.transpose(1, 2).reshape(total, self.num_heads, self.head_dim).contiguous()
        k = k.transpose(1, 2).reshape(total, self.num_kv_heads, self.head_dim).contiguous()
        v = v.transpose(1, 2).reshape(total, self.num_kv_heads, self.head_dim).contiguous()

        (q, k, v), wrap = _dtensor_unwrap(q, k, v)
        out = varlen_attn(
            q, k, v,
            cu_seq_q=cu_seqlens, cu_seq_k=cu_seqlens,
            max_q=max_seqlen, max_k=max_seqlen,
            window_size=(-1, 0),  # causal
        )
        out = _dtensor_rewrap(out, wrap)

        out = out.reshape(1, total, self.num_heads * self.head_dim)
        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen3_5TextConfig, **kwargs):
        super().__init__()
        self.n_key_heads = cfg.linear_num_key_heads
        self.n_value_heads = cfg.linear_num_value_heads
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim

        dim = cfg.hidden_size

        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = nn.Linear(dim, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, value_dim, bias=False)
        self.in_proj_a = nn.Linear(dim, cfg.linear_num_value_heads, bias=False)
        self.in_proj_b = nn.Linear(dim, cfg.linear_num_value_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=cfg.linear_conv_kernel_dim,
            groups=conv_dim,  # depthwise
            padding=0,  # causal padding applied manually in forward
        )

        self.A_log = nn.Parameter(torch.zeros(cfg.linear_num_value_heads))
        self.dt_bias = nn.Parameter(torch.ones(cfg.linear_num_value_heads))

        self.norm = RMSNormGated(cfg.linear_value_head_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(value_dim, dim, bias=False)

    @torch.compiler.disable(recursive=True)
    def forward(self, x: torch.Tensor, cu_seqlens, **kwargs) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.in_proj_qkv(x)  # (B, L, conv_dim) — channel-last in memory
        z = self.in_proj_z(x)
        a = self.in_proj_a(x)
        b = self.in_proj_b(x)
        (qkv, z, a, b), wrap = _dtensor_unwrap(qkv, z, a, b)

        # Per-token segment index for causal_conv1d_fn's packed mode. Using
        # bucketize keeps this graph-traceable with no host sync.
        seq_idx = torch.bucketize(
            torch.arange(L, device=qkv.device), cu_seqlens[1:-1], right=True
        ).to(torch.int32).unsqueeze(0).expand(B, -1).contiguous()

        # Fused causal-conv1d + SiLU. Input is (B, C, L) with channel-last
        # strides (stride(1) == 1), which is what the kernel expects when
        # seq_idx is provided. Triton kernel → needs local tensors.
        mixed_qkv = _causal_conv1d_fn(
            x=qkv.transpose(1, 2),
            weight=_local(self.conv1d.weight).squeeze(1),
            bias=_local(self.conv1d.bias),
            seq_idx=seq_idx,
            activation="silu",
        ).transpose(1, 2)  # (B, L, conv_dim)

        # Split into q, k, v and reshape to (B, L, H, D)
        key_dim = self.n_key_heads * self.key_head_dim
        value_dim = self.n_value_heads * self.value_head_dim
        q, k, v = mixed_qkv.split([key_dim, key_dim, value_dim], dim=-1)
        q = q.view(B, L, self.n_key_heads, self.key_head_dim)
        k = k.view(B, L, self.n_key_heads, self.key_head_dim)
        v = v.view(B, L, self.n_value_heads, self.value_head_dim)

        # Grouped heads: repeat q, k to match n_value_heads.
        repeat = self.n_value_heads // self.n_key_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        # Log-decay (g) and update weight (beta). A_log/dt_bias may be Replicate
        # DTensors under TP — work in local-tensor space.
        g = -torch.exp(_local(self.A_log).float()) * F.softplus(a.float() + _local(self.dt_bias))
        beta = torch.sigmoid(b)

        # Gated delta rule in (B, L, H, D) layout. Triton kernel → local only.
        output, _ = _fla_chunk_gated_delta_rule(
            q, k, v, g, beta,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cu_seqlens.to(torch.int64),
        )

        # Gated norm (Triton inside RMSNormGated, already DTensor-safe).
        z = z.view(B, L, self.n_value_heads, self.value_head_dim)
        output = self.norm(output, z)
        return self.out_proj(_dtensor_rewrap(output.reshape(B, L, -1), wrap))

class MLP(nn.Module):
    def __init__(self, cfg: Qwen3_5TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class MoeMLP(nn.Module):
    def __init__(self, config: Qwen3_5MoeConfig, intermediate_size: int):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = F.silu

    def forward(self, x):
        down_proj = self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
        return down_proj

class MoeExperts(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = config.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))
        self.act_fn = F.silu

    def forward(
        self,
        hidden_states: torch.Tensor,
        top_k_index: torch.Tensor,
        top_k_weights: torch.Tensor,
    ) -> torch.Tensor:
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = torch.nn.functional.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = nn.functional.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = self.act_fn(gate) * up
            current_hidden_states = nn.functional.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states

class TopKRouter(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.num_experts = config.num_experts
        self.hidden_dim = config.hidden_size
        self.weight = nn.Parameter(torch.zeros(self.num_experts, self.hidden_dim))

    def forward(self, hidden_states):
        hidden_states = hidden_states.reshape(-1, self.hidden_dim)
        router_logits = F.linear(hidden_states, self.weight)  # (seq_len, num_experts)
        router_probs = torch.nn.functional.softmax(router_logits, dtype=torch.float, dim=-1)
        router_top_value, router_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (seq_len, top_k)
        router_top_value /= router_top_value.sum(dim=-1, keepdim=True)
        router_top_value = router_top_value.to(router_logits.dtype)
        router_scores = router_top_value
        return router_logits, router_scores, router_indices

class MoE(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate = TopKRouter(config)
        self.experts = MoeExperts(config)
        self.shared_expert = MoeMLP(config, intermediate_size=config.shared_expert_intermediate_size)
        self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        N = batch_size * sequence_length
        num_experts = self.experts.num_experts

        hidden_states_reshaped = hidden_states.view(-1, hidden_dim)

        shared_expert_output = self.shared_expert(hidden_states_reshaped)

        router_logits, routing_weights, selected_experts = self.gate(hidden_states_reshaped)
        expert_output = self.experts(hidden_states_reshaped, selected_experts, routing_weights)

        shared_expert_output = F.sigmoid(self.shared_expert_gate(hidden_states_reshaped)) * shared_expert_output
        expert_output = expert_output + shared_expert_output
        
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.experts.num_experts)
        tokens_per_expert = expert_mask.sum(dim=(0, 1), dtype=torch.float)
        fraction_tokens = tokens_per_expert / (N * self.gate.top_k)

        router_probs = torch.nn.functional.softmax(router_logits, dim=-1).sum(dim=0)
        fraction_probs = router_probs.sum(dim=0) / N

        aux_loss = num_experts * torch.sum(fraction_tokens * fraction_probs)

        return expert_output.reshape(batch_size, sequence_length, hidden_dim), aux_loss

class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3_5TextConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == "full_attention":
            self.self_attn = SelfAttention(cfg)
        else:
            self.linear_attn = GatedDeltaNet(cfg)

        self.mlp = MoE(cfg)
        self.input_layernorm = OffsetRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        # self_attn has extra arguments, the kwargs are omitted

        attn = self.self_attn if self.layer_type == "full_attention" else self.linear_attn
        x = x + attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen
        )
        mlp_out, aux_loss = self.mlp(self.post_attention_layernorm(x))
        x = x + mlp_out
        return x, aux_loss

class LanguageModel(nn.Module):
    """HF name: `model.language_model`."""

    def __init__(self, cfg: Qwen3_5TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        layers = []
        for layer_id in range(cfg.num_hidden_layers):
            is_full = (layer_id + 1) % cfg.full_attention_interval == 0
            layer_type = "full_attention" if is_full else "linear_attention"
            layers.append(DecoderLayer(cfg, layer_type, ))
        self.layers = nn.ModuleList(layers)

        self.norm = OffsetRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = inputs_embeds

        total_aux_loss = 0
        for i, layer in enumerate(self.layers):
            x, aux_loss = layer(x, cos, sin, cu_seqlens, max_seqlen)
            total_aux_loss += aux_loss
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                x = x.clone()
                x[visual_pos_masks] = (
                    x[visual_pos_masks] + deepstack_visual_embeds[i].to(x.dtype)
                )

        return self.norm(x) if self.norm is not None else x, total_aux_loss

class VisionPatchEmbed(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.temporal_patch_size = cfg.temporal_patch_size
        self.in_channels = cfg.in_channels
        self.embed_dim = cfg.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(dtype=target_dtype)).view(-1, self.embed_dim)

class VisionMLP(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)
        if cfg.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        elif cfg.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif cfg.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported vision hidden_act: {cfg.hidden_act}")

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)  # (seqlen, dim/2)

class VisionAttention(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        S = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(S, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )  # each (S, H, D) — already the layout varlen_attn wants
        cos, sin = position_embeddings
        q, k = apply_rope_vision(q, k, cos, sin)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        (q, k, v), wrap = _dtensor_unwrap(q, k, v)
        out = varlen_attn(
            q, k, v,
            cu_seq_q=cu_seqlens, cu_seq_k=cu_seqlens,
            max_q=max_seqlen, max_k=max_seqlen,
            window_size=(-1, -1),  # non-causal
        )
        out = _dtensor_rewrap(out, wrap)
        return self.proj(out.reshape(S, self.dim))

class VisionBlock(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, cu_seqlens, max_seqlen, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens, max_seqlen, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x

class VisionPatchMerger(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = cfg.hidden_size * (cfg.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else cfg.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, cfg.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

class VisionModel(nn.Module):
    """HF name: `model.visual`. Mirrors `Qwen3VLVisionModel` exactly."""

    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.num_grid_per_side = int(cfg.num_position_embeddings ** 0.5)

        self.patch_embed = VisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)

        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = VisionPatchMerger(cfg, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [VisionPatchMerger(cfg, use_postshuffle_norm=True)
             for _ in range(len(self.deepstack_visual_indexes))]
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge = self.spatial_merge_size
        grid_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim/2)
        device = freq_table.device

        total = sum(t * h * w for t, h, w in grid_list)
        pos_ids = torch.empty((total, 2), dtype=torch.long, device=device)
        offset = 0
        for t, h, w in grid_list:
            mh, mw = h // merge, w // merge
            block_rows = torch.arange(mh, device=device)
            block_cols = torch.arange(mw, device=device)
            intra_r = torch.arange(merge, device=device)
            intra_c = torch.arange(merge, device=device)
            row_idx = block_rows[:, None, None, None] * merge + intra_r[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge + intra_c[None, None, None, :]
            row_idx = row_idx.expand(mh, mw, merge, merge).reshape(-1)
            col_idx = col_idx.expand(mh, mw, merge, merge).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            n = coords.shape[0]
            pos_ids[offset : offset + n] = coords
            offset += n

        emb = freq_table[pos_ids]  # (total, 2, dim/2)
        return emb.flatten(1)  # (total, dim)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_list = grid_thw.tolist()
        grid_ts = [r[0] for r in grid_list]
        grid_hs = [r[1] for r in grid_list]
        grid_ws = [r[2] for r in grid_list]
        device = self.pos_embed.weight.device

        idx_list: list[list[int]] = [[], [], [], []]
        weight_list: list[list[float]] = [[], [], [], []]

        for _t, h, w in grid_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
        wt = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pe = self.pos_embed(idx_t) * wt[:, :, None]
        patch_pe = pe[0] + pe[1] + pe[2] + pe[3]
        chunks = patch_pe.split([h * w for h, w in zip(grid_hs, grid_ws)])

        merge = self.spatial_merge_size
        out = []
        for pe_chunk, t, h, w in zip(chunks, grid_ts, grid_hs, grid_ws):
            pe_chunk = pe_chunk.repeat(t, 1)
            pe_chunk = (
                pe_chunk.view(t, h // merge, merge, w // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(pe_chunk)
        return torch.cat(out)

    def forward(
        self, hidden_states: torch.Tensor, grid_thw: torch.Tensor
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns (merged_hidden_states, deepstack_features)."""
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seg_lens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu = F.pad(seg_lens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
        max_seqlen = int(seg_lens.max().item())

        deepstack: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu, max_seqlen, position_embeddings)
            if i in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)]
                deepstack.append(merger(hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack

class Qwen3_5InnerLanguage(nn.Module):
    def __init__(self, cfg: Qwen3VLConfig):
        super().__init__()
        self.language_model = LanguageModel(cfg.text)

class Qwen3_5Inner(nn.Module):
    """HF name: `model`. Groups `language_model` and `visual`.
    This is only used to match the state keys. """
    def __init__(self, cfg: Qwen3VLConfig):
        super().__init__()
        self.language_model = LanguageModel(cfg.text)
        self.visual = VisionModel(cfg.vision)

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3VLConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3_5Inner(cfg)
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

        # Text rope: store only inv_freq; cos/sin are computed per-forward via
        # MRoPE (3D position ids). For text-only inputs the 3 axes share the
        # same arange, which collapses to plain 1D rope.
        head_dim = cfg.text.head_dim
        partial = cfg.text.rope_parameters.get('partial_rotary_factor', 1.0)
        rope_dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            cfg.text.rope_parameters['rope_theta'] ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        self.register_buffer("text_inv_freq", inv_freq, persistent=False)

        cfg_rope_section = cfg.text.rope_parameters['mrope_section']
        self.mrope_section = list(cfg_rope_section)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 3D MRoPE position_ids over a packed row.

        Args:
            input_ids: (1, total) packed token ids.
            cu_seqlens: (N+1,) int32 cumulative offsets (starts at 0).
            image_grid_thw, video_grid_thw: per-image/video (T,H,W) grids.

        Positions reset to 0 at each packed-sample boundary. Within a segment,
        text runs are `arange`, image/video runs are 3D per HF's algorithm.
        """
        image_id = self.cfg.image_token_id
        video_id = self.cfg.video_token_id
        spatial = self.cfg.vision.spatial_merge_size

        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        _, S = input_ids.shape
        device = input_ids.device
        mm_type = torch.zeros(S, dtype=torch.int64, device=device)
        mm_type[input_ids[0] == image_id] = 1
        mm_type[input_ids[0] == video_id] = 2
        types_all = mm_type.tolist()

        image_iter = iter(image_grid_thw) if image_grid_thw is not None else None
        video_iter = iter(video_grid_thw) if video_grid_thw is not None else None

        bounds = cu_seqlens.tolist()
        out = torch.zeros(3, 1, S, dtype=torch.int64, device=device)

        for start, end in zip(bounds[:-1], bounds[1:]):
            if start == end:
                continue
            types_seg = types_all[start:end]
            pos_list: list[torch.Tensor] = []
            current = 0
            j = 0
            while j < len(types_seg):
                k = j
                while k < len(types_seg) and types_seg[k] == types_seg[j]:
                    k += 1
                key = types_seg[j]
                length = k - j
                if key == 0:
                    p = torch.arange(length, device=device).view(1, -1).expand(3, -1) + current
                    current += length
                else:
                    grid = next(image_iter if key == 1 else video_iter)
                    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
                    llm_h, llm_w, llm_t = h // spatial, w // spatial, t
                    n = llm_t * llm_h * llm_w
                    pw = torch.arange(current, current + llm_w, device=device).repeat(llm_h * llm_t)
                    ph = torch.arange(current, current + llm_h, device=device).repeat_interleave(
                        llm_w * llm_t
                    )
                    pt = torch.full((n,), current, device=device, dtype=torch.int64)
                    p = torch.stack([pt, ph, pw], dim=0)
                    current += max(llm_h, llm_w)
                pos_list.append(p)
                j = k
            out[:, 0, start:end] = torch.cat(pos_list, dim=1)
        return out

    def _compute_cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (B, S) or (3, B, S) → cos/sin of shape (B, S, D)."""
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        return mrope_cos_sin(self.text_inv_freq, position_ids, self.mrope_section)

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> "CausalLMOutput | torch.Tensor":
        """Varlen-only forward.

        Expected layout: `input_ids` / `inputs_embeds` is a single packed row
        `(1, total)` / `(1, total, H)`. `attention_mask` is interpreted as
        `cu_seqlens`: a 1D int32 tensor of cumulative offsets starting at 0
        (same tensor consumed by `torch.nn.attention.varlen.varlen_attn`).
        If `attention_mask` is None, the whole row is treated as one sample.
        """
        if getattr(self.model.language_model, "embed_tokens", None) is not None:
            input_ids = hidden_states
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        else:
            inputs_embeds = hidden_states
            if input_ids is None:
                raise ValueError("input_ids must be passed to intermediate stages for MRoPE calculation.")

        assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1, (
            f"inputs_embeds should be (1, total, hidden_dim), got {tuple(inputs_embeds.shape)}"
        )

        total = inputs_embeds.shape[1]
        device = inputs_embeds.device

        if attention_mask is None:
            cu_seqlens = torch.tensor([0, total], device=device, dtype=torch.int32)
        else:
            assert (
                attention_mask.dim() == 1
                and attention_mask[0].item() == 0
                and attention_mask[-1].item() == total
            ), "attention_mask must be cu_seqlens: 1D int32, starts at 0, ends at total"
            cu_seqlens = attention_mask.to(torch.int32)

        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        visual_pos_masks: torch.Tensor | None = None
        deepstack_visual_embeds: list[torch.Tensor] | None = None

        if getattr(self.model, "visual", None) is not None:
            if pixel_values is not None:
                assert image_grid_thw is not None
                merged, deepstack = self.model.visual(pixel_values, image_grid_thw)
                merged = merged.to(inputs_embeds.dtype)
                image_mask = input_ids == self.cfg.image_token_id
                #assert image_mask.sum().item() == merged.shape[0], (f"image tokens={image_mask.sum().item()} vs features={merged.shape[0]}")
                inputs_embeds = inputs_embeds.masked_scatter(
                    image_mask.unsqueeze(-1).expand_as(inputs_embeds), merged
                )
                visual_pos_masks = image_mask
                deepstack_visual_embeds = deepstack

            if pixel_values_videos is not None:
                assert video_grid_thw is not None
                merged_v, deepstack_v = self.model.visual(pixel_values_videos, video_grid_thw)
                merged_v = merged_v.to(inputs_embeds.dtype)
                video_mask = input_ids == self.cfg.video_token_id
                inputs_embeds = inputs_embeds.masked_scatter(
                    video_mask.unsqueeze(-1).expand_as(inputs_embeds), merged_v
                )
                if visual_pos_masks is None:
                    visual_pos_masks = video_mask
                    deepstack_visual_embeds = deepstack_v
                else:
                    combined = visual_pos_masks | video_mask
                    image_only = visual_pos_masks[combined]
                    video_only = video_mask[combined]
                    merged_ds = []
                    for img_ds, vid_ds in zip(deepstack_visual_embeds, deepstack_v):
                        e = img_ds.new_zeros(combined.sum().item(), img_ds.shape[-1])
                        e[image_only] = img_ds
                        e[video_only] = vid_ds
                        merged_ds.append(e)
                    visual_pos_masks = combined
                    deepstack_visual_embeds = merged_ds

        if position_ids is None:
            if image_grid_thw is not None or video_grid_thw is not None:
                assert input_ids is not None, "need input_ids to compute 3D MRoPE positions"
                position_ids = self.get_rope_index(
                    input_ids,
                    cu_seqlens=cu_seqlens,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )
            else:
                # Per-segment arange, matching cu_seqlens boundaries.
                pos = torch.zeros(total, device=device, dtype=torch.int64)
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                    pos[start:end] = torch.arange(end - start, device=device)
                position_ids = pos.view(1, 1, -1).expand(3, 1, -1)

        cos, sin = self._compute_cos_sin(position_ids)
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        h, total_aux_loss = self.model.language_model(
            inputs_embeds,
            cos,
            sin,
            cu_seqlens,
            max_seqlen,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )

        if getattr(self, "lm_head", None) is not None:
            return self.lm_head(h), total_aux_loss
        else:
            return h, total_aux_loss

    @classmethod
    def from_pretrained(
        cls,
        snapshot_dir: str | Path,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        *,
        load_vision: bool = True,
        weights: bool = True,
    ) -> "Qwen3_5ForCausalLM":
        snapshot_dir = Path(snapshot_dir)
        cfg = Qwen3VLConfig.from_json(snapshot_dir / "config.json")
        if dtype is None:
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[cfg.torch_dtype]

        with torch.device("meta"):
            model = cls(cfg)

        model = model.to_empty(device='cuda').to(dtype=dtype)
        if False:
            load_safetensors_into(
                model,
                snapshot_dir,
                device=device,
                dtype=dtype,
                load_vision=load_vision,
            )
        else:
            with torch.no_grad():
                for name, module in model.named_modules():
                    if isinstance(module, torch.nn.Linear):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                    elif isinstance(module, torch.nn.Embedding):
                        torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                    elif isinstance(module, torch.nn.Conv3d) or isinstance(module, torch.nn.Conv1d):
                        torch.nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                        if module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                    
                    elif "Norm" in module.__class__.__name__:
                        if hasattr(module, "weight") and module.weight is not None:
                            if "Offset" in module.__class__.__name__:
                                torch.nn.init.zeros_(module.weight)
                            else:
                                torch.nn.init.ones_(module.weight)
                        if hasattr(module, "bias") and module.bias is not None:
                            torch.nn.init.zeros_(module.bias)
                    
                    elif "MoeExperts" in module.__class__.__name__:
                        torch.nn.init.normal_(module.gate_up_proj, mean=0.0, std=0.02)
                        torch.nn.init.normal_(module.down_proj, mean=0.0, std=0.02)
                    
                    elif "TopKRouter" in module.__class__.__name__:
                        # Router weights are typically initialized to 0 or very small values
                        torch.nn.init.zeros_(module.weight)

                    elif "RMSNormGated" in module.__class__.__name__:
                        if hasattr(module, "weight") and module.weight is not None:
                            torch.nn.init.ones_(module.weight)
                            
                    elif "OffsetRMSNorm" in module.__class__.__name__:
                        if hasattr(module, "weight") and module.weight is not None:
                            # Offset norm weights start at 0
                            torch.nn.init.zeros_(module.weight)

        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.language_model.embed_tokens.weight

        # `to_empty` also wipes non-persistent buffers. Recompute the vision
        # rotary `inv_freq` (it's not in the safetensors).
        if load_vision:
            head_dim_v = cfg.vision.hidden_size // cfg.vision.num_heads
            rdim = head_dim_v // 2
            inv_freq_v = 1.0 / (
                10000.0 ** (torch.arange(0, rdim, 2, dtype=torch.float32, device=device) / rdim)
            )
            model.model.visual.rotary_pos_emb.inv_freq = inv_freq_v

        head_dim = cfg.text.head_dim
        partial = cfg.text.rope_parameters.get('partial_rotary_factor', 1.0)
        rope_dim = int(head_dim * partial)
        text_inv = 1.0 / (
            cfg.text.rope_parameters['rope_theta']
            ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
        )
        model.text_inv_freq = text_inv
        return model


@torch.no_grad()
def initialize_missing_weights(model):
    for module in model.modules():
        if hasattr(module, 'reset_parameters'):
            module.reset_parameters()
            
        elif isinstance(module, OffsetRMSNorm):
            torch.nn.init.zeros_(module.weight)
        elif isinstance(module, RMSNormGated):
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, GatedDeltaNet):
            torch.nn.init.zeros_(module.A_log)
            torch.nn.init.ones_(module.dt_bias)
            
        elif "MoeExperts" in module.__class__.__name__:
            torch.nn.init.normal_(module.gate_up_proj, mean=0.0, std=0.02)
            torch.nn.init.normal_(module.down_proj, mean=0.0, std=0.02)
        elif "TopKRouter" in module.__class__.__name__:
            torch.nn.init.zeros_(module.weight)

    for name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"WARNING: Fallback init applied to missed parameter: {name}")
            if param.dim() >= 2:
                torch.nn.init.normal_(param, mean=0.0, std=0.02)
            else:
                torch.nn.init.zeros_(param)
