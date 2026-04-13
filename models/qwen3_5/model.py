from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from torch.nn.attention.varlen import varlen_attn

from torch.distributed.tensor import DTensor


# ------------------------------------------------------------------ outputs

@dataclass
class CausalLMOutput:
    loss: torch.Tensor
    logits: torch.Tensor


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


# ------------------------------------------------------------------- config

@dataclass
class Qwen3_5TextConfig:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float
    tie_word_embeddings: bool
    # MRoPE (used in Stage C); kept here so the config round-trips cleanly.
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = True


@dataclass
class Qwen3_5VisionConfig:
    depth: int
    hidden_size: int
    intermediate_size: int
    num_heads: int
    in_channels: int
    patch_size: int
    temporal_patch_size: int
    spatial_merge_size: int
    num_position_embeddings: int
    out_hidden_size: int
    hidden_act: str
    deepstack_visual_indexes: list[int]


@dataclass
class Qwen3VLConfig:
    text: Qwen3VLTextConfig
    vision: Qwen3VLVisionConfig
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int
    vision_end_token_id: int
    tie_word_embeddings: bool
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3VLConfig":
        with open(path, "r") as f:
            raw = json.load(f)
        tc = raw["text_config"]
        rs = tc.get("rope_scaling") or {}
        text = Qwen3VLTextConfig(
            vocab_size=tc["vocab_size"],
            hidden_size=tc["hidden_size"],
            intermediate_size=tc["intermediate_size"],
            num_hidden_layers=tc["num_hidden_layers"],
            num_attention_heads=tc["num_attention_heads"],
            num_key_value_heads=tc["num_key_value_heads"],
            head_dim=tc.get("head_dim", tc["hidden_size"] // tc["num_attention_heads"]),
            max_position_embeddings=tc["max_position_embeddings"],
            rms_norm_eps=tc["rms_norm_eps"],
            rope_theta=tc["rope_theta"],
            tie_word_embeddings=tc.get("tie_word_embeddings", raw.get("tie_word_embeddings", False)),
            mrope_section=rs.get("mrope_section"),
            mrope_interleaved=rs.get("mrope_interleaved", True),
        )
        vc = raw["vision_config"]
        vision = Qwen3VLVisionConfig(
            depth=vc["depth"],
            hidden_size=vc["hidden_size"],
            intermediate_size=vc["intermediate_size"],
            num_heads=vc["num_heads"],
            in_channels=vc["in_channels"],
            patch_size=vc["patch_size"],
            temporal_patch_size=vc["temporal_patch_size"],
            spatial_merge_size=vc["spatial_merge_size"],
            num_position_embeddings=vc["num_position_embeddings"],
            out_hidden_size=vc["out_hidden_size"],
            hidden_act=vc["hidden_act"],
            deepstack_visual_indexes=vc["deepstack_visual_indexes"],
        )
        return cls(
            text=text,
            vision=vision,
            image_token_id=raw["image_token_id"],
            video_token_id=raw["video_token_id"],
            vision_start_token_id=raw["vision_start_token_id"],
            vision_end_token_id=raw["vision_end_token_id"],
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
            torch_dtype=raw.get("torch_dtype") or tc.get("dtype", "bfloat16"),
        )


# ----------------------------------------------------------------- building blocks

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(in_dtype)


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, S, D). cos, sin: either (S, D) or (B, S, D).
    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)  # (B, 1, S, D)
    sin = sin.unsqueeze(1)

    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def mrope_cos_sin(
    inv_freq: torch.Tensor,
    position_ids: torch.Tensor,
    mrope_section: list[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Interleaved 3D MRoPE matching HF Qwen3VLTextRotaryEmbedding.

    Args:
        inv_freq: (D/2,) text rope inv frequencies.
        position_ids: (3, B, S) integer positions for T/H/W.
        mrope_section: lengths for T, H, W in the freq layout (sum = D/2).

    Returns:
        cos, sin: (B, S, D).
    """
    # freqs[d, b, s, i] = position_ids[d, b, s] * inv_freq[i]
    pid = position_ids.to(torch.float32)  # (3, B, S)
    freqs = pid.unsqueeze(-1) * inv_freq[None, None, None, :]  # (3, B, S, D/2)

    freqs_t = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]

    emb = torch.cat((freqs_t, freqs_t), dim=-1)  # (B, S, D)
    return emb.cos(), emb.sin()


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


# ------------------------------------------------------------------- text stack

class Qwen3VLTextAttention(nn.Module):
    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        # Operates exclusively in varlen layout: x is (1, total, hidden),
        # cu_seqlens is (N+1,) int32 starting with 0.
        assert x.dim() == 3 and x.shape[0] == 1, f"expected (1, total, hidden), got {tuple(x.shape)}"
        total = x.shape[1]

        q = self.q_proj(x).view(1, total, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(1, total, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(1, total, self.num_kv_heads, self.head_dim)

        # Rope expects (B, H, S, D). Apply then flatten back for varlen.
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        q = q.transpose(1, 2).reshape(total, self.num_heads, self.head_dim).contiguous()
        k = k.transpose(1, 2).reshape(total, self.num_kv_heads, self.head_dim).contiguous()
        v = v.transpose(1, 2).reshape(total, self.num_kv_heads, self.head_dim).contiguous()

        out = varlen_attn(
            q, k, v,
            cu_seq_q=cu_seqlens, cu_seq_k=cu_seqlens,
            max_q=max_seqlen, max_k=max_seqlen,
            window_size=(-1, 0),  # causal
        )  # (total, num_heads, head_dim)

        out = out.reshape(1, total, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3VLTextMLP(nn.Module):
    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3VLTextLayer(nn.Module):
    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.self_attn = Qwen3VLTextAttention(cfg)
        self.mlp = Qwen3VLTextMLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, cu_seqlens, max_seqlen)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3VLLanguageModel(nn.Module):
    """HF name: `model.language_model`."""

    def __init__(self, cfg: Qwen3VLTextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3VLTextLayer(cfg) for _ in range(cfg.num_hidden_layers)]
        )
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

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
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cu_seqlens, max_seqlen)
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                x = x.clone()
                x[visual_pos_masks] = (
                    x[visual_pos_masks] + deepstack_visual_embeds[i].to(x.dtype)
                )
        return self.norm(x)


# ------------------------------------------------------------------- vision encoder
# Mirrors HF `Qwen3VLVisionModel` (transformers/models/qwen3_vl/modeling_qwen3_vl.py).
# Parameter names match HF so checkpoints load without remapping. Input/output
# layout is the flat (seq_len, hidden_size) format with `cu_seqlens` chunking
# (not the padded FlexAttention layout used by torchtitan).


class Qwen3VLVisionPatchEmbed(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
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


class Qwen3VLVisionMLP(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
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


class Qwen3VLVisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)  # (seqlen, dim/2)


def _rotate_half_last(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope_vision(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q, orig_k = q.dtype, k.dtype
    q, k = q.float(), k.float()
    cos = cos.unsqueeze(-2).float()
    sin = sin.unsqueeze(-2).float()
    q_emb = (q * cos) + (_rotate_half_last(q) * sin)
    k_emb = (k * cos) + (_rotate_half_last(k) * sin)
    return q_emb.to(orig_q), k_emb.to(orig_k)


class Qwen3VLVisionAttention(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
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
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        S = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(S, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )  # each (S, H, D)
        cos, sin = position_embeddings
        q, k = apply_rope_vision(q, k, cos, sin)

        # (1, H, S, D)
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)

        lengths = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()
        q_chunks = torch.split(q, lengths, dim=2)
        k_chunks = torch.split(k, lengths, dim=2)
        v_chunks = torch.split(v, lengths, dim=2)

        outs = []
        for qc, kc, vc in zip(q_chunks, k_chunks, v_chunks):
            o = F.scaled_dot_product_attention(qc, kc, vc, is_causal=False)
            outs.append(o.transpose(1, 2))  # (1, Sc, H, D)
        out = torch.cat(outs, dim=1).reshape(S, self.dim)
        return self.proj(out)


class Qwen3VLVisionBlock(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = Qwen3VLVisionAttention(cfg)
        self.mlp = Qwen3VLVisionMLP(cfg)

    def forward(self, x, cu_seqlens, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x


class Qwen3VLVisionPatchMerger(nn.Module):
    def __init__(self, cfg: Qwen3VLVisionConfig, use_postshuffle_norm: bool = False):
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


class Qwen3VLVisionModel(nn.Module):
    """HF name: `model.visual`. Mirrors `Qwen3VLVisionModel` exactly."""

    def __init__(self, cfg: Qwen3VLVisionConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.num_grid_per_side = int(cfg.num_position_embeddings ** 0.5)

        self.patch_embed = Qwen3VLVisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)

        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([Qwen3VLVisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [Qwen3VLVisionPatchMerger(cfg, use_postshuffle_norm=True)
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

        cu = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
            dim=0, dtype=torch.int32
        )
        cu = F.pad(cu, (1, 0), value=0)

        deepstack: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu, position_embeddings)
            if i in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)]
                deepstack.append(merger(hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack


class Qwen3VLInner(nn.Module):
    """HF name: `model`. Groups `language_model` and `visual`."""

    def __init__(self, cfg: Qwen3VLConfig):
        super().__init__()
        self.language_model = Qwen3VLLanguageModel(cfg.text)
        self.visual = Qwen3VLVisionModel(cfg.vision)


# ------------------------------------------------------------------- top-level

class Qwen3VLForCausalLM(nn.Module):
    """
    Parameter layout (matches HF `Qwen3VLForConditionalGeneration`):
        model.language_model.embed_tokens.weight
        model.language_model.layers.{i}.input_layernorm.weight
        model.language_model.layers.{i}.post_attention_layernorm.weight
        model.language_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.language_model.layers.{i}.self_attn.{q,k}_norm.weight
        model.language_model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.language_model.norm.weight
        lm_head.weight                                   (absent when tied)
        model.visual.*                                   (added in Stage B)
    """

    def __init__(self, cfg: Qwen3VLConfig, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3VLInner(cfg)
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

        # Text rope: store only inv_freq; cos/sin are computed per-forward via
        # MRoPE (3D position ids). For text-only inputs the 3 axes share the
        # same arange, which collapses to plain 1D rope.
        head_dim = cfg.text.head_dim
        inv_freq = 1.0 / (
            cfg.text.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("text_inv_freq", inv_freq, persistent=False)

        default_mrope = [head_dim // 2 // 3, head_dim // 2 // 3, head_dim // 2 // 3]
        self.mrope_section = list(cfg.text.mrope_section) if cfg.text.mrope_section else default_mrope

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
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
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
        assert (input_ids is None) ^ (inputs_embeds is None)
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids is not None:
            assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
                f"varlen expects packed (1, total), got {tuple(input_ids.shape)}"
            )

        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1
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

        if pixel_values is not None:
            assert image_grid_thw is not None
            merged, deepstack = self.model.visual(pixel_values, image_grid_thw)
            merged = merged.to(inputs_embeds.dtype)
            image_mask = input_ids == self.cfg.image_token_id
            assert image_mask.sum().item() == merged.shape[0], (
                f"image tokens={image_mask.sum().item()} vs features={merged.shape[0]}"
            )
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

        h = self.model.language_model(
            inputs_embeds,
            cos,
            sin,
            cu_seqlens,
            max_seqlen,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        logits = self.lm_head(h)

        if labels is None:
            return logits
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        loss = causal_lm_loss(logits, labels)
        return CausalLMOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(
        cls,
        snapshot_dir: str | Path,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        *,
        load_vision: bool = True,
    ) -> "Qwen3VLForCausalLM":
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
        model = model.to_empty(device=device).to(dtype=dtype)

        load_safetensors_into(
            model,
            snapshot_dir,
            device=device,
            dtype=dtype,
            load_vision=load_vision,
        )

        # `to_empty` above re-materializes every parameter and breaks the
        # tie established in `__init__`. Re-tie here so `lm_head` (absent
        # from checkpoints when tied) shares storage with the embedding.
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

        # Recompute text inv_freq (non-persistent buffer wiped by `to_empty`).
        head_dim = cfg.text.head_dim
        text_inv = 1.0 / (
            cfg.text.rope_theta
            ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
        )
        model.text_inv_freq = text_inv
        return model


# --------------------------------------------------------------- state-dict load

def load_safetensors_into(
    model: Qwen3VLForCausalLM,
    snapshot_dir: Path,
    device: str | torch.device,
    dtype: torch.dtype,
    load_vision: bool,
) -> None:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: dict[str, list[str] | None] = {}
        for name, shard in weight_map.items():
            shards.setdefault(shard, []).append(name)  # type: ignore[arg-type]
        files = {shard: snapshot_dir / shard for shard in shards}
    else:
        single = snapshot_dir / "model.safetensors"
        assert single.exists(), f"No safetensors found in {snapshot_dir}"
        files = {single.name: single}
        shards = {single.name: None}

    state = dict(model.state_dict())
    loaded: set[str] = set()
    skipped_vision = 0

    for shard_name, shard_path in files.items():
        with safe_open(str(shard_path), framework="pt", device=str(device)) as f:
            keys = shards[shard_name] if shards[shard_name] is not None else list(f.keys())
            for k in keys:
                if (not load_vision) and k.startswith("model.visual."):
                    skipped_vision += 1
                    continue
                if k not in state:
                    # tied lm_head is absent from file; other absences are bugs.
                    continue
                tensor = f.get_tensor(k).to(dtype=dtype)
                if state[k].shape != tensor.shape:
                    raise ValueError(f"Shape mismatch for {k}: {state[k].shape} vs {tensor.shape}")
                state[k].copy_(tensor)
                loaded.add(k)

    missing = set(state.keys()) - loaded
    if model.cfg.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if not load_vision:
        missing = {m for m in missing if not m.startswith("model.visual.")}
    if missing:
        raise RuntimeError(f"Missing weights after load: {sorted(missing)[:8]} ... ({len(missing)} total)")

    if not load_vision and skipped_vision:
        # Stage A informational only.
        pass
