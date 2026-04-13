"""Torch-native Qwen3 (text-only) implementation.

Loads config.json and safetensors shards directly from an HF snapshot
directory. The parameter names mirror the HF layout so state dicts from
`Qwen3ForCausalLM` load without remapping.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open


@dataclass
class CausalLMOutput:
    loss: torch.Tensor
    logits: torch.Tensor


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Standard next-token cross-entropy: predict token t+1 from position t.

    Positions with label == ignore_index are excluded from the mean.
    """
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


@dataclass
class Qwen3Config:
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
    torch_dtype: str = "bfloat16"

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        with open(path, "r") as f:
            cfg = json.load(f)
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"]),
            max_position_embeddings=cfg["max_position_embeddings"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
            torch_dtype=cfg.get("torch_dtype", "bfloat16"),
        )


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
    """Returns (cos, sin) of shape (max_seq_len, head_dim).

    Uses the HF layout where the first half and second half of the head
    dimension carry the same frequency, matching `rotate_half`.
    """
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)  # (seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (seq, head_dim)
    return emb.cos().to(dtype), emb.sin().to(dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    # q, k: (B, H, S, D). cos/sin: (S, D) -> (1, 1, S, D)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_out = (q * cos) + (rotate_half(q) * sin)
    k_out = (k * cos) + (rotate_half(k) * sin)
    return q_out.to(q.dtype), k_out.to(k.dtype)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None, :, :].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

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
    ) -> torch.Tensor:
        B, S, _ = x.shape
        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)  # (B, H, S, D)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rope(q, k, cos, sin)

        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.transpose(1, 2).contiguous().view(B, S, self.num_heads * self.head_dim)
        return self.o_proj(out)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Block(nn.Module):
    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.self_attn = Qwen3Attention(cfg)
        self.mlp = Qwen3MLP(cfg)
        self.input_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(self, x, cos, sin):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin)
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


class Qwen3Model(nn.Module):
    """Decoder stack — parameter names mirror HF `model.*`."""

    def __init__(self, cfg: Qwen3Config):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen3Block(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x, cos, sin)
        return self.norm(x)


class Qwen3ForCausalLM(nn.Module):
    """Top-level Qwen3 text-only causal LM.

    Parameter layout (matches HF):
        model.embed_tokens.weight
        model.layers.{i}.input_layernorm.weight
        model.layers.{i}.post_attention_layernorm.weight
        model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
        model.layers.{i}.self_attn.{q,k}_norm.weight
        model.layers.{i}.mlp.{gate,up,down}_proj.weight
        model.norm.weight
        lm_head.weight   (absent when tie_word_embeddings=True)
    """

    def __init__(self, cfg: Qwen3Config, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3Model(cfg)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

        cos, sin = precompute_rope_cache(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
        )
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> "CausalLMOutput | torch.Tensor":
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        S = input_ids.shape[1]
        if position_ids is None:
            cos = self.rope_cos[:S]
            sin = self.rope_sin[:S]
        else:
            cos = self.rope_cos[position_ids[0]]
            sin = self.rope_sin[position_ids[0]]
        h = self.model(input_ids, cos, sin)
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
    ) -> "Qwen3ForCausalLM":
        snapshot_dir = Path(snapshot_dir)
        cfg = Qwen3Config.from_json(snapshot_dir / "config.json")
        if dtype is None:
            dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}[cfg.torch_dtype]

        with torch.device("meta"):
            model = cls(cfg)
        model = model.to_empty(device=device)

        load_safetensors_into(model, snapshot_dir, device=device, dtype=dtype)

        # Recompute RoPE buffers on target device/dtype (float32 for precision).
        cos, sin = precompute_rope_cache(
            head_dim=cfg.head_dim,
            max_seq_len=cfg.max_position_embeddings,
            theta=cfg.rope_theta,
            device=torch.device(device),
            dtype=torch.float32,
        )
        model.rope_cos = cos
        model.rope_sin = sin
        return model


def load_safetensors_into(
    model: Qwen3ForCausalLM,
    snapshot_dir: Path,
    device: str | torch.device,
    dtype: torch.dtype,
) -> None:
    index_path = snapshot_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: dict[str, list[str]] = {}
        for name, shard in weight_map.items():
            shards.setdefault(shard, []).append(name)
        files = {shard: snapshot_dir / shard for shard in shards}
    else:
        single = snapshot_dir / "model.safetensors"
        assert single.exists(), f"No safetensors found in {snapshot_dir}"
        files = {single.name: single}
        shards = {single.name: None}  # load all

    state = dict(model.state_dict())
    loaded: set[str] = set()

    for shard_name, shard_path in files.items():
        with safe_open(str(shard_path), framework="pt", device=str(device)) as f:
            keys = shards[shard_name] if shards[shard_name] is not None else list(f.keys())
            for k in keys:
                if k not in state:
                    # lm_head.weight may be missing when tied — skip silently.
                    continue
                tensor = f.get_tensor(k).to(dtype=dtype)
                if state[k].shape != tensor.shape:
                    raise ValueError(f"Shape mismatch for {k}: {state[k].shape} vs {tensor.shape}")
                state[k].copy_(tensor)
                loaded.add(k)

    # Handle tied embeddings: if lm_head.weight wasn't in the checkpoint but
    # is tied to embed_tokens, it's already shared via the Parameter assignment.
    expected = set(state.keys())
    missing = expected - loaded
    if model.cfg.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if missing:
        raise RuntimeError(f"Missing weights after load: {sorted(missing)[:8]} ... ({len(missing)} total)")
