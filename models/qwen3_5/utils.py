from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from torch.distributed.tensor import DTensor, Replicate

@torch.compiler.disable(recursive=True)
def _dtensor_unwrap(*tensors: torch.Tensor):
    """Strip DTensor wrapping for kernels that don't understand DTensor.

    Returns ``(local_tensors, wrap_info)`` where ``wrap_info`` is ``None`` if
    the first input was already a plain tensor, otherwise a
    ``(device_mesh, placements)`` pair captured from it. Pass the pair to
    :func:`_dtensor_rewrap` to re-wrap an output tensor with the same layout.
    """
    first = tensors[0]
    if isinstance(first, DTensor):
        wrap = (first.device_mesh, first.placements)
        return tuple(
            t.to_local() if isinstance(t, DTensor) else t for t in tensors
        ), wrap
    return tensors, None


@torch.compiler.disable(recursive=True)
def _dtensor_rewrap(tensor: torch.Tensor, wrap_info) -> torch.Tensor:
    if wrap_info is None:
        return tensor
    mesh, placements = wrap_info
    return DTensor.from_local(
        tensor, device_mesh=mesh, placements=placements, run_check=False
    )

@torch.compiler.disable(recursive=True)
def _local(param: torch.Tensor) -> torch.Tensor:
    """Return the local shard of a parameter, whether DTensor or plain."""
    return param.to_local() if isinstance(param, DTensor) else param

@dataclass
class CausalLMOutput:
    loss: torch.Tensor
    logits: torch.Tensor

def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    # Match HF ForCausalLMLoss: upcast to fp32 before CE to avoid bf16 precision issues.
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )

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
    # q, k: (B, H, S, D). cos, sin: (S, R) or (B, S, R) with R <= D (partial rotary).
    if isinstance(q, DTensor) and not isinstance(cos, DTensor):
        replicate_placements = tuple(Replicate() for _ in q.placements)
        cos = DTensor.from_local(cos, q.device_mesh, replicate_placements, run_check=False)
        sin = DTensor.from_local(sin, q.device_mesh, replicate_placements, run_check=False)

    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)  # (B, 1, S, R)
    sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]

    q_emb = (q_rot * cos) + (rotate_half(q_rot) * sin)
    k_emb = (k_rot * cos) + (rotate_half(k_rot) * sin)

    q_out = torch.cat((q_emb, q_pass), dim=-1) if q_pass.shape[-1] > 0 else q_emb
    k_out = torch.cat((k_emb, k_pass), dim=-1) if k_pass.shape[-1] > 0 else k_emb
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
    pid = position_ids.to(torch.float32)  # (3, B, S)
    freqs = pid.unsqueeze(-1) * inv_freq[None, None, None, :]  # (3, B, S, D/2)

    freqs_t = freqs[0].clone()
    for dim, offset in ((1, 1), (2, 2)):
        length = mrope_section[dim] * 3
        idx = slice(offset, length, 3)
        freqs_t[..., idx] = freqs[dim, ..., idx]

    emb = torch.cat((freqs_t, freqs_t), dim=-1)  # (B, S, D)
    return emb.cos(), emb.sin()

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

def load_safetensors_into(
    model: nn.Module,
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
