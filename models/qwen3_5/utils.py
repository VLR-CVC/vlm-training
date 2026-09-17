# DEPRECATED (Qwen3.5 model definition): the old model definitions and DTensor parallelism are no longer used by train/train_qwen.py. Training runs on models/qwen3_5_tt and models/qwen3_vl_tt with train/parallel/ (TITAN_MIGRATION_v2.md). Kept for reference until they are deleted.
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

def _local(param: torch.Tensor) -> torch.Tensor:
    """Return the local shard of a parameter, whether DTensor or plain."""
    return param.to_local() if isinstance(param, DTensor) else param

@dataclass
class CausalLMOutput:
    loss: torch.Tensor
    logits: torch.Tensor

# Cap on the fp32 working set inside the cross-entropy, in bytes. 0 runs
# `F.cross_entropy` over the whole packed row, which at seq_len 8192 means an
# 8.14 GB fp32 copy of the logits and about as much again in its backward.
_CE_CHUNK_BYTES = 0


def set_loss_chunk_mb(mb: int) -> None:
    """Cap the fp32 working set inside the loss, in MiB. 0 disables chunking."""
    global _CE_CHUNK_BYTES
    _CE_CHUNK_BYTES = max(0, int(mb)) * 1024 * 1024


def _ce_chunk(vocab_size: int) -> int:
    """Rows per chunk for the given vocabulary, from the byte budget."""
    return max(256, _CE_CHUNK_BYTES // max(1, vocab_size * 4))


class _ChunkedCrossEntropy(torch.autograd.Function):
    """Mean token cross-entropy that never holds a full fp32 (N, V) tensor.

    Both directions upcast `chunk` rows at a time. The backward recomputes the
    softmax from the saved bf16 logits rather than saving a fp32 log-softmax,
    which is what makes the peak proportional to the chunk instead of to the
    packed row.

    Ported from `models/qwen3_vl/model.py`, with the host sync removed: the
    original guarded an all-ignored batch with `if (labels != ignore).sum() == 0`,
    which reads a device value. Dividing by a clamped count gives the same answer
    -- 0.0 when nothing is supervised -- without stalling the stream.
    """

    @staticmethod
    def forward(ctx, logits, labels, ignore_index, chunk):
        n_valid = (labels != ignore_index).sum()
        total = torch.zeros((), dtype=torch.float32, device=logits.device)
        for lo in range(0, logits.shape[0], chunk):
            hi = min(lo + chunk, logits.shape[0])
            lg = logits[lo:hi].float()
            lb = labels[lo:hi]
            keep = lb != ignore_index
            # gather needs a valid index even where the row is ignored
            tgt = lg.gather(1, lb.clamp_min(0).unsqueeze(1)).squeeze(1)
            row = torch.logsumexp(lg, dim=-1) - tgt
            total = total + torch.where(keep, row, torch.zeros_like(row)).sum()
        ctx.save_for_backward(logits, labels, n_valid)
        ctx.chunk = chunk
        ctx.ignore_index = ignore_index
        return total / n_valid.clamp_min(1)

    @staticmethod
    def backward(ctx, grad_out):
        logits, labels, n_valid = ctx.saved_tensors
        scale = grad_out / n_valid.clamp_min(1)
        grad = torch.empty_like(logits)
        for lo in range(0, logits.shape[0], ctx.chunk):
            hi = min(lo + ctx.chunk, logits.shape[0])
            lb = labels[lo:hi]
            keep = (lb != ctx.ignore_index).unsqueeze(1)
            p = torch.softmax(logits[lo:hi].float(), dim=-1)
            p.scatter_add_(
                1, lb.clamp_min(0).unsqueeze(1), torch.full_like(p[:, :1], -1.0)
            )
            grad[lo:hi] = torch.where(keep, p * scale, torch.zeros_like(p)).to(
                logits.dtype
            )
        return grad, None, None, None


def causal_lm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    # `logits` is (1, T, V), so dropping the last row leaves a contiguous view
    # and `reshape` stays a view. The previous `.contiguous().float()` here made
    # a vocab-sized fp32 copy before the loss even started: 8.14 GB at 8192.
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    flat_logits = shift_logits.reshape(-1, shift_logits.size(-1))
    flat_labels = shift_labels.reshape(-1)

    if _CE_CHUNK_BYTES > 0:
        return _ChunkedCrossEntropy.apply(
            flat_logits, flat_labels, ignore_index, _ce_chunk(flat_logits.shape[-1])
        )

    # Unchunked path: upcast to fp32 before CE, matching HF's ForCausalLMLoss.
    flat_logits = flat_logits.float()

    # `reduction="sum"` divided by a clamped count is the same mean, computed
    # without the host sync that `(... != ignore_index).sum() == 0` needed --
    # and that comparison sat right after the fp32 logits materialization, the
    # worst possible place to drain the stream. It still yields exactly 0.0 when
    # every label is ignored, which is what the branch existed for.
    loss_sum = F.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    n_target = (flat_labels != ignore_index).sum()
    return loss_sum / n_target.clamp(min=1)

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

@torch.compiler.disable
def _wrap_cos_sin_as_dtensor(q: DTensor, cos: torch.Tensor, sin: torch.Tensor):
    replicate_placements = tuple(Replicate() for _ in q.placements)
    cos = DTensor.from_local(cos, q.device_mesh, replicate_placements, run_check=False)
    sin = DTensor.from_local(sin, q.device_mesh, replicate_placements, run_check=False)
    return cos, sin


def apply_rope(
    q: torch.Tensor,
    k: torch.Tensor | None,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
    """Partial-rotary RoPE. `rotary_dim` is taken from `cos.shape[-1]`.

    q, k: (B, H, S, D). cos, sin: (S, R) or (B, S, R) with R <= D.

    `k` may be None -- the QSA indexer rotates its queries and its pooled block
    keys separately, at different positions -- in which case a single tensor
    comes back instead of a pair.
    """
    if isinstance(q, DTensor) and not isinstance(cos, DTensor):
        cos, sin = _wrap_cos_sin_as_dtensor(q, cos, sin)

    if cos.dim() == 2:
        cos = cos.unsqueeze(0)
        sin = sin.unsqueeze(0)
    cos = cos.unsqueeze(1)  # (B, 1, S, R)
    sin = sin.unsqueeze(1)

    rotary_dim = cos.shape[-1]

    def _rot(x):
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        emb = (x_rot * cos) + (rotate_half(x_rot) * sin)
        out = torch.cat((emb, x_pass), dim=-1) if x_pass.shape[-1] > 0 else emb
        return out.to(x.dtype)

    if k is None:
        return _rot(q)
    return _rot(q), _rot(k)

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

    # Start from T-axis frequencies and overwrite H and W bands in-place.
    # Each axis occupies every 3rd element at its respective offset:
    #   T → [0, 3, 6, ...]  (already in freqs[0], no-op)
    #   H → [1, 4, 7, ...]  up to mrope_section[1]*3
    #   W → [2, 5, 8, ...]  up to mrope_section[2]*3
    freqs_t = freqs[0].clone()
    h_end = mrope_section[1] * 3
    w_end = mrope_section[2] * 3
    freqs_t[..., 1:h_end:3] = freqs[1, ..., 1:h_end:3]
    freqs_t[..., 2:w_end:3] = freqs[2, ..., 2:w_end:3]

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
