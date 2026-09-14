from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import safe_open
from torch.distributed.tensor import DTensor, Replicate

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


def _dtensor_rewrap(tensor: torch.Tensor, wrap_info) -> torch.Tensor:
    if wrap_info is None:
        return tensor
    mesh, placements = wrap_info
    return DTensor.from_local(
        tensor, device_mesh=mesh, placements=placements, run_check=False
    )

def iter_layers(layers) -> list[nn.Module]:
    """Decoder layers in order, whichever container is holding them.

    Without PP that is the plain `nn.ModuleList`. `apply_pp_qwen4` replaces it
    with an `nn.ModuleDict` keyed by each layer's *original* index, because a
    stage has to keep the parameter names it has in the unsplit model: slicing
    into a fresh `ModuleList` renumbers layer 4 to `layers.0`, and then two
    stages write the same `layers.0.*` keys into one checkpoint.
    """
    if isinstance(layers, nn.ModuleDict):
        return list(layers.values())
    return list(layers)


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
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)

    if (flat_labels != ignore_index).sum() == 0:
        return flat_logits.sum() * 0.0
    return F.cross_entropy(
        flat_logits,
        flat_labels,
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
    """Partial-rotary RoPE. ``rotary_dim`` is taken from ``cos.shape[-1]``.

    ``k`` may be ``None`` (the QSA indexer rotates queries and pooled keys
    separately), in which case a single tensor is returned.
    """
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

_PLE_SHARD_RE = re.compile(r"^(.*\.ngram_embedding)\.shard_(\d+)\.weight$")

def load_safetensors_into(
    model: nn.Module,
    snapshot_dir: Path,
    device: str | torch.device,
    dtype: torch.dtype,
    load_vision: bool,
    load_ple: bool = True,
) -> None:
    """Copy an HF snapshot into ``model`` in place.

    Module names mirror HF exactly, so almost every key maps 1:1. The one
    exception is the PLE n-gram table, which the checkpoint stores as
    ``split_ngram_parts`` separate ``ngram_embedding.shard_{i}.weight``
    tensors that concatenate (in numeric order) into our single embedding.
    """
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

    # PLE tables are written back shard by shard at a running row offset, so we
    # need a deterministic order and the row each shard starts at.
    ple_offsets: dict[str, int] = {}
    ple_parts: dict[str, list[tuple[int, str, str]]] = {}
    for shard_name, names in shards.items():
        if names is None:
            continue
        for k in names:
            m = _PLE_SHARD_RE.match(k)
            if m:
                ple_parts.setdefault(m.group(1), []).append((int(m.group(2)), shard_name, k))
    for target, parts in ple_parts.items():
        parts.sort()

    for shard_name, shard_path in files.items():
        with safe_open(str(shard_path), framework="pt", device=str(device)) as f:
            keys = shards[shard_name] if shards[shard_name] is not None else list(f.keys())
            for k in keys:
                if (not load_vision) and k.startswith("model.visual."):
                    continue
                if _PLE_SHARD_RE.match(k):
                    continue  # handled below
                if k not in state:
                    # tied lm_head is absent from file; `mtp.*` has no module here.
                    continue
                tensor = f.get_tensor(k)
                # PLE ships integer buffers (`layer_multipliers` holds 64-bit
                # splitmix constants, plus the per-head vocab sizes and
                # offsets). Casting those to the model dtype destroys them and
                # the n-gram gather then indexes out of bounds.
                if tensor.is_floating_point():
                    tensor = tensor.to(dtype=dtype)
                if state[k].shape != tensor.shape:
                    raise ValueError(f"Shape mismatch for {k}: {state[k].shape} vs {tensor.shape}")
                state[k].copy_(tensor)
                loaded.add(k)

    # The per-head vocabulary sizes and offsets are derived from the config in
    # `NGramEmbedding.__init__` rather than loaded (they differ per TP rank, and
    # a plain buffer that differs per rank is not safely checkpointable). They
    # are still in the file, so use them to check the derivation.
    for shard_name, names in shards.items():
        for k in names or ():
            if not k.endswith((".ngram_heads_vocab_sizes", ".ngram_heads_offsets")):
                continue
            module_name = k.rsplit(".", 1)[0]
            try:
                owner = model.get_submodule(module_name)
            except AttributeError:
                continue
            with safe_open(str(files[shard_name]), framework="pt", device="cpu") as f:
                reference = f.get_tensor(k).tolist()
            column = 0 if k.endswith("offsets") else 1
            ours = [w[column] for w in owner.checkpoint_windows]
            theirs = [reference[h] for h in owner._local_heads]
            if ours != theirs:
                raise ValueError(
                    f"{k}: derived n-gram head layout disagrees with the checkpoint "
                    f"({ours[:4]}... vs {theirs[:4]}...)"
                )

    if load_ple:
        for target, parts in ple_parts.items():
            dest_key = f"{target}.weight"
            if dest_key not in state:
                continue
            # `target` names the inner `nn.Embedding`; its parent holds the
            # head layout.
            owner = model.get_submodule(target.rsplit(".", 1)[0])
            dest = state[dest_key]
            rows_per_head = owner.rows_per_head
            windows = owner.checkpoint_windows

            # The checkpoint packs every head back to back at its true prime
            # size; the runtime table pads each head to `rows_per_head` so the
            # TP shard is even. Walk the parts once in order and copy whatever
            # of each part falls inside a local head's window.
            dest.zero_()
            row = 0
            for _idx, shard_name, k in parts:
                with safe_open(str(files[shard_name]), framework="pt", device=str(device)) as f:
                    part = f.get_slice(k)
                    n_rows, width = part.get_shape()
                    if width != dest.shape[1]:
                        raise ValueError(
                            f"PLE shard {k} has width {width}, expected {dest.shape[1]}"
                        )
                    for local_idx, (src_start, size) in enumerate(windows):
                        lo = max(row, src_start)
                        hi = min(row + n_rows, src_start + size)
                        if lo >= hi:
                            continue
                        base = local_idx * rows_per_head + (lo - src_start)
                        dest[base : base + (hi - lo)].copy_(
                            part[lo - row : hi - row].to(dtype=dtype)
                        )
                row += n_rows

            needed = max(start + size for start, size in windows)
            if row < needed:
                raise ValueError(
                    f"PLE shards for {dest_key} cover {row} rows, but this rank's heads "
                    f"end at row {needed}"
                )
            loaded.add(dest_key)

    missing = set(state.keys()) - loaded
    if model.cfg.tie_word_embeddings:
        missing.discard("lm_head.weight")
    if not load_vision:
        missing = {m for m in missing if not m.startswith("model.visual.")}
    if not load_ple:
        missing = {m for m in missing if ".ngram_embedding." not in m}
    if missing:
        raise RuntimeError(f"Missing weights after load: {sorted(missing)[:8]} ... ({len(missing)} total)")
