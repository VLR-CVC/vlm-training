"""Energon packed batch -> input of the torchtitan-port model (`models/qwen3_5_tt`).

The task encoders emit one packed row: `input_ids` (T,), `labels` (T,) equal to
`input_ids` on supervised positions and -100 elsewhere (NOT shifted), `cu_seqlens`
with any right padding as its own trailing segment, and `pixel_values` /
`image_grid_thw` for the images in the row.

The torchtitan model wants, per token:
  input            (T,)    token ids
  labels           (T,)    the id to predict at t, i.e. shifted by one; -100 on the
                           last token of every document, so nothing is predicted
                           across a packing boundary
  positions        (T,)    arange restarting at 0 per document -- the model derives
                           the varlen document offsets from these
  mrope_positions  (T, 3)  temporal/height/width positions (HF's get_rope_index,
                           per document)
plus `pixel_values`, `grid_thw`, and `num_valid_tokens` (host int) for the loss
normaliser.

Runs on the host, on CPU tensors, before the H2D copy: `mrope_positions` is
data-dependent Python and must stay out of the compiled region.
"""

from __future__ import annotations

import torch


def document_positions(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    starts = cu_seqlens[:-1].to(torch.int64)
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    return torch.arange(total) - torch.repeat_interleave(starts, lens, output_size=total)


def shift_labels(labels: torch.Tensor, cu_seqlens: torch.Tensor) -> torch.Tensor:
    shifted = torch.full_like(labels, -100)
    shifted[:-1] = labels[1:]
    ends = cu_seqlens[1:].to(torch.int64) - 1
    shifted[ends[ends >= 0]] = -100
    return shifted


def mrope_positions(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor,
    image_grid_thw: torch.Tensor | None,
    *,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    """(T, 3) MRoPE positions over a packed row; HF's algorithm per document.

    Moved from `models/qwen3_5/model.py:get_rope_index`. Text runs are `arange`;
    an image run of `t x h x w` merged tokens takes (t, row, col) offsets from the
    current position, which then advances by `max(h, w)`.
    """
    total = input_ids.shape[0]
    types = torch.zeros(total, dtype=torch.int64)
    types[input_ids == image_token_id] = 1
    types[input_ids == video_token_id] = 2
    if bool((types == 2).any()):
        raise NotImplementedError("video positions are not ported")
    types_all = types.tolist()
    grids = iter(image_grid_thw.tolist()) if image_grid_thw is not None else iter(())
    out = torch.zeros(total, 3, dtype=torch.int64)
    bounds = cu_seqlens.tolist()
    for start, end in zip(bounds[:-1], bounds[1:]):
        current = 0
        j = start
        while j < end:
            k = j
            while k < end and types_all[k] == types_all[j]:
                k += 1
            if types_all[j] == 0:
                out[j:k] = torch.arange(current, current + (k - j)).unsqueeze(-1)
                current += k - j
            else:
                t, h, w = next(grids)
                llm_h, llm_w = h // spatial_merge_size, w // spatial_merge_size
                n = t * llm_h * llm_w
                if n != k - j:
                    raise ValueError(f"image run of {k - j} tokens, grid says {n}")
                out[j:k, 0] = current
                out[j:k, 1] = torch.arange(current, current + llm_h).repeat_interleave(llm_w * t)
                out[j:k, 2] = torch.arange(current, current + llm_w).repeat(llm_h * t)
                current += max(llm_h, llm_w)
            j = k
    return out


def to_titan_batch(
    batch: dict,
    *,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
) -> dict:
    input_ids = batch["input_ids"].reshape(-1)
    cu_seqlens = batch["cu_seqlens"].reshape(-1)
    total = input_ids.shape[0]
    labels = shift_labels(batch["labels"].reshape(-1), cu_seqlens)
    grid = batch.get("image_grid_thw")
    if grid is not None and grid.ndim == 3:
        grid = grid[0]
    out = {
        "input": input_ids,
        "labels": labels,
        "positions": document_positions(cu_seqlens, total),
        "mrope_positions": mrope_positions(
            input_ids,
            cu_seqlens,
            grid,
            image_token_id=image_token_id,
            video_token_id=video_token_id,
            spatial_merge_size=spatial_merge_size,
        ),
        "num_valid_tokens": int((labels != -100).sum()),
    }
    if batch.get("pixel_values") is not None:
        pixel_values = batch["pixel_values"]  # the loader keeps a (1, P, D) batch dim
        out["pixel_values"] = pixel_values.reshape(-1, pixel_values.shape[-1])
        out["grid_thw"] = grid
    return out


if __name__ == "__main__":
    # Self-check against the old model's in-forward get_rope_index and loss shift.
    cu = torch.tensor([0, 7, 12, 16], dtype=torch.int32)
    image_id, merge = 99, 2
    ids = torch.tensor([1, 2, 99, 99, 99, 99, 3, 4, 5, 6, 7, 8, 0, 0, 0, 0])
    grid = torch.tensor([[1, 4, 4]])
    labels = ids.clone()
    labels[:2] = -100
    b = to_titan_batch(
        {"input_ids": ids, "labels": labels, "cu_seqlens": cu, "image_grid_thw": grid,
         "pixel_values": torch.zeros(16, 4)},
        image_token_id=image_id, video_token_id=98, spatial_merge_size=merge,
    )
    assert b["positions"].tolist() == [0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 0, 1, 2, 3]
    # doc 1: text 0,1 | image 2x2 merged at offset 2 | text resumes at 2+max(2,2)=4
    assert b["mrope_positions"][:7].tolist() == [
        [0, 0, 0], [1, 1, 1], [2, 2, 2], [2, 2, 3], [2, 3, 2], [2, 3, 3], [4, 4, 4]]
    assert b["labels"][6].item() == -100 and b["labels"][11].item() == -100  # doc ends
    # labels[:2] masked, so predicting token 1 (from t=0) is masked; t=1 predicts 99
    assert b["labels"][:6].tolist() == [-100, 99, 99, 99, 99, 3]

    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from types import SimpleNamespace

    from models.qwen3_5.model import Qwen3_5ForCausalLM

    stub = SimpleNamespace(cfg=SimpleNamespace(
        image_token_id=image_id, video_token_id=98, vision=SimpleNamespace(spatial_merge_size=merge)))
    old = Qwen3_5ForCausalLM.get_rope_index(stub, ids.unsqueeze(0), cu, image_grid_thw=grid)
    assert torch.equal(old[:, 0].t(), b["mrope_positions"]), "mrope mismatch vs old model"
    print("titan_batch self-check OK")
