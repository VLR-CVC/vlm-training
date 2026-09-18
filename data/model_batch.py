from __future__ import annotations

import torch

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

    HF's `get_rope_index`, per document. Text runs are `arange`;
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
