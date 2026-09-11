"""The vectorised index building in the vision tower must be value-identical.

`rot_pos_emb` and `fast_pos_embed_interpolate` were rewritten to stop issuing
per-image CUDA `arange`s and to stop round-tripping every (h*w) index through
Python scalars. Both are pure index arithmetic, so "faster" is only acceptable
if it is also exactly equal -- these tests pin that against the loops they
replaced.
"""

import torch
import pytest

from models.qwen3_vl.model import Qwen3VLVisionConfig, Qwen3VLVisionModel


def _tiny_cfg():
    return Qwen3VLVisionConfig(
        hidden_size=64,
        depth=2,
        num_heads=4,
        in_channels=3,
        patch_size=16,
        temporal_patch_size=2,
        spatial_merge_size=2,
        intermediate_size=128,
        out_hidden_size=64,
        hidden_act="gelu_pytorch_tanh",
        num_position_embeddings=64,   # 8x8 grid
        deepstack_visual_indexes=[0],
    )


def _ref_rot_pos_emb(model, grid_thw):
    merge = model.spatial_merge_size
    grid_list = grid_thw.tolist()
    max_hw = max(max(h, w) for _, h, w in grid_list)
    freq_table = model.rotary_pos_emb(max_hw)
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
    emb = freq_table[pos_ids]
    return emb.flatten(1)


def _ref_fast_pos_embed_interpolate(model, grid_thw):
    grid_list = grid_thw.tolist()
    grid_ts = [r[0] for r in grid_list]
    grid_hs = [r[1] for r in grid_list]
    grid_ws = [r[2] for r in grid_list]
    device = model.pos_embed.weight.device

    idx_list: list[list[int]] = [[], [], [], []]
    weight_list: list[list[float]] = [[], [], [], []]

    for _t, h, w in grid_list:
        h_idxs = torch.linspace(0, model.num_grid_per_side - 1, h)
        w_idxs = torch.linspace(0, model.num_grid_per_side - 1, w)
        h_floor = h_idxs.int()
        w_floor = w_idxs.int()
        h_ceil = (h_idxs.int() + 1).clip(max=model.num_grid_per_side - 1)
        w_ceil = (w_idxs.int() + 1).clip(max=model.num_grid_per_side - 1)
        dh = h_idxs - h_floor
        dw = w_idxs - w_floor
        base_h = h_floor * model.num_grid_per_side
        base_h_ceil = h_ceil * model.num_grid_per_side
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
    wt = torch.tensor(weight_list, dtype=model.pos_embed.weight.dtype, device=device)
    pe = model.pos_embed(idx_t) * wt[:, :, None]
    patch_pe = pe[0] + pe[1] + pe[2] + pe[3]
    chunks = patch_pe.split([h * w for h, w in zip(grid_hs, grid_ws)])

    merge = model.spatial_merge_size
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


GRIDS = [
    [[1, 4, 4]],
    [[1, 4, 6], [1, 8, 2], [1, 2, 2]],
    [[2, 4, 4], [1, 6, 8]],          # temporal repeat
    [[1, 2, 2]] * 7,                  # many small images, the packed-row case
]


@pytest.mark.parametrize("grid", GRIDS)
def test_rot_pos_emb_identical(grid):
    torch.manual_seed(0)
    model = Qwen3VLVisionModel(_tiny_cfg())
    grid_thw = torch.tensor(grid, dtype=torch.long)
    assert torch.equal(model.rot_pos_emb(grid_thw), _ref_rot_pos_emb(model, grid_thw))


@pytest.mark.parametrize("grid", GRIDS)
def test_fast_pos_embed_interpolate_identical(grid):
    torch.manual_seed(0)
    model = Qwen3VLVisionModel(_tiny_cfg())
    grid_thw = torch.tensor(grid, dtype=torch.long)
    assert torch.equal(
        model.fast_pos_embed_interpolate(grid_thw),
        _ref_fast_pos_embed_interpolate(model, grid_thw),
    )


@pytest.mark.parametrize("grid", GRIDS)
def test_shapes_line_up_with_token_count(grid):
    torch.manual_seed(0)
    model = Qwen3VLVisionModel(_tiny_cfg())
    grid_thw = torch.tensor(grid, dtype=torch.long)
    total = sum(t * h * w for t, h, w in grid)
    assert model.rot_pos_emb(grid_thw).shape[0] == total
    assert model.fast_pos_embed_interpolate(grid_thw).shape[0] == total
