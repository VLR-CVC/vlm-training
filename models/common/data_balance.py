# Adapted from VeOmni 894f7444: veomni/utils/data_balance/ (Qwen3VLEncoderDataBalance)
# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Even out vision-encoder work over a group of DP ranks.

The group is one node (`dp_local` in train/parallel/parallel_dims.py), so the
exchange stays on NVLink and its cost does not grow with the cluster.

    pixel_values, grid_thw, restore = balance(pixel_values, grid_thw, merge_unit, group)
    out = restore(encoder(pixel_values, grid_thw))

Every rank gathers all grids and computes the same plan, sends its patches to
their assigned ranks, and `restore` sends the encoder output back in the
original order. The return trip is differentiable.
"""

import heapq

import torch
import torch.distributed as dist
from torch.distributed._functional_collectives import all_to_all_single_autograd


def plan(lengths: list[list[int]]) -> list[list[int]]:
    """`dest[r][i]`: the rank that encodes image `i` of rank `r`.

    Longest first onto the least-loaded rank, load = sum of patches**2. Ties go
    to the lower rank, so every rank computes the same plan.
    """
    heap = [(0, r) for r in range(len(lengths))]
    dest = [[0] * len(ls) for ls in lengths]
    items = [(n, r, i) for r, ls in enumerate(lengths) for i, n in enumerate(ls)]
    for n, r, i in sorted(items, key=lambda item: -item[0]):  # stable: ties in (r, i) order
        load, d = heapq.heappop(heap)
        dest[r][i] = d
        heapq.heappush(heap, (load + n * n, d))
    return dest


def balance(pixel_values: torch.Tensor, grid_thw: torch.Tensor, merge_unit: int, group: dist.ProcessGroup):
    """Returns this rank's balanced `(pixel_values, grid_thw)` and `restore`.

    `restore(out)` takes the encoder output for the balanced images, `k` stacked
    blocks (merged tokens, then deepstack) of one row per merged token, and returns
    the same layout for the original images.
    """
    rank, world = group.rank(), group.size()
    grid_thw = grid_thw.long()
    device = grid_thw.device

    counts = torch.empty(world, dtype=torch.long, device=device)
    dist.all_gather_into_tensor(counts, torch.tensor([grid_thw.shape[0]], device=device), group=group)
    counts = counts.tolist()
    padded = grid_thw.new_zeros(max(counts), 3)
    padded[: counts[rank]] = grid_thw
    gathered = grid_thw.new_empty(world * max(counts), 3)
    dist.all_gather_into_tensor(gathered, padded, group=group)
    grids = [g[:c] for g, c in zip(gathered.view(world, -1, 3).tolist(), counts)]
    lengths = [[t * h * w for t, h, w in g] for g in grids]
    dest = plan(lengths)

    # send in destination order, keeping the original order within a destination;
    # all_to_all_single then delivers source-major, which is the order of recv_grids
    order = sorted(range(counts[rank]), key=dest[rank].__getitem__)
    send = [0] * world
    for i, d in enumerate(dest[rank]):
        send[d] += lengths[rank][i]
    recv, recv_grids = [0] * world, []
    for r in range(world):
        for i, d in enumerate(dest[r]):
            if d == rank:
                recv[r] += lengths[r][i]
                recv_grids.append(grids[r][i])
    # holds while every rank brings at least one image (the dummy, if nothing else)
    assert recv_grids, f"rank {rank} got no images to encode"

    chunks = pixel_values.split(lengths[rank])
    send_px = torch.cat([chunks[i] for i in order])
    recv_px = send_px.new_empty(sum(recv), send_px.shape[1])
    dist.all_to_all_single(recv_px, send_px, recv, send, group=group)

    def restore(out: torch.Tensor) -> torch.Tensor:
        tokens, dim = sum(recv) // merge_unit, out.shape[1]
        k = out.shape[0] // tokens
        # one row per token holding all k blocks, so one exchange carries them
        rows = out.unflatten(0, (k, tokens)).transpose(0, 1).reshape(tokens, k * dim)
        back = all_to_all_single_autograd(
            rows,
            [n // merge_unit for n in send],
            [n // merge_unit for n in recv],
            group,
        )
        restored = [None] * len(order)
        for piece, i in zip(back.split([lengths[rank][i] // merge_unit for i in order]), order):
            restored[i] = piece
        return torch.cat(restored).unflatten(1, (k, dim)).transpose(0, 1).reshape(-1, dim)

    return recv_px, torch.tensor(recv_grids, device=device), restore
