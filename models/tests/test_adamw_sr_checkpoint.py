"""AdamWSR's moments must reach a checkpoint as DTensors.

Kept as plain local shards they are saved with `chunks=1` and no offsets: all
ranks write one key, one rank's moments survive, and an unevenly sharded
parameter fails the load outright --

    Size mismatch between saved torch.Size([1, 1152]) and current: torch.Size([5, 1152])

which is what every checkpoint of the 35B-A3B run contains (2508 optimizer
tensors, all chunks=1). The parameter here is deliberately uneven: global dim0=5
over 2 ranks gives shards of 3 and 2, so a regression reproduces the failure
rather than hiding behind an even split.
"""

import os
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from train.adamw_sr import AdamWSR

GLOBAL_SHAPE = (5, 4)  # 5 rows over 2 ranks -> 3 + 2, uneven on purpose


def _worker(rank: int, world: int, ckpt: str, port: str) -> None:
    import torch.distributed as dist
    import torch.distributed.checkpoint as dcp
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard, distribute_tensor

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port,
                      RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))

    full = torch.arange(20, dtype=torch.bfloat16).reshape(GLOBAL_SHAPE) / 20
    p = torch.nn.Parameter(distribute_tensor(full, mesh, [Shard(0)]))
    p.grad = distribute_tensor(torch.ones(GLOBAL_SHAPE, dtype=torch.bfloat16), mesh, [Shard(0)])

    opt = AdamWSR([p], lr=1e-3, betas=(0.9, 0.95), weight_decay=0.1)
    opt.step()

    st = opt.state[p]
    assert isinstance(st["exp_avg"], torch.distributed.tensor.DTensor), "exp_avg is not a DTensor"
    assert isinstance(st["exp_avg_sq"], torch.distributed.tensor.DTensor), "exp_avg_sq is not a DTensor"
    assert tuple(st["exp_avg"].shape) == GLOBAL_SHAPE, st["exp_avg"].shape
    assert tuple(st["exp_avg_sq"].shape) == GLOBAL_SHAPE, st["exp_avg_sq"].shape
    # the moments actually moved; a zero tensor would pass the shape checks alone
    assert st["exp_avg"].to_local().abs().sum() > 0

    dcp.save({"exp_avg": st["exp_avg"], "exp_avg_sq": st["exp_avg_sq"]}, checkpoint_id=ckpt)
    dist.barrier()
    dist.destroy_process_group()


def test_moments_are_saved_with_sharding_metadata(tmp_path):
    world, port = 2, "29763"
    torch.multiprocessing.spawn(_worker, args=(world, str(tmp_path), port), nprocs=world, join=True)

    from torch.distributed.checkpoint import FileSystemReader

    sd = FileSystemReader(str(tmp_path)).read_metadata().state_dict_metadata
    for key in ("exp_avg", "exp_avg_sq"):
        md = sd[key]
        assert tuple(md.size) == GLOBAL_SHAPE, f"{key}: saved global size {tuple(md.size)}"
        # the bug: one chunk covering only the writing rank's shard
        assert len(md.chunks) == world, f"{key}: {len(md.chunks)} chunks, expected {world}"
        rows = sorted((tuple(c.offsets), tuple(c.sizes)) for c in md.chunks)
        assert rows == [((0, 0), (3, 4)), ((3, 0), (2, 4))], rows


def test_step_still_matches_adamw_on_a_plain_tensor():
    """The hot path now reads moments through _local; guard the arithmetic."""
    torch.manual_seed(0)
    init = torch.randn(8, 3)
    grad = torch.randn(8, 3)

    ours = torch.nn.Parameter(init.clone())
    ours.grad = grad.clone()
    AdamWSR([ours], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
            stochastic_round=False).step()

    ref = torch.nn.Parameter(init.clone())
    ref.grad = grad.clone()
    torch.optim.AdamW([ref], lr=1e-2, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1).step()

    torch.testing.assert_close(ours.detach(), ref.detach(), rtol=1e-4, atol=1e-6)
