"""Row-sharded Engram lookup (`models/common/engram_sharded.py`) against the dense
lookup it replaces: same rows forward, and each owner's shard gradient equals
its slice of the dense gradient summed over every rank (FSDP's SUM
reduce-scatter). CPU, gloo, several processes."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from models.common.engram_sharded import RowShardedLookup, row_chunk, shard_table_rows

DIM = 8
HEADS = 4
TOKENS = 64


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ids(rank: int, num_rows: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(100 + rank)
    # few distinct rows, so ids repeat inside a rank and across ranks
    return torch.randint(0, num_rows, (TOKENS, HEADS), generator=g)


def _upstream(rank: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(200 + rank)
    return torch.randn(TOKENS, HEADS, DIM, generator=g)


def _worker(rank, world, port, num_rows, accumulate, errors):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("world",))
        full = torch.randn(num_rows, DIM, generator=torch.Generator().manual_seed(0))
        weight = shard_table_rows(torch.nn.Parameter(full.clone()), mesh)
        lookup = RowShardedLookup(mesh, num_rows, compute_dtype=torch.float32)

        steps = 2 if accumulate else 1
        for step in range(steps):
            ids = _ids(rank + 10 * step, num_rows)
            out = lookup(weight, ids)
            torch.testing.assert_close(out, full[ids], rtol=0, atol=0)
            (out * _upstream(rank + 10 * step)).sum().backward()

        # dense reference: every rank's contribution, summed
        ref = torch.zeros_like(full)
        for step in range(steps):
            for r in range(world):
                ids = _ids(r + 10 * step, num_rows)
                ref.index_add_(0, ids.reshape(-1), _upstream(r + 10 * step).reshape(-1, DIM))
        chunk = row_chunk(num_rows, world)
        lo, hi = min(rank * chunk, num_rows), min((rank + 1) * chunk, num_rows)
        grad = weight.grad.to_local()
        assert grad.shape == (hi - lo, DIM)
        torch.testing.assert_close(grad, ref[lo:hi], rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(weight.grad.full_tensor(), ref, rtol=1e-5, atol=1e-5)
    except Exception as e:  # noqa: BLE001 -- reported to the parent
        errors.put(f"rank {rank}: {type(e).__name__}: {e}")
        raise
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    "world,num_rows,accumulate",
    [(1, 50, False), (2, 50, False), (3, 50, True), (4, 37, False)],
)
def test_row_sharded_lookup_matches_dense(world, num_rows, accumulate):
    ctx = mp.get_context("spawn")
    errors = ctx.Queue()
    mp.start_processes(
        _worker, args=(world, _free_port(), num_rows, accumulate, errors),
        nprocs=world, join=True, start_method="spawn",
    )
    assert errors.empty(), errors.get()


def test_row_chunk():
    assert row_chunk(128, 4) == 32
    assert row_chunk(37, 4) == 10
    assert row_chunk(5, 8) == 1
