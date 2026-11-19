"""Host-memory Engram tables (`train/engram_host.py`) against a dense reference
of lazy AdamW: same rows in every forward, same clipped gradient norm, same
table and moments after several steps with gradient accumulation, lookahead
prefetch, misses and eviction. CPU, gloo, several processes."""

import os
import socket

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

DIM = 8
HEADS = 4
TOKENS = 48
ROWS = (97, 40)  # two tables, uneven last shard
ACCUM = (2, 1, 3, 1, 2)
LR, BETAS, EPS = 0.05, (0.9, 0.99), 1e-8
MAX_NORM = 0.5


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _ids(rank: int, j: int, k: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(1000 * rank + 10 * j + k)
    return torch.randint(0, ROWS[k], (TOKENS, HEADS), generator=g)


def _up(rank: int, j: int, k: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(7 + 1000 * rank + 10 * j + k)
    return torch.randn(TOKENS, HEADS, DIM, generator=g)


class _Table(torch.nn.Module):
    def __init__(self, weight, num_rows):
        super().__init__()
        self.weight = weight
        self.num_embeddings = num_rows
        self.embedding_dim = DIM
        self.row_lookup = None


class _Module:
    def __init__(self, table):
        self.table = table


class _Model:
    def __init__(self, tables):
        self.engram = {str(i): _Module(t) for i, t in enumerate(tables)}


def _reference(world, steps_done, clip):
    """Dense lazy AdamW over every rank's micro-batches; returns tables, m, v
    and the per-step pre-clip norms."""
    tables = [torch.randn(n, DIM, generator=torch.Generator().manual_seed(50 + k))
              for k, n in enumerate(ROWS)]
    m = [torch.zeros_like(t) for t in tables]
    v = [torch.zeros_like(t) for t in tables]
    norms, j = [], 0
    for step in range(steps_done):
        grads = [torch.zeros_like(t) for t in tables]
        touched = [set() for _ in tables]
        for _ in range(ACCUM[step % len(ACCUM)]):
            for r in range(world):
                for k in range(len(tables)):
                    ids = _ids(r, j, k)
                    grads[k].index_add_(0, ids.reshape(-1), _up(r, j, k).reshape(-1, DIM))
                    touched[k] |= set(ids.reshape(-1).tolist())
            j += 1
        sq = sum(float(g[sorted(t)].pow(2).sum()) for g, t in zip(grads, touched))
        norm = sq ** 0.5
        norms.append(norm)
        scale = min(1.0, MAX_NORM / (norm + 1e-6)) if clip else 1.0
        t = step + 1
        for k in range(len(tables)):
            rows = torch.tensor(sorted(touched[k]))
            g = grads[k][rows] * scale
            m[k][rows] = m[k][rows] * BETAS[0] + (1 - BETAS[0]) * g
            v[k][rows] = v[k][rows] * BETAS[1] + (1 - BETAS[1]) * g * g
            mh = m[k][rows] / (1 - BETAS[0] ** t)
            vh = v[k][rows] / (1 - BETAS[1] ** t)
            tables[k][rows] -= LR * mh / (vh.sqrt() + EPS)
    return tables, m, v, norms


def _worker(rank, world, port, lookahead, clip, errors):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh

        from models.common.engram_sharded import row_chunk
        from train.engram_host import EngramHostOffload, host_table_mesh, shard_table_rows_host

        row_mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("world",))
        host_mesh = host_table_mesh(world)
        init, _, _, _ = _reference(world, 0, clip)
        tables = []
        for k, n in enumerate(ROWS):
            meta = torch.nn.Parameter(torch.empty(n, DIM, device="meta"))
            p = shard_table_rows_host(meta, host_mesh, seed=k)
            chunk = row_chunk(n, world)
            p.detach().to_local().copy_(init[k][rank * chunk:(rank + 1) * chunk])
            tables.append(_Table(p, n))
        model = _Model(tables)
        off = EngramHostOffload(
            model, row_mesh=row_mesh, ids_fn=lambda b: b["ids"], compute_dtype=torch.float32,
            device=torch.device("cpu"), lookahead=lookahead,
        )
        opt = torch.optim.AdamW([t.weight for t in tables], lr=LR, betas=BETAS, eps=EPS,
                                weight_decay=0.0)
        off.register_optimizer(opt)

        n_batches = sum(ACCUM)
        data = [{"ids": [_ids(rank, j, k) for k in range(len(ROWS))], "j": j}
                for j in range(n_batches)]
        it = off.wrap(iter(data))
        for step, accum in enumerate(ACCUM):
            ref_tables, _, _, norms = _reference(world, step, clip)
            batches = [next(it) for _ in range(accum)]
            for b in batches:
                loss = 0
                for k, table in enumerate(tables):
                    out = table.row_lookup(table.weight, b["ids"][k])
                    torch.testing.assert_close(out, ref_tables[k][b["ids"][k]], rtol=1e-5, atol=1e-6)
                    loss = loss + (out * _up(rank, b["j"], k)).sum()
                loss.backward()
            if step == 2:
                # an eval-style lookup of rows nobody prefetched: a miss, still exact
                with torch.no_grad():
                    ids = torch.arange(ROWS[0]).reshape(-1, 1).repeat(1, HEADS)
                    out = tables[0].row_lookup(tables[0].weight, ids)
                    torch.testing.assert_close(out, ref_tables[0][ids], rtol=1e-5, atol=1e-6)
            off.finish_backward()
            norm = off.grad_norm()
            _, _, _, norms = _reference(world, step + 1, clip)
            torch.testing.assert_close(float(norm), norms[-1], rtol=1e-5, atol=1e-6)
            if clip:
                torch.nn.utils.clip_grad_norm_(off.grad_params(), MAX_NORM)
            opt.step()
            opt.zero_grad()
        off.flush()
        ref_tables, ref_m, ref_v, _ = _reference(world, len(ACCUM), clip)
        for k, table in enumerate(tables):
            chunk = row_chunk(ROWS[k], world)
            sl = slice(rank * chunk, (rank + 1) * chunk)
            state = opt.state[table.weight]
            torch.testing.assert_close(table.weight.to_local(), ref_tables[k][sl], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(state["exp_avg"].to_local(), ref_m[k][sl], rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(state["exp_avg_sq"].to_local(), ref_v[k][sl], rtol=1e-5, atol=1e-6)
            assert float(state["step"]) == len(ACCUM)
        if lookahead >= max(ACCUM):
            # every training row came from a prefetch; the only misses are the eval lookup's
            assert off.stats["miss_fetches"] <= len(tables), dict(off.stats)
        off.close()
    except Exception as e:  # pragma: no cover - reported by the parent
        import traceback

        errors.put(f"rank {rank}: {traceback.format_exc()}")
        raise e
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("world", [1, 2, 3])
@pytest.mark.parametrize("lookahead,clip", [(0, False), (3, True), (6, False)])
def test_host_table_matches_dense_lazy_adamw(world, lookahead, clip):
    ctx = mp.get_context("spawn")
    errors = ctx.SimpleQueue()
    mp.start_processes(
        _worker, args=(world, _free_port(), lookahead, clip, errors),
        nprocs=world, join=True, start_method="spawn",
    )
    assert errors.empty()


def test_chunked_row_scatter_is_exact_under_deterministic_mode():
    from concurrent.futures import ThreadPoolExecutor
    from types import SimpleNamespace

    from train.engram_host import EngramHostOffload, host_zeros

    fake = SimpleNamespace(scatter_threads=4, scatter_pool=ThreadPoolExecutor(4))
    g = torch.Generator().manual_seed(0)
    dst = host_zeros(500_000, DIM)
    dst.copy_(torch.randn(500_000, DIM, generator=g))
    ids = torch.unique(torch.randint(0, 500_000, (120_000,), generator=g))
    src = torch.randn(ids.numel(), DIM, generator=g)
    ref = dst.clone().index_copy_(0, ids, src)
    prev = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        EngramHostOffload._scatter_rows(fake, dst, ids, src)
    finally:
        torch.use_deterministic_algorithms(prev)
    assert torch.equal(dst, ref)
