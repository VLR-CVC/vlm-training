"""Host-memory Engram tables with prefetch (`training.engram_table_device = "host"`).

The row-sharded table (`engram_sharding = "rows"`, `models/common/engram_sharded.py`)
keeps each rank's shard -- weights and both AdamW moments, 12 B per parameter --
in GPU memory. Here the shard lives in host RAM. The GPU holds only a **row cache**
of the rows the next micro-batches read, as DeepSeek's Engram does for inference
(arXiv 2601.07372 sec. 2.5 and 6.4): the row ids are a function of the token ids
only, so they are known as soon as a batch leaves the data loader, long before
its forward pass.

Per step::

    data loader --(lookahead: L batches early)--> worker thread, FIFO:
        hash ids on CPU -> all-to-all the unique ids to their owners (gloo)
        -> owner gathers w, m, v rows from host -> H2D on a side stream
    forward     route ids (NCCL, as in "rows") -> owner reads its GPU cache
                (a row missing from it is fetched synchronously: slower, still exact)
    backward    row gradients go back to the owner, into the cache's grad
    clip        the cache gradients join the global norm (`grad_params`)
    step        lazy AdamW on the touched rows, in the cache -> D2H -> the
                worker copies them into the host shard (FIFO, before any later gather)
                -> merge the prefetched rows, evict rows no pending batch needs

Consistency. The cache is always at least as new as host memory, so on a merge
the cached copy of a row wins. Host memory lags only by the write-backs still
queued, and every host read (prefetch or miss) runs on the same FIFO worker,
behind them. A prefetch gathered before step ``k``'s write-back is merged at
the end of step ``k``, before eviction, while step ``k``'s rows are still
cached. So no stale row ever reaches a lookup, and the result is the same
lazy AdamW as `engram_table_optim = "lazy_adamw"` on the GPU.

Requirements. `engram_sharding = "rows"` and `engram_table_optim =
"lazy_adamw"`: dense AdamW changes every row's moments every step, which
would mean streaming the whole table through PCIe.

Checkpoints. The table is a ``Shard(0)`` DTensor over a CPU mesh, with the
same layout as "rows" and "fsdp", and the moments sit in the optimizer state
under the usual keys. A DCP checkpoint therefore moves between all three
modes. The batches the lookahead pulled from the loader but did not hand out
yet are saved next to the dataloader state, so a resume does not skip them.

Cost per step and rank: ~rows read x row width x 12 B each way over PCIe,
plus the CPU gather and scatter. For 16 heads x 16k tokens x 128-wide rows
that is <= 400 MB, against a ~3 s step at 2B (docs/ARCHITECTURE.md,
"Engram training at scale").
"""

from __future__ import annotations

import collections
import contextlib
import mmap
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

from models.common.engram_sharded import _RowShardedLookupFn, _Route, row_chunk
from train.logger import logger

__all__ = [
    "ENGRAM_TABLE_DEVICES",
    "EngramHostOffload",
    "host_table_mesh",
    "shard_table_rows_host",
]

ENGRAM_TABLE_DEVICES = ("gpu", "host")


# ---------------------------------------------------------------------------
# setup (meta model, before FSDP)


def host_zeros(rows: int, dim: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """A zeroed ``(rows, dim)`` host tensor on transparent huge pages, faulted in.

    The shard is read and written at random rows every step. With 4 KB pages
    every row is a TLB miss: scattering 250k rows into 3 x 17 GB took 96 ms
    with malloc and 8 ms with 2 MB pages (T20, cudahpc51). The pages are
    touched here (a parallel ``zero_``): left to the first write-back, faulting
    in ~34 GB of moments per rank took ~13 s spread over the first steps.
    Falls back to malloc where madvise is unavailable (THP "never" makes the
    advice a no-op)."""
    nbytes = rows * dim * torch.empty((), dtype=dtype).element_size()
    if nbytes == 0 or not hasattr(mmap, "MADV_HUGEPAGE"):
        return torch.zeros(rows, dim, dtype=dtype)
    buf = mmap.mmap(-1, nbytes, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    try:
        buf.madvise(mmap.MADV_HUGEPAGE)
    except OSError:
        pass
    # the tensor keeps `buf` alive
    t = torch.frombuffer(buf, dtype=dtype).view(rows, dim)
    t.zero_()
    return t


def host_table_mesh(world: int) -> DeviceMesh:
    """1D CPU mesh over every rank, which the host tables' DTensors live on.

    On its own gloo group: `init_device_mesh("cpu", ...)` over the whole world
    reuses the default group, which is NCCL-only in training, so a collective on
    a host DTensor (`full_tensor`, export) would find no CPU backend."""
    if world != dist.get_world_size():
        raise ValueError(f"the host mesh spans every rank ({dist.get_world_size()}), got {world}")
    group = dist.new_group(backend="gloo")
    return DeviceMesh.from_group(group, "cpu", mesh_dim_names=("engram_host",))


def shard_table_rows_host(weight: torch.nn.Parameter, mesh: DeviceMesh, *, seed: int) -> torch.nn.Parameter:
    """The ``Shard(0)`` DTensor parameter of a (meta) table, with this rank's rows
    allocated in host memory (pageable: only the staging buffers are pinned).
    ``seed`` makes the per-rank init of `EngramTable` reproducible."""
    if isinstance(weight, DTensor):
        raise ValueError("the table is already a DTensor")
    chunk = row_chunk(weight.shape[0], mesh.size())
    start = min(mesh.get_local_rank() * chunk, weight.shape[0])
    stop = min(start + chunk, weight.shape[0])
    local = host_zeros(stop - start, weight.shape[1], dtype=weight.dtype)
    dt = DTensor.from_local(local, mesh, [Shard(0)], run_check=False,
                            shape=weight.shape, stride=weight.stride())
    param = torch.nn.Parameter(dt, requires_grad=weight.requires_grad)
    param.host_init_seed = seed * 1_000_003 + mesh.get_local_rank()
    return param


# ---------------------------------------------------------------------------
# worker-side data


@dataclass
class _Rows:
    """Rows of one table on the GPU (or CPU in tests): sorted local ids, w, m, v."""

    ids: torch.Tensor
    w: torch.Tensor
    m: torch.Tensor
    v: torch.Tensor


@dataclass
class _Plan:
    index: int  # batch index, -1 for a synchronous miss fetch
    rows: list  # one `_Rows` per table
    event: object  # CUDA event of the H2D copy, None on CPU


class _StepGrads:
    """The gradient leaves one table's lookups used during the current step."""

    def __init__(self):
        self.entries = []  # [leaf, ids, [slot tensors]]

    def record(self, leaf, ids, slots):
        if self.entries and self.entries[-1][0] is leaf:
            self.entries[-1][2].append(slots)
        else:
            self.entries.append([leaf, ids, [slots]])

    def clear(self):
        self.entries.clear()


# ---------------------------------------------------------------------------
# one table


class _HostRowStore:
    """Owner side of one row-sharded host table: host shard + GPU row cache."""

    def __init__(self, offload: "EngramHostOffload", table, index: int):
        self.offload = offload
        self.table = table
        self.index = index
        self.param = table.weight
        self.dim = table.embedding_dim
        self.chunk = row_chunk(table.num_embeddings, offload.world)
        local = self.param.detach().to_local()
        # moments: own zeros until the optimizer hook adopts them into its state
        self.host_m = host_zeros(*local.shape)
        self.host_v = host_zeros(*local.shape)
        self.step_grads = _StepGrads()
        self.update_ids = None  # rows this step's update touches (local ids)
        self.update_grad = None  # their summed gradient (fp32)
        self.carrier = None
        self.clear_cache()

    # host tensors -----------------------------------------------------------
    def host_tensors(self):
        return self.param.detach().to_local(), self.host_m, self.host_v

    # cache -------------------------------------------------------------------
    def clear_cache(self):
        dev = self.offload.device
        self.cache = _Rows(
            torch.empty(0, dtype=torch.long, device=dev),
            *(torch.empty(0, self.dim, dtype=torch.float32, device=dev) for _ in range(3)),
        )
        self._mark_leaf()

    def _mark_leaf(self):
        if self.param.requires_grad:
            self.cache.w.requires_grad_(True)

    @torch.no_grad()
    def merge(self, rows: _Rows):
        """Add ``rows`` to the cache; rows already cached keep the cached copy."""
        if rows.ids.numel() == 0:
            return
        old = self.cache
        n_old = old.ids.numel()
        if n_old == 0:
            self.cache = _Rows(rows.ids, rows.w, rows.m, rows.v)
        else:
            ids, inverse = torch.unique(torch.cat([old.ids, rows.ids]), return_inverse=True)
            new_slots, old_slots = inverse[n_old:], inverse[:n_old]
            out = []
            for a, b in ((old.w, rows.w), (old.m, rows.m), (old.v, rows.v)):
                t = a.new_empty(ids.numel(), self.dim)
                t[new_slots] = b
                t[old_slots] = a.detach()  # written last: the cache wins
                out.append(t)
            self.cache = _Rows(ids, *out)
        self._mark_leaf()

    @torch.no_grad()
    def keep_only(self, keep_ids: torch.Tensor):
        c = self.cache
        if c.ids.numel() == 0:
            return
        mask = torch.isin(c.ids, keep_ids)
        self.cache = _Rows(c.ids[mask], c.w.detach()[mask], c.m[mask], c.v[mask])
        self._mark_leaf()

    def slots(self, local_ids: torch.Tensor):
        """Cache slots of ``local_ids`` and whether all are cached (host sync)."""
        ids = self.cache.ids
        if ids.numel() == 0:
            return None, local_ids.numel() == 0
        pos = torch.searchsorted(ids, local_ids)
        hit = ids[pos.clamp_max(ids.numel() - 1)] == local_ids
        return pos, bool(hit.all())

    # lookup ------------------------------------------------------------------
    def lookup(self, ids: torch.Tensor) -> torch.Tensor:
        off = self.offload
        off.merge_ready()
        with torch.no_grad():
            route = _Route(ids, chunk=self.chunk, group=off.group)
        local_ids = route.local_ids
        pos, ok = self.slots(local_ids)
        if not ok:
            off.fetch_missing(self, local_ids)
            pos, ok = self.slots(local_ids)
            assert ok, "Engram host table: a fetched row is still missing from the cache"
        if pos is None:
            pos = local_ids
        leaf = self.cache.w
        if torch.is_grad_enabled() and leaf.requires_grad:
            self.step_grads.record(leaf, self.cache.ids, pos)
        else:
            leaf = leaf.detach()
        route.local_ids = pos
        return _RowShardedLookupFn.apply(
            leaf, ids, route, off.group, off.compute_dtype, torch.float32
        )

    # gradients -----------------------------------------------------------------
    @torch.no_grad()
    def finish_backward(self):
        """Sum this step's row gradients per cached row: ``update_ids`` /
        ``update_grad``, exposed to the clip through ``carrier``."""
        ids_list, grads = [], []
        for leaf, ids, slots in self.step_grads.entries:
            if leaf.grad is None:
                continue
            s = torch.unique(torch.cat(slots))
            ids_list.append(ids[s])
            grads.append(leaf.grad[s].float())
            leaf.grad = None
        self.step_grads.clear()
        dev = self.offload.device
        if ids_list:
            u, inverse = torch.unique(torch.cat(ids_list), return_inverse=True)
            g = torch.zeros(u.numel(), self.dim, dtype=torch.float32, device=dev)
            g.index_add_(0, inverse, torch.cat(grads))
        else:
            u = torch.empty(0, dtype=torch.long, device=dev)
            g = torch.empty(0, self.dim, dtype=torch.float32, device=dev)
        self.update_ids, self.update_grad = u, g
        mesh = self.offload.row_mesh
        p = torch.nn.Parameter(DTensor.from_local(g, mesh, [Shard(0)], run_check=False),
                               requires_grad=True)
        p.grad = DTensor.from_local(g, mesh, [Shard(0)], run_check=False)
        self.carrier = p

    def discard_grads(self):
        for leaf, _, _ in self.step_grads.entries:
            leaf.grad = None
        self.step_grads.clear()
        self.update_ids = self.update_grad = self.carrier = None

    @torch.no_grad()
    def adamw_update(self, state, *, lr, betas, eps, weight_decay):
        """Lazy AdamW (`train/engram_optim.py`) on the rows of ``update_ids``, in
        the cache; returns the updated rows for the write-back."""
        u, g = self.update_ids, self.update_grad
        self.update_ids = self.update_grad = self.carrier = None
        if u is None:
            return None
        state["step"] += 1
        step = float(state["step"])
        if u.numel() == 0:
            return None
        c = self.cache
        pos = torch.searchsorted(c.ids, u)
        beta1, beta2 = betas
        m = c.m.index_select(0, pos).mul_(beta1).add_(g, alpha=1 - beta1)
        v = c.v.index_select(0, pos).mul_(beta2).addcmul_(g, g, value=1 - beta2)
        w = c.w.detach().index_select(0, pos)
        if weight_decay:
            w.mul_(1 - lr * weight_decay)
        bc1 = 1 - beta1 ** step
        bc2 = 1 - beta2 ** step
        w.sub_(lr * (m / bc1) / ((v / bc2).sqrt_().add_(eps)))
        c.m.index_copy_(0, pos, m)
        c.v.index_copy_(0, pos, v)
        c.w.detach().index_copy_(0, pos, w)
        return _Rows(u, w, m, v)


class _HostLookup:
    """`EngramTable.row_lookup` of a host table."""

    def __init__(self, store: _HostRowStore):
        self.store = store

    def __call__(self, weight, ids):
        return self.store.lookup(ids)


# ---------------------------------------------------------------------------
# the lookahead iterator


class _Lookahead:
    """Wraps the raw data-loader iterator: pulls ``depth`` batches ahead and
    submits a prefetch plan for each one as it is pulled.

    Exactly one plan is submitted per ``__next__`` on every rank (an empty one
    once this rank's loader is exhausted), so the worker's collectives line up
    across ranks as long as the training loop pulls in lockstep."""

    def __init__(self, source, offload: "EngramHostOffload", depth: int, replay=()):
        self.source = source
        self.offload = offload
        self.depth = depth
        self.buffer = collections.deque()
        self.exhausted = False
        self.started = False
        for batch in replay:  # restored from a checkpoint: already pulled from the loader
            self.buffer.append((batch, offload.submit_plan(batch)))

    def _pull(self):
        batch = None
        if not self.exhausted:
            try:
                batch = next(self.source)
            except StopIteration:
                self.exhausted = True
        index = self.offload.submit_plan(batch)
        if batch is not None:
            self.buffer.append((batch, index))

    def __iter__(self):
        return self

    def __next__(self):
        # a fixed number of pulls (= plans) per call, the same on every rank:
        # fill the window on the first call, then one per call
        pulls = 1
        if not self.started:
            self.started = True
            pulls = max(1, self.depth + 1 - len(self.buffer))
        for _ in range(pulls):
            self._pull()
        if not self.buffer:
            raise StopIteration
        batch, index = self.buffer.popleft()
        self.offload.handed_out = index + 1
        return batch

    def pending_batches(self):
        return [b for b, _ in self.buffer]


# ---------------------------------------------------------------------------
# the coordinator


class EngramHostOffload:
    """Every host table of a model, one FIFO worker thread, the prefetch plans.

    Built after the model is materialised; attaches itself as the tables'
    ``row_lookup``. Integration points in the trainer: `wrap` (data iterator),
    `finish_backward` + `grad_params` (before the clip), `register_optimizer`
    (the update and write-back run in a step pre-hook), `flush` (before a
    checkpoint save), `reload` (after a load), `close`.
    """

    def __init__(self, model, *, row_mesh: DeviceMesh, ids_fn, compute_dtype: torch.dtype,
                 device: torch.device, lookahead: int = 4, cpu_threads: int = 8):
        tables = [m.table for m in (getattr(model, "engram", None) or {}).values()]
        if not tables:
            raise ValueError("engram_table_device='host' but the model has no Engram tables")
        for t in tables:
            if not (isinstance(t.weight, DTensor) and t.weight.to_local().device.type == "cpu"):
                raise ValueError("the Engram tables must be host DTensors (shard_table_rows_host)")
        self.row_mesh = row_mesh
        self.group = row_mesh.get_group()
        self.world = dist.get_world_size(self.group)
        self.rank = dist.get_rank(self.group)
        # the worker's own collectives: never shared with the main thread
        self.gloo = dist.new_group(backend="gloo")
        self.ids_fn = ids_fn
        self.compute_dtype = compute_dtype
        self.device = torch.device(device)
        self.cuda = self.device.type == "cuda"
        self.lookahead = lookahead
        self.copy_stream = torch.cuda.Stream(self.device) if self.cuda else None
        self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="engram-host")
        # row scatters run in chunks on these (see `_scatter_rows`)
        self.scatter_threads = max(1, cpu_threads)
        self.scatter_pool = ThreadPoolExecutor(
            max_workers=self.scatter_threads, thread_name_prefix="engram-host-scatter"
        )
        self.stores = [_HostRowStore(self, t, i) for i, t in enumerate(tables)]
        for s in self.stores:
            s.table.row_lookup = _HostLookup(s)
        self.plans: collections.deque[tuple[int, Future]] = collections.deque()
        self.live: collections.deque[tuple[int, list]] = collections.deque()  # merged, not yet consumed
        self.submitted = 0
        self.handed_out = 0
        self.iterator: _Lookahead | None = None
        self._replay = []
        self._writebacks: list[Future] = []
        self.stats = collections.Counter()
        self.times = collections.Counter()  # seconds per phase, see `close`
        self._lock = threading.Lock()
        # the CPU gathers / scatters are intra-op parallel; torchrun sets
        # OMP_NUM_THREADS=1, which makes them ~10x slower than with 8 threads
        if cpu_threads and torch.get_num_threads() < cpu_threads:
            torch.set_num_threads(cpu_threads)
        shard_gb = sum(s.param.to_local().numel() for s in self.stores) * 12 / 1e9
        logger.info(
            f"Engram host tables: {len(tables)} table(s), {shard_gb:.2f} GB per rank in host "
            f"RAM (w + AdamW m, v), GPU row cache, lookahead {lookahead} micro-batches, "
            f"{torch.get_num_threads()} CPU threads (worker thread sees "
            f"{self.worker.submit(torch.get_num_threads).result()})"
        )

    @contextlib.contextmanager
    def _timed(self, key):
        # wall time, and this thread's CPU time (``key/cpu``): wall >> cpu means
        # the thread was waiting (GIL, another rank, the GPU)
        t, c = time.perf_counter(), time.thread_time()
        try:
            yield
        finally:
            self.times[key] += time.perf_counter() - t
            if key.startswith("worker"):
                self.times[key + "/cpu"] += time.thread_time() - c

    # ---- data ----------------------------------------------------------------
    def wrap(self, source):
        self.iterator = _Lookahead(source, self, self.lookahead, replay=self._replay)
        self._replay = []
        return self.iterator

    def pending_batches(self):
        return [] if self.iterator is None else self.iterator.pending_batches()

    def restore_pending_batches(self, batches):
        self._replay = list(batches)

    def submit_plan(self, batch) -> int:
        index = self.submitted
        self.submitted += 1
        ids = None
        if batch is not None:
            with self._timed("main_hash"):
                ids = [t.reshape(-1, t.shape[-1]).cpu() for t in self.ids_fn(batch)]
        hosts = [s.host_tensors() for s in self.stores]
        self.plans.append((index, self.worker.submit(self._plan_job, index, ids, hosts)))
        return index

    # ---- worker jobs ---------------------------------------------------------
    def _route_owner_ids(self, ids):
        """gloo all-to-all: each table's unique ids to their owners. Returns, per
        table, the sorted unique local ids this rank owns and must serve."""
        n = len(self.stores)
        world = self.world
        sends, counts = [], torch.zeros(world, n, dtype=torch.long)
        per_table = []
        for i, s in enumerate(self.stores):
            u = torch.empty(0, dtype=torch.long) if ids is None else torch.unique(ids[i])
            owner = torch.div(u, s.chunk, rounding_mode="floor")
            counts[:, i] = torch.bincount(owner, minlength=world)
            per_table.append((u, owner))
        # ids grouped by owner, then by table
        for r in range(world):
            for i, (u, owner) in enumerate(per_table):
                sends.append(u[owner == r])
        send = torch.cat(sends) if sends else torch.empty(0, dtype=torch.long)
        recv_counts = torch.empty_like(counts)
        dist.all_to_all_single(recv_counts.view(-1), counts.view(-1), group=self.gloo)
        recv = torch.empty(int(recv_counts.sum()), dtype=torch.long)
        dist.all_to_all_single(
            recv, send, recv_counts.sum(1).tolist(), counts.sum(1).tolist(), group=self.gloo
        )
        pieces = torch.split(recv, recv_counts.view(-1).tolist())
        out = []
        for i, s in enumerate(self.stores):
            mine = torch.cat([pieces[r * n + i] for r in range(world)])
            out.append(torch.unique(mine - self.rank * s.chunk))
        return out

    def _gather(self, local_ids, hosts) -> _Plan:
        rows = []
        for ids, (w, m, v) in zip(local_ids, hosts):
            k = ids.numel()
            pin = self.cuda
            bufs = [torch.empty(k, w.shape[1], dtype=torch.float32, pin_memory=pin) for _ in range(3)]
            for src, dst in zip((w, m, v), bufs):
                torch.index_select(src, 0, ids, out=dst)
            if pin:
                ids = ids.pin_memory()
            rows.append((ids, *bufs))
            self.stats["fetched_rows"] += k
        event = None
        if self.cuda:
            with torch.cuda.stream(self.copy_stream):
                rows = [_Rows(*(t.to(self.device, non_blocking=True) for t in r)) for r in rows]
                event = torch.cuda.Event()
                event.record(self.copy_stream)
        else:
            rows = [_Rows(*r) for r in rows]
        return _Plan(-1, rows, event)

    def _plan_job(self, index, ids, hosts):
        with self._timed("worker_route"):
            local = self._route_owner_ids(ids)
        with self._timed("worker_gather_h2d"):
            plan = self._gather(local, hosts)
        plan.index = index
        return plan

    def _missing_job(self, store_index, ids_cpu, hosts):
        local = [torch.empty(0, dtype=torch.long) for _ in self.stores]
        local[store_index] = ids_cpu
        return self._gather(local, hosts)

    def _scatter_rows(self, dst, ids, src):
        """``dst[ids] = src`` for unique ``ids``, in disjoint chunks on several
        threads. Under `torch.use_deterministic_algorithms(True)` (the training
        default) every CPU row scatter -- index_copy_, index_put_, scatter_ --
        takes a single-threaded sorted path: 40 ms per 250k x 64 rows instead
        of 3 ms, and ~1 s per step inside a busy training job (T20). Disjoint
        unique rows make the chunked result exact and deterministic."""
        n = ids.numel()
        k = min(self.scatter_threads, max(1, n // 16384))
        if k == 1:
            dst.index_copy_(0, ids, src)
            return
        bounds = [n * i // k for i in range(k + 1)]
        futs = [
            self.scatter_pool.submit(dst.index_copy_, 0, ids[a:b], src[a:b])
            for a, b in zip(bounds[:-1], bounds[1:])
        ]
        for f in futs:
            f.result()

    def _writeback_job(self, event, store_index, rows, hosts):
        if event is not None:
            with self._timed("worker_wb_wait_gpu"):
                event.synchronize()
        w, m, v = hosts[store_index]
        with self._timed("worker_wb_scatter"):
            for dst, src in ((w, rows.w), (m, rows.m), (v, rows.v)):
                self._scatter_rows(dst, rows.ids, src)
        self.stats["written_rows"] += rows.ids.numel()

    # ---- main-thread merging ----------------------------------------------------
    def _merge(self, plan: _Plan, *, live: bool):
        if plan.event is not None:
            stream = torch.cuda.current_stream(self.device)
            stream.wait_event(plan.event)
            for r in plan.rows:
                for t in (r.ids, r.w, r.m, r.v):
                    t.record_stream(stream)
        for store, rows in zip(self.stores, plan.rows):
            store.merge(rows)
        if live and plan.index >= self.handed_out:
            self.live.append((plan.index, [r.ids for r in plan.rows]))

    def merge_ready(self):
        """Merge every plan whose host gather has finished (no waiting)."""
        while self.plans and self.plans[0][1].done():
            _, fut = self.plans.popleft()
            self._merge(fut.result(), live=True)

    def merge_all(self):
        while self.plans:
            _, fut = self.plans.popleft()
            self._merge(fut.result(), live=True)

    def fetch_missing(self, store: _HostRowStore, local_ids: torch.Tensor):
        """Synchronous fetch of the rows of ``local_ids`` the cache lacks. Correct
        (FIFO: after every queued write-back) but on the critical path."""
        self.merge_all()
        pos, ok = store.slots(local_ids)
        if ok:
            return
        if pos is None:
            missing = torch.unique(local_ids)
        else:
            hit = store.cache.ids[pos.clamp_max(store.cache.ids.numel() - 1)] == local_ids
            missing = torch.unique(local_ids[~hit])
        self.stats["miss_rows"] += missing.numel()
        self.stats["miss_fetches"] += 1
        hosts = [s.host_tensors() for s in self.stores]
        plan = self.worker.submit(self._missing_job, store.index, missing.cpu(), hosts).result()
        self._merge(plan, live=False)

    # ---- step --------------------------------------------------------------------
    def finish_backward(self):
        for s in self.stores:
            s.finish_backward()

    def grad_params(self):
        """Parameters carrying the step's table gradients (DTensors over the row
        mesh), for the gradient clip and its norm."""
        return [s.carrier for s in self.stores if s.carrier is not None]

    def grad_norm(self):
        grads = [p.grad for p in self.grad_params()]
        if not grads:
            return None
        norm = torch.nn.utils.get_total_norm(grads)
        return norm.full_tensor() if hasattr(norm, "full_tensor") else norm

    def discard_grads(self):
        for s in self.stores:
            s.discard_grads()

    def register_optimizer(self, optimizer: torch.optim.Optimizer):
        """Run the lazy AdamW update, write-back, merge and eviction in a step
        pre-hook. The tables must be in ``optimizer``'s param groups (their LR,
        betas, eps and weight decay are used); the main AdamW never sees a
        table gradient."""
        group_of = {id(p): g for g in optimizer.param_groups for p in g["params"]}
        for s in self.stores:
            if id(s.param) not in group_of:
                raise ValueError("an Engram host table is not in the optimizer's param groups")
        self.optimizer = optimizer

        def pre_step(opt, args, kwargs):
            with self._timed("main_step_hook"):
                _pre_step(opt)

        def _pre_step(opt):
            any_update = False
            for s in self.stores:
                s.param.grad = None  # `_init_optim_state` sets zeros; never ours
                state = opt.state[s.param]
                if not state:
                    self._adopt_state(s, state)
                group = group_of[id(s.param)]
                rows = s.adamw_update(
                    state, lr=group["lr"], betas=group["betas"], eps=group["eps"],
                    weight_decay=group["weight_decay"],
                )
                if rows is not None:
                    any_update = True
                    self._write_back(s, rows)
            if any_update or self.plans:
                self._after_step()

        optimizer.register_step_pre_hook(pre_step)

    def _adopt_state(self, store, state):
        mesh = store.param.device_mesh
        shape, stride = store.param.shape, store.param.stride()

        def wrap(t):
            return DTensor.from_local(t, mesh, [Shard(0)], run_check=False, shape=shape, stride=stride)

        state["step"] = torch.zeros((), dtype=torch.float32)
        state["exp_avg"] = wrap(store.host_m)
        state["exp_avg_sq"] = wrap(store.host_v)

    def _write_back(self, store, rows: _Rows):
        if self.cuda:
            pinned = _Rows(*(torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
                             for t in (rows.ids, rows.w, rows.m, rows.v)))
            for dst, src in zip((pinned.ids, pinned.w, pinned.m, pinned.v),
                                (rows.ids, rows.w, rows.m, rows.v)):
                dst.copy_(src, non_blocking=True)
            event = torch.cuda.Event()
            event.record(torch.cuda.current_stream(self.device))
        else:
            pinned, event = _Rows(rows.ids.clone(), rows.w.clone(), rows.m.clone(), rows.v.clone()), None
        hosts = [s.host_tensors() for s in self.stores]
        self._writebacks.append(
            self.worker.submit(self._writeback_job, event, store.index, pinned, hosts)
        )
        self._writebacks = [f for f in self._writebacks if not f.done()]

    def _after_step(self):
        """Merge every submitted plan (their gathers may predate this step's
        write-back: the freshly updated cache wins), then evict every row no
        pending batch needs."""
        with self._timed("main_wait_plans"):
            self.merge_all()
        while self.live and self.live[0][0] < self.handed_out:
            self.live.popleft()
        for i, s in enumerate(self.stores):
            keep = [ids[i] for _, ids in self.live]
            keep_ids = torch.unique(torch.cat(keep)) if keep else s.cache.ids[:0]
            s.keep_only(keep_ids)
        self.stats["steps"] += 1
        self.stats["cache_rows"] += sum(s.cache.ids.numel() for s in self.stores)

    # ---- checkpoint ------------------------------------------------------------
    def flush(self):
        """Wait until every queued write-back (and gather) has run: the host
        tables and moments are then exactly the trained state."""
        self.worker.submit(lambda: None).result()

    def reload(self):
        """After a checkpoint load: adopt the loaded moments, drop every cached
        or prefetched row (they predate the load)."""
        self.flush()
        self.merge_all()
        self.live.clear()
        for s in self.stores:
            s.discard_grads()
            s.clear_cache()
            state = getattr(self, "optimizer", None) and self.optimizer.state.get(s.param)
            if state and "exp_avg" in state:
                # keep the huge-page buffers: copy the loaded moments into them
                # and point the optimizer state back at them
                for key, mine in (("exp_avg", s.host_m), ("exp_avg_sq", s.host_v)):
                    loaded = _to_local(state[key])
                    if loaded.data_ptr() != mine.data_ptr():
                        mine.copy_(loaded)
                step = state["step"]
                state.clear()
                self._adopt_state(s, state)
                state["step"].copy_(step)

    def close(self):
        self.flush()
        self.worker.shutdown(wait=True)
        self.scatter_pool.shutdown(wait=True)
        steps = max(self.stats["steps"], 1)
        logger.info(
            "Engram host tables, ms per step: "
            + ", ".join(f"{k} {1e3 * v / steps:.1f}" for k, v in sorted(self.times.items()))
        )
        logger.info(
            "Engram host tables: rows prefetched %d, fetched on a miss %d (%d fetches), "
            "written back %d, mean cache rows after a step %.0f"
            % (self.stats["fetched_rows"] - self.stats["miss_rows"], self.stats["miss_rows"],
               self.stats["miss_fetches"], self.stats["written_rows"],
               self.stats["cache_rows"] / steps)
        )


def _to_local(t):
    return t.detach().to_local() if isinstance(t, DTensor) else t
