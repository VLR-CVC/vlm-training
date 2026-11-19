"""Row-sharded Engram table: all-to-all lookup instead of an FSDP all-gather.

Under plain FSDP the table is a dense parameter: every micro-batch all-gathers
the whole table in bf16 on every rank, backward builds a full-size bf16
gradient and reduce-scatters it, and AdamW walks every row every step. All of it
scales with the table, none of it with the tokens, and the gathered copy plus
its gradient are full-size on every rank however many ranks there are.

Here the table stays sharded -- the same ``Shard(0)`` DTensor over the DP ranks
that FSDP stores, so checkpoints are interchangeable -- and only the rows a
micro-batch reads move, the recommender-system pattern (DLRM / TorchRec
row-wise sharding):

forward
    1. ``unique`` this rank's ``(T, heads)`` row ids (sorted, so they are already
       grouped by owner: rank ``r`` owns rows ``[r*chunk, (r+1)*chunk)``);
    2. exchange per-owner counts (one host sync), then the ids (all-to-all);
    3. each owner gathers its rows in the compute dtype and sends them back;
    4. scatter the unique rows back to ``(T, heads, dim)``.

backward
    1. sum the incoming gradient per unique id (fp32);
    2. all-to-all the row gradients back to their owners;
    3. the owner adds them into its shard's gradient. Rows hit by several ranks
       are summed there, which is exactly FSDP's SUM reduce-scatter
       (`disable_fsdp_gradient_division`).

Communication per micro-batch is O(unique ids x dim), independent of the table
size. The gradient is a dense tensor of the *shard* (rows/N); the optimizer is
unchanged (AdamW over the shard, as under FSDP).
"""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor, Shard

__all__ = ["RowShardedLookup", "row_chunk", "shard_table_rows"]


def row_chunk(num_rows: int, world: int) -> int:
    """Rows per rank under ``Shard(0)`` (``torch.chunk`` sizes: ceil, last short)."""
    return -(-num_rows // world)


def _a2a(output, input, output_splits, input_splits, group):
    dist.all_to_all_single(output, input, output_splits, input_splits, group=group)
    return output


class _Route:
    """Who sends which unique id where, for one micro-batch's ids."""

    __slots__ = ("inverse", "num_unique", "send_splits", "recv_splits", "local_ids")

    def __init__(self, ids: torch.Tensor, *, chunk: int, group) -> None:
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        unique, self.inverse = torch.unique(ids.reshape(-1), sorted=True, return_inverse=True)
        self.num_unique = unique.numel()
        owner = torch.div(unique, chunk, rounding_mode="floor")
        send_counts = torch.bincount(owner, minlength=world)
        recv_counts = torch.empty_like(send_counts)
        dist.all_to_all_single(recv_counts, send_counts, group=group)
        counts = torch.stack([send_counts, recv_counts]).cpu()  # the one host sync
        self.send_splits = counts[0].tolist()
        self.recv_splits = counts[1].tolist()
        recv_ids = unique.new_empty(sum(self.recv_splits))
        _a2a(recv_ids, unique, self.recv_splits, self.send_splits, group)
        self.local_ids = recv_ids - rank * chunk


class _RowShardedLookupFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, local_weight, ids, route, group, compute_dtype, grad_dtype):
        dim = local_weight.shape[1]
        rows = local_weight.index_select(0, route.local_ids).to(compute_dtype)
        unique_rows = rows.new_empty(route.num_unique, dim)
        _a2a(unique_rows, rows, route.send_splits, route.recv_splits, group)
        ctx.route = route
        ctx.group = group
        ctx.grad_dtype = grad_dtype
        ctx.weight_meta = (local_weight.shape, local_weight.dtype)
        return unique_rows.index_select(0, route.inverse).view(*ids.shape, dim)

    @staticmethod
    def backward(ctx, grad_out):
        route, group = ctx.route, ctx.group
        shape, dtype = ctx.weight_meta
        dim = shape[1]
        grad = grad_out.reshape(-1, dim).to(ctx.grad_dtype)
        unique_grad = grad.new_zeros(route.num_unique, dim).index_add_(0, route.inverse, grad)
        recv = unique_grad.new_empty(route.local_ids.numel(), dim)
        _a2a(recv, unique_grad, route.recv_splits, route.send_splits, group)
        weight_grad = torch.zeros(shape, dtype=dtype, device=grad.device)
        weight_grad.index_add_(0, route.local_ids, recv.to(dtype))
        return weight_grad, None, None, None, None, None


class RowShardedLookup:
    """State a row-sharded `EngramTable` keeps: the group and the chunk size."""

    def __init__(self, mesh: DeviceMesh, num_rows: int, *, compute_dtype: torch.dtype,
                 grad_dtype: torch.dtype = torch.float32) -> None:
        if mesh.ndim != 1:
            raise ValueError(f"the table shards over a 1D mesh, got {mesh.ndim}D")
        self.mesh = mesh
        self.group = mesh.get_group()
        self.chunk = row_chunk(num_rows, mesh.size())
        self.compute_dtype = compute_dtype
        self.grad_dtype = grad_dtype

    def __call__(self, weight: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
        """``(T, heads)`` global row ids -> ``(T, heads, dim)`` rows.

        Records the shard rows it read in ``weight.touched_rows`` (a list, one
        tensor per call) when the attribute exists: a row-sparse optimizer
        (`train/engram_optim.py`) updates only those and clears the list."""
        with torch.no_grad():
            route = _Route(ids, chunk=self.chunk, group=self.group)
        touched = getattr(weight, "touched_rows", None)
        if touched is not None and torch.is_grad_enabled():
            touched.append(route.local_ids)
        local = weight.to_local() if isinstance(weight, DTensor) else weight
        return _RowShardedLookupFn.apply(
            local, ids, route, self.group, self.compute_dtype, self.grad_dtype
        )


def shard_table_rows(weight: torch.nn.Parameter, mesh: DeviceMesh) -> torch.nn.Parameter:
    """The ``Shard(0)`` DTensor parameter for a (meta or real) dense table."""
    if isinstance(weight, DTensor):
        raise ValueError("the table is already a DTensor")
    dt = DTensor.from_local(
        _local_rows(weight, mesh), mesh, [Shard(0)], run_check=False,
        shape=weight.shape, stride=weight.stride(),
    )
    return torch.nn.Parameter(dt, requires_grad=weight.requires_grad)


def _local_rows(weight: torch.Tensor, mesh: DeviceMesh) -> torch.Tensor:
    chunk = row_chunk(weight.shape[0], mesh.size())
    start = min(mesh.get_local_rank() * chunk, weight.shape[0])
    stop = min(start + chunk, weight.shape[0])
    return weight.detach()[start:stop].clone()
