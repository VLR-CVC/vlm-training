"""How the Engram tables are distributed for training (`training.engram_sharding`).

* ``"fsdp"`` (default): the table is an ordinary parameter of its Engram FSDP
  unit -- all-gathered in bf16 for every micro-batch, dense gradient,
  reduce-scattered.
* ``"rows"``: the table is a ``Shard(0)`` DTensor over every rank, outside FSDP,
  and looked up with an all-to-all of ids and rows
  (`models/common/engram_sharded.py`). Same storage layout as FSDP's, so a DCP
  checkpoint loads in either mode.
"""

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh

from models.common.engram_sharded import RowShardedLookup, row_chunk, shard_table_rows
from train.logger import logger
from train.parallel.parallel_dims import ParallelDims

ENGRAM_SHARDING_MODES = ("fsdp", "rows")


def engram_row_mesh(parallel_dims: ParallelDims):
    """The 1D mesh of every rank, which the tables are row-sharded over."""
    if parallel_dims.tp > 1 or parallel_dims.cp > 1 or parallel_dims.ep > 1:
        raise NotImplementedError(
            "engram_sharding='rows' needs tp = cp = ep = 1 (the table would be "
            "sharded over TP and DP at once); use engram_sharding='fsdp'"
        )
    return parallel_dims.world_mesh


def apply_engram_row_sharding(
    model: nn.Module,
    mesh: DeviceMesh,
    *,
    compute_dtype: torch.dtype,
    grad_dtype: torch.dtype = torch.float32,
) -> set[nn.Parameter]:
    """Row-shard every Engram table over all ranks, on the meta model, before
    FSDP, over ``mesh`` (1D: `engram_row_mesh`). Returns the table parameters,
    which FSDP must then ignore.

    Every rank is a shard: the table is spread over nodes as well as GPUs, so
    per-rank memory is table/world. The all-to-all payload is ~ unique ids x
    row width per micro-batch, small enough to cross nodes (see the T20 notes in
    docs/ARCHITECTURE.md).
    """
    engram = getattr(model, "engram", None)
    if engram is None:
        return set()
    tables = set()
    for module in engram.values():
        table = module.table
        table.weight = shard_table_rows(table.weight, mesh)
        table.row_lookup = RowShardedLookup(
            mesh, table.num_embeddings, compute_dtype=compute_dtype, grad_dtype=grad_dtype
        )
        tables.add(table.weight)
        logger.info(
            f"Engram table row-sharded over {mesh.size()} ranks: "
            f"{row_chunk(table.num_embeddings, mesh.size()):,} of {table.num_embeddings:,} "
            f"rows x {table.embedding_dim} per rank, all-to-all lookup"
        )
    return tables
