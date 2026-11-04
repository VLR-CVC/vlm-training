"""Prove an ep_size>1 trainer run really dispatches over the EP axis.

    MODE=wiring scripts/local_ep_parity.sh

An EP=1 fallback produces the SAME loss as a correct EP=2 run -- that is the
point of the parity -- so a matching loss curve is not evidence that the
all-to-all ran. This builds the model the way the trainer does and asserts on
the wiring instead: which dispatcher class, which mesh it holds, and where the
expert parameters actually live.
"""

from __future__ import annotations

import os

import torch
import torch.distributed as dist

from models.common.moe import GroupedExperts
from models.common.token_dispatcher import AllToAllTokenDispatcher
from models.qwen3_5.checkpoint import build_meta, materialize
from train.parallel.parallel_dims import ParallelDims
from train.parallel.parallelize import parallelize_qwen3_5

CONFIG = "configs/models/qwen3_5_moe_debug.json"


def main() -> None:
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(
        int(os.environ.get("LOCAL_RANK", 0)) % torch.cuda.device_count()
    )
    ep = int(os.environ.get("EP", "2"))
    tp = int(os.environ.get("TP", "1"))
    config = os.environ.get("MODEL_CONFIG", CONFIG)
    assert world % tp == 0, (world, tp)

    pd = ParallelDims(
        dp_replicate=1, dp_shard=world // tp, cp=1, tp=tp, pp=1, ep=ep,
        world_size=world,
    )
    pd.build_mesh()

    model = build_meta(config, seq_len=512, with_vision=False, tp=tp, ep=ep)
    materialize(model, "cuda")
    parallelize_qwen3_5(
        model,
        pd,
        mode="fsdp",
        compile=False,
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
    )

    moes = [m for m in model.modules() if isinstance(m, GroupedExperts)]
    assert moes, "no GroupedExperts in the model"
    dispatchers = [
        m.token_dispatcher
        for m in model.modules()
        if hasattr(m, "token_dispatcher")
    ]
    assert dispatchers, "no token dispatcher in the model"

    if ep > 1:
        for d in dispatchers:
            assert isinstance(d, AllToAllTokenDispatcher), type(d).__name__
            assert d.ep_mesh is not None, "dispatcher has no ep_mesh"
            assert d.ep_mesh.size() == ep, d.ep_mesh.size()
    else:
        for d in dispatchers:
            assert getattr(d, "ep_mesh", None) is None

    w1 = moes[0].w1_EFD
    experts_per_rank = w1.to_local().shape[0] if hasattr(w1, "to_local") else w1.shape[0]
    print(
        f"rank {rank}: tp={tp} ep={ep}  dispatcher={type(dispatchers[0]).__name__}  "
        f"ep_mesh={None if ep == 1 else dispatchers[0].ep_mesh.size()}  "
        f"w1 global {tuple(w1.shape)} local {tuple(w1.to_local().shape)}  "
        f"placements {getattr(w1, 'placements', None)}  "
        f"experts_held {experts_per_rank}",
        flush=True,
    )
    if ep > 1:
        total = moes[0].num_experts
        assert experts_per_rank == total // ep, (experts_per_rank, total, ep)

    # TP is on a different axis than EP, and nothing above would notice if it
    # were silently off: check a dense weight is column-sharded too.
    wq = next(
        m.weight
        for name, m in model.named_modules()
        if name.endswith("attn.wq") and hasattr(m, "weight")
    )
    local_rows = wq.to_local().shape[0] if hasattr(wq, "to_local") else wq.shape[0]
    print(
        f"rank {rank}: dense attn.wq global {tuple(wq.shape)} local "
        f"{tuple(wq.to_local().shape)} placements {getattr(wq, 'placements', None)}",
        flush=True,
    )
    if tp > 1:
        assert local_rows == wq.shape[0] // tp, (local_rows, wq.shape, tp)

    dist.barrier()
    if rank == 0:
        print(f"EP wiring OK (tp={tp} ep={ep})", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
