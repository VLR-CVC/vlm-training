"""S2 gate (TITAN_MIGRATION_v2.md): loss normalisation is independent of DP size.

Four micro-batches with very different valid-token counts. One rank accumulates
all four; two ranks accumulate two each. With torchtitan's normalisation (sum / global
valid tokens, gradients reduced as a SUM) the step loss and the gradient norm
must match. A per-micro-batch mean averaged over ranks would not.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2   python -m torch.distributed.run --nproc_per_node=1 models/tests/test_dp_parity.py fsdp
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 models/tests/test_dp_parity.py fsdp
    (and `ddp`) -- each run appends one JSON line to $PARITY_OUT; compare them.
    ENGRAM=1 adds an Engram module; ENGRAM=1 ENGRAM_SHARDING=rows row-shards its
    table (fsdp mode only) and must match ENGRAM=1 at every world size.
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.checkpoint import build_meta, load_hf, materialize
from train.parallel.fsdp import apply_data_parallel
from train.step import forward_backward

SNAPSHOT = os.environ.get(
    "QWEN3_5_SNAPSHOT", "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b"
)
T = 2048
VALID = (200, 1800, 900, 50)  # supervised tokens per micro-batch


def make_batch(i: int) -> dict:
    g = torch.Generator().manual_seed(1234 + i)
    ids = torch.randint(1000, 50000, (T,), generator=g)
    labels = torch.full((T,), -100)
    labels[: T - 1] = ids[1:]
    labels[VALID[i]:] = -100
    pos = torch.arange(T)
    return {
        "input": ids.cuda(),
        "labels": labels.cuda(),
        "positions": pos.cuda(),
        "mrope_positions": pos.unsqueeze(-1).expand(-1, 3).contiguous().cuda(),
        "num_valid_tokens": int((labels != -100).sum()),
    }


def main() -> None:
    mode = sys.argv[1]
    torch.distributed.init_process_group("nccl")
    rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (world,), mesh_dim_names=("dp",))

    # ENGRAM=1: an Engram module (small tables, deterministic non-zero weights), so
    # its own FSDP unit and its gradient reduction go through the same check
    engram = None
    if os.environ.get("ENGRAM") == "1":
        from models.qwen3_5.engram import EngramArgs

        engram = EngramArgs(rows_per_head=4099)
    model = build_meta(SNAPSHOT, seq_len=T, engram=engram)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    # ENGRAM_SHARDING=rows: the tables row-sharded outside FSDP, looked up by
    # all-to-all (`train/parallel/engram.py`); must match the FSDP table exactly
    ignored = set()
    if engram is not None and os.environ.get("ENGRAM_SHARDING", "fsdp") == "rows":
        from train.parallel.engram import apply_engram_row_sharding

        ignored = apply_engram_row_sharding(
            model, mesh, compute_dtype=torch.bfloat16, grad_dtype=torch.float32
        )
    apply_data_parallel(
        model, mesh, mode=mode, param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
        ignored_params=ignored,
    )
    materialize(model, "cuda")
    load_hf(model, SNAPSHOT)
    if model.engram is not None:
        from torch.distributed.tensor import distribute_tensor

        from models.qwen3_5.checkpoint import init_engram

        init_engram(model, "cuda")
        g = torch.Generator().manual_seed(1234)
        with torch.no_grad():
            for name, p in sorted(model.named_parameters()):
                # every random-initialised Engram weight: `init_engram` draws on the
                # sharded tensor, so its values depend on the world size
                stds = {"table.weight": 0.5, "key_proj.weight": 0.05, "value_proj.weight": 0.002}
                std = next((s for k, s in stds.items() if name.endswith(k)), None)
                if name.startswith("engram.") and std is not None:
                    full = (torch.randn(p.shape, generator=g) * std).to(p.dtype).cuda()
                    p.to_local().copy_(distribute_tensor(full, p.device_mesh, p.placements).to_local())

    # both world sizes accumulate, so the no-sync path runs in both
    per_rank = len(VALID) // world
    batches = [make_batch(rank * per_rank + i) for i in range(per_rank)]
    loss, local_tokens, global_tokens = forward_backward(
        model, batches, dp_group=mesh, special_tokens={"image_id": -1}, ddp=mode == "ddp",
        loss_chunks=int(os.environ.get("LOSS_CHUNKS", "8")),
    )
    torch.distributed.all_reduce(loss)  # SUM over DP = global per-token mean
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    gn = torch.nn.utils.get_total_norm([g.float() for g in grads])
    gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
    table_gn = None
    if model.engram is not None:
        tg = torch.nn.utils.get_total_norm(
            [m.table.weight.grad.float() for m in model.engram.values()]
        )
        table_gn = (tg.full_tensor() if hasattr(tg, "full_tensor") else tg).item()
    if rank == 0:
        rec = {"mode": mode, "world": world, "loss_chunks": int(os.environ.get("LOSS_CHUNKS", "8")),
               "engram_sharding": os.environ.get("ENGRAM_SHARDING", "fsdp") if engram else None,
               "table_grad_norm": table_gn,
               "loss": loss.item(), "grad_norm": gn.item(),
               "global_tokens": int(global_tokens.item())}
        print(json.dumps(rec))
        out = os.environ.get("PARITY_OUT")
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
