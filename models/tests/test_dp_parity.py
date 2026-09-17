"""S2 gate (TITAN_MIGRATION_v2.md): loss normalisation is independent of DP size.

Four micro-batches with very different valid-token counts. One rank accumulates
all four; two ranks accumulate two each. With torchtitan's normalisation (sum / global
valid tokens, gradients reduced as a SUM) the step loss and the gradient norm
must match. A per-micro-batch mean averaged over ranks would not.

    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2   python -m torch.distributed.run --nproc_per_node=1 models/tests/test_titan_dp_parity.py fsdp
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --nproc_per_node=2 models/tests/test_titan_dp_parity.py fsdp
    (and `ddp`) -- each run appends one JSON line to $PARITY_OUT; compare them.
"""

import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5_tt.checkpoint import build_meta, load_hf, materialize
from train.parallel.fsdp import apply_data_parallel
from train.titan_step import forward_backward

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

    model = build_meta(SNAPSHOT, seq_len=T)
    for p in model.parameters():
        p.data = p.data.to(torch.bfloat16)
    apply_data_parallel(
        model, mesh, mode=mode, param_dtype=torch.bfloat16, reduce_dtype=torch.float32
    )
    materialize(model, "cuda")
    load_hf(model, SNAPSHOT)

    # both world sizes accumulate, so the no-sync path runs in both
    per_rank = len(VALID) // world
    batches = [make_batch(rank * per_rank + i) for i in range(per_rank)]
    loss, local_tokens, denom = forward_backward(
        model, batches, dp_group=mesh, special_tokens={"image_id": -1}, ddp=mode == "ddp",
        loss_chunks=int(os.environ.get("LOSS_CHUNKS", "8")),
    )
    torch.distributed.all_reduce(loss)  # SUM over DP = global per-token mean
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    gn = torch.nn.utils.get_total_norm([g.float() for g in grads])
    gn = gn.full_tensor() if hasattr(gn, "full_tensor") else gn
    if rank == 0:
        rec = {"mode": mode, "world": world, "loss_chunks": int(os.environ.get("LOSS_CHUNKS", "8")),
               "loss": loss.item(), "grad_norm": gn.item(),
               "global_tokens": int(denom.item())}
        print(json.dumps(rec))
        out = os.environ.get("PARITY_OUT")
        if out:
            with open(out, "a") as f:
                f.write(json.dumps(rec) + "\n")
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
