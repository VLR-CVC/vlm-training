"""DCP checkpoints move between engram_sharding = "fsdp", "rows" and "rows" with
the table in host RAM (`training.engram_table_device = "host"`) (T20).

Saves a model + AdamW state in one mode, loads it into a model in another, and
checks the table, the optimizer moments and the loss on one batch. The host
model has no prefetch here, so every lookup takes the miss path. GPU, any
world size:

    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 models/tests/test_engram_rows_dcp.py /tmp/dcp_dir
"""

import os
import sys
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.checkpoint import build_meta, init_engram, load_hf, materialize
from models.qwen3_5.engram import EngramArgs
from train.parallel.engram import apply_engram_row_sharding
from train.parallel.fsdp import apply_data_parallel
from train.step import forward_backward

SNAPSHOT = os.environ.get(
    "QWEN3_5_SNAPSHOT",
    "/data/151-2/users/allabres/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/"
    "2fc06364715b967f1860aea9cf38778875588b17",
)
T = 1024


def build(mesh, sharding):
    model = build_meta(SNAPSHOT, seq_len=T, engram=EngramArgs(rows_per_head=4099))
    for p in model.parameters():
        p.data = p.data.float()
        p.requires_grad = False
    for p in model.engram.parameters():
        p.requires_grad = True
    ignored = set()
    if sharding in ("rows", "host"):
        ignored = apply_engram_row_sharding(model, mesh, compute_dtype=torch.bfloat16,
                                            host=sharding == "host")
    apply_data_parallel(model, mesh, mode="fsdp", param_dtype=torch.bfloat16,
                        reduce_dtype=torch.float32, ignored_params=ignored)
    materialize(model, "cuda")
    load_hf(model, SNAPSHOT)
    init_engram(model, "cuda")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    model.host = None
    if sharding == "host":
        from train.engram_host import EngramHostOffload

        model.host = EngramHostOffload(
            model, row_mesh=mesh, ids_fn=None, compute_dtype=torch.bfloat16,
            device=torch.device("cuda"), lookahead=0,
        )
        model.host.register_optimizer(opt)
    return model, opt


def batch():
    g = torch.Generator().manual_seed(7 + torch.distributed.get_rank())
    ids = torch.randint(1000, 50000, (T,), generator=g)
    return {"input": ids.cuda(), "labels": torch.roll(ids, -1).cuda(),
            "positions": torch.arange(T).cuda(), "num_valid_tokens": T}


def step(model, opt, mesh):
    loss, _, _ = forward_backward(model, [batch()], dp_group=mesh, special_tokens={"image_id": -1})
    if model.host is not None:
        model.host.finish_backward()
    opt.step()
    opt.zero_grad()
    torch.distributed.all_reduce(loss)
    return loss.item()


def eval_loss(model, mesh):
    loss, _, _ = forward_backward(model, [batch()], dp_group=mesh, special_tokens={"image_id": -1})
    for p in model.parameters():
        p.grad = None
    if model.host is not None:
        model.host.discard_grads()
    torch.distributed.all_reduce(loss)
    return loss.item()


def main():
    out = sys.argv[1]
    torch.distributed.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    rank = torch.distributed.get_rank()
    from torch.distributed.device_mesh import init_device_mesh

    mesh = init_device_mesh("cuda", (torch.distributed.get_world_size(),), mesh_dim_names=("dp",))
    pairs = (("fsdp", "rows"), ("rows", "fsdp"), ("rows", "host"), ("host", "fsdp"), ("host", "host"))
    for src, dst in pairs:
        torch.manual_seed(0)
        a, opt_a = build(mesh, src)
        for _ in range(2):  # W_V moves off zero, AdamW state exists
            step(a, opt_a, mesh)
        ref_loss = eval_loss(a, mesh)
        if a.host is not None:
            a.host.flush()
        ref_table = a.engram["2"].table.weight.full_tensor().cpu()
        ref_m = opt_a.state[a.engram["2"].table.weight]["exp_avg"].full_tensor().cpu()
        path = f"{out}/{src}-{dst}"
        dcp.save({"model": a.state_dict(), "optim": opt_a.state_dict()}, checkpoint_id=path)
        a_host = a.host
        del a, opt_a

        torch.manual_seed(1)
        b, opt_b = build(mesh, dst)
        step(b, opt_b, mesh)  # create optimizer state to load into
        sd = {"model": b.state_dict(), "optim": opt_b.state_dict()}
        dcp.load(sd, checkpoint_id=path)
        b.load_state_dict(sd["model"])
        opt_b.load_state_dict(sd["optim"])
        if b.host is not None:
            b.host.reload()
        table = b.engram["2"].table.weight.full_tensor().cpu()
        m = opt_b.state[b.engram["2"].table.weight]["exp_avg"].full_tensor().cpu()
        loss = eval_loss(b, mesh)
        ok = torch.equal(table, ref_table) and torch.equal(m, ref_m) and abs(loss - ref_loss) < 1e-4
        if rank == 0:
            print(f"{src} -> {dst}: table equal {torch.equal(table, ref_table)}, "
                  f"exp_avg equal {torch.equal(m, ref_m)}, loss {ref_loss:.6f} -> {loss:.6f}: "
                  f"{'OK' if ok else 'FAIL'}")
        assert ok
        for x in (a_host, b.host):
            if x is not None:
                x.close()
        del b, opt_b
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
