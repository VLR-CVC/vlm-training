"""Check optimizer="muon_fused" (microsoft/dion Muon via train.optimizer.build_dion_muon) against "muon".

  torchrun --nproc_per_node=2 scripts/test_muon_fused.py          # agreement, FSDP and DDP layouts
  torchrun --nproc_per_node=2 scripts/test_muon_fused.py --bench  # step time on ElliotVL 2B Muon shapes

Same math; bf16 Newton-Schulz rounds differently with batch size (one GPU's matrix share), so ~1%.
"""
import os
import sys
import time
from types import SimpleNamespace

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate, Shard, distribute_tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.optimizer import MuonSR, build_dion_muon  # noqa: E402

# (rows, cols, count): uneven shards, an empty shard (1 row), fewer matrices than ranks, real shapes
TEST_SHAPES = [(9, 32, 3), (1, 64, 1), (64, 48, 1), (16, 2048, 5), (4304, 1152, 3), (1152, 4304, 2),
               (6144, 2048, 2)]
# Muon matrices of elliot-vl-2b-v1.2 (350)
ELLIOT_SHAPES = [(6144, 2048, 66), (2048, 6144, 24), (2048, 2048, 43), (1152, 1152, 108), (4304, 1152, 27),
                 (1152, 4304, 27), (4096, 2048, 6), (512, 2048, 12), (2048, 4608, 1), (16, 2048, 36)]
ARGS = SimpleNamespace(lr_llm=2e-2, muon_momentum=0.95, adam_betas=(0.9, 0.99), weight_decay=0.1, eps=1e-8,
                       muon_nesterov=True, muon_adjust_lr="match_rms_adamw", muon_ns_steps=5)


def build(variant, shapes, placement, mesh):
    g = torch.Generator().manual_seed(0)
    mats = []
    for r, c, n in shapes:
        for _ in range(n):
            w = (torch.randn(r, c, generator=g) * 0.02).cuda()
            mats.append(torch.nn.Parameter(distribute_tensor(w, mesh, [placement])))
    bias = torch.nn.Parameter(distribute_tensor(torch.zeros(64, device="cuda"), mesh, [Replicate()]))
    groups = [{"params": mats, "lr": 2e-2, "weight_decay": 0.1, "use_muon": True},
              {"params": [bias], "lr": 1e-3, "weight_decay": 0.0}]
    if variant == "muon_fused":
        return mats, bias, build_dion_muon(groups, ARGS)
    return mats, bias, MuonSR(groups, lr=2e-2, betas=(0.9, 0.99), eps=1e-8, weight_decay=0.1)


def set_grads(mats, bias, placement, mesh, step):
    g = torch.Generator().manual_seed(1000 + step)
    for p in mats:
        p.grad = distribute_tensor(torch.randn(p.shape, generator=g).cuda(), mesh, [placement])
    bias.grad = distribute_tensor(torch.randn(64, generator=g).cuda(), mesh, [Replicate()])


def check(mesh, rank):
    for name, placement in (("fsdp Shard(0)", Shard(0)), ("ddp Replicate", Replicate())):
        out = {}
        for variant in ("muon", "muon_fused"):
            mats, bias, opt = build(variant, TEST_SHAPES, placement, mesh)
            p0 = [p.full_tensor().clone() for p in mats]
            for step in range(3):
                set_grads(mats, bias, placement, mesh, step)
                opt.step()
            out[variant] = ([p.full_tensor() - q for p, q in zip(mats, p0)], bias.full_tensor().clone())
        cos = [torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()
               for a, b in zip(out["muon"][0], out["muon_fused"][0])]
        rms = [(b.norm() / a.norm()).item() for a, b in zip(out["muon"][0], out["muon_fused"][0])]
        adam = ((out["muon"][1] - out["muon_fused"][1]).norm() / out["muon"][1].norm()).item()
        if rank == 0:
            print(f"{name:14s}: Muon update cosine min {min(cos):.3f} mean {sum(cos) / len(cos):.3f}, "
                  f"norm ratio {min(rms):.3f}-{max(rms):.3f}; AdamW rel diff {adam:.1e}")
        assert min(cos) > 0.999 and 0.98 < min(rms) and max(rms) < 1.02 and adam < 1e-4, (cos, rms, adam)
    if rank == 0:
        print("OK")


def bench(mesh, rank):
    for name, placement in (("fsdp Shard(0)", Shard(0)), ("ddp Replicate", Replicate())):
        for variant in ("muon", "muon_fused"):
            mats, bias, opt = build(variant, ELLIOT_SHAPES, placement, mesh)
            times = []
            for step in range(6):
                set_grads(mats, bias, placement, mesh, step)
                torch.cuda.synchronize(); dist.barrier()
                t = time.perf_counter()
                opt.step()
                torch.cuda.synchronize(); dist.barrier()
                times.append(time.perf_counter() - t)
            if rank == 0:
                print(f"{name:14s} {variant:10s} step {sorted(times[2:])[2] * 1e3:8.1f} ms  (W={mesh.size()}, "
                      f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB)")
            del mats, bias, opt
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    dist.init_process_group("nccl")
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    mesh = init_device_mesh("cuda", (dist.get_world_size(),))
    (bench if "--bench" in sys.argv else check)(mesh, dist.get_rank())
    dist.destroy_process_group()
