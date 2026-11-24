"""Parity of the batched MuonSR step (one all-gather + one batched Newton-Schulz per bucket of
same-shaped matrices) against the per-matrix step it replaced, kept verbatim below as
`PerMatrixMuonSR`: unsharded, FSDP-sharded (uneven Shard(0), 2 gloo ranks), and on the GPU
with ElliotVL 2B's matrix shapes.

    python models/tests/test_muon_batched.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from torch.distributed.tensor import DTensor, distribute_tensor  # noqa: E402

from train.adamw_sr import _local, _zeros_like_sharded, stochastic_round_  # noqa: E402
from train.optimizer import MuonSR, adjust_lr_ratio, zeropower_via_newtonschulz  # noqa: E402


class PerMatrixMuonSR(MuonSR):
    """The previous implementation, verbatim."""

    @torch.no_grad()
    def step(self, closure=None):
        loss = super(MuonSR, self).step(closure)
        for group in self.param_groups:
            if group.get("use_muon"):
                for p in group["params"]:
                    if p.grad is not None:
                        self._muon_update(p, group["lr"], group["weight_decay"])
        return loss

    def _muon_update(self, p, lr: float, wd: float):
        state = self.state[p]
        if "momentum_buffer" not in state:
            state["momentum_buffer"] = _zeros_like_sharded(p)
        g, buf = _local(p.grad), _local(state["momentum_buffer"])
        buf.lerp_(g, 1 - self.momentum)
        u = torch.lerp(g, buf, self.momentum) if self.nesterov else buf
        u = u.to(torch.bfloat16)
        if isinstance(p, DTensor):
            full = DTensor.from_local(u, p.device_mesh, p.placements, shape=p.shape,
                                      stride=p.stride()).full_tensor()
            d = zeropower_via_newtonschulz(full, self.ns_steps)
            d = distribute_tensor(d, p.device_mesh, p.placements, src_data_rank=None).to_local()
        else:
            d = zeropower_via_newtonschulz(u, self.ns_steps)
        ratio = adjust_lr_ratio(self.adjust_lr_fn, *p.shape[-2:])
        lp = _local(p)
        w = lp.float()
        w.mul_(1 - lr * wd).add_(d.float(), alpha=-lr * ratio)
        if self.stochastic_round and lp.dtype == torch.bfloat16:
            stochastic_round_(w.view(-1), torch.randint_like(w.view(-1), 0, 1 << 16, dtype=torch.int32))
        lp.copy_(w)


# repeated shapes (buckets of 3 and 2), tall and wide, odd row counts (uneven over 2 ranks),
# a 3-D expert stack, one param left without a grad, and a 1-D AdamW param
SMALL = [(5, 4), (5, 4), (5, 4), (4, 7), (4, 7), (9, 3), (3, 6, 4), (7, 7)]
NO_GRAD = 7  # index into SMALL that never gets a grad


def _data(shapes, steps, dtype=torch.float32, device="cpu"):
    g = torch.Generator().manual_seed(0)
    init = [torch.randn(s, generator=g) for s in shapes] + [torch.randn(11, generator=g)]
    grads = [[torch.randn(t.shape, generator=g) for t in init] for _ in range(steps)]
    cast = lambda t: t.to(device, dtype)
    return [cast(t) for t in init], [[cast(t) for t in gs] for gs in grads]


def _run(cls, params, grads, wrap=lambda t: t, no_grad=None):
    *mats, norm = params
    opt = cls([{"params": mats, "lr": 1e-2, "weight_decay": 0.1, "use_muon": True},
               {"params": [norm], "lr": 1e-2, "weight_decay": 0.0}],
              betas=(0.9, 0.95), stochastic_round=False)
    for gs in grads:
        for i, (p, g) in enumerate(zip(params, gs)):
            p.grad = None if i == no_grad else wrap(g.clone())
        opt.step()
    return opt


def _compare(new, ref, atol, label):
    worst = max((a.double() - b.double()).abs().max().item() for a, b in zip(new, ref))
    print(f"{label:34s} max |batched - per-matrix| = {worst:.2e}  (atol {atol:.0e})")
    for i, (a, b) in enumerate(zip(new, ref)):
        torch.testing.assert_close(a, b, rtol=0, atol=atol, msg=f"{label}: param {i}")


def test_unsharded():
    for dtype in (torch.float32, torch.bfloat16):
        init, grads = _data(SMALL, 3, dtype)
        out = {}
        for cls in (MuonSR, PerMatrixMuonSR):
            params = [torch.nn.Parameter(t.clone()) for t in init]
            _run(cls, params, grads, no_grad=NO_GRAD)
            out[cls] = [p.detach() for p in params]
        assert torch.equal(out[MuonSR][NO_GRAD], init[NO_GRAD])  # no grad, no step
        _compare(out[MuonSR], out[PerMatrixMuonSR], 0.0, f"unsharded cpu {dtype}")


def _worker(rank, world, port, out):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.tensor import Shard

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=port, RANK=str(rank), WORLD_SIZE=str(world))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    mesh = init_device_mesh("cpu", (world,))
    shard = lambda t: distribute_tensor(t, mesh, [Shard(0)])
    init, grads = _data(SMALL, 3)
    res = {}
    for name, cls in (("new", MuonSR), ("ref", PerMatrixMuonSR)):
        params = [torch.nn.Parameter(shard(t.clone())) for t in init]
        opt = _run(cls, params, grads, shard, no_grad=NO_GRAD)
        assert isinstance(opt.state[params[0]]["momentum_buffer"], DTensor)
        res[name] = [p.detach().full_tensor() for p in params]
    if rank == 0:
        torch.save(res, out)
    dist.barrier()
    dist.destroy_process_group()


def test_sharded(tmp_path=None):
    out = str(Path(tmp_path or "/tmp") / f"muon_batched_{os.getpid()}.pt")
    torch.multiprocessing.spawn(_worker, args=(2, "29783", out), nprocs=2, join=True)
    res = torch.load(out)
    os.remove(out)
    _compare(res["new"], res["ref"], 0.0, "fsdp Shard(0) x2 gloo")
    init, grads = _data(SMALL, 3)
    params = [torch.nn.Parameter(t.clone()) for t in init]
    _run(MuonSR, params, grads, no_grad=NO_GRAD)
    _compare(res["new"], [p.detach() for p in params], 1e-6, "fsdp vs unsharded")


def test_gpu_2b_shapes():
    """ElliotVL 2B's Muon matrices (LLM GDN/attention/MLP + so400m ViT), fp32 masters."""
    if not torch.cuda.is_available():
        print("no GPU: skipped")
        return
    shapes = ([(6144, 2048)] * 2 + [(2048, 6144)] + [(4096, 2048)] * 2 + [(2048, 4096)]) * 4 \
        + ([(4304, 1152)] + [(1152, 4304)] + [(1152, 1152)] * 4) * 4
    init, grads = _data(shapes, 2, device="cuda")
    out, secs = {}, {}
    for cls in (MuonSR, PerMatrixMuonSR):
        params = [torch.nn.Parameter(t.clone()) for t in init]
        opt = _run(cls, params, grads[:1])  # warm-up step, then time the second
        for p, g in zip(params, grads[1]):
            p.grad = g.clone()
        torch.cuda.synchronize(); t = time.perf_counter()
        opt.step()
        torch.cuda.synchronize(); secs[cls] = time.perf_counter() - t
        out[cls] = [p.detach() for p in params]
    # bf16 GEMMs may pick other kernels batched vs one by one: compare against the update size
    step = max((a - b).abs().max().item() for a, b in zip(out[PerMatrixMuonSR], init))
    diff = max((a - b).abs().max().item() for a, b in zip(out[MuonSR], out[PerMatrixMuonSR]))
    print(f"gpu 2B shapes ({len(shapes)} matrices): max diff {diff:.2e} vs max update {step:.2e}; "
          f"step {secs[PerMatrixMuonSR] * 1e3:.1f} ms -> {secs[MuonSR] * 1e3:.1f} ms")
    assert diff <= 1e-2 * step, (diff, step)


if __name__ == "__main__":
    test_unsharded()
    test_sharded()
    test_gpu_2b_shapes()
    print("ok")
