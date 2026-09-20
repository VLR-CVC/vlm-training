"""Deferred gradient reduction (gradient accumulation) survives an unused FSDP unit.

`train/step.py` reduces gradients once per accumulation window, with
`set_requires_gradient_sync(False)` on every micro-batch but the last. On torch < 2.15
that path crashes unless `train/parallel/fsdp.py` patches it -- this is the smallest
thing that fails if that patch is dropped or breaks.

Two ingredients are load-bearing and both come from the real model: an FSDP unit that
is wrapped but never runs forward (the vision tower on a text-only micro-batch), and
`reduce_dtype != param_dtype`, so the accumulate path does not return early. Remove
either and it passes on a torch that would otherwise fail.

    torchrun --nproc_per_node=1 models/tests/test_fsdp_nosync.py
    torchrun --nproc_per_node=2 models/tests/test_fsdp_nosync.py

One rank issues no collectives, so it only covers the crash; run two for the rest.
NOSYNC=0 reduces on every micro-batch instead -- the control, which must also pass.

On a box whose GPUs lack working P2P a bare 2-rank all_reduce hangs, with or without
any of this; export NCCL_P2P_DISABLE=1 there before concluding anything from a hang.
"""

import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import train.parallel.fsdp  # noqa: F401  (imported for its torch < 2.15 patch)

D = 256
MICRO_BATCHES = 2


class Block(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.lin = nn.Linear(d, d)

    def forward(self, x):
        return self.lin(x)


class Model(nn.Module):
    """`unused` stands in for the vision encoder: wrapped, but never called."""

    def __init__(self, d):
        super().__init__()
        self.used = Block(d)
        self.unused = Block(d)

    def forward(self, x):
        return self.used(x)


def main() -> None:
    torch.distributed.init_process_group("nccl")
    rank = torch.distributed.get_rank()
    world = torch.distributed.get_world_size()
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

    torch.manual_seed(0)
    model = Model(D).cuda()
    mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
    for unit in (model.used, model.unused):
        fully_shard(unit, mp_policy=mp)
    fully_shard(model, mp_policy=mp)
    # the trainer divides by the global token count itself, so gradients reduce as a
    # SUM -- which is what makes one reduction per window identical to one per
    # micro-batch (train/parallel/fsdp.py:disable_fsdp_gradient_division)
    for module in model.modules():
        if hasattr(module, "set_gradient_divide_factor"):
            module.set_gradient_divide_factor(1.0)

    defer = os.environ.get("NOSYNC", "1") == "1"
    for i in range(MICRO_BATCHES):
        last = i == MICRO_BATCHES - 1
        model.set_requires_gradient_sync(last or not defer)
        x = torch.randn(8, D, device="cuda", dtype=torch.bfloat16)
        model(x).sum().backward()

    grads = {n: p.grad for n, p in model.named_parameters() if p.grad is not None}
    assert set(grads) == {"used.lin.weight", "used.lin.bias"}, sorted(grads)
    for name, g in grads.items():
        assert torch.isfinite(g.to_local() if hasattr(g, "to_local") else g).all(), name

    if rank == 0:
        print(f"OK torch={torch.__version__} ranks={world} defer={defer} "
              f"grads={len(grads)}", flush=True)
    torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
