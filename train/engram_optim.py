"""Row-sparse ("lazy") AdamW for row-sharded Engram tables
(`training.engram_table_optim = "lazy_adamw"`, needs `engram_sharding = "rows"`).

Dense AdamW walks every row of the table every step: ~28 B of HBM traffic per
parameter, even though a step reads a small fraction of the rows. Here only the
rows the step looked up are updated (TF LazyAdam / TorchRec row-wise Adam):

    m[r] = b1 m[r] + (1-b1) g[r]        v[r] = b2 v[r] + (1-b2) g[r]^2
    p[r] -= lr (m[r]/(1-b1^t)) / (sqrt(v[r]/(1-b2^t)) + eps)        for r touched

with the global step ``t``. Untouched rows keep ``p``, ``m`` and ``v`` as they
are; dense AdamW would decay ``m`` and ``v`` there and keep moving ``p`` along the
stale momentum. That is a change of training dynamics, hence opt-in.

The table stays in the main optimizer's param groups (same LR schedule, same
``state_dict`` keys ``step`` / ``exp_avg`` / ``exp_avg_sq``, so checkpoints
are interchangeable with dense AdamW). A step pre-hook does the sparse update and
then drops the table's ``grad``, which every AdamW implementation skips.
"""

import torch
from torch.distributed.tensor import DTensor

__all__ = ["ENGRAM_TABLE_OPTIMS", "register_lazy_adamw"]

ENGRAM_TABLE_OPTIMS = ("adamw", "lazy_adamw")


def _local(t: torch.Tensor) -> torch.Tensor:
    return t.to_local() if isinstance(t, DTensor) else t


@torch.no_grad()
def _lazy_adamw_update(param, state, *, lr, betas, eps, weight_decay):
    touched = param.touched_rows
    if not touched:
        param.grad = None
        return
    rows = torch.unique(torch.cat(touched))
    touched.clear()
    if not state:
        state["step"] = torch.zeros((), dtype=torch.float32, device=_local(param).device)
        state["exp_avg"] = torch.zeros_like(param, memory_format=torch.preserve_format)
        state["exp_avg_sq"] = torch.zeros_like(param, memory_format=torch.preserve_format)
    state["step"] += 1
    step = state["step"]
    beta1, beta2 = betas
    p = _local(param)
    g = _local(param.grad).index_select(0, rows).to(torch.float32)
    m_all, v_all = _local(state["exp_avg"]), _local(state["exp_avg_sq"])
    m = m_all.index_select(0, rows).float().mul_(beta1).add_(g, alpha=1 - beta1)
    v = v_all.index_select(0, rows).float().mul_(beta2).addcmul_(g, g, value=1 - beta2)
    m_all.index_copy_(0, rows, m.to(m_all.dtype))
    v_all.index_copy_(0, rows, v.to(v_all.dtype))
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    w = p.index_select(0, rows).float()
    if weight_decay:
        w.mul_(1 - lr * weight_decay)
    w.sub_(lr * (m / bc1) / ((v / bc2).sqrt_().add_(eps)))
    p.index_copy_(0, rows, w.to(p.dtype))
    param.grad = None  # the main AdamW skips it


def register_lazy_adamw(optimizer: torch.optim.Optimizer, tables) -> None:
    """Update ``tables`` (row-sharded Engram table parameters, which must be in
    ``optimizer``'s param groups) row-sparsely before each ``optimizer.step``."""
    tables = list(tables)
    if not tables:
        return
    group_of = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            group_of[id(p)] = group
    for t in tables:
        if id(t) not in group_of:
            raise ValueError("an Engram table is not in the optimizer's param groups")
        if not hasattr(t, "touched_rows"):
            t.touched_rows = []  # filled by RowShardedLookup

    def pre_step(opt, args, kwargs):
        for t in tables:
            if t.grad is None:
                t.touched_rows.clear()
                continue
            group = group_of[id(t)]
            _lazy_adamw_update(
                t, opt.state[t], lr=group["lr"], betas=group["betas"],
                eps=group["eps"], weight_decay=group["weight_decay"],
            )

    optimizer.register_step_pre_hook(pre_step)
