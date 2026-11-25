from __future__ import annotations

import math
import re

import torch
from torch.distributed.tensor import DTensor, Replicate, Shard, distribute_tensor
from torch.optim.lr_scheduler import LambdaLR

from train.config import Training as TrainArgs
from train.logger import logger
from train.utils import register_moe_load_balancing_hook

def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t

def _zeros_like_sharded(p: torch.Tensor) -> torch.Tensor:
    """Zeros shaped like ``p``'s local shard, wrapped back into a DTensor when ``p`` is one.

    DCP records sharding only for DTensors: state kept as a plain local shard is saved with
    ``chunks=1`` and no offsets, so every rank writes the same key, one rank's values survive,
    and uneven shards fail to load (``Size mismatch between saved torch.Size([1, 1152]) and
    current: torch.Size([5, 1152])``). ``shape``/``stride`` are explicit: inferring them costs
    a collective and gets uneven shards wrong.
    """
    z = torch.zeros_like(_local(p))
    if isinstance(p, DTensor):
        return DTensor.from_local(z, p.device_mesh, p.placements, shape=p.shape, stride=p.stride())
    return z

def create_optimizer(model, training_args: TrainArgs, parallel_dims, moe_stats=None, log: bool = False):
    """Optimizer over `model`'s trainable parameters and its LR scheduler."""
    t = training_args
    if t.optimizer not in ("adamw", "muon"):
        raise ValueError(f"optimizer must be 'adamw' or 'muon', got {t.optimizer!r}")
    groups = build_optimizer_param_groups(
        model.named_parameters(), {"mlp": t.lr_mlp, "vit": t.lr_vit, "llm": t.lr_llm},
        t.weight_decay, log=log, muon=t.optimizer == "muon",
    )
    if t.optimizer == "muon":
        optimizer = MuonSR(
            groups, lr=t.lr_llm, betas=tuple(t.adam_betas), eps=t.eps, weight_decay=t.weight_decay,
            momentum=t.muon_momentum,
            nesterov=t.muon_nesterov, ns_steps=t.muon_ns_steps, adjust_lr_fn=t.muon_adjust_lr,
        )
    else:
        optimizer = build_adamw(
            groups, lr=t.lr_llm, weight_decay=t.weight_decay, impl=t.adamw_impl,
            betas=t.adam_betas, eps=t.eps,
        )
    register_moe_load_balancing_hook(optimizer, model, parallel_dims, stats=moe_stats)
    return optimizer, get_scheduler(optimizer, t)

def create_WSD_scheduler(optimizer, training_args: TrainArgs):
    total_steps = training_args.total_steps
    warmup_steps = training_args.warmup_steps
    
    if training_args.wsd_decay_steps > 0:
        decay_steps = training_args.wsd_decay_steps
    else:
        decay_steps = int(training_args.wsd_decay_ratio * total_steps)
    stable_steps = total_steps - warmup_steps - decay_steps
    
    def lr_lambda(current_step):
        # warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # stable
        if current_step < warmup_steps + stable_steps:
            return 1.0
        
        # decay
        decay_current = current_step - (warmup_steps + stable_steps)
        progress = float(decay_current) / float(max(1, decay_steps))
        
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return LambdaLR(optimizer, lr_lambda)

def create_cosine_scheduler(optimizer, training_args: TrainArgs):
    total_steps = training_args.cosine_schedule_steps or training_args.total_steps
    warmup_steps = training_args.warmup_steps
    min_lr_ratio = training_args.min_lr_ratio
    offset = training_args.cosine_schedule_offset

    def lr_lambda(current_step):
        current_step += offset
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

def get_scheduler(optimizer, training_args: TrainArgs):
    if training_args.scheduler_type.lower() == "wsd":
        return create_WSD_scheduler(optimizer, training_args)
    elif training_args.scheduler_type.lower() == "cosine":
        return create_cosine_scheduler(optimizer, training_args)
    else:
        raise ValueError(f"Unknown scheduler type: {training_args.scheduler_type}")

MUON_EXCLUDE = re.compile(r"embed|lm_head|conv")

def build_optimizer_param_groups(named_parameters, lr_by_group: dict, weight_decay: float, log: bool = False,
                                 muon: bool = False):
    """Bucket trainable params by part (mlp / vit / llm) and by weight-decay
    eligibility, returning the ``optimizer_grouped_parameters`` list for AdamW.

    Only >=2D tensors (weight matrices, embeddings) get weight decay; biases and
    norm scales are 1D and must never be pulled toward zero.

    ``muon=True`` moves the decayed matrices, except embeddings / lm_head / conv
    kernels (``MUON_EXCLUDE``), into extra groups flagged ``use_muon``.
    """
    groups = ("mlp", "vit", "llm")
    decay_params = {g: [] for g in groups}
    no_decay_params = {g: [] for g in groups}
    muon_params = {g: [] for g in groups}

    for n, p in named_parameters:
        if not p.requires_grad:
            continue
        if ("visual.merger" in n or "visual.deepstack_merger_list" in n
                or "vision_encoder.merger" in n or "vision_encoder.deepstack_mergers" in n):
            group = "mlp"
        elif "visual.patch_embed" in n or "visual.blocks" in n or "vision_encoder." in n:
            group = "vit"
        else:
            group = "llm"

        if muon and p.dim() >= 2 and not MUON_EXCLUDE.search(n):
            muon_params[group].append(p)
        else:
            (decay_params if p.dim() >= 2 else no_decay_params)[group].append(p)

    if log:
        for group in groups:
            logger.info(
                f"optimizer group {group} (lr={lr_by_group[group]}) -> "
                f"decay:{len(decay_params[group])} (wd={weight_decay}) "
                f"no_decay:{len(no_decay_params[group])} (wd=0.0) muon:{len(muon_params[group])}"
            )

    param_groups = []
    for group in groups:
        if muon_params[group]:
            param_groups.append({
                "params": muon_params[group], "lr": lr_by_group[group], "weight_decay": weight_decay,
                "use_muon": True,
            })
        if decay_params[group]:
            param_groups.append({
                "params": decay_params[group], "lr": lr_by_group[group], "weight_decay": weight_decay,
            })
        if no_decay_params[group]:
            param_groups.append({
                "params": no_decay_params[group], "lr": lr_by_group[group], "weight_decay": 0.0,
            })
    return param_groups

TORCHAO_ADAMW = {
    "torchao": "_AdamW",
    "fp8": "AdamWFp8",
    "8bit": "AdamW8bit",
    "4bit": "AdamW4bit",
}
ADAMW_IMPLS = ("foreach", "fused", "forloop", *TORCHAO_ADAMW)

def build_adamw(param_groups, lr: float, weight_decay: float, impl: str,
                betas=(0.9, 0.999), eps: float = 1e-8):
    """AdamW over ``param_groups``, picking the implementation by name."""
    if impl not in ADAMW_IMPLS:
        raise ValueError(f"adamw_impl must be one of {ADAMW_IMPLS}, got {impl!r}")
    if impl in TORCHAO_ADAMW:
        import torchao.optim as ao_optim

        return getattr(ao_optim, TORCHAO_ADAMW[impl])(param_groups, lr=lr, weight_decay=weight_decay)
    return torch.optim.AdamW(
        param_groups, lr=lr, betas=tuple(betas), eps=eps, weight_decay=weight_decay,
        foreach=impl == "foreach", fused=impl == "fused",
    )

NS_COEFFICIENTS = (3.4445, -4.7750, 2.0315)
ADJUST_LR_FNS = ("original", "match_rms_adamw", "spectral_unclamped")

def adjust_lr_ratio(adjust_lr_fn: str, rows: int, columns: int) -> float:
    if adjust_lr_fn == "original":
        return math.sqrt(max(1.0, rows / columns))
    if adjust_lr_fn == "match_rms_adamw":
        return 0.2 * math.sqrt(max(rows, columns))
    if adjust_lr_fn == "spectral_unclamped":
        return math.sqrt(rows / columns)
    raise ValueError(f"muon_adjust_lr must be one of {ADJUST_LR_FNS}, got {adjust_lr_fn!r}")

def zeropower_via_newtonschulz(update: torch.Tensor, ns_steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """Approximate polar factor of each trailing 2-D matrix, in bf16."""
    x = update.to(torch.bfloat16, copy=True)
    transposed = x.shape[-2] > x.shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    x = x / x.norm(dim=(-2, -1), keepdim=True).clamp_min(eps)
    a, b, c = NS_COEFFICIENTS
    shape = x.shape
    x = x.reshape(-1, *shape[-2:])  # fused baddbmm, same bf16 rounding as torchtitan's
    for _ in range(ns_steps):
        gram = x @ x.transpose(-2, -1)
        x = torch.baddbmm(x, torch.baddbmm(gram, gram, gram, beta=b, alpha=c), x, beta=a)
    x = x.reshape(shape)
    return x.transpose(-2, -1) if transposed else x

class MuonSR(torch.optim.AdamW):
    """Param groups with ``use_muon=True`` get Muon; the rest torch's AdamW. fp32 parameters."""

    def __init__(self, params, *, momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5,
                 adjust_lr_fn: str = "match_rms_adamw", **adamw_kwargs):
        adjust_lr_ratio(adjust_lr_fn, 1, 1)  # validate early
        super().__init__(params, **adamw_kwargs)
        self.momentum, self.nesterov, self.ns_steps, self.adjust_lr_fn = momentum, nesterov, ns_steps, adjust_lr_fn

    @torch.no_grad()
    def step(self, closure=None):
        muon = [g for g in self.param_groups if g.get("use_muon")]
        for g in muon:  # hide Muon groups from AdamW
            g["_params"], g["params"] = g["params"], []
        try:
            loss = super().step(closure)
        finally:
            for g in muon:
                g["params"] = g.pop("_params")
        for group in muon:
            params = [p for p in group["params"] if p.grad is not None]
            if params:
                self._muon_group(params, group["lr"], group["weight_decay"])
        return loss

    def _muon_group(self, params, lr: float, wd: float):
        """Muon over one param group, batched: the arithmetic of a per-matrix update, but
        one all-gather and one batched Newton-Schulz per bucket of same-shaped matrices
        (one per matrix was ~360 collectives and ~14k small kernels per step on ElliotVL 2B)."""
        for p in params:
            if "momentum_buffer" not in self.state[p]:
                self.state[p]["momentum_buffer"] = _zeros_like_sharded(p)
        gs = [_local(p.grad) for p in params]
        bufs = [_local(self.state[p]["momentum_buffer"]) for p in params]
        torch._foreach_lerp_(bufs, gs, 1 - self.momentum)
        us = torch._foreach_lerp(gs, bufs, self.momentum) if self.nesterov else bufs

        buckets = {}
        for p, u in zip(params, us):
            layout = (p.device_mesh, tuple(p.placements)) if isinstance(p, DTensor) else None
            # only plain Shard/Replicate stack into one DTensor (Shard(d) -> Shard(d+1));
            # anything else (e.g. _StridedShard under TP+FSDP) is gathered matrix by matrix
            plain = layout is None or all(type(pl) in (Shard, Replicate) for pl in layout[1])
            key = (tuple(p.shape), tuple(u.shape), p.dtype, layout, None if plain else id(p))
            buckets.setdefault(key, []).append((p, u))

        for (shape, _, _, layout, alone), items in buckets.items():
            u = torch.stack([u.to(torch.bfloat16) for _, u in items])  # (k, *local shape)
            if layout is None:
                d = zeropower_via_newtonschulz(u, self.ns_steps)
            elif alone is not None:  # the per-matrix path, k = 1
                p = items[0][0]
                full = DTensor.from_local(u[0], *layout, shape=p.shape, stride=p.stride()).full_tensor()
                d = zeropower_via_newtonschulz(full, self.ns_steps)
                d = distribute_tensor(d, *layout, src_data_rank=None).to_local().unsqueeze(0)
            else:
                mesh, placements = layout
                stacked = tuple(Shard(pl.dim + 1) if isinstance(pl, Shard) else pl for pl in placements)
                full_shape = (len(items), *shape)
                full = DTensor.from_local(u, mesh, stacked, shape=full_shape,
                                          stride=torch.empty(full_shape, device="meta").stride()).full_tensor()
                d = zeropower_via_newtonschulz(full, self.ns_steps)
                # every rank holds the full result: take the local shards, no communication
                d = distribute_tensor(d, mesh, stacked, src_data_rank=None).to_local()
            ratio = adjust_lr_ratio(self.adjust_lr_fn, *shape[-2:])
            lps = [_local(p) for p, _ in items]
            if lps[0].dtype != torch.float32:
                raise ValueError(f"MuonSR needs fp32 parameters, got {lps[0].dtype}")
            torch._foreach_mul_(lps, 1 - lr * wd)
            torch._foreach_add_(lps, [x.float() for x in d.unbind(0)], alpha=-lr * ratio)
