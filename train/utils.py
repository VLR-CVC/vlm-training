import torch
import math
from torch.optim.lr_scheduler import LambdaLR
import os
import random
from train.logger import logger
import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor import DTensor
from train.config import Training as TrainArgs
from train.config import Model as ModelArgs

def generate_accumulation_pattern(target_multiplier: float, pattern_length: int = 100) -> list[int]:
    if target_multiplier < 1.0:
        raise ValueError("Multiplier must be >= 1.0")

    pattern = []
    current_cumulative = 0.0
    for i in range(pattern_length):
        next_cumulative = (i + 1) * target_multiplier
        steps_this_cycle = math.floor(next_cumulative) - math.floor(current_cumulative)

        pattern.append(int(steps_this_cycle))
        current_cumulative = next_cumulative

        if math.isclose(current_cumulative, round(current_cumulative)):
            break

    return pattern

def set_determinism(
    world_mesh,
    seed: int | None = None,
    deterministic: bool = True,
    debug_mode: bool = False,
) -> None:
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if not debug_mode:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if seed is None: seed = 42

    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    if not debug_mode:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.distributed.tensor._random.manual_seed(seed, world_mesh)

def set_trainable_parts(model_args: ModelArgs, model):
    """Freezing for `models/qwen3_5_tt` and `models/qwen3_vl_tt`, same policy as
    `set_model_qwen3_5` / `set_model_qwen3vl` under the torchtitan module names. Call
    on the meta model, before sharding."""
    for n, p in model.named_parameters():
        if n.startswith(("vision_encoder.merger.", "vision_encoder.deepstack_mergers.")):
            p.requires_grad = model_args.train_mlp
        elif n.startswith("vision_encoder."):
            p.requires_grad = model_args.train_vit
        else:
            p.requires_grad = model_args.train_llm
    return model

def dist_sum(x: torch.Tensor, group) -> torch.Tensor:
    """SUM all-reduce over `group`, no host sync.

    This replaces five separate `dist_mean`/`dist_max`/`dist_sum` calls, each of
    which was typed `-> float` and therefore ended in `.item()`. The collectives
    themselves were never the expensive part -- a handful of small all-reduces
    is microseconds. The `.item()` was: it forces the CPU to wait for the GPU to
    reach that point in the stream, which on a launch-bound model means the CPU
    stops running ahead and the GPU starts going idle between kernels.

    Returns a device tensor. The caller is expected to stage it to pinned memory
    and read it on a later step. A mean is a sum divided by the group size on the
    host side; there is no reason to spend a separate collective on `ReduceOp.AVG`.
    """
    return funcol.all_reduce(x, reduceOp=c10d.ReduceOp.SUM.name, group=group)

def dist_max(x: torch.Tensor, group) -> torch.Tensor:
    """MAX all-reduce over `group`, no host sync.

    Separate from `dist_sum` because the two have different scopes: token and
    FLOP counts are summed over the DP axis (TP ranks share a micro-batch, so a
    world sum would count it twice), while a saturating quantity like peak memory
    is only honest as a MAX over every rank -- an OOM on one rank of 512 is
    invisible in a DP-only reduction.
    """
    return funcol.all_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, group=group)

def dist_all_gather(x: torch.Tensor, group) -> torch.Tensor:
    """Gather a 1-D per-rank tensor across `group`.

    Returns a [world_size, x.numel()] tensor available on every rank. This is a
    collective, so it MUST be called on all ranks of `group`.
    """
    x = x.contiguous()
    return funcol.all_gather_tensor(x, gather_dim=0, group=group).reshape(-1, x.numel())

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
    total_steps = training_args.total_steps
    warmup_steps = training_args.warmup_steps
    min_lr_ratio = training_args.min_lr_ratio
    
    def lr_lambda(current_step):
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

def build_optimizer_param_groups(named_parameters, lr_by_group: dict, weight_decay: float, log: bool = False):
    """Bucket trainable params by part (mlp / vit / llm) and by weight-decay
    eligibility, returning the ``optimizer_grouped_parameters`` list for AdamW.

    Only >=2D tensors (weight matrices, embeddings) get weight decay; biases and
    norm scales are 1D and must never be pulled toward zero.
    """
    groups = ("mlp", "vit", "llm")
    decay_params = {g: [] for g in groups}
    no_decay_params = {g: [] for g in groups}

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

        (decay_params if p.dim() >= 2 else no_decay_params)[group].append(p)

    if log:
        for group in groups:
            logger.info(
                f"optimizer group {group} (lr={lr_by_group[group]}) -> "
                f"decay:{len(decay_params[group])} (wd={weight_decay}) "
                f"no_decay:{len(no_decay_params[group])} (wd=0.0)"
            )

    param_groups = []
    for group in groups:
        if decay_params[group]:
            param_groups.append({
                "params": decay_params[group],
                "lr": lr_by_group[group],
                "weight_decay": weight_decay,
            })
        if no_decay_params[group]:
            param_groups.append({
                "params": no_decay_params[group],
                "lr": lr_by_group[group],
                "weight_decay": 0.0,
            })
    return param_groups

def clip_grad_norm_mixed(parameters, max_norm: float, norm_type: float = 2.0):
    """`clip_grad_norm_` for a model whose grads live on more than one mesh."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.zeros(())

    # grouped by the mesh the *gradient* lives on, but keyed to the parameters:
    # `clip_grads_with_norm_` below takes parameters, not gradients.
    groups: dict[object, list[torch.Tensor]] = {}
    for p in params:
        key = p.grad.device_mesh if isinstance(p.grad, DTensor) else None
        groups.setdefault(key, []).append(p)

    if len(groups) == 1:
        # common case, no DTensors
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    # Multi-mesh case. Everything below stays on device: this used to do
    # `float(n.item())` once per group, which is a host sync in the middle of
    # the optimizer step, plus a `full_tensor()` all-gather per group. The
    # caller now stages the returned tensor and reads it a step later.
    total_sq = None
    for ps in groups.values():
        n = torch.nn.utils.get_total_norm([p.grad for p in ps], norm_type)
        if isinstance(n, DTensor):
            n = n.full_tensor()
        n_sq = n.to(torch.float32).pow(norm_type)
        total_sq = n_sq if total_sq is None else total_sq + n_sq

    total_norm = total_sq.pow(1.0 / norm_type)

    if max_norm > 0:
        # upstream's own clipping half, so the DTensor handling matches what
        # `clip_grad_norm_` does on the single-mesh path above. It clamps the
        # scale to <= 1 on device, so the `total_norm > max_norm` test that
        # used to gate this no longer needs a host value.
        for ps in groups.values():
            torch.nn.utils.clip_grads_with_norm_(ps, max_norm, total_norm)

    return total_norm

MASTER_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}

def cast_master_weights(model, master_dtype: str) -> torch.dtype:
    """Cast the optimizer's master copy of the *parameters* to ``master_dtype``.
    It only casts the weights and not other parameters/buffer (e.g. RoPE)"""
    if master_dtype not in MASTER_DTYPES:
        raise ValueError(
            f"master_dtype must be one of {sorted(MASTER_DTYPES)}, got {master_dtype!r}"
        )
    dtype = MASTER_DTYPES[master_dtype]
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return dtype

# torchao's AdamW variants, by the `adamw_impl` name that selects them. These are
# the only implementations that can do `adamw_stochastic_round` -- torch.optim.AdamW
# always rounds to nearest, which silently drops updates smaller than ~2^-9 of the
# weight once `master_dtype = "bfloat16"`.
#
# "torchao" is the *unquantized* one. `_AdamW` passes `block_size=inf`, so the
# `local_p.numel() % block_size == 0` test in `_new_buffer` never holds and the
# moments fall through to a plain `torch.zeros_like` in the parameter dtype. No
# `OptimState8bit` subclass, hence none of the `AttributeError: 'Tensor' object has
# no attribute 'codes'` that killed AdamW8bit on a dp x tp mesh.
TORCHAO_ADAMW = {
    "torchao": "_AdamW",
    "fp8": "AdamWFp8",
    "8bit": "AdamW8bit",
    "4bit": "AdamW4bit",
}
# "foreach_sr" is ours: `train/adamw_sr.py`. Same update as torchao's `_AdamW`
# and the same stochastic rounding, but bucketed through flat fp32 scratch
# instead of a Python loop with one compiled call per parameter -- ~18 kernel
# launches per bucket rather than ~500 per step. It is the only implementation
# that is both fast and safe at `master_dtype = "bfloat16"`.
ADAMW_IMPLS = ("foreach_sr", "foreach", "fused", "forloop", *TORCHAO_ADAMW)

def build_adamw(param_groups, lr: float, weight_decay: float, impl: str,
                stochastic_round: bool = False, betas=(0.9, 0.999), eps: float = 1e-8):
    """AdamW over ``param_groups``, picking the implementation by name."""
    if impl not in ADAMW_IMPLS:
        raise ValueError(
            f"adamw_impl must be one of {ADAMW_IMPLS}, got {impl!r}"
        )

    if impl == "foreach_sr":
        from train.adamw_sr import AdamWSR

        return AdamWSR(
            param_groups,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
            stochastic_round=stochastic_round,
        )

    if impl in TORCHAO_ADAMW:
        import torchao.optim as ao_optim

        return getattr(ao_optim, TORCHAO_ADAMW[impl])(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            bf16_stochastic_round=stochastic_round,
        )

    if stochastic_round:
        # warn, not raise: `adamw_stochastic_round` defaults to true, so every
        # config that picks a torch.optim impl would otherwise fail to build.
        logger.warning(
            f"adamw_stochastic_round ignored: adamw_impl={impl!r} is "
            "torch.optim.AdamW, which always rounds to nearest. Use "
            f"'foreach_sr' or one of {sorted(TORCHAO_ADAMW)} to get stochastic "
            "rounding. With master_dtype='bfloat16' this is not a preference: "
            "at lr 2e-5 a weight of typical magnitude never moves."
        )

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        foreach=impl == "foreach",
        fused=impl == "fused",
        weight_decay=weight_decay,
    )

# column order produced by the perf gather; True == higher is better
PERF_METRIC_NAMES = ("tps", "step_time", "fwd_bwd_time", "tflops", "mfu", "mem_gib")
PERF_HIGHER_IS_BETTER = (True, False, False, True, True, False)

def topk_metrics(gathered, top_k: int) -> dict:
    """Build the ``perf_topk/*`` dict: K slowest and K fastest ranks per metric,
    from the per-rank rows gathered across the world."""
    metrics = {}
    # One device-to-host copy instead of 96. The loop below ends in `.item()`
    # per value (6 metrics x k x 4), and on a CUDA tensor each of those drains
    # the stream. The gathered rows are a few KiB; topk over them on the host
    # is free.
    gathered = gathered.cpu()
    k = min(top_k, gathered.shape[0])
    for j, name in enumerate(PERF_METRIC_NAMES):
        col = gathered[:, j]
        higher_better = PERF_HIGHER_IS_BETTER[j]
        worst_v, worst_i = torch.topk(col, k, largest=not higher_better)
        best_v, best_i = torch.topk(col, k, largest=higher_better)
        for r in range(k):
            metrics[f"perf_topk/{name}_slow_{r}"] = worst_v[r].item()
            metrics[f"perf_topk/{name}_slow_{r}_rank"] = int(worst_i[r].item())
            metrics[f"perf_topk/{name}_fast_{r}"] = best_v[r].item()
            metrics[f"perf_topk/{name}_fast_{r}_rank"] = int(best_i[r].item())
    return metrics