import torch
import math
from collections.abc import Iterator
from torch.optim.lr_scheduler import LambdaLR
import os
import random
from typing import cast, Protocol

import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl

from train.logger import logger
from train.parallel.parallel_dims import ParallelDims
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
    for n, p in model.named_parameters():
        if n.startswith(("vision_encoder.merger.", "vision_encoder.deepstack_mergers.")):
            p.requires_grad = model_args.train_mlp
        elif n.startswith("vision_encoder."):
            p.requires_grad = model_args.train_vit
        else:
            p.requires_grad = model_args.train_llm
    return model

def dist_sum(x: torch.Tensor, group) -> torch.Tensor:
    return funcol.all_reduce(x, reduceOp=c10d.ReduceOp.SUM.name, group=group)

def dist_max(x: torch.Tensor, group) -> torch.Tensor:
    return funcol.all_reduce(x, reduceOp=c10d.ReduceOp.MAX.name, group=group)

def dist_all_gather(x: torch.Tensor, group) -> torch.Tensor:
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
                "params": decay_params[group], "lr": lr_by_group[group], "weight_decay": weight_decay,
            })
        if no_decay_params[group]:
            param_groups.append({
                "params": no_decay_params[group], "lr": lr_by_group[group], "weight_decay": 0.0,
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

    total_sq = None
    for ps in groups.values():
        n = torch.nn.utils.get_total_norm([p.grad for p in ps], norm_type)
        if isinstance(n, DTensor):
            n = n.full_tensor()
        n_sq = n.to(torch.float32).pow(norm_type)
        total_sq = n_sq if total_sq is None else total_sq + n_sq

    total_norm = total_sq.pow(1.0 / norm_type)

    if max_norm > 0:
        for ps in groups.values():
            torch.nn.utils.clip_grads_with_norm_(ps, max_norm, total_norm)

    return total_norm

# TODO: change name to `clip_grad_norm`
def clip_grad_norm_split_mtp(model, max_norm: float, norm_type: float = 2.0):
    """Clip the MTP depths on their own gradient norm, the rest of the model on
    its own, and return ``(main_norm, mtp_norm)``
    
    return MTP and MODEL norm has separate things
    """
    params = [p for p in model.parameters() if p.grad is not None]
    mtp_layers = getattr(model, "mtp_layers", None)
    if mtp_layers is None:
        return clip_grad_norm_mixed(params, max_norm, norm_type), None

    depth_params = {id(p) for p in mtp_layers.parameters()}
    main = [p for p in params if id(p) not in depth_params]
    depths = [p for p in params if id(p) in depth_params]
    if not depths:  # depths exist but are frozen
        return clip_grad_norm_mixed(main, max_norm, norm_type), None
    return (
        clip_grad_norm_mixed(main, max_norm, norm_type),
        clip_grad_norm_mixed(depths, max_norm, norm_type),
    )

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

TORCHAO_ADAMW = {
    "torchao": "_AdamW",
    "fp8": "AdamWFp8",
    "8bit": "AdamW8bit",
    "4bit": "AdamW4bit",
}
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

PERF_METRIC_NAMES = ("tps", "step_time", "fwd_bwd_time", "tflops", "mfu", "mem_gib")
PERF_HIGHER_IS_BETTER = (True, False, False, True, True, False)

def topk_metrics(gathered, top_k: int) -> dict:
    metrics = {}
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

class _MoELike(Protocol):
    load_balance_coeff: float | None
    tokens_per_expert_E: torch.Tensor  # noqa: N815
    expert_bias_E: torch.Tensor  # noqa: N815

def _iter_moe_layers(model: nn.Module) -> Iterator[tuple[nn.Module, _MoELike]]:
    layers = model.get_submodule("layers")
    assert isinstance(layers, (nn.ModuleDict, nn.ModuleList))
    for block in layers.children():
        if getattr(block, "moe_enabled", False):
            yield block, cast(_MoELike, block.moe)

def _is_recomputation_enabled(module: nn.Module) -> bool:
    return getattr(module, "checkpoint_impl", None) is CheckpointImpl.NO_REENTRANT

MOE_STAT_NAMES = (
    "max_over_mean",       # mean over layers; 1.0 is perfect balance, E is total collapse
    "max_over_mean_worst", # worst layer
    "worst_layer",         # which layer that was
    "cv",                  # coefficient of variation (std/mean), mean over layers
    "dead_expert_frac",    # experts that got zero tokens, mean over layers
    "dead_expert_frac_worst",
    "expert_bias_absmax",  # the hook's own state: is the bias saturating?
    "expert_bias_std",
)

def _moe_load_balance_stats(
    tokens_per_expert_E_by_layer: torch.Tensor, expert_bias_by_layer: torch.Tensor
) -> torch.Tensor:
    counts = tokens_per_expert_E_by_layer.float()
    mean_L = counts.mean(dim=-1).clamp_min(1e-9)
    max_over_mean_L = counts.max(dim=-1).values / mean_L
    cv_L = counts.std(dim=-1) / mean_L
    dead_L = (counts == 0).float().mean(dim=-1)
    return torch.stack(
        [
            max_over_mean_L.mean(),
            max_over_mean_L.max(),
            max_over_mean_L.argmax().float(),
            cv_L.mean(),
            dead_L.mean(),
            dead_L.max(),
            expert_bias_by_layer.abs().max(),
            expert_bias_by_layer.std(dim=-1).mean(),
        ]
    )

def _update_expert_bias(
    model: nn.Module,
    parallel_dims: ParallelDims,
    stats: dict | None = None,
) -> None:
    loss_mesh = parallel_dims.get_optional_mesh("loss")
    # TODO: Currently this sync is blocking (thus exposed) and happens on the
    # default compute stream. Need to assess if this is OK performance-wise.
    tokens_per_expert_E_list = []
    for block, moe in _iter_moe_layers(model):
        tokens_per_expert_E = moe.tokens_per_expert_E
        if _is_recomputation_enabled(block):
            tokens_per_expert_E = tokens_per_expert_E // 2
        tokens_per_expert_E_list.append(tokens_per_expert_E)

    if not tokens_per_expert_E_list:
        return

    tokens_per_expert_E_by_layer = torch.vstack(tokens_per_expert_E_list)

    if parallel_dims.ep_enabled and parallel_dims.tp > 1:
        torch.distributed.all_reduce(
            tokens_per_expert_E_by_layer, group=parallel_dims.get_dense_tp_mesh().get_group(),
        )
    if loss_mesh is not None:
        torch.distributed.all_reduce(
            tokens_per_expert_E_by_layer, group=loss_mesh.get_group(), op=torch.distributed.ReduceOp.SUM,
        )
    moe_layer_idx = 0
    with torch.no_grad():
        for _block, moe in _iter_moe_layers(model):
            load_balance_coeff = moe.load_balance_coeff
            assert load_balance_coeff is not None

            tokens_per_expert_E = tokens_per_expert_E_by_layer[moe_layer_idx].float()
            moe_layer_idx += 1

            # update the expert bias
            # this is not exactly the same as https://arxiv.org/pdf/2408.15664 proposed
            expert_bias_delta_E = load_balance_coeff * torch.sign(tokens_per_expert_E.mean() - tokens_per_expert_E)
            expert_bias_delta_E = expert_bias_delta_E - expert_bias_delta_E.mean()
            moe.expert_bias_E.add_(expert_bias_delta_E)
            moe.tokens_per_expert_E.zero_()

        if stats is not None:
            stats["vec"] = _moe_load_balance_stats(
                tokens_per_expert_E_by_layer,
                torch.vstack([moe.expert_bias_E for _b, moe in _iter_moe_layers(model)]),
            )

def register_moe_load_balancing_hook(
    optimizer: torch.optim.Optimizer,
    model: nn.Module,
    parallel_dims: ParallelDims,
    stats: dict | None = None,
) -> bool:
    """Register an optimizer step pre-hook for MoE auxiliary-loss-free load
    balancing (https://arxiv.org/abs/2408.15664). Returns whether it registered"""
    moe_layers = list(_iter_moe_layers(model))
    if not moe_layers:
        return False

    enabled = moe_layers[0][1].load_balance_coeff is not None
    for _block, moe in moe_layers[1:]:
        if (moe.load_balance_coeff is not None) != enabled:
            raise ValueError(
                "MoE load_balance_coeff must be configured consistently across "
                "all MoE layers. Either set it for every MoE layer or leave it "
                "unset for all MoE layers."
            )
    if not enabled:
        return False

    optimizer.register_step_pre_hook(
        lambda *args, **kwargs: _update_expert_bias(
            model, parallel_dims=parallel_dims, stats=stats
        )
    )
    logger.info(
        f"MoE load balancing: expert-bias hook on {len(moe_layers)} layers "
        f"(coeff {moe_layers[0][1].load_balance_coeff})"
    )
    return True