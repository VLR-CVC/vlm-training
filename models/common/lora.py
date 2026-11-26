"""Minimal LoRA for the decoder's linear layers (research code, `exp/e01`).

    model = build_meta(...)
    apply_lora(model, rank=64, alpha=128, targets=DEFAULT_TARGETS)   # on meta
    ... parallelize, materialize, load_hf ...
    init_lora(model)          # A ~ U(+-1/sqrt(in)), B = 0: step 0 is the base model
    ...
    merge_lora(model)         # W += (alpha / r) B A, then the model is plain again

`apply_lora` swaps the class of each targeted `Linear` to `LoRALinear` in place,
so the base weight keeps its name (`...wq.weight`) and the HF load / export paths
see the same keys as before; the adapters are two extra parameters on the same
module (`...wq.lora_A`, `...wq.lora_B`), which FSDP shards with the block they
live in. The fused SwiGLU `w13` gets ONE rank-r adapter on the fused
(gate|up) projection -- the state-dict hook splits only `weight`/`bias`, so it
must be merged before export (`merge_lora`), which `utils/dcp_to_hf.py` does.

Single-node, `tp_size = 1` only: with TP the adapters would need their own
colwise/rowwise layouts; they get the base weight's layout, which is only right
when that layout is trivial.
"""

from __future__ import annotations

import dataclasses
import math

import torch
import torch.nn.functional as F
from torch import nn

from models.common.nn_modules import Linear

# module-name suffixes (the last FQN component) adapted by default: full-attention
# q/k/v/o, GatedDeltaNet q/k/v/z in-projections and out-projection, and the
# SwiGLU MLP. GDN's in_proj_a / in_proj_b (1024 -> 16 heads) are left out: a
# rank-r adapter on a 16-wide output is mostly wasted parameters.
DEFAULT_TARGETS = (
    "wq", "wk", "wv", "wo",
    "in_proj_q", "in_proj_k", "in_proj_v", "in_proj_z", "out_proj",
    "w13", "w2",
)
LORA_SUFFIXES = (".lora_A", ".lora_B")


class LoRALinear(Linear):
    """`Linear` + a low-rank update: y = x W^T (+ b) + scale * (x A^T) B^T."""

    lora_rank: int
    lora_scale: float

    def forward(self, input: torch.Tensor) -> torch.Tensor:  # noqa: A002
        out = F.linear(input, self.weight, self.bias)
        return out + F.linear(F.linear(input, self.lora_A), self.lora_B) * self.lora_scale


def _is_target(name: str, module: nn.Module, targets) -> bool:
    if not isinstance(module, Linear) or isinstance(module, LoRALinear):
        return False
    if type(module) is not Linear:  # router gates, partial-bias linears: leave alone
        return False
    if not name.startswith("layers."):  # decoder blocks only (no ViT, Engram, MTP)
        return False
    return name.rsplit(".", 1)[-1] in targets


def apply_lora(model: nn.Module, *, rank: int, alpha: float, targets=DEFAULT_TARGETS) -> list[str]:
    """Turn every targeted decoder `Linear` into a `LoRALinear` (in place, any
    device; on meta before `parallelize`). Returns the adapted module names."""
    if rank <= 0:
        raise ValueError(f"lora rank must be > 0, got {rank}")
    adapted = []
    for name, module in list(model.named_modules()):
        if not _is_target(name, module, targets):
            continue
        w = module.weight
        module.__class__ = LoRALinear
        module.lora_rank = rank
        module.lora_scale = alpha / rank
        module.lora_A = nn.Parameter(torch.empty(rank, module.in_features, device=w.device, dtype=w.dtype))
        module.lora_B = nn.Parameter(torch.empty(module.out_features, rank, device=w.device, dtype=w.dtype))
        cfg = module._sharding_config
        if cfg is not None:
            # tp = 1: the weight's layout is trivial, reuse it for the adapters
            layout = cfg.state_shardings["weight"]
            module._sharding_config = dataclasses.replace(
                cfg, state_shardings={**cfg.state_shardings, "lora_A": layout, "lora_B": layout}
            )
        adapted.append(name)
    if not adapted:
        raise ValueError(f"no decoder Linear matched lora targets {list(targets)}")
    return adapted


def lora_modules(model: nn.Module):
    return [(n, m) for n, m in model.named_modules() if isinstance(m, LoRALinear)]


@torch.no_grad()
def init_lora(model: nn.Module) -> None:
    """A ~ U(-1/sqrt(in), 1/sqrt(in)) (PEFT's kaiming_uniform(a=sqrt(5))), B = 0.
    Elementwise, so it works on FSDP-sharded parameters."""
    for _, m in lora_modules(model):
        bound = 1.0 / math.sqrt(m.in_features)
        m.lora_A.uniform_(-bound, bound)
        m.lora_B.zero_()


@torch.no_grad()
def merge_lora(model: nn.Module) -> int:
    """W += scale * B A in place (fp32 math), drop the adapters and make every
    `LoRALinear` a plain `Linear` again. Unsharded models only. Returns the
    number of merged modules."""
    n = 0
    for _, m in lora_modules(model):
        delta = (m.lora_B.float() @ m.lora_A.float()) * m.lora_scale
        m.weight.copy_((m.weight.float() + delta).to(m.weight.dtype))
        del m.lora_A, m.lora_B
        m.__class__ = Linear
        n += 1
    return n


def lora_params_per_rank(model: nn.Module, targets=DEFAULT_TARGETS) -> int:
    """Trainable parameters per unit of rank: sum(in + out) over the target Linears
    (call on a model without adapters)."""
    return sum(m.in_features + m.out_features
               for name, m in model.named_modules() if _is_target(name, m, targets))
