"""AdamW with stochastic rounding into bfloat16 parameters.

Why this exists
---------------
With ``master_dtype = "bfloat16"`` the optimizer's copy of the weights *is* the
bf16 tensor, and bf16 carries 7 explicit mantissa bits. At the model's typical
weight scale (1/sqrt(4096) = 2^-6) one ULP is 2^-13 = 1.22e-4, so half a ULP is
6.1e-5 -- three times ``lr_llm = 2e-5``. Under round-to-nearest an Adam update of
that size lands back on the same bf16 value and the weight never moves: measured,
200 consecutive same-direction steps at lr 2e-5 move 0 of 4096 elements.

Stochastic rounding fixes it by rounding away from zero with probability equal to
the fractional distance: a 2e-5 update against a 1.22e-4 ULP moves the weight one
ULP about 16% of the time, so the drift is right in expectation.

``torchao.optim._AdamW`` does this correctly but costs ~0.53 s/step on the 9B
model, because it loops over parameters in Python and calls a separately
``torch.compile``d function per parameter -- ~500 guard-chain evaluations and
launches, none of which get faster with more work per tensor.

Design
------
Parameters are bucketed, and each bucket is processed through **flat** fp32
scratch buffers: gather with one ``_foreach_copy_``, do the Adam math with plain
tensor ops on a single contiguous tensor, stochastically round, scatter back with
one ``_foreach_copy_``. That is ~18 kernel launches per bucket *regardless of how
many tensors are in it*.

Flat buffers rather than ``_foreach_*`` throughout for one specific reason: there
is no ``torch._foreach_bitwise_and_``, and stochastic rounding is a bit-level
operation. On a flat tensor it is three kernels; per-tensor it would be four
launches times the parameter count, which is the cost we are trying to escape.

Moments are stored in the parameter dtype (bf16), matching ``torchao._AdamW``, so
the footprint stays at 8 B/param: 2 param + 2 grad + 2 + 2.

Memory: the scratch is 4 flat fp32 buffers plus an int32 of random bits, i.e.
``20 bytes per bf16 parameter in the largest bucket``. ``bucket_mb`` is in
megabytes of *parameter* bytes, so the default 64 costs ~640 MB of scratch,
allocated once and reused across buckets and steps.
"""

from __future__ import annotations

import math

import torch
from torch.distributed.tensor import DTensor


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def stochastic_round_(flat_f32: torch.Tensor, rand_i32: torch.Tensor) -> None:
    """Round ``flat_f32`` in place to values exactly representable in bf16,
    stochastically, so that ``E[result] == input``.

    An fp32 number is ``[a31..a16][a15..a0]``; the bf16 neighbours are the upper
    half with the lower half zeroed (towards zero) and that plus one (away from
    zero). Adding a uniform 16-bit value before truncating carries into the upper
    half with probability ``[a15..a0] / 2^16``, which is exactly the fractional
    distance to the neighbour away from zero.

    Sign needs no special handling: the fp32 bit pattern is monotone in magnitude
    within each sign, so "carry into the upper half" means "away from zero" for
    both. Same construction as ``torchao.optim.quant_utils._fp32_to_bf16_sr``,
    done on one flat tensor instead of per parameter.

    The result still has fp32 dtype, but its low 16 mantissa bits are zero, so
    the later cast to bf16 is exact and its rounding mode is irrelevant.
    """
    bits = flat_f32.view(torch.int32)
    bits.add_(rand_i32)
    bits.bitwise_and_(-65536)  # 0xFFFF0000 as a signed int32


class AdamWSR(torch.optim.Optimizer):
    """AdamW that writes bf16 parameters with stochastic rounding.

    Numerically equivalent to ``torchao.optim._AdamW`` (same update, same fp32
    intermediates, same moment dtype), without the per-parameter Python loop.
    Set ``stochastic_round=False`` to get plain round-to-nearest, which is only
    useful for parity testing -- at the learning rates this codebase uses it does
    not train, see the module docstring.
    """

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        *,
        stochastic_round: bool = True,
        bucket_mb: int = 64,
    ):
        if lr < 0.0:
            raise ValueError(f"lr must be >= 0, got {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"betas must be in [0, 1), got {betas}")
        if eps < 0.0:
            raise ValueError(f"eps must be >= 0, got {eps}")
        if bucket_mb <= 0:
            raise ValueError(f"bucket_mb must be > 0, got {bucket_mb}")

        super().__init__(
            params,
            dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay),
        )
        self.stochastic_round = stochastic_round
        self.bucket_elems = bucket_mb * 1024 * 1024 // 2  # bf16 params per bucket
        self._scratch: dict[torch.device, list[torch.Tensor]] = {}
        self._buckets: list | None = None


    # ---------------------------------------------------------------- setup --
    def _build_buckets(self):
        """Group parameter *slices* into buckets of at most `bucket_elems`.

        Slices, not whole tensors: `lm_head` and `embed_tokens` are 254M-element
        shards at tp=4, and a bucket built around one of those would need ~5 GB
        of fp32 scratch. Every operation below is elementwise, so a parameter can
        be split across buckets with no effect on the result -- it just has to be
        split consistently across p, grad, exp_avg and exp_avg_sq, which the
        `(param, offset, length)` triple guarantees.

        Buckets never span param groups: `lr` and `weight_decay` differ per group
        (the trainer runs separate rates for llm / mlp / vit) and the math applies
        them to a whole bucket at once.
        """
        buckets = []
        for gi, group in enumerate(self.param_groups):
            cur, cur_elems = [], 0
            for p in group["params"]:
                if p.grad is None:
                    continue
                lp = _local(p)
                if not lp.is_contiguous():
                    raise RuntimeError(
                        f"AdamWSR needs contiguous local shards; got a "
                        f"{tuple(lp.shape)} parameter with strides {lp.stride()}"
                    )
                n, off = lp.numel(), 0
                while off < n:
                    take = min(n - off, self.bucket_elems - cur_elems)
                    cur.append((p, off, take))
                    cur_elems += take
                    off += take
                    if cur_elems >= self.bucket_elems:
                        buckets.append((gi, cur))
                        cur, cur_elems = [], 0
            if cur:
                buckets.append((gi, cur))
        return buckets

    def _get_scratch(self, device, n_elems):
        """Four fp32 buffers plus one int32 of random bits, allocated once at the
        bucket bound and reused for every bucket and every step."""
        if device not in self._scratch:
            cap = self.bucket_elems
            self._scratch[device] = [
                torch.empty(cap, dtype=torch.float32, device=device)
                for _ in range(4)
            ] + [torch.empty(cap, dtype=torch.int32, device=device)]
        return [b[:n_elems] for b in self._scratch[device]]

    @staticmethod
    def _slices(flat, items, pick):
        """Matching 1-D windows into `flat` and into each source tensor.

        Returns (dst_views, src_views) for `_foreach_copy_`, both flat, so the
        copy is a single multi-tensor launch in each direction.
        """
        dst, src, pos = [], [], 0
        for p, off, n in items:
            dst.append(flat[pos : pos + n])
            src.append(pick(p).view(-1).narrow(0, off, n))
            pos += n
        return dst, src

    # ----------------------------------------------------------------- step --
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if self._buckets is None:
            self._buckets = self._build_buckets()
            for p, _, _ in (it for _, items in self._buckets for it in items):
                s = self.state[p]
                if "exp_avg" not in s:
                    lp = _local(p)
                    s["exp_avg"] = torch.zeros_like(lp)
                    s["exp_avg_sq"] = torch.zeros_like(lp)

        # One counter per param group rather than per parameter: every parameter
        # steps together, so the numbers would be identical, and `param_groups`
        # is what `state_dict()` serializes -- so this survives a checkpoint
        # without any custom (de)serialization.
        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

        for gi, items in self._buckets:
            group = self.param_groups[gi]
            beta1, beta2 = group["betas"]
            lr, eps, wd = group["lr"], group["eps"], group["weight_decay"]
            step = group["step"]

            n = sum(it[2] for it in items)
            device = _local(items[0][0]).device
            f_p, f_g, f_m, f_v, f_rand = self._get_scratch(device, n)

            d_p, s_p = self._slices(f_p, items, lambda p: _local(p))
            d_g, s_g = self._slices(f_g, items, lambda p: _local(p.grad))
            d_m, s_m = self._slices(f_m, items, lambda p: self.state[p]["exp_avg"])
            d_v, s_v = self._slices(f_v, items, lambda p: self.state[p]["exp_avg_sq"])

            # gather: one multi-tensor launch each, bf16 -> fp32
            torch._foreach_copy_(d_p, s_p)
            torch._foreach_copy_(d_g, s_g)
            torch._foreach_copy_(d_m, s_m)
            torch._foreach_copy_(d_v, s_v)

            # --- Adam math, on flat contiguous tensors -----------------------
            # Matches torchao.optim._AdamW's `single_param_adam` term for term.
            bc1 = 1 - beta1**step
            bc2 = 1 - beta2**step

            f_p.mul_(1 - lr * wd)                 # decoupled weight decay
            f_m.lerp_(f_g, 1 - beta1)             # exp_avg
            f_g.mul_(f_g)                         # grad^2, in place
            f_v.lerp_(f_g, 1 - beta2)             # exp_avg_sq

            # moments back to bf16 before f_m / f_v are consumed further
            torch._foreach_copy_(s_m, d_m)
            torch._foreach_copy_(s_v, d_v)

            denom = f_g                           # reuse: grad^2 is spent
            torch.sqrt(f_v, out=denom)
            denom.div_(math.sqrt(bc2)).add_(eps)
            f_p.addcdiv_(f_m, denom, value=-lr / bc1)

            # --- write back --------------------------------------------------
            if self.stochastic_round:
                torch.randint(
                    0, 1 << 16, (n,), out=f_rand, device=device, dtype=torch.int32
                )
                stochastic_round_(f_p, f_rand)
            torch._foreach_copy_(s_p, d_p)        # fp32 -> bf16, one launch

        return loss
