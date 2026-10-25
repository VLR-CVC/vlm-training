from __future__ import annotations

import math

import torch
from torch.distributed.tensor import DTensor


def _local(t):
    return t.to_local() if isinstance(t, DTensor) else t


def stochastic_round_(flat_f32: torch.Tensor, rand_i32: torch.Tensor) -> None:
    bits = flat_f32.view(torch.int32)
    bits.add_(rand_i32)
    bits.bitwise_and_(-65536)  # 0xFFFF0000 as a signed int32


class AdamWSR(torch.optim.Optimizer):
    """AdamW with FP32 moments and replica-consistent BF16 stochastic rounding"""

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
        self._rounding_generators: dict[torch.device, torch.Generator] = {}

    def load_state_dict(self, state_dict):
        # Optimizer.load_state_dict normally casts floating state to parameter
        # dtype. Exclude moments from that path: casting to BF16 and then back
        # to FP32 would already have destroyed precision in a new checkpoint.
        moment_names = ("exp_avg", "exp_avg_sq")
        stripped = dict(state_dict)
        stripped["state"] = {
            key: {name: value for name, value in state.items()
                  if name not in moment_names}
            for key, state in state_dict["state"].items()
        }
        super().load_state_dict(stripped)
        for saved_group, group in zip(state_dict["param_groups"], self.param_groups):
            for saved_id, p in zip(saved_group["params"], group["params"]):
                saved = state_dict["state"].get(saved_id, {})
                for name in moment_names:
                    if name in saved:
                        self.state[p][name] = _local(saved[name]).to(
                            device=_local(p).device, dtype=torch.float32, copy=True
                        )
        self._buckets = None
        self._rounding_generators.clear()

    def _rounding_generator(self, device, step, bucket_index):
        # Fixed optimizer-only seed; never use rank or the default RNG. Each
        # bucket is keyed by its stable index and checkpointed group step.
        # Replicas must have identical parameter/bucket layouts. Shards may
        # differ, but corresponding replicas of each shard use the same bits.
        generator = self._rounding_generators.get(device)
        if generator is None:
            generator = torch.Generator(device=device)
            self._rounding_generators[device] = generator
        seed = (42 + step * 6364136223846793005
                + bucket_index * 1442695040888963407) % (1 << 63)
        generator.manual_seed(seed)
        return generator

    def _build_buckets(self):
        """Group parameter *slices* into buckets of at most `bucket_elems`"""
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
                if cur and _local(cur[0][0]).dtype != lp.dtype:
                    raise RuntimeError(
                        f"AdamWSR bucket mixes dtypes: {_local(cur[0][0]).dtype} "
                        f"and {lp.dtype}"
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
                    s["exp_avg"] = torch.zeros_like(lp, dtype=torch.float32)
                    s["exp_avg_sq"] = torch.zeros_like(lp, dtype=torch.float32)

        for group in self.param_groups:
            group["step"] = group.get("step", 0) + 1

        for bucket_index, (gi, items) in enumerate(self._buckets):
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

            # Gather into FP32 scratch: parameters/gradients may be BF16;
            # persistent moments are already FP32. One launch per copy.
            torch._foreach_copy_(d_p, s_p)
            torch._foreach_copy_(d_g, s_g)
            torch._foreach_copy_(d_m, s_m)
            torch._foreach_copy_(d_v, s_v)

            bc1 = 1 - beta1**step
            bc2 = 1 - beta2**step

            f_p.mul_(1 - lr * wd)                 # decoupled weight decay
            f_m.lerp_(f_g, 1 - beta1)             # exp_avg
            f_g.mul_(f_g)                         # grad^2, in place
            f_v.lerp_(f_g, 1 - beta2)             # exp_avg_sq

            # Preserve FP32 moments before scratch is consumed by the update
            torch._foreach_copy_(s_m, d_m)
            torch._foreach_copy_(s_v, d_v)

            denom = f_g                           # reuse: grad^2 is spent
            torch.sqrt(f_v, out=denom)
            denom.div_(math.sqrt(bc2)).add_(eps)
            f_p.addcdiv_(f_m, denom, value=-lr / bc1)

            # conditional stochastic round
            if self.stochastic_round and _local(items[0][0]).dtype != torch.float32:
                torch.randint(
                    0, 1 << 16, (n,), out=f_rand, device=device, dtype=torch.int32,
                    generator=self._rounding_generator(device, step, bucket_index),
                )
                stochastic_round_(f_p, f_rand)
            torch._foreach_copy_(s_p, d_p)        # fp32 -> bf16, one launch

        return loss
