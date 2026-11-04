"""Is our fp8 grouped-GEMM integration correct, on hardware that has the kernel?

    srun ... python models/tests/fp8_grouped_mm_check.py      # GH200 (sm_90)

Not a pytest file: `torch._scaled_grouped_mm` needs compute capability 9.0/10.0,
which no card on the dev box has.

WHAT THIS DOES NOT ASK: "is fp8 close to bf16". fp8 e4m3 carries 3 mantissa
bits, so its quantization step is ~6% and an aggregate deviation of a few
percent from bf16 is the FORMAT, not a defect. Gating on a tolerance there just
measures how arbitrary the tolerance was -- which is exactly what the first two
versions of this script did.

WHAT IT ASKS INSTEAD, all crisp pass/fail and all about the integration rather
than torchao's kernel:

  1. padding invariance -- real tokens must produce identical results however
     much the dispatcher pads. This is THE property our integration can break:
     `TorchAOTokenDispatcher` pads the groups and `offs` is a cumsum of the
     padded counts, so a mismatch between the two silently mixes padded rows
     into an expert's output.
  2. direction preserved -- cosine against bf16, forward and backward. A wrong
     scale or a transposed operand destroys the cosine while leaving the
     magnitude plausible.
  3. unbiased error -- a correct rounding scheme has |mean error| far below the
     error's own spread. A systematic offset means a broken scale.
  4. finite -- no NaN/Inf, including at the minimum 16-token group.

Deviation from bf16 is printed for information at every shape, never asserted.
"""

from __future__ import annotations

import torch

from models.common.moe import GroupedExperts

# 35B-A3B expert shapes: hidden 2048, moe_intermediate 512
D, F, E = 2048, 512, 8
PAD = 16


def _experts(kind: str):
    return GroupedExperts.Config(
        dim=D, hidden_dim=F, num_experts=E, param_init={}, grouped_mm=kind
    ).build()


def _run(A, weight, offs, kind):
    a = A.clone().requires_grad_(True)
    w = weight.clone().requires_grad_(True)
    out = _experts(kind)._grouped_mm(A=a, weight_EOI=w, offs=offs)
    out.float().pow(2).sum().backward()
    return out.float(), a.grad.float()


def _offsets(counts):
    return (
        torch.tensor(counts, dtype=torch.int32, device="cuda")
        .cumsum(0)
        .to(torch.int32)
    )


def compare(counts: list[int], label: str) -> None:
    assert all(c % PAD == 0 for c in counts), "groups must arrive pre-padded"
    torch.manual_seed(0)
    total = sum(counts)
    A = torch.randn(total, D, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(E, F, D, device="cuda", dtype=torch.bfloat16) * 0.02
    offs = _offsets(counts)

    out8, grad8 = _run(A, w, offs, "fp8_rowwise")
    out16, grad16 = _run(A, w, offs, "bf16")

    assert torch.isfinite(out8).all(), f"{label}: non-finite forward"
    assert torch.isfinite(grad8).all(), f"{label}: non-finite backward"

    def report(x, y, name):
        err = x - y
        cos = torch.nn.functional.cosine_similarity(
            x.flatten().unsqueeze(0), y.flatten().unsqueeze(0)
        ).item()
        frob = (err.norm() / y.norm()).item()
        bias = (err.mean() / err.std()).item()  # mean relative to its own spread
        print(
            f"    {name}  cos {cos:.6f}  frob {frob:.3e}  bias(mean/std) {bias:+.4f}",
            flush=True,
        )
        # direction must survive; a transposed operand or dead scale kills this
        assert cos > 0.99, f"{label} {name}: cosine {cos} -- not just quantization"
        # rounding must be unbiased; a broken scale shows up as a systematic shift
        assert abs(bias) < 0.1, f"{label} {name}: error biased, mean/std {bias}"

    print(f"{label}: groups {counts} total {total}", flush=True)
    report(out8, out16, "fwd")
    report(grad8, grad16, "bwd")


def padding_invariance() -> None:
    """The property our integration owns: more padding must not change a real
    token's output. Same 48 real tokens per expert, padded to 64 then to 128."""
    torch.manual_seed(1)
    real = 48
    w = torch.randn(E, F, D, device="cuda", dtype=torch.bfloat16) * 0.02
    base = torch.randn(real * E, D, device="cuda", dtype=torch.bfloat16)

    results = []
    for group in (64, 128):
        A = torch.zeros(group * E, D, device="cuda", dtype=torch.bfloat16)
        for e in range(E):  # real rows first in each group, rest left as padding
            A[e * group : e * group + real] = base[e * real : (e + 1) * real]
        out, _ = _run(A, w, _offsets([group] * E), "fp8_rowwise")
        results.append(
            torch.cat([out[e * group : e * group + real] for e in range(E)])
        )

    diff = (results[0] - results[1]).abs().max().item()
    print(f"padding invariance (64 vs 128 per group): max diff {diff:.3e}", flush=True)
    assert diff == 0.0, (
        f"padded rows leak into real tokens: max diff {diff}. The dispatcher's "
        "padded counts and the `offs` cumsum disagree."
    )


def main() -> None:
    cap = torch.cuda.get_device_capability()
    print(f"device {torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}", flush=True)
    padding_invariance()
    compare([1024] * E, "balanced")
    compare([4096, 256, 256, 256, 1024, 1024, 512, 512], "skewed (3.9x, like the router)")
    compare([PAD] * E, "minimum aligned group")
    print("fp8 grouped mm integration OK", flush=True)


if __name__ == "__main__":
    main()
