"""Why is an fp8 expert GEMM step 100x slower than bf16? Time it per call.

    srun --nodes=1 --gpus-per-node=1 ... python models/tests/fp8_autotune_probe.py

The 1-node A/B (jobs 2219865 bf16 vs 2219866 fp8) showed 0.31s/step against
37.2s/step on an identical 3B model, with the cost flat and independent of how
many samples the step had. `TORCH_LOGS=recompiles` showed only bounded,
warmup-time dynamo recompiles, so the cost is BELOW dynamo.

Hypothesis: triton autotuning inside torchao's kernels keys on the row count,
which `permute_and_pad` makes data-dependent, so every new M re-benchmarks.

This separates the three possibilities directly:
  * first call at a NEW M slow, repeat at the SAME M fast  -> autotune per shape
  * every call slow regardless                              -> the kernel is slow
  * all calls fast                                          -> the cost is elsewhere
"""

from __future__ import annotations

import time

import torch

D, F, E = 2048, 512, 64   # debug_1n expert shapes


def timed(fn, *, warmup=0):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    fn()
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def make(M):
    A = torch.randn(M, D, device="cuda", dtype=torch.bfloat16)
    W = torch.randn(E, F, D, device="cuda", dtype=torch.bfloat16) * 0.02
    per = M // E
    offs = (torch.full((E,), per, dtype=torch.int32, device="cuda")
            .cumsum(0).to(torch.int32))
    return A, W, offs


def call_fp8(A, W, offs):
    from torchao.prototype.moe_training.fp8_grouped_mm import (
        _to_fp8_rowwise_then_scaled_grouped_mm as f,
    )
    return lambda: f(A, W.transpose(-2, -1), offs,
                     pad_token_groups_for_grouped_mm=False)


def call_bf16(A, W, offs):
    return lambda: torch._grouped_mm(A, W.transpose(-2, -1), offs=offs)


def main() -> None:
    cap = torch.cuda.get_device_capability()
    print(f"{torch.cuda.get_device_name()} sm_{cap[0]}{cap[1]}\n", flush=True)

    base = 64 * E  # 4096 rows, 64 per expert, a multiple of 16
    for name, builder in (("bf16", call_bf16), ("fp8 ", call_fp8)):
        A, W, offs = make(base)
        fn = builder(A, W, offs)
        first = timed(fn)
        same = [timed(fn) for _ in range(3)]
        print(f"{name} M={base}: first {first*1e3:9.1f} ms | "
              f"repeats {' '.join(f'{s*1e3:.1f}' for s in same)} ms", flush=True)

    print("\nnew M each time (what the router actually produces):", flush=True)
    for name, builder in (("bf16", call_bf16), ("fp8 ", call_fp8)):
        for M in (base + 16 * E, base + 32 * E, base + 48 * E):
            A, W, offs = make(M)
            fn = builder(A, W, offs)
            first = timed(fn)
            again = timed(fn)
            print(f"  {name} M={M}: first {first*1e3:9.1f} ms | again {again*1e3:8.1f} ms",
                  flush=True)


def probe_permute() -> None:
    """Time `permute_and_pad` -- the half of the fp8 switch the GEMM probe missed.

    The 1-node A/B says an fp8 fwd+bwd costs ~37s against bf16's ~0.25s, and the
    grouped GEMMs account for ~12ms of that. `permute_and_pad` runs once per
    layer per forward, so ~4.6s per call would explain the whole gap. The plain
    `_permute` it replaces (cumsum + repeat_interleave) is the bf16 baseline.
    """
    from torchao.prototype.moe_training.ep.permute import permute_and_pad

    ep_degree, local_e = 4, 16          # debug_1n: 64 experts over ep=4
    rows = 8192 * 8 // ep_degree        # tokens/rank * top_k, arriving per rank
    D_ = 2048
    counts = torch.full((ep_degree * local_e,), rows // (ep_degree * local_e),
                        dtype=torch.int32, device="cuda")
    x = torch.randn(int(counts.sum()), D_, device="cuda", dtype=torch.bfloat16)

    def ao():
        return permute_and_pad(x, counts, ep_degree, local_e, 16)

    def plain():
        # what AllToAllTokenDispatcher._permute does, same arithmetic
        t_mat = counts.view(ep_degree, local_e)
        starts = (counts.cumsum(0) - counts).view(ep_degree, local_e)
        lens = t_mat.t().reshape(-1)
        st = starts.t().reshape(-1)
        seg = torch.arange(lens.shape[0], device="cuda").repeat_interleave(lens)
        out_st = lens.cumsum(0) - lens
        idx = st[seg] + torch.arange(seg.shape[0], device="cuda") - out_st[seg]
        return x[idx, :]

    print(f"\npermute, rows={x.shape[0]} dim={D_} ep={ep_degree} local_experts={local_e}:",
          flush=True)
    for name, fn in (("plain (bf16 path)", plain), ("permute_and_pad   ", ao)):
        first = timed(fn)
        rep = [timed(fn) for _ in range(3)]
        print(f"  {name}: first {first*1e3:9.1f} ms | repeats "
              f"{' '.join(f'{r*1e3:.2f}' for r in rep)} ms", flush=True)


if __name__ == "__main__":
    main()
    probe_permute()
