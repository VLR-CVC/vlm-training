"""Bucket a torch.profiler chrome trace by GPU kernel category.

    python scripts/scaling/profile_breakdown.py <trace_rank_N_step_M.json>

The `key_averages` summary the profiler writes aggregates every collective under
one `record_param_comms` row, which says "80% is comms" but not WHICH comms. The
levers differ: inter-node FSDP weight all-gather is fixed per step and shrinks
with `reshard_after_forward`/precision, while many small intra-node SP
redistributions are a latency problem with a different fix entirely.

This reads the GPU-side kernel events instead and splits them by NCCL op, so the
breakdown is unambiguous. Chrome traces are plain JSON; no profiler needed.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

# kernel-name substrings -> bucket. First match wins, so order matters.
BUCKETS = (
    ("all_gather / AG", ("allgather", "all_gather", "AllGather")),
    ("reduce_scatter", ("reducescatter", "reduce_scatter", "ReduceScatter")),
    ("all_to_all", ("alltoall", "all_to_all", "AllToAll")),
    ("all_reduce", ("allreduce", "all_reduce", "AllReduce")),
    ("other nccl", ("nccl", "Nccl", "NCCL")),
    ("grouped_mm (experts)", ("grouped", "GroupedGemm", "group_gemm")),
    ("gemm / matmul", ("gemm", "Gemm", "GEMM", "cutlass", "sm90_", "sm80_", "wgrad", "dgrad")),
    ("attention", ("flash", "fmha", "attn", "attention", "chunk_", "delta")),
    ("elementwise / norm", ("elementwise", "vectorized", "norm", "Norm", "silu", "mul", "add")),
    ("copy / memset", ("memcpy", "Memcpy", "memset", "Memset", "copy", "Copy")),
)


def bucket_of(name: str) -> str:
    for label, needles in BUCKETS:
        if any(n in name for n in needles):
            return label
    return "other"


def main(path: str) -> None:
    print(f"loading {path} ...", flush=True)
    with open(path) as f:
        trace = json.load(f)
    events = trace.get("traceEvents", [])
    print(f"{len(events):,} events", flush=True)

    # GPU-side work only: cat 'kernel' plus the gpu memcpy/memset categories
    gpu_cats = {"kernel", "gpu_memcpy", "gpu_memset", "gpu_user_annotation"}
    per_bucket: dict[str, list] = defaultdict(lambda: [0.0, 0])
    per_kernel: dict[str, list] = defaultdict(lambda: [0.0, 0])
    total = 0.0
    for e in events:
        if e.get("cat") not in gpu_cats or e.get("ph") != "X":
            continue
        dur = float(e.get("dur") or 0.0)
        if e["cat"] == "gpu_user_annotation":
            continue  # annotations overlap real kernels; would double count
        name = e.get("name", "?")
        b = bucket_of(name)
        per_bucket[b][0] += dur
        per_bucket[b][1] += 1
        per_kernel[name][0] += dur
        per_kernel[name][1] += 1
        total += dur

    print(f"\nGPU kernel time total: {total/1e6:.3f} s (sum of kernel durations)\n")
    print(f"{'bucket':24} {'time (s)':>10} {'share':>8} {'calls':>9} {'avg (us)':>10}")
    print("-" * 66)
    for b, (dur, n) in sorted(per_bucket.items(), key=lambda kv: -kv[1][0]):
        print(f"{b:24} {dur/1e6:10.3f} {dur/total:8.1%} {n:9,} {dur/max(n,1):10.1f}")

    print(f"\ntop 18 kernels by total GPU time:")
    print(f"{'kernel':<72} {'time (s)':>9} {'calls':>8} {'avg (us)':>9}")
    print("-" * 102)
    for name, (dur, n) in sorted(per_kernel.items(), key=lambda kv: -kv[1][0])[:18]:
        print(f"{name[:72]:<72} {dur/1e6:9.3f} {n:8,} {dur/max(n,1):9.1f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else
         "/data/151-2/users/tockier/profiling/jup_prof_8n/trace_rank_0_step_14.json")
