#!/usr/bin/env python3
"""Turn a sweep's logs into a scaling table.

    python scripts/scaling/collect.py logs_scaling/20260911_120000
    python scripts/scaling/collect.py logs_scaling/20260911_120000 --skip 15 --csv out.csv

Reads the `log.out` under each `<N>n/` directory, parses the per-step training
line, drops the warmup steps and reports the median of the rest. Median, not
mean: one straggler step at 256 nodes drags a mean far more than it reflects
what the run actually sustains.

Speedup and efficiency are relative to the smallest node count present, so a
sweep that starts at 16 reads as 16 -> 1.00x. Perfect weak scaling is a flat
tok/s/GPU column and 100% efficiency.
"""
from __future__ import annotations

import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

# `<step> - loss 1.234 tps 5678.90 mfu 41.2% tflops 407.6 gnorm ... time 1.234s
#  fwd 0.987s data_pct 1.20% nsamples 4096 batch_util 99.1%`
# wrapped in ANSI colour, hence the strip below.
ANSI = re.compile(r"\x1b\[[0-9;]*m")
# `search`, not `match`: the logger prefixes "%(asctime)s - " and srun may add
# its own prefix on top of that.
STEP = re.compile(
    r"(?P<step>\d+)\s+-\s+"
    r"loss\s+(?P<loss>[\d.]+)\s+"
    r"tps\s+(?P<tps>[\d.]+)\s+"
    r"mfu\s+(?P<mfu>[\d.]+)%\s+"
    r"tflops\s+(?P<tflops>[\d.]+)\s+"
    r".*?time\s+(?P<step_time>[\d.]+)s\s+"
    r"fwd\s+(?P<fwd>[\d.]+)s\s+"
    r"data_pct\s+(?P<data_pct>[\d.]+)%"
)
# the `[scaling]` banner the sbatch script prints; authoritative for the GPU
# count, since the directory name is only a label.
BANNER = re.compile(r"\[scaling\]\s+(?P<nodes>\d+)\s+nodes\s*/\s*(?P<gpus>\d+)\s+GPUs")

def parse_log(path: Path, skip: int) -> dict | None:
    nodes = gpus = None
    rows = []
    for raw in path.read_text(errors="replace").splitlines():
        line = ANSI.sub("", raw).strip()
        if banner := BANNER.search(line):
            nodes = int(banner["nodes"])
            gpus = int(banner["gpus"])
            continue
        if m := STEP.search(line):
            rows.append({k: float(v) for k, v in m.groupdict().items()})

    if not rows:
        return None

    # `skip` counts steps, not lines: a restarted rendezvous can repeat steps.
    kept = [r for r in rows if r["step"] > skip]
    if not kept:
        return None

    if gpus is None:  # no banner (hand-run job): fall back to the directory name
        nodes = int(re.sub(r"\D", "", path.parent.name) or 0)
        gpus = nodes * 4

    med = lambda key: statistics.median(r[key] for r in kept)
    return {
        "nodes": nodes,
        "gpus": gpus,
        "steps_used": len(kept),
        "tflops_per_gpu": med("tflops"),
        "tok_s_per_gpu": med("tps"),
        "mfu_pct": med("mfu"),
        "step_time_s": med("step_time"),
        "fwd_bwd_s": med("fwd"),
        "data_pct": med("data_pct"),
        "loss_last": kept[-1]["loss"],
    }

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sweep_dir", type=Path, nargs="?",
                    help="logs_scaling/<sweep id>")
    ap.add_argument("--skip", type=int, default=10,
                    help="drop steps <= this (torch.compile + autotune warmup)")
    ap.add_argument("--csv", type=Path, help="also write the table here")
    args = ap.parse_args()

    if args.sweep_dir is None:
        ap.error("sweep_dir is required")

    logs = sorted(args.sweep_dir.glob("*n/log.out"))
    if not logs:
        print(f"no <N>n/log.out under {args.sweep_dir}", file=sys.stderr)
        return 1

    results, skipped = [], []
    for log in logs:
        row = parse_log(log, args.skip)
        (results if row else skipped).append(row or log.parent.name)
    if not results:
        print("every log parsed empty -- did the jobs get past startup?", file=sys.stderr)
        return 1

    results.sort(key=lambda r: r["gpus"])
    base = results[0]
    for r in results:
        ratio = r["gpus"] / base["gpus"]
        total = r["tok_s_per_gpu"] * r["gpus"]
        r["speedup"] = total / (base["tok_s_per_gpu"] * base["gpus"])
        r["efficiency_pct"] = 100.0 * r["speedup"] / ratio

    head = (f"{'nodes':>6} {'gpus':>6} {'steps':>6} {'TFLOP/s/GPU':>12} "
            f"{'tok/s/GPU':>11} {'MFU%':>6} {'step s':>7} {'data%':>6} "
            f"{'speedup':>8} {'eff%':>6}")
    print(head)
    print("-" * len(head))
    for r in results:
        print(f"{r['nodes']:>6} {r['gpus']:>6} {r['steps_used']:>6} "
              f"{r['tflops_per_gpu']:>12.1f} {r['tok_s_per_gpu']:>11.0f} "
              f"{r['mfu_pct']:>6.1f} {r['step_time_s']:>7.3f} {r['data_pct']:>6.2f} "
              f"{r['speedup']:>7.2f}x {r['efficiency_pct']:>6.1f}")
    print(f"\nbaseline {base['nodes']} nodes, median over steps > {args.skip}")
    if skipped:
        print(f"no usable steps in: {', '.join(skipped)}  "
              f"(check the matching errors/rank_*.err)")

    if args.csv:
        with args.csv.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0]))
            w.writeheader()
            w.writerows(results)
        print(f"wrote {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
