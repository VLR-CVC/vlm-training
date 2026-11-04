"""Step-time distributions: how long the recompile warmup actually lasts.

    python scripts/scaling/steptime_plots.py <series>... [--out DIR]

Each `<series>` is `label=path` or just `path`, where the file is
`HH:MM:SS<ms> step tps tflops` per line, one line per step (see the awk in
`scripts/scaling/collect_steady.py`'s docstring, or extract from a torchtitan log).

Three panels per run:

1. **step time vs step** -- every step as a point, full-rate steps and stalled steps
   in different colours. The argument lives here: stalls are dense early and thin out.
2. **histogram of step times** -- bimodal, and the two modes are "nobody was
   compiling" and "somebody was". Not a spread around a mean, so a mean over a short
   window measures the mixing ratio rather than the throughput.
3. **stall rate in a rolling window** -- with the warmup boundary marked: the first
   step after which the stall rate stays under `--settled` for the rest of the run.
   That boundary is the number this script exists to measure.

A step counts as stalled when it takes more than `--full-rate` x the run's fastest
step; the fastest step is full-rate by construction, which keeps the classifier
honest even when most steps stall (a median-based threshold does not -- see
collect_steady.py).
"""

import argparse
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

FULL = "#2E7D32"
STALL = "#C62828"
ACCENT = "#1565C0"


def load(path: str) -> tuple[np.ndarray, np.ndarray]:
    """-> (step, step_time_seconds).

    Two input shapes. `step<space>seconds` uses the durations as given -- our trainer
    logs `time` per step, which beats differencing its one-second-resolution
    timestamps. `HH:MM:SS<ms> step ...` (a torchtitan log) differences the timestamps
    instead, and the first step is dropped because it has no predecessor.
    """
    lines = [l.split() for l in Path(path).read_text().splitlines() if l.split()]
    if lines and ":" not in lines[0][0]:
        s = np.array([int(p[0]) for p in lines])
        return s, np.array([float(p[1]) for p in lines])

    steps, times = [], []
    for parts in lines:
        if len(parts) < 2:
            continue
        hms = parts[0]
        h, m, rest = hms.split(":")
        sec = float(rest[:2]) + float(rest[2:] or 0) / 1000.0
        times.append(int(h) * 3600 + int(m) * 60 + sec)
        steps.append(int(parts[1]))
    t = np.array(times)
    s = np.array(steps)
    dt = np.diff(t)
    # a day rollover shows up as a single large negative gap
    dt[dt < 0] += 86400
    return s[1:], dt


sys.path.insert(0, str(Path(__file__).resolve().parent))
from collect_steady import warmup_end as warmup_boundary  # noqa: E402  (same definition as the table)


def plot(label: str, path: str, args, out_dir: Path) -> dict:
    step, dt = load(path)
    floor = float(np.min(dt))
    stalled = dt > args.full_rate * floor
    full_dt, stall_dt = dt[~stalled], dt[stalled]
    boundary = warmup_boundary(stalled, args.window, args.settled)

    fig, (ax1, ax2, ax3) = plt.subplots(
        3, 1, figsize=(9, 9), gridspec_kw={"height_ratios": [2, 1.5, 1.2]}
    )
    fig.suptitle(f"Step time and recompile warmup — {label}", fontsize=12)

    # --- 1. step time vs step ------------------------------------------------
    ax1.scatter(step[~stalled], full_dt, s=9, color=FULL, label="full rate", zorder=3)
    ax1.scatter(step[stalled], stall_dt, s=9, color=STALL, label="stalled", zorder=3)
    ax1.set_ylabel("step time (s)")
    ax1.set_xlabel("step")
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.axhline(floor, color="grey", linewidth=0.8, linestyle=":")
    if boundary is not None:
        ax1.axvline(step[boundary], color=ACCENT, linewidth=1.5)
        ax1.text(step[boundary], ax1.get_ylim()[1] * 0.92,
                 f"  warmup ends ~step {step[boundary]}", color=ACCENT,
                 fontsize=9, fontweight="bold", va="top")
    ax1.legend(loc="upper right", framealpha=1.0, edgecolor="black", fancybox=False)

    # --- 2. histogram --------------------------------------------------------
    bins = np.linspace(0, float(np.percentile(dt, 99)) * 1.05, 60)
    ax2.hist(full_dt, bins=bins, color=FULL, alpha=0.85, label=f"full rate (n={len(full_dt)})")
    ax2.hist(stall_dt, bins=bins, color=STALL, alpha=0.85, label=f"stalled (n={len(stall_dt)})")
    ax2.set_xlabel("step time (s)")
    ax2.set_ylabel("steps")
    ax2.grid(True, linestyle="--", alpha=0.4, axis="y")
    ax2.legend(loc="upper right", framealpha=1.0, edgecolor="black", fancybox=False, fontsize=9)

    # --- 3. rolling stall rate ----------------------------------------------
    if len(stalled) >= args.window:
        rate = np.convolve(stalled.astype(float), np.ones(args.window) / args.window, "valid") * 100
        ax3.plot(step[args.window - 1:], rate, color=STALL, linewidth=1.6)
        ax3.fill_between(step[args.window - 1:], rate, color=STALL, alpha=0.2)
    ax3.axhline(args.settled * 100, color=ACCENT, linestyle="--", linewidth=1,
                label=f"settled ({args.settled * 100:.0f}%)")
    if boundary is not None:
        ax3.axvline(step[boundary], color=ACCENT, linewidth=1.5)
    ax3.set_ylabel(f"stalled steps\nin {args.window}-step window (%)")
    ax3.set_xlabel("step")
    ax3.set_ylim(0, 105)
    ax3.grid(True, linestyle="--", alpha=0.4)
    ax3.legend(loc="upper right", framealpha=1.0, edgecolor="black", fancybox=False, fontsize=9)

    # --- how to read the three panels ---------------------------------------
    # Spelled out on the figure itself: these plots get pasted into docs and
    # messages without the script, and "stalled" is a threshold, not an
    # observation -- a reader who assumes a different one misreads every panel.
    note = (
        f"stalled:  step_time > {args.full_rate:g} x min(step_time)   "
        f"= {args.full_rate * floor:.2f}s   (min = {floor:.2f}s, dotted in panel 1)\n"
        f"warmup ends:  last step whose trailing {args.window}-step stall rate "
        f"> {args.settled * 100:.0f}%,  + {args.window}\n"
        f"reported throughput:  mean over steps after that boundary"
    )
    fig.text(0.5, -0.02, note, ha="center", va="top", fontsize=8,
             family="monospace", color="#333333", linespacing=1.6)

    fig.tight_layout(rect=(0, 0.02, 1, 0.97))
    path_out = out_dir / f"steptime_{label}.pdf"
    fig.savefig(path_out, format="pdf", bbox_inches="tight")
    fig.savefig(path_out.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)

    # steady / effective on the settled tail, and on everything, for comparison
    def rates(mask):
        d = dt[mask]
        return (len(d), float(np.mean(d)) if len(d) else float("nan"))

    tail = slice(boundary, None) if boundary is not None else slice(0, 0)
    return dict(
        label=label, steps=len(dt), floor=floor,
        stalled=int(stalled.sum()), stall_pct=100 * stalled.mean(),
        full_mean=float(np.mean(full_dt)) if len(full_dt) else float("nan"),
        stall_mean=float(np.mean(stall_dt)) if len(stall_dt) else float("nan"),
        boundary=int(step[boundary]) if boundary is not None else None,
        tail_stall_pct=100 * float(np.mean(stalled[tail])) if boundary is not None else float("nan"),
        overall_mean=float(np.mean(dt)),
        tail_mean=float(np.mean(dt[tail])) if boundary is not None else float("nan"),
        out=str(path_out),
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("series", nargs="+", help="label=path or path")
    p.add_argument("--out", default="/data/151-2/users/tockier/figures")
    p.add_argument("--full-rate", type=float, default=1.5,
                   help="step counts as stalled above this x the fastest step")
    p.add_argument("--window", type=int, default=50, help="rolling window, steps")
    p.add_argument("--settled", type=float, default=0.10,
                   help="stall rate below which the run counts as warm")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for spec in args.series:
        label, _, path = spec.rpartition("=")
        path = path or spec
        label = label or Path(path).stem
        results.append(plot(label, path, args, out_dir))

    w = max(len(r["label"]) for r in results)
    print(f"{'run':<{w}} {'steps':>6} {'stalled':>12} {'fast_s':>7} {'stall_s':>8} "
          f"{'warmup ends':>12} {'tail stall':>11} {'mean all':>9} {'mean tail':>10}")
    for r in results:
        b = r["boundary"] if r["boundary"] is not None else "never"
        print(f"{r['label']:<{w}} {r['steps']:>6} {r['stalled']:>5} "
              f"({r['stall_pct']:>4.1f}%) {r['full_mean']:>7.2f} {r['stall_mean']:>8.2f} "
              f"{str(b):>12} {r['tail_stall_pct']:>10.1f}% {r['overall_mean']:>9.2f} "
              f"{r['tail_mean']:>10.2f}")
    for r in results:
        print(r["out"])


if __name__ == "__main__":
    sys.exit(main())
