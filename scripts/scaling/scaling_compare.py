"""One-panel scaling comparison: tokens/s/GPU against GPU count, several runs.

    python scripts/scaling/scaling_compare.py [output_dir]

`scalability_plot.py` draws one rich two-panel figure per study and cannot hold
two frameworks at once. This is the stripped version: one axes per `Figure`, one
line per `Run` in it, so adding a series means adding a `Run(...)` to that list
and adding a whole figure means adding a `Figure(...)` to `FIGURES`.

**The plotted line is the full-rate steps** -- every step after warmup that runs
within 1.5x the run's fastest step -- and MFU and efficiency are quoted against
that same set, so the three numbers at each point describe one thing. The
post-warmup mean, which includes residual recompile stalls and is what a finite
run actually delivers, is drawn dashed behind it without labels. Where the two
diverge, the gap is stall time, not step time.

Efficiency is computed here, not transcribed: `tps[i] / tps[0]`, i.e. against the
smallest point of that series. Weak and strong scaling share the formula because
both hold the *per-GPU* ideal fixed -- weak by construction, strong because the
fixed global batch is spread over more GPUs.

MFU is each framework's own, and **the denominators are not the same**:
torchtitan bills a constant 54.29 GFLOP/token against the full row, while
`train/flops_estimation.py` counts per-document causal pairs plus the actual ViT
patch count. Compare MFU down a column, never across the two.
"""

import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


@dataclass
class Run:
    label: str
    gpus: list[int]
    # full-rate steps: the line that is drawn, labelled and measured
    tps_per_gpu: list[float]
    mfu: list[float]
    color: str
    # post-warmup mean over every step, stalls included; dashed, unlabelled
    mean_tps: list[float] = field(default_factory=list)
    # label placement in points; flip it where two series meet at a point
    label_dy: int = 10

    def efficiency(self) -> list[float]:
        return [100 * t / self.tps_per_gpu[0] for t in self.tps_per_gpu]


@dataclass
class Figure:
    key: str
    title: str
    note: str
    runs: list[Run]


# Every figure's numbers come from the JUPITER logs named in the comment above it,
# read with scripts/scaling/collect_steady.py. A GH200 node is 4 GPUs, so the
# "(Nn)" half of each x tick is GPUs/4 regardless of the TP degree.
FIGURES = [
    # Qwen3.5-9B, plotqa_cot, tp=4, FSDP shard=-1, fp32 master, 32,768 tokens per
    # DP rank (= per node) at every point, so the global batch grows with the
    # machine and the two frameworks are at matched tokens/step.
    #
    # ours:       logs_scaling/s4rows_20260917_{084242,091515}. 1 row of 32768.
    #             The 32n mean is its last 100 steps: that run never left warmup,
    #             so it has no post-warmup window.
    # torchtitan: TORCHTITAN_BENCHMARK.md 9.1 for the full-rate line and its MFU,
    #             9.3's re-analysis of the same sweep for the mean.
    Figure(
        key="scaling_compare",
        title="Qwen3.5-9B on JUPITER, 32,768 tokens/DP rank, TP 4, FSDP",
        note=("ours: 1x32768 rows, 500 steps @16 GPUs / 409 @128 (32n never fully "
              "warm). torchtitan: 2x16384 rows, 500 steps. "
              "MFU denominators differ between the two -- see the module docstring."),
        runs=[
            Run("ours 9B", [16, 128], [4894, 4490], [32.4, 29.7], "#1f77b4",
                mean_tps=[4825, 3736]),
            Run("torchtitan 9B", [16, 32, 64, 128],
                [4864, 4553, 4351, 4269], [26.7, 25.0, 23.9, 23.4], "#d62728",
                mean_tps=[4799, 4377, 4106, 3935], label_dy=-30),
        ],
    ),
    # Qwen3-VL-2B, plotqa_cot, tp=2, fp32 master, seq_len 32768, one row per
    # micro-batch, `reshard_after_forward=never`, 500 steps (2000 at 256/512 GPUs).
    # No torchtitan counterpart exists: torchtitan has no Qwen3-VL model.
    #
    # weak:   logs_scaling/weak_vl2b_fsdp (16-128 GPUs) + weak_vl2b_fsdp_big
    #         (256, 512). Same config throughout -- FSDP shard=-1, accum 1, 32,768
    #         tokens/DP rank -- so the two sweeps join into one curve, and the
    #         efficiency of the last two points is against the 16-GPU base here
    #         rather than against `_big`'s own first point.
    # strong: logs_scaling/strong_vl2b_500. HSDP (replicate=nodes, shard=2), global
    #         batch pinned at 2,097,152 tokens, accumulation 8/4/2/1 absorbs scale.
    Figure(
        key="scaling_vl2b",
        title="Qwen3-VL-2B on JUPITER, TP 2, seq_len 32768",
        note=("weak: FSDP, 32,768 tokens/DP rank, global batch grows. "
              "strong: HSDP, global batch fixed at 2,097,152 tokens, accum 8/4/2/1."),
        runs=[
            Run("weak scaling, FSDP", [16, 32, 64, 128, 256, 512],
                [14369, 13831, 13000, 12172, 11688, 10956],
                [21.3, 20.5, 19.4, 18.1, 17.4, 16.4], "#1f77b4",
                mean_tps=[14228, 13718, 12575, 12085, 11659, 10852], label_dy=-30),
            Run("strong scaling, HSDP", [16, 32, 64, 128],
                [14201, 15302, 14052, 13026], [21.2, 22.8, 20.9, 19.4], "#2ca02c",
                mean_tps=[14201, 15253, 13924, 12778]),
        ],
    ),
]


def plot(spec: Figure, out_dir: str) -> list[str]:
    fig, ax = plt.subplots(figsize=(9, 5.6))

    for run in spec.runs:
        if run.mean_tps:
            ax.plot(run.gpus, run.mean_tps, marker="D", markersize=4.5,
                    linestyle="--", linewidth=1.3, alpha=0.6, color=run.color,
                    label=f"{run.label} - post-warmup mean", zorder=2)
        ax.plot(run.gpus, run.tps_per_gpu, marker="o", markersize=6,
                linewidth=2.0, color=run.color,
                label=f"{run.label} - full-rate steps", zorder=3)
        for x, tps, mfu, eff in zip(run.gpus, run.tps_per_gpu, run.mfu,
                                    run.efficiency()):
            ax.annotate(f"{tps:,.0f}\n{mfu:.1f}% MFU | {eff:.0f}% eff",
                        (x, tps), textcoords="offset points",
                        xytext=(0, run.label_dy), ha="center", fontsize=7.5,
                        linespacing=1.3, color=run.color, fontweight="bold")

    ax.set_title(spec.title, fontsize=11)
    ax.set_xlabel("GPUs")
    ax.set_ylabel("Tokens/s/GPU")
    ax.set_xscale("log", base=2)
    # only the measured points get a tick; a log2 axis otherwise invents 24, 48...
    ax.set_xticks(sorted({g for run in spec.runs for g in run.gpus}))
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(
        lambda x, pos: f"{x:.0f}\n({x / 4:.0f}n)"))
    ax.xaxis.set_minor_locator(ticker.NullLocator())
    # the end labels are wider than a default margin; widen the axis, do not
    # re-anchor them, or they collide with their neighbour's label instead
    ticks = sorted({g for run in spec.runs for g in run.gpus})
    ax.set_xlim(ticks[0] / 1.9, ticks[-1] * 1.9)
    # headroom for the two-line point labels
    ax.set_ylim(0, max(max(r.tps_per_gpu) for r in spec.runs) * 1.3)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="lower left", framealpha=1.0, edgecolor="black",
              fancybox=False, fontsize=8.5)
    if spec.note:
        fig.text(0.5, -0.03, spec.note, ha="center", fontsize=7.5,
                 style="italic", wrap=True)

    paths = []
    for ext in ("pdf", "png"):
        path = f"{out_dir}/{spec.key}.{ext}"
        plt.savefig(path, format=ext, bbox_inches="tight", dpi=200)
        paths.append(path)
    plt.close(fig)
    return paths


if __name__ == "__main__":
    import os

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/151-2/users/tockier/figures"
    os.makedirs(out_dir, exist_ok=True)
    # lists that disagree in length silently truncate in matplotlib and in zip()
    for spec in FIGURES:
        for run in spec.runs:
            n = len(run.gpus)
            assert len(run.tps_per_gpu) == n, (spec.key, run.label, "tps")
            assert len(run.mfu) == n, (spec.key, run.label, "mfu")
            assert not run.mean_tps or len(run.mean_tps) == n, (spec.key, run.label)
    for spec in FIGURES:
        for path in plot(spec, out_dir):
            print(path)
