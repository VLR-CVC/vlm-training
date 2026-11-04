"""Throughput/efficiency plots for the torchtitan scaling studies.

One figure per study (TORCHTITAN_BENCHMARK.md 9.1, 9.2, 9.3). Top panel: total
tokens/s against ideal scaling, with weak-scaling efficiency on the right axis.
Bottom panel: the per-GPU view -- TFLOP/s as bars, tokens/s/GPU as a line.

    python scripts/scaling/scalability_plot.py [output_dir]

Numbers are transcribed from the tables in TORCHTITAN_BENCHMARK.md 9. Every run
there uses `tensor-parallel-degree 4`, so a "DP rank" is a node and the per-GPU
figures are already per GPU (see the note at the top of section 9).
"""

import sys
from dataclasses import dataclass, field

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

PEAK_TFLOPS = 989.0  # what torchtitan prints at startup, for the MFU it logs


@dataclass
class Study:
    key: str
    title: str
    gpus: list[int]
    tps_per_gpu: list[float]
    tflops_per_gpu: list[float]
    # second line of each x tick, e.g. the sequence length or the accumulation count
    sublabel: list[str]
    sublabel_name: str
    note: str = ""
    mfu: list[float] = field(default_factory=list)


STUDIES = [
    Study(
        key="9_1_weak",
        title="Weak scaling, Qwen3.5-9B (FSDP, TP 4, JUPITER)",
        gpus=[16, 32, 64, 128],
        tps_per_gpu=[4864, 4553, 4351, 4269],
        tflops_per_gpu=[264.1, 247.2, 236.2, 231.8],
        mfu=[26.7, 25.0, 23.9, 23.4],
        sublabel=["16384"] * 4,
        sublabel_name="Seq",
        note="32768 tokens/rank, batch 2, no accumulation",
    ),
    Study(
        key="9_2_strong",
        title="Strong scaling, Qwen3.5-9B (DDP, TP 4, JUPITER)",
        gpus=[16, 32, 64, 128],
        tps_per_gpu=[4596, 4125, 3759, 3031],
        tflops_per_gpu=[249.5, 223.9, 204.1, 164.6],
        mfu=[25.2, 22.6, 20.6, 16.6],
        sublabel=["8", "4", "2", "1"],
        sublabel_name="Accum",
        note="global batch fixed at 524,288 tokens; accumulation absorbs the scale",
    ),
    Study(
        key="9_3_grow",
        title="Weak scaling with growing seq_len, Qwen3.5-9B (FSDP, TP 4, JUPITER)",
        gpus=[16, 32, 64, 128],
        tps_per_gpu=[4927, 4719, 4700, 4342],
        tflops_per_gpu=[268.5, 258.1, 258.0, 238.8],
        mfu=[27.1, 26.1, 26.1, 24.2],
        sublabel=["16896", "17408", "17920", "18176"],
        sublabel_name="Seq",
        note="seq_len grows into the memory FSDP frees as the optimizer shards thin out",
    ),
]


def plot_study(study: Study, out_dir: str) -> str:
    gpus = np.array(study.gpus)
    tps_per_gpu = np.array(study.tps_per_gpu, dtype=float)
    tflops_per_gpu = np.array(study.tflops_per_gpu, dtype=float)

    measured_total = gpus * tps_per_gpu
    optimal_total = gpus * tps_per_gpu[0]
    efficiency = (measured_total / optimal_total) * 100

    x_pos = np.arange(len(gpus))

    fig, (ax1, ax3) = plt.subplots(
        2, 1, gridspec_kw={"height_ratios": [3, 1.4]}, figsize=(8, 6.5), sharex=True
    )
    fig.subplots_adjust(hspace=0.08)

    # --- TOP PANEL: total throughput -------------------------------------------
    ax1.set_title(study.title, fontsize=11)
    ax1.set_ylabel("Tokens/sec")
    ax1.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax1.yaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f"{x * 1e-6:.1f}M" if x else "0M")
    )

    bar_width = 0.4
    ax1.bar(x_pos, measured_total, width=bar_width, color="#A9DBF4", zorder=2)
    (l1,) = ax1.plot(x_pos, measured_total, marker="o", color="blue",
                     label="Measured", zorder=3)
    (l2,) = ax1.plot(x_pos, optimal_total, marker="o", color="orange",
                     linestyle="--", label="Ideal scaling", zorder=3)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Efficiency (%)")
    ax2.set_ylim(0, 108)
    (l3,) = ax2.plot(x_pos, efficiency, marker="s", color="red",
                     label="Efficiency", zorder=4)
    # the baseline point is 100% by construction and sits behind the legend, so it
    # is not labelled
    for x, y in zip(x_pos[1:], efficiency[1:]):
        ax2.text(x, y - 3, f"{y:.1f}%", ha="center", va="top",
                 fontsize=9, color="red", fontweight="bold")

    ax1.legend([l1, l2, l3], [line.get_label() for line in (l1, l2, l3)],
               loc="upper left", framealpha=1.0, edgecolor="black", fancybox=False)

    # --- BOTTOM PANEL: per-GPU metrics -----------------------------------------
    ax3.set_xlabel("Number of GPUs")
    ax3.set_ylabel("TFLOP/s/GPU")
    ax3.grid(True, linestyle="--", alpha=0.5, axis="y")
    ax3.set_ylim(0, max(tflops_per_gpu) * 1.35)

    bars = ax3.bar(x_pos, tflops_per_gpu, width=bar_width, color="#43A047",
                   zorder=2, label="TFLOP/s/GPU")
    for bar, mfu in zip(bars, study.mfu or [None] * len(bars)):
        height = bar.get_height()
        text = f"{height:,.1f}" + (f"\n{mfu:.1f}% MFU" if mfu is not None else "")
        ax3.text(bar.get_x() + bar.get_width() / 2, height * 0.06, text,
                 ha="center", va="bottom", fontsize=8, color="white",
                 fontweight="bold")

    ax4 = ax3.twinx()
    ax4.set_ylabel("Tokens/s/GPU")
    ax4.set_ylim(min(tps_per_gpu) * 0.7, max(tps_per_gpu) * 1.3)
    (l4,) = ax4.plot(x_pos, tps_per_gpu, marker="D", color="purple",
                     label="Tokens/s/GPU", zorder=4)
    for x, y in zip(x_pos, tps_per_gpu):
        ax4.text(x, y + max(tps_per_gpu) * 0.04, f"{y:,.0f}", ha="center",
                 va="bottom", fontsize=9, color="purple", fontweight="bold")

    ax3.legend([bars, l4], [bars.get_label(), l4.get_label()], loc="upper right",
               framealpha=1.0, edgecolor="black", fancybox=False, fontsize=9)

    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(
        [f"{g}\n({study.sublabel_name}: {s})" for g, s in zip(gpus, study.sublabel)]
    )
    if study.note:
        fig.text(0.5, 0.005, study.note, ha="center", fontsize=8, style="italic")

    path = f"{out_dir}/{study.key}.pdf"
    plt.savefig(path, format="pdf", bbox_inches="tight")
    plt.close(fig)
    return path


if __name__ == "__main__":
    import os

    out_dir = sys.argv[1] if len(sys.argv) > 1 else "/data/151-2/users/tockier/figures"
    os.makedirs(out_dir, exist_ok=True)
    for study in STUDIES:
        # the MFU torchtitan logs, recomputed from the two columns as a consistency
        # check on the transcription
        for tflops, mfu in zip(study.tflops_per_gpu, study.mfu):
            assert abs(tflops / PEAK_TFLOPS * 100 - mfu) < 0.1, (study.key, tflops, mfu)
        print(plot_study(study, out_dir))
