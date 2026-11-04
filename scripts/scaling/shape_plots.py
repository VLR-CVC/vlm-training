"""Plot the dataloader's shape distributions and what bucketing would do to them.

    python scripts/scaling/shape_plots.py shapes.txt [--out DIR] [--label NAME]

`shapes.txt` is the dump from `scripts/scaling/shape_histogram.py`: one line per
micro-batch, `tokens docs longest patches`.

Each distinct value on an axis that reaches a compiled region costs one graph on the
first rank to see it, and every other rank waits at the next collective -- so the
count of distinct values, not the spread, is what sets warmup (`BENCHMARKS.md`).
Each panel therefore shows the distribution and, in the corner, how many distinct
values survive rounding up to each candidate bucket.

`longest` is the worked example: `create_varlen_metadata_for_document` already
rounds it to a power of two, which is why `max_k` does not appear in the recompile
guards while the vision patch count does.
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BUCKETS = (256, 1024, 4096)
BAR = "#1565C0"
POW2 = "#C62828"


def distinct_after(v: np.ndarray, bucket: int) -> int:
    return len(np.unique(-(-v // bucket) * bucket))


def distinct_pow2(v: np.ndarray) -> int:
    return len(np.unique([1 if x <= 1 else 1 << int(x - 1).bit_length() for x in v]))


def panel(ax, v: np.ndarray, title: str, note: str, *, bucketed: bool) -> None:
    ax.hist(v, bins=min(60, max(10, len(np.unique(v)))), color=BAR, alpha=0.85)
    ax.set_title(title, fontsize=10.5, loc="left")
    ax.set_ylabel("micro-batches")
    ax.grid(True, linestyle="--", alpha=0.35, axis="y")

    lines = [f"distinct: {len(np.unique(v))} of {len(v)}"]
    lines += [f"  ->{b:>5,}: {distinct_after(v, b)}" for b in BUCKETS]
    lines.append(f"  -> pow2: {distinct_pow2(v)}")
    ax.text(0.985, 0.95, "\n".join(lines), transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, family="monospace",
            bbox=dict(boxstyle="round", facecolor="white",
                      edgecolor=POW2 if bucketed else "black", linewidth=1.4))
    ax.text(0.985, 0.30, note, transform=ax.transAxes, ha="right", va="top",
            fontsize=8.5, style="italic", color=POW2 if bucketed else "black")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("dump")
    p.add_argument("--out", default="/data/151-2/users/tockier/figures/shapes")
    p.add_argument("--label", default="")
    args = p.parse_args()

    rows = [l.split() for l in Path(args.dump).read_text().splitlines()[1:] if l.strip()]
    cols = {k: np.array([int(r[i]) for r in rows])
            for i, k in enumerate(("tokens", "docs", "longest", "patches"))}
    label = args.label or Path(args.dump).stem

    fig, axes = plt.subplots(3, 1, figsize=(9, 9.5))
    fig.suptitle(f"Dataloader shape distributions — {label}  (n={len(rows)} micro-batches)",
                 fontsize=12)

    panel(axes[0], cols["patches"],
          "vision patches  ->  Q_LEN in create_block_mask, ViT input size",
          "NOT bucketed today: every distinct value is a compile", bucketed=False)
    panel(axes[1], cols["docs"],
          "documents per row  ->  len(cu_seq_q)",
          "NOT bucketed today; small range, so few graphs", bucketed=False)
    panel(axes[2], cols["longest"],
          "longest document  ->  max_k",
          "already rounded to a power of two, so max_k costs 5 graphs, not 293",
          bucketed=True)
    axes[2].set_xlabel("value")

    fig.tight_layout(rect=(0, 0.01, 1, 0.97))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"shapes_{label}.pdf"
    fig.savefig(path, format="pdf", bbox_inches="tight")
    fig.savefig(path.with_suffix(".png"), dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(path)
    print(path.with_suffix(".png"))


if __name__ == "__main__":
    main()
