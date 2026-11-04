"""Plots from `oracle_sparse_recall.py` json output.

    python utils/plot_oracle.py --out-dir figs 2B=run_2b/*.json 35B=run_35b/*.json

Each positional argument is `label=glob`. One panel per label, one line per
full-attention layer.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm


def load(spec: str):
    label, pattern = spec.split("=", 1)
    runs = {}
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        runs[d["seq_len"]] = d
    if not runs:
        raise SystemExit(f"no json matched {pattern!r}")
    return label, runs


def layers(runs):
    return [L["layer"] for L in runs[max(runs)]["layers"]]


def xlen(d):
    """Context actually measured. `seq_len` is the row length, which the loader
    path only reaches when a sample fills it; the 120-image cap holds the long
    multimodal samples near 112K inside a 131K row, so plotting against
    `seq_len` would flatter them."""
    return d.get("doc_tokens_mean") or d["seq_len"]


def series(runs, layer, key):
    xs, ys = [], []
    for n in sorted(runs):
        row = {L["layer"]: L for L in runs[n]["layers"]}.get(layer)
        if row and key in row:
            xs.append(xlen(runs[n]))
            ys.append(row[key])
    return xs, ys


def exponent(runs, layer):
    """Least-squares slope of log(support) on log(length)."""
    xs, ys = series(runs, layer, "eff_support_tok")
    if len(xs) < 2:
        return float("nan")
    lx = [math.log(x) for x in xs]
    ly = [math.log(y) for y in ys]
    mx, my = sum(lx) / len(lx), sum(ly) / len(ly)
    return sum((a - mx) * (b - my) for a, b in zip(lx, ly)) / sum((a - mx) ** 2 for a in lx)


def colors(n):
    return [cm.viridis(i / max(n - 1, 1)) for i in range(n)]


def panels(specs, title, ylabel, plotter, out, logx=True, logy=False, figsize=(6.0, 4.2)):
    fig, axes = plt.subplots(
        1, len(specs), figsize=(figsize[0] * len(specs), figsize[1]), squeeze=False
    )
    for ax, (label, runs) in zip(axes[0], specs):
        names = layers(runs)
        for name, c in zip(names, colors(len(names))):
            plotter(ax, runs, name, c)
        ax.set_title(f"{label}  ({len(names)} full-attention layers)")
        ax.set_xlabel("context length (tokens)")
        if logx:
            ax.set_xscale("log", base=2)
        if logy:
            ax.set_yscale("log", base=2)
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=6, ncol=2)
    axes[0][0].set_ylabel(ylabel)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="label=glob")
    ap.add_argument("--out-dir", default="figs")
    ap.add_argument("--budget", type=int, default=2048)
    ap.add_argument("--profile-at", type=int, default=0, help="0 = each panel's longest")
    a = ap.parse_args()
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    specs = [load(s) for s in a.runs]
    K = a.budget

    # 1. effective support vs context, with a reference slope
    def sup(ax, runs, name, c):
        xs, ys = series(runs, name, "eff_support_tok")
        ax.plot(xs, ys, "o-", color=c, label=name, ms=3, lw=1.2)

    def sup_panel(ax, runs, name, c):
        sup(ax, runs, name, c)
        if name == layers(runs)[0]:
            xs = [xlen(runs[n]) for n in sorted(runs)]
            e = sum(exponent(runs, l) for l in layers(runs)) / len(layers(runs))
            ref = [500 * (x / xs[0]) ** e for x in xs]
            ax.plot(xs, ref, "k--", lw=1, label="$n^{%.2f}$ (mean fit)" % e)
            ax.axhline(K, color="r", ls=":", lw=1, label=f"budget K={K}")

    panels(
        specs,
        "Effective support exp(H) of the attention distribution — how many tokens a row actually needs",
        "effective support (tokens)",
        sup_panel,
        out / "support_vs_context.png",
        logy=True,
    )

    # 2. recall at the budget, mean with the p5 band
    def rec(ax, runs, name, c):
        xs, ys = series(runs, name, f"blk_sum@{K}")
        _, lo = series(runs, name, f"blk_sum@{K}_p5")
        ax.plot(xs, ys, "o-", color=c, label=name, ms=3, lw=1.2)
        ax.fill_between(xs, lo, ys, color=c, alpha=0.12)

    panels(
        specs,
        f"QSA oracle ceiling at K={K} (r=4 blocks): mean, shaded down to p5",
        "attention mass captured",
        rec,
        out / "recall_vs_context.png",
    )

    # 3. budget sweep, each panel at its own longest context
    def sweep_panel(ax, runs, name, c):
        n = max(runs)
        row = {L["layer"]: L for L in runs[n]["layers"]}[name]
        ks = runs[n]["budgets"]
        ax.plot(ks, [row[f"blk_sum@{k}"] for k in ks], "o-", color=c, label=name, ms=3, lw=1.2)
        ax.set_xlabel("token budget K")
        if name == layers(runs)[0]:
            ax.text(0.03, 0.03, f"{xlen(runs[n]) / 1024:.0f}K tokens",
                    transform=ax.transAxes, fontsize=8, color="0.3")

    panels(
        specs,
        "Budget sweep at each panel's longest context — where the knee sits",
        "attention mass captured",
        sweep_panel,
        out / "budget_sweep.png",
    )

    # 4. depth profile: the U-shape, both models on one axis
    fig, ax = plt.subplots(figsize=(7, 4.2))
    for (label, runs), mk in zip(specs, ["o-", "s--", "^:"]):
        n = a.profile_at if a.profile_at in runs else max(runs)
        length = xlen(runs[n])
        rows = runs[n]["layers"]
        idx = [int(L["layer"].replace("layer", "")) for L in rows]
        depth = [i / max(idx) for i in idx]
        ax.plot(depth, [L["eff_support_tok"] for L in rows], mk,
                label=f"{label} @ {length / 1024:.0f}K", ms=4, lw=1.3)
    ax.axhline(K, color="r", ls=":", lw=1, label=f"budget K={K}")
    ax.set_xlabel("relative depth of the full-attention layer (0 = first, 1 = last)")
    ax.set_ylabel("effective support (tokens)")
    ax.set_title("Early and late attention layers are diffuse, middle ones are sharp")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "depth_profile.png", dpi=150)
    plt.close(fig)
    print(f"wrote {out / 'depth_profile.png'}")

    # 5. what r=4 granularity costs. Raw tok - blk_sum: blk_sum gets the r-1
    # tail tokens for free, so this understates the gap slightly, but it is the
    # number that answers "what would DSA's token granularity buy".
    def gran(ax, runs, name, c):
        xs, tok = series(runs, name, f"tok@{K}")
        _, blk = series(runs, name, f"blk_sum@{K}")
        ax.plot(xs, [t - b for t, b in zip(tok, blk)], "o-", color=c, label=name, ms=3, lw=1.2)

    panels(
        specs,
        f"Cost of r=4 block granularity at K={K}:  tok − blk_sum  (what DSA would recover)",
        "attention mass lost vs free token choice",
        gran,
        out / "granularity_cost.png",
    )


if __name__ == "__main__":
    main()
