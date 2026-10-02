from __future__ import annotations

import argparse
import glob
import os

import pandas as pd

lab = lambda L: f"{L // 1024}K" if L % 1024 == 0 else str(L)   # noqa: E731

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 24576])
    ap.add_argument("--csv")
    a = ap.parse_args()

    d = pd.concat((pd.read_parquet(f) for f in sorted(glob.glob(os.path.join(a.index_dir, "*.parquet")))),
                  ignore_index=True)
    bad = d[d.skip == "error"]
    if len(bad):
        print(f"WARNING: {len(bad)} samples errored, e.g. {bad.err.iloc[0]!r}")
    d = d[d.skip != "error"]
    sup = d.n_loss > 0                      # encode_sample also drops samples with no assistant tokens
    rows = []
    for name, g in d.groupby("subset"):
        r = dict(subset=name, samples=len(g), mean_len=g.n_tokens.mean(), p50=g.n_tokens.median(),
                 p95=g.n_tokens.quantile(.95), p99=g.n_tokens.quantile(.99), max=g.n_tokens.max(),
                 mean_images=g.n_images.mean(), loss_frac=g.n_loss.sum() / g.n_tokens.sum(),
                 tokens_B=g.n_tokens.sum() / 1e9, loss_B=g.n_loss.sum() / 1e9,
                 visual_B=g.n_visual.sum() / 1e9)
        for L in a.seq_lens:
            keep = g[(g.n_tokens <= L) & (g.n_loss > 0)]
            r[f"kept%@{lab(L)}"] = 100 * len(keep) / len(g)
            r[f"tokens_B@{lab(L)}"] = keep.n_tokens.sum() / 1e9
        rows.append(r)
    out = pd.DataFrame(rows).sort_values("tokens_B", ascending=False)
    tot = {"subset": "TOTAL", "samples": len(d), "tokens_B": d.n_tokens.sum() / 1e9,
           "loss_B": d.n_loss.sum() / 1e9, "visual_B": d.n_visual.sum() / 1e9,
           "loss_frac": d.n_loss.sum() / d.n_tokens.sum()}
    for L in a.seq_lens:
        keep = d[(d.n_tokens <= L) & sup]
        tot[f"kept%@{lab(L)}"] = 100 * len(keep) / len(d)
        tot[f"tokens_B@{lab(L)}"] = keep.n_tokens.sum() / 1e9
    out = pd.concat([out, pd.DataFrame([tot])], ignore_index=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_rows", 500)
    print(out.round(3).to_string(index=False))
    if a.csv:
        out.to_csv(a.csv, index=False)

if __name__ == "__main__":
    main()