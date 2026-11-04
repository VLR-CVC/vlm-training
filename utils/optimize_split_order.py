"""Choose the tar order in each energon dataset's split.yaml so a run uses as much of the
data as possible before the first dataloader worker runs dry.

The model (it reproduced a real 64-rank / 2-worker OV-Instruct + FineVision run to ~4%
on tokens and 0.4% on samples): energon splits each dataset's tar list, in split.yaml
order, into equal-*sample* contiguous slices, one per dataloader worker; a worker holds
one slice of every dataset in the blend; the run stops when the first worker is empty.
Every worker consumes tokens at the same rate, so

    tokens consumed at stop  =  W x min over workers( tokens in that worker's slices )

and the fraction of the pool that gets trained on is that over the pool total. Sorted
or subset-ordered tars make some workers token-poor; interleaving heavy and light tars
raises the minimum. Tars bigger than a slice cannot be balanced by ordering -- split
them first (utils/split_tars.py).

Inputs are length indexes (utils/length_index.py output: one parquet per tar, named after
the tar). `n_tokens` of samples the trainer will keep count as data; `no_labels` samples
count too by default (they become trainable once the cooker null-role fix is deployed;
`--old-cooker` drops them).

    python utils/optimize_split_order.py \\
        --dataset ov=/gpfs/.../length_index/ov_instruct \\
        --dataset ov_chunks=/gpfs/.../length_index/ov_chunks \\
        --fixed fv=/gpfs/.../length_index/finevision_v2 \\
        --workers 64,128,256 --seconds 120 --out orders/

    --dataset NAME=DIR[@MAX]   reorderable; @MAX keeps only tars with <= MAX samples (the
                               rest belong in another dataset, e.g. the split chunks)
    --fixed   NAME=DIR         contributes tokens but keeps its own (sorted) order
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


# ---------------------------------------------------------------- loading

def _load_one(path, old_cooker):
    t = pq.read_table(path, columns=["n_tokens", "skip"])
    keep = pa.array([""] if old_cooker else ["", "no_labels"])
    ok = pc.is_in(t.column("skip"), value_set=keep).to_numpy(zero_copy_only=False)
    return np.where(ok, t.column("n_tokens").to_numpy(), 0).astype(np.int64)


def load_dataset(name, index_dir, max_samples, old_cooker):
    files = sorted(glob.glob(os.path.join(index_dir, "*.parquet")))
    if not files:
        raise SystemExit(f"{name}: no parquet files in {index_dir}")
    with ThreadPoolExecutor(16) as pool:
        per_tar = list(pool.map(lambda f: _load_one(f, old_cooker), files))
    tars = [os.path.basename(f)[:-len(".parquet")].replace("__", "/") + ".tar" for f in files]
    if max_samples:
        keep = [i for i, a in enumerate(per_tar) if len(a) <= max_samples]
        tars, per_tar = [tars[i] for i in keep], [per_tar[i] for i in keep]
    return dict(name=name, tars=np.array(tars), arrays=per_tar,
                cnt=np.array([len(a) for a in per_tar]), tok=np.array([a.sum() for a in per_tar], float))


# ---------------------------------------------------------------- the model

def slice_tokens(cnt, tok, order, W):
    """Tokens per worker for one dataset in `order`, slice edges inside a tar interpolated."""
    c, t = cnt[order], tok[order]
    cs, ct = np.concatenate([[0], np.cumsum(c)]), np.concatenate([[0], np.cumsum(t)])
    b = (np.arange(W + 1) * cs[-1]) // W
    idx = np.clip(np.searchsorted(cs, b, side="right") - 1, 0, len(c) - 1)
    return np.diff(ct[idx] + (b - cs[idx]) / np.maximum(c[idx], 1) * t[idx])


def slice_tokens_exact(arrays, order, W):
    """Same, but on the per-sample token arrays (no interpolation): the verification."""
    a = np.concatenate([arrays[i] for i in order])
    c = np.concatenate([[0], np.cumsum(a)])
    b = (np.arange(W + 1) * len(a)) // W
    return np.diff(c[b]).astype(float)


def coverage(datasets, orders, fixed_tokens, Ws, pool, exact=False):
    """Fraction of the pool consumed when the first worker is empty, per worker count."""
    out = []
    for W in Ws:
        tw = fixed_tokens[W].copy()
        for d, o in zip(datasets, orders):
            tw = tw + (slice_tokens_exact(d["arrays"], o, W) if exact
                       else slice_tokens(d["cnt"], d["tok"], o, W))
        out.append(tw.min() * W / pool)
    return out


def objective(cov):
    return min(cov) + 0.1 * float(np.mean(cov))


# ---------------------------------------------------------------- search

def optimize(datasets, fixed_tokens, Ws, pool, seconds, restarts, seed):
    rng = np.random.default_rng(seed)
    sorted_orders = [np.argsort(d["tars"]) for d in datasets]
    base = coverage(datasets, sorted_orders, fixed_tokens, Ws, pool)

    best, best_obj = None, -1
    for _ in range(restarts):
        o = [rng.permutation(len(d["tars"])) for d in datasets]
        v = objective(coverage(datasets, o, fixed_tokens, Ws, pool))
        if v > best_obj:
            best, best_obj = o, v
    orders = [x.copy() for x in best]
    cur = coverage(datasets, orders, fixed_tokens, Ws, pool)
    score, t0, proposals, accepted = objective(cur), time.time(), 0, 0
    sizes = np.array([len(d["tars"]) for d in datasets], float)
    while time.time() - t0 < seconds:
        k = rng.choice(len(datasets), p=sizes / sizes.sum())
        n = len(orders[k])
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        proposals += 1
        orders[k][i], orders[k][j] = orders[k][j], orders[k][i]
        s = coverage(datasets, orders, fixed_tokens, Ws, pool)
        v = objective(s)
        if v >= score:
            score, cur, accepted = v, s, accepted + 1
        else:
            orders[k][i], orders[k][j] = orders[k][j], orders[k][i]
    return sorted_orders, base, orders, cur, proposals, accepted


def write_split(path, tars):
    with open(path, "w") as f:
        f.write("split_parts:\n  train:\n")
        f.writelines(f"  - {t}\n" for t in tars)
        f.write("  val: []\n  test: []\nexclude: []\n")


def parse_spec(spec):
    name, _, rest = spec.partition("=")
    if not rest:
        raise SystemExit(f"bad spec {spec!r}, want NAME=DIR[@MAX]")
    path, _, mx = rest.partition("@")
    return name, path, int(mx) if mx else 0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", action="append", default=[], help="NAME=DIR[@MAX] (reorderable)")
    ap.add_argument("--fixed", action="append", default=[], help="NAME=DIR (order kept)")
    ap.add_argument("--workers", default="64,128,256", help="dataloader worker counts to score")
    ap.add_argument("--seconds", type=float, default=90)
    ap.add_argument("--restarts", type=int, default=200, help="random orders tried before the swap search")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--old-cooker", action="store_true", help="samples with no assistant tokens do not count")
    ap.add_argument("--out", default="", help="write <name>.split.yaml here")
    ap.add_argument("--no-verify", action="store_true")
    a = ap.parse_args()
    if not a.dataset:
        ap.error("need at least one --dataset")
    Ws = [int(x) for x in a.workers.split(",")]

    t0 = time.time()
    datasets = [load_dataset(*parse_spec(s), a.old_cooker) for s in a.dataset]
    fixed = [load_dataset(*parse_spec(s), a.old_cooker) for s in a.fixed]
    for d in datasets + fixed:
        print(f"  {d['name']:12s} {len(d['tars']):6d} tars {d['cnt'].sum():>12,} samples {d['tok'].sum() / 1e9:7.2f}B tokens", flush=True)
    pool = sum(d["tok"].sum() for d in datasets + fixed)
    fixed_tokens = {W: sum((slice_tokens(d["cnt"], d["tok"], np.argsort(d["tars"]), W) for d in fixed),
                           np.zeros(W)) for W in Ws}
    print(f"loaded in {time.time() - t0:.0f}s; pool {pool / 1e9:.2f}B tokens; scoring W={Ws}", flush=True)

    sorted_orders, base, orders, cur, proposals, accepted = optimize(
        datasets, fixed_tokens, Ws, pool, a.seconds, a.restarts, a.seed)
    fmt = lambda c: "  ".join(f"W={W}: {100 * x:5.1f}%" for W, x in zip(Ws, c))      # noqa: E731
    print(f"\npool consumed before the first worker runs dry (model):")
    print(f"  sorted order   {fmt(base)}")
    print(f"  optimized      {fmt(cur)}   ({proposals:,} swaps tried, {accepted:,} kept)")

    if not a.no_verify:                        # per-sample arrays, no tar-level interpolation
        vb = coverage(datasets, sorted_orders, fixed_tokens_exact(fixed, Ws), Ws, pool, exact=True)
        vo = coverage(datasets, orders, fixed_tokens_exact(fixed, Ws), Ws, pool, exact=True)
        print("verified on per-sample tokens (no interpolation):")
        print(f"  sorted order   {fmt(vb)}")
        print(f"  optimized      {fmt(vo)}")

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        for d, o in zip(datasets, orders):
            new = [str(t) for t in d["tars"][o]]
            assert len(new) == len(set(new)) == len(d["tars"])
            p = os.path.join(a.out, f"{d['name']}.split.yaml")
            write_split(p, new)
            print(f"wrote {p} ({len(new)} tars)")


def fixed_tokens_exact(fixed, Ws):
    return {W: sum((slice_tokens_exact(d["arrays"], np.argsort(d["tars"]), W) for d in fixed), np.zeros(W))
            for W in Ws}


if __name__ == "__main__":
    main()
