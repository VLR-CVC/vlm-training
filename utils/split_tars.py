"""Split oversized energon WebDataset tars into small tars, subset by subset.

Resumable: a chunk set is written under a `.tmp` name and renamed when the source tar's
sample count matches `.info.json`; finished tars are skipped on rerun.

    python utils/split_tars.py --src /gpfs/scratch/ehpc543/onevision_instruct \\
        --dst /gpfs/scratch/ehpc543/onevision_instruct_split --min-samples 10000 --workers 32
    python utils/split_tars.py ... --dry-run          # what would be split, and how big
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def split_one(src_tar, dst_dir, expect, chunk):
    stem = os.path.basename(src_tar)[:-4]
    done = os.path.join(dst_dir, f".done.{stem}")
    if os.path.exists(done):
        return stem, 0, 0, "skipped"
    for f in os.listdir(dst_dir):                 # leftovers of a killed run
        if f.startswith(stem + "-c") and (f.endswith(".tmp") or f.endswith(".tar")):
            os.remove(os.path.join(dst_dir, f))

    out, n_chunks, n_samples, in_chunk, cur_key = None, 0, 0, 0, None
    written = []

    def open_chunk():
        nonlocal out, n_chunks, in_chunk
        path = os.path.join(dst_dir, f"{stem}-c{n_chunks:04d}.tar")
        out = tarfile.open(path + ".tmp", "w")
        written.append(path)
        n_chunks += 1
        in_chunk = 0

    def close_chunk():
        if out is not None:
            out.close()

    with tarfile.open(src_tar, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            d, base = os.path.split(m.name)
            if "." not in base:
                # Extension-less member (some OV-Instruct tars open with a few, e.g.
                # "<stem>/000000"): energon ignores them, so they are not samples and must
                # not be counted or start a chunk -- copied through unchanged.
                if out is None:
                    open_chunk()
                orphan = tarfile.TarInfo(m.name)
                orphan.size, orphan.mtime, orphan.mode = m.size, m.mtime, m.mode
                orphan.uid, orphan.gid, orphan.uname, orphan.gname = m.uid, m.gid, m.uname, m.gname
                out.addfile(orphan, tf.extractfile(m))
                continue
            key = os.path.join(d, base.partition(".")[0]) if d else base.partition(".")[0]
            if key != cur_key:                    # a new sample begins
                cur_key = key
                n_samples += 1
                if out is None or in_chunk >= chunk:
                    close_chunk()
                    open_chunk()
                in_chunk += 1
            info = tarfile.TarInfo(m.name)
            info.size, info.mtime, info.mode = m.size, m.mtime, m.mode
            info.uid, info.gid, info.uname, info.gname = m.uid, m.gid, m.uname, m.gname
            out.addfile(info, tf.extractfile(m))
    close_chunk()

    if n_samples != expect:
        for p in written:
            os.remove(p + ".tmp")
        raise RuntimeError(f"{stem}: wrote {n_samples} samples, .info.json says {expect}")
    for p in written:
        os.replace(p + ".tmp", p)
    open(done, "w").close()
    return stem, n_samples, n_chunks, "ok"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="energon dataset dir (has .nv-meta/.info.json)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min-samples", type=int, default=10000)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    counts = json.load(open(os.path.join(a.src, ".nv-meta", ".info.json")))["shard_counts"]
    big = sorted((t for t, n in counts.items() if n > a.min_samples), key=lambda t: -counts[t])
    total = sum(counts[t] for t in big)
    gb = sum(os.path.getsize(os.path.join(a.src, t)) for t in big) / 1e9
    print(f"{len(big)} of {len(counts)} tars have > {a.min_samples} samples: {total:,} samples "
          f"({100 * total / sum(counts.values()):.0f}% of the dataset), {gb:.0f} GB "
          f"-> ~{sum(-(-counts[t] // a.chunk) for t in big)} chunk tars of ~{a.chunk}", flush=True)
    if a.dry_run:
        return
    os.makedirs(a.dst, exist_ok=True)
    mine = big[a.task_id::a.num_tasks]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as pool:
        futs = {pool.submit(split_one, os.path.join(a.src, t), a.dst, counts[t], a.chunk): t for t in mine}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                stem, n, k, status = fut.result()
                print(f"[{i}/{len(mine)}] {stem}: {status} {n} samples -> {k} chunks "
                      f"({time.time() - t0:.0f}s)", flush=True)
            except Exception as exc:                                   # noqa: BLE001
                print(f"[{i}/{len(mine)}] FAILED {futs[fut]}: {exc}", flush=True)


if __name__ == "__main__":
    main()
