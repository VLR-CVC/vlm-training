"""Split OV-Instruct into per-subset energon datasets via symlinks -- no tar is read or
rewritten. Subset name comes straight from the tar name (`<subset>_train-NNNNN-of-MMMMM
[.tar|-cNNNN.tar]`)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

_SUBSET_RE = re.compile(r"^(.+?)_train-")
_PART_RE = re.compile(r"-part-\d+-of-\d+$")

DATASET_YAML = ("__module__: megatron.energon\n__class__: CrudeWebdataset\n"
               "subflavors:\n  type_dataset: onevision_instruct\n")

def subset_of(tar_name):
    """Same rule as length_index.py::subset_of -- keep them in sync."""
    m = _SUBSET_RE.match(tar_name)
    head = m.group(1) if m else "unknown"
    return _PART_RE.sub("", head)


def load_shard_counts(root):
    with open(os.path.join(root, ".nv-meta", ".info.json")) as f:
        return json.load(f)["shard_counts"]

def plan_groups(src, src_chunks, giant_threshold, small_threshold):
    orig = load_shard_counts(src)
    chunks = load_shard_counts(src_chunks)
    members = {}    # tar_name -> (source_root, count)
    for name, count in orig.items():
        if count <= giant_threshold:      # over threshold: superseded by src_chunks
            members[name] = (src, count)
    for name, count in chunks.items():
        members[name] = (src_chunks, count)

    totals = {}
    for name, (_root, count) in members.items():
        s = subset_of(name)
        totals[s] = totals.get(s, 0) + count

    groups = {}    # group -> [(root, tar_name), ...]
    for name, (root, _count) in members.items():
        s = subset_of(name)
        group = s if totals[s] >= small_threshold else "small_subsets"
        groups.setdefault(group, []).append((root, name))
    return groups, totals

def build_group(dst, group, members, dry_run, fix_duplicates):
    out = os.path.join(dst, group)
    info = os.path.join(out, ".nv-meta", ".info.json")
    if os.path.exists(info):
        return group, "skipped (already prepared)", 0

    if not dry_run and os.path.isdir(os.path.join(out, ".nv-meta")):
        shutil.rmtree(os.path.join(out, ".nv-meta"))

    os.makedirs(out, exist_ok=True)
    for root, tar_name in members:
        if dry_run:
            continue
        link_name = f"{os.path.basename(root)}_{tar_name}"
        link_path = os.path.join(out, link_name)
        if not os.path.islink(link_path):
            os.symlink(os.path.join(root, tar_name), link_path)
    if dry_run:
        return group, f"would link {len(members)} shards", len(members)

    cmd = ["energon", "prepare", out, "--non-interactive",
          "--sample-type", "CrudeWebdataset", "--split-ratio", "1,0,0"]
    if fix_duplicates:
        cmd.append("--fix-duplicates")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"energon prepare failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    with open(os.path.join(out, ".nv-meta", "dataset.yaml"), "w") as f:
        f.write(DATASET_YAML)
    return group, f"prepared, {len(members)} shards", len(members)

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True)
    ap.add_argument("--src-chunks", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--small-threshold", type=int, default=5000)
    ap.add_argument("--giant-threshold", type=int, default=10000,
                    help="must match split_tars_mn5.sh's --min-samples")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--fix-duplicates", action="store_true",
                    help="energon's duplicate-key remediation (needs numba); OV-Instruct's own "
                         "keys embed the full tar stem so this shouldn't be needed, but pass it "
                         "if `energon prepare` complains about duplicate keys")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    groups, totals = plan_groups(a.src, a.src_chunks, a.giant_threshold, a.small_threshold)
    bucketed = [s for s, t in totals.items() if t < a.small_threshold]
    print(f"{len(groups)} groups from {len(totals)} subsets "
          f"({len(bucketed)} subsets bucketed into 'small_subsets', {sum(totals[s] for s in bucketed):,} samples)",
          flush=True)

    if not a.dry_run:
        os.makedirs(a.dst, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(build_group, a.dst, g, m, a.dry_run, a.fix_duplicates): g
                for g, m in groups.items()}
        for fut in as_completed(futs):
            g = futs[fut]
            try:
                _, status, n = fut.result()
                print(f"  {g:20s} {status}", flush=True)
            except Exception as exc:                                        # noqa: BLE001
                print(f"  {g:20s} FAILED: {exc}", flush=True)

    print(f"\n-> {a.dst}" if not a.dry_run else "\n(dry run, nothing written)")


if __name__ == "__main__":
    main()
