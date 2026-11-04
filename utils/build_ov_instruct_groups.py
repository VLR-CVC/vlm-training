"""Split OV-Instruct into per-subset energon datasets via symlinks -- no tar is read or
rewritten. Subset name comes straight from the tar name (`<subset>_train-NNNNN-of-MMMMM
[.tar|-cNNNN.tar]`), same regex `length_index.py::subset_of` already uses for reporting,
so groups match the numbers you've already seen (133 folded subsets, e.g. Docmatix's 10
`-part-NN-of-10` pieces become one `Docmatix` group).

Subsets under `--small-threshold` samples are merged into one `small_subsets` bucket
instead of standing alone: a lone dataset of a few hundred samples starves almost every
dataloader worker under W=128/256 equal-sample slicing (see optimize_split_order.py's
worker model) -- it would sit in the blend but be invisible to most ranks in any one
pass. Everything else keeps its own directory for per-subset blend weighting.

Two physical sources feed the groups. `--src` (the original onevision_instruct tars) is
filtered to tars with <= `--giant-threshold` samples -- must match whatever
`split_tars_mn5.sh --min-samples` was run with, since tars over that were split into
`--src-chunks` instead and would otherwise be double-counted. A subset whose original
tar was split (orca_agentinstruct, openmathinstruct, ...) draws shards from both sources
into the same group directory.

    python utils/build_ov_instruct_groups.py \\
        --src /gpfs/scratch/ehpc543/onevision_instruct \\
        --src-chunks /gpfs/scratch/ehpc543/onevision_instruct_split \\
        --dst /gpfs/scratch/ehpc543/onevision_instruct_grouped \\
        --small-threshold 5000 --giant-threshold 10000 [--dry-run] [--workers 4]

Idempotent like build_nemotron_groups.py: a group with `.nv-meta/.info.json` already
present is skipped; rerun after a kill or after raising --small-threshold picks up only
what's missing (an already-built subset that later falls into the bucket is not undone
automatically -- delete its directory to fold it in).
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

    # A prior attempt on this exact path (crash, kill, or an energon prepare failure)
    # can leave a partial .nv-meta/index.sqlite with no .info.json. energon's own
    # dataset-type detection (index.sqlite present, no .info.json -> "FILESYSTEM
    # dataset") then misfires on the very next attempt with an unrelated-looking
    # "Filesystem datasets must be prepared using 'energon prepare-media'" error.
    # Clear any leftover prep state before every real (non-skip) attempt.
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
    # energon prepare regenerates .nv-meta/dataset.yaml itself (keeping only
    # __module__/__class__), so a subflavors block written before this point -- into
    # the group root, or even straight into .nv-meta -- gets silently dropped. It has
    # to be written here, after prepare, into .nv-meta/dataset.yaml specifically:
    # that's the file dataset_config.py actually reads at load time, not the root one.
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
