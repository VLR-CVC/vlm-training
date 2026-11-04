"""Generate an energon MetadatasetV2 yaml blending every prepared dataset from the
per-subset/per-group split work: FineVision (as one dataset until its own per-source
split lands), all OV-Instruct groups (utils/build_ov_instruct_groups.py), and all
Nemotron groups (utils/build_nemotron_groups.py).

Only datasets with a valid `.nv-meta/.info.json` at generation time are included --
anything still mid-`energon prepare` (or not yet built) is skipped and named at the
end, not silently left out of the run without a trace. Rerun after the rest finish to
pick them up; this never edits a yaml in place, only writes a fresh one.

Every path gets `repetitions: 1` (blend_epochized default -- roughly proportional to
each dataset's own sample count, same convention the original two-dataset instruct.yaml
uses). Adjust by hand afterward if you want a different mixture.

    python utils/build_metadataset.py --out /gpfs/scratch/ehpc543/metadatasets/full_grouped.yaml
"""

from __future__ import annotations

import argparse
import glob
import json
import os


def is_ready(path):
    return os.path.exists(os.path.join(path, ".nv-meta", ".info.json"))


def sample_count(entry):
    """Real sample count, for --min-samples filtering. Monolithic datasets (bare path)
    keep it in .nv-meta/.info.json's shard_counts; the jsonl+aux ones (Nemotron-v2 style)
    don't have an energon-side count for the jsonl itself, so it's a line count."""
    if isinstance(entry, dict):
        with open(entry["jsonl"], "rb") as f:
            return sum(1 for _ in f)
    with open(os.path.join(entry, ".nv-meta", ".info.json")) as f:
        return sum(json.load(f).get("shard_counts", {}).values())


def is_ready_v2(subdir):
    """Nemotron-VLM-Dataset-v2 layout: `<name>/<name>.jsonl` + `<name>/media/` prepared
    as its own energon dataset (joined in as an `aux` FileStore, not a blend path)."""
    name = os.path.basename(subdir.rstrip("/"))
    jsonl = os.path.join(subdir, f"{name}.jsonl")
    media = os.path.join(subdir, "media")
    return os.path.isfile(jsonl) and is_ready(media), jsonl, media


def collect(a):
    ready, skipped = [], []

    def add(path):
        (ready if is_ready(path) else skipped).append(path)

    if a.finevision:
        add(a.finevision)
    for g in sorted(glob.glob(os.path.join(a.ov_grouped, "*"))):
        if os.path.isdir(g):
            add(g)
    for g in sorted(glob.glob(os.path.join(a.nemotron_grouped, "*"))):
        if os.path.isdir(g):
            add(g)
    if a.nemotron_v2:
        for g in sorted(glob.glob(os.path.join(a.nemotron_v2, "*"))):
            if not os.path.isdir(g):
                continue
            okay, jsonl, media = is_ready_v2(g)
            if okay:
                ready.append({"jsonl": jsonl, "media": media})
            else:
                skipped.append(g)
    return ready, skipped


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--finevision", default="/gpfs/scratch/ehpc543/finevision_energon")
    ap.add_argument("--ov-grouped", default="/gpfs/scratch/ehpc543/onevision_instruct_grouped")
    ap.add_argument("--nemotron-grouped", default="/gpfs/scratch/ehpc543/nemotron_grouped")
    ap.add_argument("--nemotron-v2", default=None,
                    help="root of Nemotron-VLM-Dataset-v2 (e.g. /p/data1/mmlaion/Nemotron-VLM-Dataset-v2); "
                         "each <name>/<name>.jsonl + <name>/media/ subset becomes a jsonl+aux blend entry")
    ap.add_argument("--min-samples", type=int, default=0,
                    help="drop any dataset with fewer real samples than this -- a tiny dataset under "
                         "blend_epochized's repetitions:1 still contributes almost nothing to an epoch, "
                         "but one with fewer samples than the rank count (e.g. 256) starves most ranks "
                         "of it entirely every pass")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    ready, skipped = collect(a)
    if not ready:
        raise SystemExit("no prepared datasets found -- check --finevision/--ov-grouped/--nemotron-grouped")

    too_small = []
    if a.min_samples:
        kept = []
        for entry in ready:
            n = sample_count(entry)
            if n < a.min_samples:
                too_small.append((entry["jsonl"] if isinstance(entry, dict) else entry, n))
            else:
                kept.append(entry)
        ready = kept

    lines = ["__module__: megatron.energon", "__class__: MetadatasetV2", "splits:", "  train:",
            "    blend_epochized:"]
    for entry in ready:
        if isinstance(entry, dict):  # Nemotron-VLM-Dataset-v2: jsonl + separate media webdataset
            lines += [
                f"      - path: {entry['jsonl']}",
                "        repetitions: 1",
                f"        subflavors: {{type_dataset: nemotron_v2}}",
                "        aux:",
                f"          media_source: {entry['media']}",
            ]
        else:
            lines += [f"      - path: {entry}", "        repetitions: 1"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"{len(ready)} datasets -> {a.out}")
    if skipped:
        print(f"skipped ({len(skipped)}, no .nv-meta yet):")
        for s in skipped:
            print(f"  {s}")
    if too_small:
        print(f"dropped ({len(too_small)}, < {a.min_samples} samples):")
        for name, n in sorted(too_small, key=lambda x: x[1]):
            print(f"  {name}: {n}")


if __name__ == "__main__":
    main()
