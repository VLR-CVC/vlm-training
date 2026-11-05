"""Merge Nemotron's 30 valid (hosted-media, non-video) subsets into fewer, coarser
energon datasets, by symlinking existing shards -- no tar is read or rewritten.

Why: v1/v2/v3 each ship the same logical subset under a different name and version
(`long_document_sec_1..4`, `plotqa_cot` + `plotqa_1/2`, ...). Training one dataset per
raw subset means 30 tiny blend entries with no natural weight; grouping them into the
handful of things they actually are (one OCR set, one long-document set, ...) gives a
blend you can reason about and weight. Every member was already repacked to the same
`<key>.json` (`messages`) + `<key>.NN.<ext>` shape by `prepare_nemotron_energon.py` /
`prepare_nemotron_v1_energon.py`, so merging v1/v2/v3 members byte-for-byte is safe --
`cooker_nemotron` (subflavor `type_dataset: nemotron`) reads all of them the same way.

The wiki_en / wiki_multilingual split is deliberate, not a naming artifact: this model
is English-only for OCR SFT, so the other 9 languages are an out-of-distribution probe,
not additional in-distribution weight, and are kept as their own group rather than
merged into wiki_en or left as 9 separate tiny blend entries.

    python utils/build_nemotron_groups.py --src /gpfs/scratch/ehpc543/nemotron_valid \\
        --dst /gpfs/scratch/ehpc543/nemotron_grouped [--workers 4] [--dry-run]

Idempotent: a group already prepared (its `.nv-meta/.info.json` exists) is skipped;
rerun after adding a member or after a kill picks up where it left off (existing
symlinks are left alone, `energon prepare` reruns only for groups still missing output).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

GROUPS: dict[str, list[str]] = {
    "ocr": ["v1/ocr_1", "v1/ocr_2", "v1/ocr_3", "v1/ocr_4", "v1/ocr_5"],
    "vqa_9": ["v1/vqa_9"],
    "plotqa": ["v2/plotqa_cot", "v3/plotqa_1", "v3/plotqa_2"],
    "clevr": ["v3/clevr_1", "v3/clevr_2"],
    "long_document_sec": [f"v3/long_document_sec_{i}" for i in range(1, 5)],
    "zhwiki": ["v3/zhwiki_1", "v3/zhwiki_2"],
    "wiki_en": ["v2/wiki_en"],
    # Model trains English-only OCR SFT: the other 9 languages are an OOD probe, kept
    # as one bucket rather than merged into wiki_en or left as 9 tiny blend entries.
    "wiki_multilingual": [f"v2/wiki_{l}" for l in
                          ("de", "es", "fr", "it", "ja", "ko", "nl", "pt", "zh")],
    "nights_cot": ["v2/nights_cot"],
    "sparsetables": ["v2/sparsetables"],
    "turing": ["v3/turing"],
}

DATASET_YAML = "__module__: megatron.energon\n__class__: CrudeWebdataset\nsubflavors:\n  type_dataset: nemotron\n"

def link_group(src, dst, group, members, dry_run):
    out = os.path.join(dst, group)
    info = os.path.join(out, ".nv-meta", ".info.json")
    if os.path.exists(info):
        return group, "skipped (already prepared)", 0

    os.makedirs(out, exist_ok=True)
    n = 0
    for member in members:
        key, subset = member.split("/", 1)
        src_dir = os.path.join(src, key, subset)
        if not os.path.isdir(src_dir):
            raise FileNotFoundError(f"{group}: missing member dir {src_dir}")
        for f in sorted(os.listdir(src_dir)):
            if not f.endswith(".tar"):
                continue
            link_name = f"{key}_{subset}_{f}"     # unique across members, incl. same-index shards
            link_path = os.path.join(out, link_name)
            if dry_run:
                n += 1
                continue
            if not os.path.islink(link_path):
                os.symlink(os.path.join(src_dir, f), link_path)
            n += 1
    if dry_run:
        return group, f"would link {n} shards from {len(members)} member(s)", n

    cmd = ["energon", "prepare", out, "--non-interactive",
          "--sample-type", "CrudeWebdataset", "--split-ratio", "1,0,0"]
    if len(members) > 1:
        cmd.append("--fix-duplicates")
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"energon prepare failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    with open(os.path.join(out, ".nv-meta", "dataset.yaml"), "w") as f:
        f.write(DATASET_YAML)
    return group, f"prepared, {n} shards from {len(members)} member(s)", n

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="nemotron_valid root (has <key>/<subset>/shard-*.tar)")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    if not a.dry_run:
        os.makedirs(a.dst, exist_ok=True)
    with ThreadPoolExecutor(a.workers) as pool:
        futs = {pool.submit(link_group, a.src, a.dst, g, m, a.dry_run): g for g, m in GROUPS.items()}
        for fut in as_completed(futs):
            g = futs[fut]
            try:
                _, status, n = fut.result()
                print(f"  {g:20s} {status}", flush=True)
            except Exception as exc:                                        # noqa: BLE001
                print(f"  {g:20s} FAILED: {exc}", flush=True)

    total_members = sum(len(m) for m in GROUPS.values())
    print(f"\n{len(GROUPS)} groups from {total_members} subsets"
          + (" (dry run, nothing written)" if a.dry_run else f" -> {a.dst}"))


if __name__ == "__main__":
    main()
