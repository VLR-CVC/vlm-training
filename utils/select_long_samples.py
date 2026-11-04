"""Pick the longest samples out of polylithic Nemotron-style jsonl datasets.

The loader skips any sample longer than one row (`PackedBatchEncoder` rejects
`length > row_length`), so the longest documents never reach the model and
streaming to *find* long ones is hopeless: every rejected sample still costs a
full image decode. This scans the jsonl text instead -- no images touched -- ranks
by an estimated token length, and writes filtered jsonls plus a metadataset
pointing at the same, untouched media stores.

    python utils/select_long_samples.py --root <staging dir> --glob 'long_document_*' \
        --out-dir <dir> --target-tokens 58000 --per-dataset 60

Estimate: `chars/CHARS_PER_TOKEN + images * TOKENS_PER_IMAGE`. Both are crude, so
`--target-tokens` should sit below the real row length to leave margin, and
over-selecting is free -- the probe takes what it needs and the rest are ignored.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

CHARS_PER_TOKEN = 4.0
# ~700-900 observed on CCPDF pages (job 2432310: 52049 tokens / 62 images,
# 37315 / 47, 29760 / 28, minus a few thousand tokens of text each).
TOKENS_PER_IMAGE = 800


def measure(record: dict) -> tuple[int, int]:
    """(characters of text, number of images) in one `{id, messages}` record."""
    chars = images = 0
    for turn in record.get("messages", []):
        content = turn.get("content", [])
        if isinstance(content, str):
            chars += len(content)
            continue
        for part in content:
            if isinstance(part, str):
                chars += len(part)
            elif part.get("type") == "image":
                images += 1
            elif part.get("text"):
                chars += len(part["text"])
    return chars, images


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir holding one subdir per dataset")
    ap.add_argument("--glob", default="long_document_*")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--target-tokens", type=int, default=58000,
                    help="keep the longest samples estimated at or below this")
    ap.add_argument("--min-tokens", type=int, default=0)
    ap.add_argument("--per-dataset", type=int, default=60)
    ap.add_argument("--metadataset", default="long_document_longest.yaml")
    a = ap.parse_args()

    root, out = Path(a.root), Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    entries = []

    for sub in sorted(root.glob(a.glob)):
        jsonl = sub / f"{sub.name}.jsonl"
        media = sub / "media"
        if not jsonl.is_file() or not (media / ".nv-meta").is_dir():
            continue
        kept, n, over = [], 0, 0
        with jsonl.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                n += 1
                rec = json.loads(line)
                chars, images = measure(rec)
                est = int(chars / CHARS_PER_TOKEN + images * TOKENS_PER_IMAGE)
                if est > a.target_tokens:
                    over += 1
                    continue
                if est < a.min_tokens:
                    continue
                kept.append((est, images, line))
        kept.sort(key=lambda x: -x[0])
        kept = kept[: a.per_dataset]
        if not kept:
            print(f"{sub.name:>26}: {n:>7} samples, none in range")
            continue
        dst = out / f"{sub.name}_longest.jsonl"
        # energon reads `<jsonl>.idx`: little-endian int64 byte offsets, one per
        # line plus a trailing offset at EOF. Without it the join dies with
        # `FileNotFoundError: ... .jsonl.idx` (job 2432499).
        offsets = []
        with dst.open("wb") as fh:
            for _est, _img, line in kept:
                offsets.append(fh.tell())
                fh.write(line.encode() + b"\n")
            offsets.append(fh.tell())
        dst.with_suffix(".jsonl.idx").write_bytes(
            struct.pack(f"<{len(offsets)}q", *offsets))
        entries.append((dst, media))
        print(f"{sub.name:>26}: {n:>7} samples, {over:>6} over target, kept {len(kept):>3} "
              f"est {kept[-1][0]:>6}-{kept[0][0]:>6} tok, {kept[0][1]:>3} images max")

    lines = ["__module__: megatron.energon", "__class__: MetadatasetV2",
             "splits:", "  train:", "    blend_epochized:"]
    for jsonl, media in entries:
        lines += [f"      - path: {jsonl}", "        repetitions: 1",
                  "        subflavors:", "          type_dataset: nemotron_v2",
                  "        aux:", f"          media_source: {media}"]
    mds = out / a.metadataset
    mds.write_text("\n".join(lines) + "\n")
    print(f"\nwrote {mds}: {len(entries)} datasets")


if __name__ == "__main__":
    main()
