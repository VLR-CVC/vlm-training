"""Repack a Nemotron-VLM-Dataset-v2 subset into an energon CrudeWebdataset.

The subset as published is *two* halves that energon cannot join on its own:

    <subset>/media/           an energon CrudeWebdataset of the images alone,
                              keyed by bare filename ("4189.png")
    <subset>/<subset>.jsonl   the conversations, each referencing an image by
                              that filename

`media/.nv-meta` describes the image store, not the training samples, so
pointing the trainer at it yields images with no conversations. This script
joins the two and writes shards whose entries are what `PackedBatchEncoder`
expects: one `.json` conversation plus its `.png`, sharing a basename.

Images are duplicated where several samples cite the same file (plotqa_cot:
16,256 samples over 8,212 images). That is deliberate -- energon streams
sequentially and cannot reach sideways to a shared image, and the duplication
costs ~260 MB here.

    python utils/prepare_nemotron_energon.py RAW_DIR OUT_DIR [--per-shard N]
    energon prepare OUT_DIR --non-interactive --sample-type CrudeWebdataset \
        --split-ratio 1,0,0
"""

from __future__ import annotations

import argparse
import json
import os
import io
import tarfile


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", help="downloaded subset dir (holds media/ and *.jsonl)")
    ap.add_argument("out_dir")
    ap.add_argument("--per-shard", type=int, default=4000)
    args = ap.parse_args()

    raw, out = args.raw_dir, args.out_dir
    name = os.path.basename(raw.rstrip("/"))
    jsonl = os.path.join(raw, f"{name}.jsonl")
    media = os.path.join(raw, "media")
    os.makedirs(out, exist_ok=True)

    # Every sample needs random access into the media store, but the store is
    # not always small enough to hold (271 MB for plotqa_cot, 13.4 GB for
    # clevr_1). Index it by member and seek per read instead: ~200 B/entry.
    index: dict[str, tuple[tarfile.TarFile, tarfile.TarInfo]] = {}
    for shard in sorted(f for f in os.listdir(media) if f.endswith(".tar")):
        t = tarfile.open(os.path.join(media, shard))
        for m in t:
            if m.isfile():
                index[m.name] = (t, m)
    print(f"{len(index)} media files across {len(set(id(t) for t, _ in index.values()))} shards")

    def blob(name: str) -> bytes:
        t, m = index[name]
        return t.extractfile(m).read()

    written = skipped = shard_idx = 0
    tar = None

    def open_shard(i: int) -> tarfile.TarFile:
        return tarfile.open(os.path.join(out, f"shard-{i:06d}.tar"), "w")

    counts: dict[str, int] = {}
    for line in open(jsonl):
        rec = json.loads(line)

        imgs = [
            c["image"]
            for m in rec["messages"]
            for c in m["content"]
            # v3 interleaves bare strings with the part dicts
            if isinstance(c, dict) and c.get("type") == "image" and c.get("image")
        ]
        # A sample whose image is absent would train as text-only against a
        # conversation that says "look at this chart". Drop it rather than
        # silently mislabel it.
        if len(imgs) != 1 or imgs[0] not in index:
            skipped += 1
            continue

        if written % args.per_shard == 0:
            if tar is not None:
                tar.close()
                shard_idx += 1
            tar = open_shard(shard_idx)
            counts[f"shard-{shard_idx:06d}.tar"] = 0

        key = f"{written:08d}"
        ext = os.path.splitext(imgs[0])[1].lstrip(".").lower()
        payload = json.dumps(rec, ensure_ascii=False).encode()

        for suffix, data in ((ext, blob(imgs[0])), ("json", payload)):
            info = tarfile.TarInfo(f"{key}.{suffix}")
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))

        counts[f"shard-{shard_idx:06d}.tar"] += 1
        written += 1

    if tar is not None:
        tar.close()

    print(f"wrote {written} samples across {shard_idx + 1} shards, skipped {skipped}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
