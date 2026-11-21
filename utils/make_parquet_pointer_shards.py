"""Parquet rows -> webdataset tars of JSON pointers, for `cooker_parquet_ref`.

Each sample is {"kind": "parquet_image", "parquet_path", "row_group", "row_index",
"image_column", "messages_column"}, the format of ElliotVL's own manifests, so the
images stay in the parquet files and only a few MB of pointers are written. Reads the
parquet footers only (row counts per row group), never the data.

    python utils/make_parquet_pointer_shards.py OUT_DIR file1.parquet [file2.parquet ...] \\
        --image-column images --messages-column conversations
    energon prepare OUT_DIR --non-interactive --split-ratio 1,0,0 \\
        --sample-type CrudeWebdataset
"""

import argparse
import io
import json
import tarfile
import time
from pathlib import Path

import pyarrow.parquet as pq


def pointers(files, image_column, messages_column):
    for path in files:
        meta = pq.ParquetFile(path).metadata
        for rg in range(meta.num_row_groups):
            for row in range(meta.row_group(rg).num_rows):
                yield {
                    "kind": "parquet_image",
                    "parquet_path": str(Path(path).resolve()),
                    "row_group": rg,
                    "row_index": row,
                    "image_column": image_column,
                    "messages_column": messages_column,
                }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("files", nargs="+")
    ap.add_argument("--image-column", required=True)
    ap.add_argument("--messages-column", required=True)
    ap.add_argument("--per-shard", type=int, default=10000)
    ap.add_argument("--prefix", default="ptr")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=False)  # never write into an existing dataset
    shard, tar, n = -1, None, 0
    for n, ref in enumerate(pointers(sorted(args.files), args.image_column, args.messages_column)):
        if n % args.per_shard == 0:
            if tar:
                tar.close()
            shard += 1
            tar = tarfile.open(out / f"{args.prefix}-{shard:06d}.tar", "w")
        data = json.dumps(ref).encode()
        info = tarfile.TarInfo(f"{args.prefix}_{n:09d}.json")
        info.size, info.mtime = len(data), int(time.time())
        tar.addfile(info, io.BytesIO(data))
    if tar:
        tar.close()
    print(f"{n + 1} pointers in {shard + 1} shards -> {out}")


if __name__ == "__main__":
    main()
