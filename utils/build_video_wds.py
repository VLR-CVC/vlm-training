"""LLaVA-style video annotations -> webdataset tars of pre-sampled JPEG frames, for
`cooker_video_frames` (type_dataset `video_frames`).

Built for LLaVA-Video-178K (`lmms-lab/LLaVA-Video-178K`: `*_processed.json` rows of
{"id", "video", "conversations", "data_source"}, videos in `*_videos_N.tar.gz`), but any
json/jsonl of that shape works. Rows that share a video become one multi-turn sample
(ActivityNet-QA has 10 one-question rows per clip): the clip is decoded and stored once.

Frames are sampled uniformly from t=0 at `--fps`; a clip longer than
`--max-frames / --fps` seconds is covered by `--max-frames` frames at a lower rate, and
the rate actually used goes in the sample, so the processor's `<t seconds>` stamps are
right. Decoding needs PyAV, so run this in a data-prep env, not the training env.

    python utils/build_video_wds.py OUT_DIR --video-root VIDEOS \\
        --ann 0_30_s_academic_v0_1/*_processed.json --prefix 0_30_s_academic --workers 32
    energon prepare OUT_DIR --non-interactive --split-ratio 1,0,0 --sample-type CrudeWebdataset
    printf '__module__: megatron.energon\\n__class__: CrudeWebdataset\\nsubflavors:\\n  type_dataset: video_frames\\n' \\
        > OUT_DIR/.nv-meta/dataset.yaml
"""

import argparse
import io
import json
import os
import re
import tarfile
import time
from multiprocessing import Pool

import av
from PIL import Image

PLACEHOLDER = re.compile(r"<(image|video)>\s*")


def sample_frames(path, fps, max_frames, max_side):
    """(JPEG bytes per frame, rate used). Frames at k / rate s, nearest decoded frame at or after."""
    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        if stream.duration:
            duration = float(stream.duration * stream.time_base)
        else:
            duration = container.duration / av.time_base
        n = max(1, min(max_frames, int(duration * fps)))
        n -= n % 2 if n > 1 else 0  # temporal patches hold 2 frames; an odd last one is padded
        rate = n / duration if duration * fps > max_frames else fps
        targets = [k / rate for k in range(n)]
        out = []
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            if float(frame.pts * stream.time_base) + 1e-3 < targets[len(out)]:
                continue
            img = frame.to_image()
            img.thumbnail((max_side, max_side), Image.BICUBIC)
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=90)
            out.append(buf.getvalue())
            if len(out) == n:
                break
    if len(out) > 1 and len(out) % 2:
        out.pop()  # metadata promised more frames than the stream had
    return out, rate


def to_messages(rows, video_part):
    """Merge the rows' conversations; the video goes at the start of the first user turn."""
    messages = []
    for row in rows:
        for turn in row["conversations"]:
            role = "user" if turn["from"] in ("human", "user") else "assistant"
            text = PLACEHOLDER.sub("", turn["value"]).strip()
            content = [text] if text else []
            if not messages:
                content.insert(0, video_part)
            messages.append({"role": role, "content": content})
    return messages


def build(job):
    key, path, rows, a = job
    if a.raw:
        ext = os.path.splitext(path)[1].lstrip(".").lower() or "mp4"
        try:
            with open(path, "rb") as f:
                raw_bytes = f.read()
        except Exception as e:
            return key, None, f"{type(e).__name__}: {e}"
        if not raw_bytes:
            return key, None, "empty video file"
        sample = {"messages": to_messages(rows, {"type": "video", "video": ext}),
                  "source": rows[0].get("data_source"),
                  "ids": [r["id"] for r in rows], "video": rows[0]["video"]}
        return key, [(ext, raw_bytes), ("json", json.dumps(sample).encode())], None
    try:
        frames, rate = sample_frames(path, a.fps, a.max_frames, a.max_side)
    except Exception as e:  # corrupt or unreadable clip: skip it, say why
        return key, None, f"{type(e).__name__}: {e}"
    if not frames:
        return key, None, "no frames decoded"
    names = [f"f{i:03d}.jpg" for i in range(len(frames))]
    video_part = {"type": "video", "frames": names, "fps": rate}
    sample = {"messages": to_messages(rows, video_part), "source": rows[0].get("data_source"),
              "ids": [r["id"] for r in rows], "video": rows[0]["video"]}
    return key, [*zip(names, frames), ("json", json.dumps(sample).encode())], None


def load_rows(paths):
    for p in paths:
        with open(p) as f:
            data = json.load(f) if p.endswith(".json") else [json.loads(line) for line in f]
        yield from data


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out")
    ap.add_argument("--ann", nargs="+", required=True, help="annotation .json / .jsonl files")
    ap.add_argument("--video-root", required=True, help="where the video tars were extracted")
    ap.add_argument("--prefix", default="video", help="shard name prefix")
    ap.add_argument("--raw", action="store_true", help="store raw video files for online AVDecoder (type_dataset: video)")
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--max-frames", type=int, default=32)
    ap.add_argument("--max-side", type=int, default=448, help="longest frame side, px")
    ap.add_argument("--shard-size", type=int, default=500, help="samples (clips) per tar")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    # annotation paths do not always match the tar layout (ActivityNet: train/v1-3/train_val/x.mp4
    # in the json, train/x.mp4 in the tar): fall back to the file name
    by_name = {}
    for root, _, files in os.walk(a.video_root):
        for f in files:
            by_name.setdefault(f, os.path.join(root, f))
    groups = {}
    for row in load_rows(a.ann):
        if row.get("video"):
            groups.setdefault(row["video"], []).append(row)
    jobs, missing = [], 0
    for video, rows in groups.items():
        path = os.path.join(a.video_root, video)
        path = path if os.path.isfile(path) else by_name.get(os.path.basename(video))
        if path is None:
            missing += 1
            continue
        key = re.sub(r"[^A-Za-z0-9_-]", "_", f"{a.prefix}_{os.path.splitext(video)[0]}")
        jobs.append((key, path, rows, a))
    print(f"{sum(map(len, groups.values()))} rows, {len(groups)} videos, {missing} not found under {a.video_root}")

    os.makedirs(a.out, exist_ok=True)
    t0, written, failed, tar = time.time(), 0, 0, None
    with Pool(a.workers) as pool:
        for key, members, err in pool.imap(build, jobs, chunksize=4):
            if members is None:
                failed += 1
                print(f"skip {key}: {err}")
                continue
            if written % a.shard_size == 0:
                if tar:
                    tar.close()
                tar = tarfile.open(os.path.join(a.out, f"{a.prefix}-{written // a.shard_size:06d}.tar"), "w")
            for name, data in members:
                info = tarfile.TarInfo(f"{key}.{name}")
                info.size, info.mtime = len(data), int(t0)
                tar.addfile(info, io.BytesIO(data))
            written += 1
            if written % 1000 == 0:
                print(f"{written}/{len(jobs)} clips, {time.time() - t0:.0f}s")
    if tar:
        tar.close()
    print(f"wrote {written} clips in {-(-written // a.shard_size)} tars to {a.out}, {failed} failed, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
