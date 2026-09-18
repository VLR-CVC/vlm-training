import argparse
import io
import json
import os
import statistics
import sys
import tarfile

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.cookers import nemotron_messages  # noqa: E402
from data.energon_dataloader import cap_image_size  # noqa: E402

MAX_PIXELS = 1048576  # train_qwen.py:308


def iter_samples(dataset_dir, limit):
    """Yield (key, messages, PIL image) from the energon shards, in shard order.

    Shard order, not shuffled: the comparison is between tokenizers on a fixed
    sample set, so which samples are drawn matters less than that every model
    sees the same ones.
    """
    shards = sorted(f for f in os.listdir(dataset_dir) if f.endswith(".tar"))
    n = 0
    for shard in shards:
        with tarfile.open(os.path.join(dataset_dir, shard)) as t:
            pending = {}
            for m in t:
                if not m.isfile():
                    continue
                key, _, ext = m.name.rpartition(".")
                if ext not in ("json", "png", "jpg", "jpeg"):
                    continue
                pending.setdefault(key, {})[ext] = t.extractfile(m).read()
                blobs = pending[key]
                img_ext = next((e for e in ("png", "jpg", "jpeg") if e in blobs), None)
                if "json" not in blobs or img_ext is None:
                    continue
                meta = json.loads(blobs["json"])
                image = Image.open(io.BytesIO(blobs[img_ext])).convert("RGB")
                del pending[key]
                yield key, nemotron_messages(meta["messages"]), image
                n += 1
                if limit and n >= limit:
                    return


def image_token_id(processor):
    """Resolve the image placeholder id across processor layouts.

    Raises rather than guessing: a wrong id silently reports every image token as
    text, which is the exact confusion this script exists to settle.
    """
    tok = processor.tokenizer
    for attr in ("image_token", "image_pad_token"):
        s = getattr(processor, attr, None)
        if s:
            tid = tok.convert_tokens_to_ids(s)
            if tid is not None and tid != tok.unk_token_id:
                return tid
    for s in ("<|image_pad|>", "<image>", "<|image|>"):
        tid = tok.convert_tokens_to_ids(s)
        if tid is not None and tid != tok.unk_token_id:
            return tid
    raise SystemExit(f"cannot resolve image token id for {processor.__class__.__name__}")


def greedy_pack(lengths, seq_len):
    """Documents per row and occupancy under sequential first-fit.

    The trainer packs out of a shuffled buffer, so this is not identical to it --
    but it is a measurement of the same quantity rather than `seq_len / mean`,
    which is what §17.6 used and which ignores that a document too long to fit
    leaves the remainder of the row empty.
    """
    rows, cur, used = 0, 0, 0
    for n in lengths:
        if n > seq_len:
            continue  # SkipSample in training; counted separately
        if cur + n > seq_len:
            rows += 1
            used += cur
            cur = 0
        cur += n
    if cur:
        rows += 1
        used += cur
    if rows == 0:
        return 0.0, 0.0
    packed = sum(1 for n in lengths if n <= seq_len)
    return packed / rows, 100.0 * used / (rows * seq_len)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--limit", type=int, default=2000)
    ap.add_argument("--seq-len", type=int, default=10240)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    from transformers import AutoProcessor

    procs = {}
    for d in args.models:
        name = os.path.basename(d.rstrip("/"))
        p = AutoProcessor.from_pretrained(d, max_pixels=MAX_PIXELS)
        procs[name] = (p, image_token_id(p))
        print(f"loaded {name}: vocab {len(p.tokenizer)} image_token_id {procs[name][1]}",
              file=sys.stderr)

    stats = {n: {"total": [], "text": [], "image": []} for n in procs}

    for i, (_key, messages, image) in enumerate(iter_samples(args.dataset, args.limit)):
        capped = cap_image_size(image)
        for name, (p, img_id) in procs.items():
            text = p.apply_chat_template(
                conversation=messages, tokenize=False, add_generation_prompt=False
            )
            ids = p(text=[text], images=[capped], padding=False,
                    return_tensors="pt")["input_ids"][0]
            n_img = int((ids == img_id).sum())
            stats[name]["total"].append(len(ids))
            stats[name]["image"].append(n_img)
            stats[name]["text"].append(len(ids) - n_img)
        if (i + 1) % 200 == 0:
            print(f"  {i + 1} samples", file=sys.stderr)

    n = len(next(iter(stats.values()))["total"])
    if n == 0:
        raise SystemExit("no samples read -- check --dataset")

    out = {"dataset": args.dataset, "samples": n, "seq_len": args.seq_len, "models": {}}
    print(f"\n{args.dataset}  n={n}  seq_len={args.seq_len}\n")
    hdr = (f"{'model':<24} {'vocab':>8} {'tot med':>8} {'txt med':>8} {'img med':>8} "
           f"{'tot mean':>9} {'p90':>7} {'docs/row':>9} {'util%':>7} {'skip%':>6}")
    print(hdr)
    print("-" * len(hdr))

    for name, (p, _) in procs.items():
        s = stats[name]
        docs, util = greedy_pack(s["total"], args.seq_len)
        skipped = sum(1 for x in s["total"] if x > args.seq_len)
        row = {
            "vocab": len(p.tokenizer),
            "total_median": statistics.median(s["total"]),
            "total_mean": statistics.mean(s["total"]),
            "total_p90": sorted(s["total"])[int(0.9 * (n - 1))],
            "text_median": statistics.median(s["text"]),
            "image_median": statistics.median(s["image"]),
            "docs_per_row": docs,
            "batch_util_pct": util,
            "skip_pct": 100.0 * skipped / n,
            # Raw per-sample totals, so the JSON can be re-read at a different
            # seq_len without re-tokenizing. The skip rate is a function of
            # seq_len and the summary alone cannot recover it.
            "lengths": s["total"],
        }
        out["models"][name] = row
        print(f"{name:<24} {row['vocab']:>8} {row['total_median']:>8.0f} "
              f"{row['text_median']:>8.0f} {row['image_median']:>8.0f} "
              f"{row['total_mean']:>9.0f} {row['total_p90']:>7} "
              f"{docs:>9.2f} {util:>7.1f} {row['skip_pct']:>6.1f}")

    # The quantity §17.6 actually argued from. Varlen attention is per-document
    # and O(L^2), so a row of few long documents costs more than the same token
    # budget spread over many short ones -- in "L^2 units", not FLOPs and not
    # seconds, which is the whole limit of this number.
    print(f"\nattention work per row, sum(L^2) over documents in a {args.seq_len} row:")
    base = None
    for name in procs:
        s = stats[name]
        fit = [x for x in s["total"] if x <= args.seq_len]
        docs, _ = greedy_pack(s["total"], args.seq_len)
        mean_sq = statistics.mean(x * x for x in fit) if fit else 0.0
        work = docs * mean_sq
        base = work if base is None else base
        out["models"][name]["attn_work_per_row"] = work
        print(f"  {name:<24} {work / 1e6:>8.1f}M   {work / base:>5.2f}x")

    if args.json:
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
