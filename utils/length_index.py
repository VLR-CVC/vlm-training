from __future__ import annotations

import argparse
import glob
import io
import json
import os
import re
import sys
import tarfile
import time
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

COLUMNS = ["key", "subset", "n_tokens", "n_loss", "n_images", "n_visual", "img_w", "img_h",
           "skip", "err"]
IMG_EXTS = ("jpg", "jpeg", "png", "webp", "bmp", "tif", "tiff", "gif")
# "jpg" (one image) or "00.png", "01.png", ... "105.png" (Nemotron: one field per image, message order)
_IMG_PART = re.compile(r"^(\d+\.)?(%s)$" % "|".join(IMG_EXTS))

def image_parts(parts):
    return sorted((k for k in parts if _IMG_PART.match(k)),
                  key=lambda k: (int(k.split(".", 1)[0]) if "." in k else 0, k))

class HeaderImage:
    """Just enough of a PIL image for `cap_image_size`: `.size` and `.resize`.
    Lets the real cap function run without decoding a single pixel."""

    def __init__(self, size):
        self.size = size

    def resize(self, size, *_args, **_kw):
        return HeaderImage(size)

def header_size(data: bytes):
    # we read only the header size
    with Image.open(io.BytesIO(data)) as im:
        return im.size

def iter_samples(path):
    """Yield (key, {suffix: bytes}) from a WebDataset tar, forward-only.
    A key is the member path without everything after the first dot of its
    basename, so "stem/000012.json" and "stem/000012.jpg" group together."""
    cur, parts = None, {}
    with tarfile.open(path, "r|") as tf:
        for m in tf:
            if not m.isfile():
                continue
            d, base = os.path.split(m.name)
            stem, _, ext = base.partition(".")
            key = os.path.join(d, stem) if d else stem
            if key != cur:
                if cur is not None:
                    yield cur, parts
                cur, parts = key, {}
            parts[ext] = tf.extractfile(m).read()
    if cur is not None:
        yield cur, parts

class Measurer:
    def __init__(self, a):
        from transformers import AutoProcessor

        from data import energon_dataloader as edl
        from data.cookers import COOKERS

        self.edl = edl
        self.seq_len = a.seq_len
        self.mode = a.mode
        self.processor = AutoProcessor.from_pretrained(a.model_dir, max_pixels=a.max_pixels)
        if a.chat_template and a.chat_template != "NULL":
            with open(a.chat_template) as f:
                self.processor.chat_template = f.read()
        tok = self.processor.tokenizer
        from transformers import AutoConfig
        self.image_token_id = getattr(AutoConfig.from_pretrained(a.model_dir), "image_token_id",
                                      None) or tok.convert_tokens_to_ids("<|image_pad|>")
        merge = self.processor.image_processor.merge_size
        self.merge2 = merge * merge
        self.encoder = edl.PackedBatchEncoder(
            self.processor, a.seq_len, a.seq_len, image_token_id=self.image_token_id,
            video_token_id=-1, spatial_merge_size=merge)
        want = {"type_dataset": a.cooker}
        self.cooker = next(c.cook for c in COOKERS
                           if (c.has_subflavors or {}).items() >= want.items())
        self.ip = self.processor.image_processor

    # -- sample -> EnergonSample through the real cooker
    def cook(self, key, parts, images):
        """`images`: {part name: image object}, under the same names the tar uses."""
        sample = {"__key__": key, "__restore_key__": ("Webdataset", 0), "__subflavors__": {},
                  "__sources__": (),
                  "json": parts["_json"] if "_json" in parts else json.loads(parts.get("json", b"{}"))}
        sample.update(images)
        if "txt" in parts:                       # energon decodes .txt to str (cooker_idl, cooker_olmo_ocr)
            sample["txt"] = parts["txt"].decode("utf-8")
        return self.cooker(sample)

    def _grid_tokens(self, w, h):
        n = self.ip.get_number_of_image_patches(h, w, {})
        return n // self.merge2

    def measure(self, key, parts):
        names = image_parts(parts)
        w = h = 0
        if self.mode == "fast":
            sizes = {k: header_size(parts[k]) for k in names}
            cooked = self.cook(key, parts, {k: HeaderImage(sz) for k, sz in sizes.items()})
        else:
            imgs = {}
            for k in names:
                with Image.open(io.BytesIO(parts[k])) as im:
                    imgs[k] = im.convert("RGB")
            sizes = {k: im.size for k, im in imgs.items()}
            cooked = self.cook(key, parts, imgs)
        if names:
            w, h = sizes[names[0]]
        text = self.processor.apply_chat_template(
            conversation=cooked.messages, tokenize=False, add_generation_prompt=False)
        images = list(cooked.images) if getattr(cooked, "images", None) else (
            [cooked.image] if cooked.image is not None else [])
        images = [self.edl.cap_image_size(im) for im in images]

        if self.mode == "fast":
            ids = self.processor.tokenizer(text)["input_ids"]
            per_image = [self._grid_tokens(*im.size) for im in images]
            if ids.count(self.image_token_id) != len(per_image):
                # the real processor raises on this; do not silently miscount
                raise ValueError(f"{ids.count(self.image_token_id)} image placeholders "
                                 f"for {len(per_image)} images")
            it, ids_full = iter(per_image), []
            for t in ids:
                ids_full.extend([t] * next(it) if t == self.image_token_id else [t])
            loss_ids, n_visual = ids, sum(per_image)
        else:
            inputs = self.processor(text=[text], images=images or None, padding=False,
                                    return_tensors="pt")
            ids_full = loss_ids = inputs["input_ids"][0].tolist()
            grid = inputs.get("image_grid_thw")
            n_visual = 0 if grid is None else int((grid.prod(-1) // self.merge2).sum())
        length = len(ids_full)
        n_loss = int((self.encoder.assistant_labels(loss_ids) != -100).sum())
        skip = "too_long" if length > self.seq_len else ("no_labels" if n_loss == 0 else "")
        return dict(n_tokens=length, n_loss=n_loss, n_images=len(images), n_visual=n_visual,
                    img_w=w, img_h=h, skip=skip)

_M = None
_A = None

def _init(a):
    global _M, _A
    _A = a
    _M = Measurer(a)

def subset_of(tar_name, regex):
    m = re.match(regex, tar_name)
    return re.sub(r"-part-\d+-of-\d+$", "", m.group(1)) if m else "unknown"

def subset_from_dirs(path, depth):
    return "/".join(os.path.dirname(path).split("/")[-depth:])

def do_tar(path):
    a = _A
    name = os.path.basename(path)
    # relative path, not the bare tar name: Nemotron's <key>/<subset>/shard-000000.tar
    # would otherwise collide across subsets
    out = os.path.join(a.out, os.path.relpath(path, a.data)[:-4].replace(os.sep, "__") + ".parquet")
    if os.path.exists(out):
        return name, 0, 0.0, "skipped"
    tar_subset = (a.subset_const or (subset_from_dirs(path, a.subset_dirs) if a.subset_dirs
                                     else subset_of(name, a.subset_regex)))
    extras = [f for f in a.extra_json_fields.split(",") if f]
    cols = COLUMNS + extras
    t0, rows = time.time(), {c: [] for c in cols}
    for key, parts in iter_samples(path):
        rec = dict(key=key, subset=tar_subset, err="")
        try:
            obj = json.loads(parts.get("json", b"{}"))
            parts["_json"] = obj
            if a.subset_json_key:
                rec["subset"] = str(obj.get(a.subset_json_key) or "unknown")
            for f in extras:
                v = obj.get(f)
                rec[f] = float(v) if isinstance(v, (int, float)) else None
            rec.update(_M.measure(key, parts))
        except Exception as exc:                                    # noqa: BLE001
            rec.update(n_tokens=-1, n_loss=-1, n_images=-1, n_visual=-1, img_w=0, img_h=0,
                       skip="error", err=f"{type(exc).__name__}: {exc}"[:200])
        for c in cols:
            rows[c].append(rec.get(c))
    tmp = out + ".tmp"
    pq.write_table(pa.table(rows), tmp)
    os.replace(tmp, out)
    return name, len(rows["key"]), time.time() - t0, "ok"

def validate(a):
    from megatron.energon import SkipSample, WorkerConfig

    tars = sorted(glob.glob(os.path.join(a.data, a.glob)))
    fast_a, exact_a = argparse.Namespace(**vars(a)), argparse.Namespace(**vars(a))
    fast_a.mode, exact_a.mode = "fast", "exact"
    fast, exact = Measurer(fast_a), Measurer(exact_a)
    wc = WorkerConfig(rank=0, world_size=1, num_workers=0)
    wc.worker_activate(0)
    n = bad = real_skips = 0
    worst = []
    try:
        for path in tars[: a.validate_tars]:
            for key, parts in iter_samples(path):
                f, e = fast.measure(key, parts), exact.measure(key, parts)
                # the real encoder, on the exact-mode cooked sample
                imgs = {}
                for k in image_parts(parts):
                    with Image.open(io.BytesIO(parts[k])) as im:
                        imgs[k] = im.convert("RGB")
                try:
                    real = exact.encoder.encode_sample(exact.cook(key, parts, imgs))
                    r = dict(n_tokens=int(real.length), n_loss=int((real.labels != -100).sum()),
                             skip="")
                except SkipSample:
                    r, real_skips = None, real_skips + 1
                n += 1
                problems = []
                for k in ("n_tokens", "n_loss", "n_visual", "skip"):
                    if f[k] != e[k]:
                        problems.append(f"fast!=exact {k}: {f[k]} vs {e[k]}")
                if r is None and f["skip"] == "":
                    problems.append("real encoder skipped, index did not")
                if r is not None:
                    if f["skip"] != "":
                        problems.append(f"index skipped ({f['skip']}), real encoder did not")
                    for k in ("n_tokens", "n_loss"):
                        if f[k] != r[k]:
                            problems.append(f"fast!=real {k}: {f[k]} vs {r[k]}")
                if problems:
                    bad += 1
                    if len(worst) < 5:
                        worst.append((key, problems))
                if n >= a.validate:
                    break
            if n >= a.validate:
                break
    finally:
        wc.worker_deactivate()
    print(f"validated {n} samples against the real encode_sample: {bad} mismatches "
          f"({real_skips} skipped by the real encoder at seq_len={a.seq_len})")
    for w in worst:
        print("  e.g.", w)
    return bad == 0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir holding the .tar shards")
    ap.add_argument("--out", default="", help="required unless --validate")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--chat-template", default="assets/chat_template_sft.jinja")
    ap.add_argument("--seq-len", type=int, default=24576)
    ap.add_argument("--max-pixels", type=int, default=1048576, help="as train_qwen.py")
    ap.add_argument("--cooker", default="onevision_instruct", help="type_dataset subflavor")
    ap.add_argument("--mode", choices=["fast", "exact"], default="fast")
    ap.add_argument("--glob", default="*.tar")
    ap.add_argument("--subset-regex", default=r"^(.+?)_train-",
                    help="group 1 of this regex on the tar name is the subset")
    ap.add_argument("--subset-const", default="",
                    help="one fixed subset name for everything (IDL: 'idl')")
    ap.add_argument("--subset-dirs", type=int, default=0,
                    help="take the subset from the last N directory names above each tar "
                         "(Nemotron: 2 -> 'v3/clevr_1'); glob then needs to reach them, "
                         "e.g. --glob '*/*/*.tar'")
    ap.add_argument("--subset-json-key", default="",
                    help="take the subset from this key of each sample's json instead of the tar "
                         "name (FineVision: 'source')")
    ap.add_argument("--extra-json-fields", default="",
                    help="comma list of numeric json fields to copy into the index as columns")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--task-id", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--num-tasks", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_COUNT", 1)))
    ap.add_argument("--limit", type=int, default=0, help="only the first N tars of this task")
    ap.add_argument("--validate", type=int, default=0, help="compare fast vs exact on N samples")
    ap.add_argument("--validate-tars", type=int, default=3)
    a = ap.parse_args()

    if a.validate:
        sys.exit(0 if validate(a) else 1)

    if not a.out:
        ap.error("--out is required unless --validate is given")
    os.makedirs(a.out, exist_ok=True)
    tars = sorted(glob.glob(os.path.join(a.data, a.glob)))[a.task_id::a.num_tasks]
    if a.limit:
        tars = tars[: a.limit]
    print(f"task {a.task_id}/{a.num_tasks}: {len(tars)} tars, {a.workers} workers, mode={a.mode}",
          flush=True)
    done = samples = 0
    t0 = time.time()
    with Pool(a.workers, initializer=_init, initargs=(a,)) as pool:
        for name, n, secs, status in pool.imap_unordered(do_tar, tars):
            done += 1
            samples += n
            el = time.time() - t0
            print(f"[{done}/{len(tars)}] {name}: {status} {n} samples {secs:.0f}s | "
                  f"{samples / max(el, 1e-9):.0f} samples/s total", flush=True)

if __name__ == "__main__":
    main()