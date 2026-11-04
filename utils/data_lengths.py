"""Token length of every training document of the DSA runs (`indexer_2b/stage2_mix.yaml`).

  text   the longctx documents `utils/train_indexer.py` trains on (>= 3*16384 chars, the
         30 eval documents excluded): full length, and the first 240K chars that
         `datasets/longctx_text_energon` keeps for stage 2
  arxiv  `long_document_arxiv_crude`, shards 0-24 (train) and 25 (held out): pages, and
         tokens as `PackedBatchEncoder` makes them (cooker_arxiv_crude, SFT chat template,
         images capped by `cap_image_size` and resized by the processor), counted from the
         PNG headers instead of decoding 100 GB of pages

`--check N` runs the real processor on N papers and asserts the counts match.

    python utils/data_lengths.py --out lengths.json
"""

from __future__ import annotations

import argparse
import io
import json
import os
import struct
import sys
import tarfile
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b"
ARXIV = "/data/151-2/users/tockier/long_document_arxiv_crude/media"
MAX_PIXELS = 1048576


class _Png:
    """Stands in for a decoded page: `cap_image_size` only reads `.size`."""

    def __init__(self, size):
        self.size = size

    def resize(self, size, _resample):
        self.size = size
        return self


def image_tokens(size, image_processor) -> int:
    from data.energon_dataloader import cap_image_size
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    w, h = cap_image_size(_Png(size)).size
    f = image_processor.patch_size * image_processor.merge_size
    rh, rw = smart_resize(h, w, factor=f, min_pixels=image_processor.size["shortest_edge"], max_pixels=MAX_PIXELS)
    return rh * rw // f**2


def papers(shard: Path):
    """(paper id, crude sample) per paper; pages keep only their PNG header."""
    groups = defaultdict(dict)
    with tarfile.open(shard) as tar:
        for m in tar:
            pid, _, part = m.name.partition(".")
            if part.endswith(".png"):
                groups[pid][part] = tar.extractfile(m).read(24)
            elif part.endswith(".md") or part == "meta.json":
                groups[pid][part] = tar.extractfile(m).read()
    for pid, parts in groups.items():
        parts["meta.json"] = json.loads(parts["meta.json"])
        yield pid, parts


def arxiv_lengths(processor, shard: Path) -> list[dict]:
    from data.cookers import cooker_arxiv_crude

    tok = processor.tokenizer
    pad = processor.image_token
    rows = []
    for pid, parts in papers(shard):
        sample = cooker_arxiv_crude({**parts, "__key__": pid, "__restore_key__": (), "__subflavors__": {}})
        text = processor.apply_chat_template(sample.messages, tokenize=False, add_generation_prompt=False)
        n_text = len(tok(text, add_special_tokens=False)["input_ids"])
        pngs = [p for p in parts if p.endswith(".png")]
        n_img = sum(image_tokens(struct.unpack(">II", parts[p][16:24]), processor.image_processor) for p in pngs)
        rows.append({"id": pid, "pages": len(pngs), "tokens": n_text - len(pngs) + n_img, "image_tokens": n_img})
    return rows


def check(processor, n: int) -> None:
    """The header count equals the processor's on real pages."""
    from PIL import Image

    from data.cookers import cooker_arxiv_crude
    from data.energon_dataloader import cap_image_size

    shard = Path(ARXIV, "shard_000025.tar")
    want = {r["id"]: r["tokens"] for r in arxiv_lengths(processor, shard)[:n]}
    with tarfile.open(shard) as tar:
        groups = defaultdict(dict)
        for m in tar:
            pid, _, part = m.name.partition(".")
            if pid in want:
                groups[pid][part] = tar.extractfile(m).read()
    for pid, parts in groups.items():
        parts["meta.json"] = json.loads(parts["meta.json"])
        for p in [p for p in parts if p.endswith(".png")]:
            parts[p] = Image.open(io.BytesIO(parts[p])).convert("RGB")
        s = cooker_arxiv_crude({**parts, "__key__": pid, "__restore_key__": (), "__subflavors__": {}})
        images = [cap_image_size(im) for im in (s.images or [s.image])]
        text = processor.apply_chat_template(s.messages, tokenize=False, add_generation_prompt=False)
        got = processor(text=[text], images=images, padding=False, return_tensors="pt")["input_ids"].shape[1]
        assert got == want[pid], (pid, got, want[pid])
        print(f"check {pid}: {got} tokens ok", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    ap.add_argument("--check", type=int, default=0)
    ap.add_argument("--workers", type=int, default=26)
    args = ap.parse_args()

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(MODEL, max_pixels=MAX_PIXELS)
    processor.chat_template = Path("assets/chat_template_sft.jinja").read_text()
    if args.check:
        check(processor, args.check)
        return

    from concurrent.futures import ProcessPoolExecutor

    from utils.train_indexer import load_documents

    shards = sorted(Path(ARXIV).glob("shard_*.tar"))
    with ProcessPoolExecutor(args.workers) as pool:
        arxiv = list(pool.map(arxiv_lengths, [processor] * len(shards), shards))
    out = {"arxiv": {s.name: rows for s, rows in zip(shards, arxiv)}}
    print("arxiv", sum(map(len, arxiv)), "papers", flush=True)

    train, _ = load_documents("/data/151-1/datasets/longctx", "/data/151-2/users/tockier/oracle_loss/longctx_top.jsonl", 30)
    train = [t for t in train if len(t) >= 3 * 16384]
    tok = processor.tokenizer
    full = [len(x) for x in tok(train, add_special_tokens=False)["input_ids"]]
    cut = [len(x) for x in tok([t[:240_000] for t in train], add_special_tokens=False)["input_ids"]]
    out["text"] = [{"tokens": f, "tokens_240k_chars": c} for f, c in zip(full, cut)]
    print("text", len(train), "documents", flush=True)
    Path(args.out).write_text(json.dumps(out))


if __name__ == "__main__":
    main()
