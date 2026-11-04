"""Long-input benchmarks with the DSA indexer actually choosing the keys.

Benchmarks under the top-`K` budget (most VQA/MCQ prompts are < 2048 tokens) cannot
see the indexer: every row has <= K keys, so DSA is dense. These tasks are long:

  niah      needle in a haystack: `--niah-needles` "magic number" needles (one
            asked, the rest distractors) in held-out longctx text, at `--niah-lengths` tokens,
            asked depth spread evenly over 0..1.
  niah_mv   the same, but the asked key holds 4 numbers spread over the text and
            all are asked for; scored hits / max(asked, numbers said), so listing
            distractors costs as much as missing values
  mpdocvqa  MP-DocVQA val, documents of >= `--min-pages` pages (~1K tokens a page
            at 1M pixels), ANLS and exact accuracy as lmms-eval scores them.

The model is the HF Qwen3.5 checkpoint (KV cache, real greedy generation). Its six
full-attention layers run `oracle_attention` -- the same fp32 path as
oracle_sparse_loss.py -- in one of three modes, chosen per run:

  dense     no mask
  idx       top-`K` keys ranked by the DSA indexer (`--indexer`), every step,
            prefill and decode alike, shared by all heads
  oracle    top-`K` keys ranked by the head-averaged attention itself

Every answer goes to <out-dir>/<task>_<mode>.jsonl; the summary to summary.json.

    CUDA_VISIBLE_DEVICES=0 python utils/dsa_bench.py --model-dir HF_DIR \\
        --indexer HF_DIR/indexer.pt --modes dense,idx,oracle --budgets 2048,512,128 --out-dir OUT
"""

from __future__ import annotations

import argparse
import ast
import glob
import io
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `utils` is shadowed by an installed package
from oracle_sparse_loss import load_indexers, oracle_attention  # noqa: E402

MPDOCVQA = "/data/151-2/users/tockier/hf_cache/hub/datasets--lmms-lab-encoder--MP-DocVQA/snapshots/*/data/val-*.parquet"
WORDS = (
    "amber anchor badger cactus canyon cobalt comet copper coral cypress dolphin ember "
    "falcon fern garnet glacier harbor hazel heron indigo iris jasper juniper kestrel "
    "lantern lilac maple marble meadow nectar onyx orchid otter pebble pepper quartz "
    "raven saffron sable spruce tangerine thistle tundra velvet walnut willow zephyr"
).split()


def project(m, h_TD: torch.Tensor, start: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Indexer q/k for rows `start..start+T` of a document, as the training kernels
    build them (`_project_simplified_*`), but at an offset, for decode steps."""
    from models.common.dsa.dsa_min_memory import _apply_rope_at_positions

    pos = torch.arange(start, start + h_TD.shape[0], device=h_TD.device)
    out = []
    for w in (m.linear_q.weight, m.linear_k.weight):
        x = (h_TD @ w.T).reshape(h_TD.shape[0], 1, 1, m.index_head_dim)
        x = _apply_rope_at_positions(x, pos, m.index_head_dim, m.index_rotary_dim, m.rotary_pos_emb, False)
        out.append(x[:, 0, 0])
    return out[0], out[1]


def install_dsa(model, indexers: dict, chunk: int) -> dict:
    """Route the full-attention layers through `oracle_attention`. Returns the state
    the caller flips: mode, budget, and the per-layer indexer key cache."""
    from transformers import AttentionInterface
    from transformers.masking_utils import ALL_MASK_ATTENTION_FUNCTIONS, sdpa_mask

    lm = model.model.language_model
    state = {"mode": "dense", "budget": None, "qi": {}, "ki": {}, "calls": 0}
    full = [i for i, t in enumerate(lm.config.layer_types) if t == "full_attention"]

    def attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kw):
        assert query.shape[0] == 1, "batch 1 only: the mask is causal over one document"
        i, mode = module.layer_idx, state["mode"]
        index = None
        if mode == "idx":
            index = (state["qi"][i], state["ki"][i])
            assert index[1].shape[0] == key.shape[2], (index[1].shape, key.shape)
        out = oracle_attention(
            query[0].transpose(0, 1), key[0].transpose(0, 1), value[0].transpose(0, 1),
            scale=scaling, budget=None if mode == "dense" else state["budget"], chunk=chunk, index=index,
        )
        state["calls"] += 1
        return out[None], None

    def pre_hook(module, args, kwargs):
        if state["mode"] != "idx":
            return
        h = (kwargs["hidden_states"] if "hidden_states" in kwargs else args[0])[0]
        i = module.layer_idx
        prev = state["ki"].get(i)
        qi, ki = project(indexers[i], h, 0 if prev is None else prev.shape[0])
        state["qi"][i] = qi
        state["ki"][i] = ki if prev is None else torch.cat([prev, ki])

    AttentionInterface.register("dsa", attention)
    ALL_MASK_ATTENTION_FUNCTIONS.register("dsa", sdpa_mask)  # ignored: the path is causal itself
    lm.config._attn_implementation = "dsa"
    for i in full:
        lm.layers[i].self_attn.register_forward_pre_hook(pre_hook, with_kwargs=True)
    if indexers:
        assert sorted(indexers) == full, (sorted(indexers), full)
    print(f"DSA on full-attention layers {full}", flush=True)
    return state


def niah_samples(tok, data: str, lengths: list[int], n: int, needles: int = 8, values: int = 1):
    docs = [json.loads(l)["text"] for l in open(data)]
    ids = [tok(d, add_special_tokens=False)["input_ids"] for d in docs]
    for L in lengths:
        for j in range(n):
            rng = random.Random(L * 1000 + j)
            hay = []
            d = j % len(ids)
            while len(hay) < L - 250:
                hay += ids[d]
                d = (d + 1) % len(ids)
            hay = hay[: L - 250]
            keys = rng.sample(WORDS, 2 * needles)
            keys = [f"{keys[2 * a]}-{keys[2 * a + 1]}" for a in range(needles)]
            # the asked key holds `values` numbers; the first sits at the swept depth
            keys = [keys[0]] * (values - 1) + keys
            vals = [str(rng.randint(1_000_000, 9_999_999)) for _ in keys]
            depths = [j / max(n - 1, 1)] + [rng.random() for _ in keys[1:]]
            gold = [v for k, v in zip(keys, vals) if k == keys[0]]
            # insert deepest first so earlier insert points stay valid
            for k, v, dep in sorted(zip(keys, vals, depths), key=lambda x: -x[2]):
                at = int(dep * len(hay))
                hay[at:at] = tok(f" One of the special magic numbers for {k} is: {v}. ", add_special_tokens=False)["input_ids"]
            q = (
                f"What is the special magic number for {keys[0]} mentioned in the provided text? Answer with the number only."
                if values == 1 else
                f"What are all the special magic numbers for {keys[0]} mentioned in the provided text? "
                "Answer with the numbers only, separated by commas."
            )
            yield {
                "id": f"L{L}_{j}", "length": L, "depth": round(depths[0], 3), "gold": gold,
                "messages": [{"role": "user", "content": [{"type": "text", "text": tok.decode(hay) + "\n\n" + q}]}],
            }


def mpdocvqa_samples(n: int, min_pages: int):
    import pyarrow.parquet as pq
    from PIL import Image

    files = sorted(glob.glob(MPDOCVQA))
    assert files, MPDOCVQA
    seen = 0
    for f in files:
        for row in pq.read_table(f).to_pylist():
            pages = [row[f"image_{i}"] for i in range(1, 21) if row[f"image_{i}"] is not None]
            if len(pages) < min_pages:
                continue
            imgs = [{"type": "image", "image": Image.open(io.BytesIO(p["bytes"])).convert("RGB")} for p in pages]
            q = row["question"] + "\nAnswer the question using a single word or phrase."
            yield {
                "id": row["questionId"], "pages": len(pages), "answer_page": int(row["answer_page_idx"]),
                "gold": ast.literal_eval(row["answers"]),
                "messages": [{"role": "user", "content": imgs + [{"type": "text", "text": q}]}],
            }
            seen += 1
            if seen >= n:
                return


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def score(task: str, pred: str, gold: list[str]) -> dict:
    if task.startswith("niah"):
        # extra numbers (distractors) cost as much as missed ones
        said = set(re.findall(r"\d{7}", pred))
        return {"correct": len(said & set(gold)) / max(len(gold), len(said))}
    # lmms-eval multidocvqa: lowercase + strip, ANLS threshold 0.5
    p, gs = pred.lower().strip(), [g.lower().strip() for g in gold]
    if not p or p == "none":
        return {"anls": 0.0, "accuracy": 0.0}
    sim = max(1 - _lev(g, p) / max(len(g), len(p)) for g in gs)
    return {"anls": sim if sim >= 0.5 else 0.0, "accuracy": float(p in gs)}


def selftest() -> None:
    assert score("mpdocvqa", " Answer", ["answer"]) == {"anls": 1.0, "accuracy": 1.0}
    assert score("mpdocvqa", "answr", ["answer"])["anls"] > 0.8
    assert score("mpdocvqa", "xyz", ["answer"])["anls"] == 0.0
    assert score("niah", "It is 1234567.", ["1234567"])["correct"] == 1.0
    assert score("niah_mv", "1111111, 2222222", ["1111111", "2222222", "3333333", "4444444"])["correct"] == 0.5
    assert score("niah", "1111111 or 2222222", ["1111111"])["correct"] == 0.5
    # the offset projection equals the slice of the whole-document projection
    from models.common.dsa.dsa_min_memory import (
        _project_simplified_k_index_block,
        _project_simplified_q_index_tile,
    )
    from models.common.dsa.inner_attention import DSAIndexer

    torch.manual_seed(0)
    m = DSAIndexer.Config(dim=64, index_head_dim=32, index_rotary_dim=16).build()
    m._init_self_buffers(buffer_device=torch.device("cpu"))
    torch.nn.init.normal_(m.linear_q.weight)
    torch.nn.init.normal_(m.linear_k.weight)
    h = torch.randn(40, 64)
    common = (m.index_head_dim, m.index_rotary_dim, m.rotary_pos_emb, False, True)
    q_ref = _project_simplified_q_index_tile(h[:, None], 0, 40, m.linear_q.weight, *common)[:, 0, 0]
    k_ref = _project_simplified_k_index_block(h[:, None], 0, 40, m.linear_k.weight, *common)[:, 0, 0]
    q, k = project(m, h[25:], 25)
    assert torch.allclose(q, q_ref[25:], atol=1e-4) and torch.allclose(k, k_ref[25:], atol=1e-4)
    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir")
    ap.add_argument("--indexer", help="indexer.pt; required for mode idx")
    ap.add_argument("--modes", default="dense,idx,oracle")
    ap.add_argument("--budgets", default="2048,512,128", help="idx/oracle run at each; dense once")
    ap.add_argument("--tasks", default="niah,niah_mv,mpdocvqa")
    ap.add_argument("--niah-data", default="/data/151-2/users/tockier/oracle_loss/longctx_top.jsonl")
    ap.add_argument("--niah-lengths", default="8192,16384,32768,65536")
    ap.add_argument("--niah-n", type=int, default=20, help="samples per length")
    ap.add_argument("--niah-needles", type=int, default=8)
    ap.add_argument("--doc-n", type=int, default=150)
    ap.add_argument("--min-pages", type=int, default=6)
    ap.add_argument("--max-pixels", type=int, default=1048576, help="per page; training used 1M")
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    from transformers import AutoModelForImageTextToText, AutoProcessor

    modes = args.modes.split(",")
    assert "idx" not in modes or args.indexer, "mode idx needs --indexer"
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.model_dir)
    proc.image_processor.size = {"shortest_edge": 65536, "longest_edge": args.max_pixels}
    model = AutoModelForImageTextToText.from_pretrained(args.model_dir, dtype=torch.bfloat16).to("cuda").eval()
    state = install_dsa(model, load_indexers(args.indexer) if args.indexer else {}, args.chunk)
    # the snapshots ship no generation_config: eos is <|endoftext|> alone, and an SFT'd
    # model writes "\nassistant\n..." after <|im_end|> instead of stopping
    stop = [proc.tokenizer.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")]
    runs = [("dense", None)] if "dense" in modes else []
    runs += [(m, int(K)) for K in args.budgets.split(",") for m in modes if m != "dense"]

    summary = json.loads((out / "summary.json").read_text()) if (out / "summary.json").exists() else {}
    for task in args.tasks.split(","):
        if task.startswith("niah"):
            samples = list(niah_samples(
                proc.tokenizer, args.niah_data, [int(x) for x in args.niah_lengths.split(",")], args.niah_n,
                args.niah_needles, 4 if task == "niah_mv" else 1,
            ))
            max_new = 16 if task == "niah" else 48
        else:
            samples = list(mpdocvqa_samples(args.doc_n, args.min_pages))
            max_new = 32
        for mode, K in runs:
            state["mode"], state["budget"] = mode, K
            rows, t0 = [], time.time()
            path = out / f"{task}_{mode}{K or ''}.jsonl"
            with open(path, "w") as f:
                for s in samples:
                    enc = proc.apply_chat_template(
                        s["messages"], add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt",
                    ).to("cuda")
                    state["ki"].clear()
                    state["qi"].clear()
                    with torch.no_grad():
                        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, eos_token_id=stop)
                    pred = proc.tokenizer.decode(gen[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)
                    r = {k: v for k, v in s.items() if k != "messages"}
                    r.update(mode=mode, tokens=enc["input_ids"].shape[1], pred=pred, **score(task, pred, s["gold"]))
                    rows.append(r)
                    f.write(json.dumps(r) + "\n")
                    f.flush()
            keys = ["correct"] if task.startswith("niah") else ["anls", "accuracy"]
            res = {k: sum(r[k] for r in rows) / len(rows) for k in keys}
            res["n"], res["mean_tokens"] = len(rows), sum(r["tokens"] for r in rows) / len(rows)
            if task.startswith("niah"):
                for L in sorted({r["length"] for r in rows}):
                    sub = [r["correct"] for r in rows if r["length"] == L]
                    res[f"correct@{L}"] = sum(sub) / len(sub)
            summary[f"{task}/{mode}@{K or 'all'}"] = res
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            print(f"{task} {mode}@{K or 'all'}: {json.dumps(res)} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
