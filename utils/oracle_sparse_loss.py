"""Loss under an oracle sparse-attention mask: what a perfect DSA indexer would cost.

`oracle_sparse_recall.py` measures how much attention mass a top-`K` budget keeps.
This measures the thing that matters: the next-token loss when the full-attention
layers attend ONLY to that top-`K`. No training; the dense checkpoint is run as is.

Selection is DSA's: per query row, the top-`K` keys of the teacher (per-head softmax
averaged over heads, the recall probe's `tok` oracle), shared by all heads. Each
head then renormalises its own softmax over the selected keys. Rows with fewer than
`K` visible keys are dense, as in DSA.

The oracle needs the full score row, so sparse layers run a chunked fp32 path
instead of the flash kernel. `dense_fp32` runs that same path with no mask: it is
the control. Deltas are read against it, so the fp32-vs-bf16 path difference is not
counted as a sparsity cost; `dense_fp32 - dense` reports that difference.

Configs per document: `dense`, `dense_fp32`, `all@K` (every full-attention layer
sparse) and `L<i>@K` (layer i alone sparse). Loss is per target position, reported
in position buckets; the headline is the mean over positions >= `--min-pos`.

Usage:
    CUDA_VISIBLE_DEVICES=2 python utils/oracle_sparse_loss.py \
        --model-dir /path/to/hf_snapshot --data long_docs.jsonl \
        --seq-len 65536 --budgets 1024,2048,4096 --out loss.json

    python utils/oracle_sparse_loss.py --selftest   # GPU if present, no model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_NEG = float("-inf")


@torch.no_grad()
def oracle_attention(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    v_THV: torch.Tensor,
    *,
    scale: float,
    budget: int | None,
    chunk: int,
    index: tuple[torch.Tensor, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Causal attention over one document, each row restricted to its top-`budget` keys.
    `budget=None` is plain dense attention through the same fp32 path.

    Keys are ranked by the oracle (head-averaged attention) unless `index` gives the
    indexer's projected `(q_idx [T, d], k_idx [T, d])`, which rank by their dot product.
    The attention itself is identical either way, so oracle and indexer runs differ in
    the selection only.

    Keys may outnumber queries (a KV cache): the queries are then the LAST rows of the
    document, and `index[0]` covers the queries while `index[1]` covers every key."""
    T, H, _ = q_THK.shape
    off = k_THK.shape[0] - T
    g = H // k_THK.shape[1]
    kf = k_THK.float().repeat_interleave(g, dim=1)
    vf = v_THV.float().repeat_interleave(g, dim=1)
    keys = torch.arange(k_THK.shape[0], device=q_THK.device)
    out = torch.empty(T, H, v_THV.shape[-1], dtype=torch.float32, device=q_THK.device)
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        pos, kb = keys[off + a : off + b], off + b
        s = torch.einsum("chd,thd->cht", q_THK[a:b].float(), kf[:kb]) * scale
        s.masked_fill_(keys[None, None, :kb] > pos[:, None, None], _NEG)
        p = s.softmax(dim=-1)
        del s
        if budget is not None and kb > budget:
            if index is None:
                rank = p.mean(dim=1)
            else:
                rank = index[0][a:b].float() @ index[1][:kb].float().T
                rank.masked_fill_(keys[None, :kb] > pos[:, None], _NEG)
            idx = rank.topk(budget, dim=-1).indices
            keep = torch.zeros(b - a, kb, dtype=torch.bool, device=p.device)
            keep.scatter_(1, idx, True)
            p.mul_(keep[:, None, :])
            p.div_(p.sum(dim=-1, keepdim=True))
        out[a:b] = torch.einsum("cht,thd->chd", p, vf[:kb])
        del p
    return out.to(q_THK.dtype)


def selftest() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    T, H, Hkv, D = 300, 4, 2, 32
    q, k, v = (torch.randn(T, h, D, device=dev) for h in (H, Hkv, Hkv))
    ref = F.scaled_dot_product_attention(
        q.transpose(0, 1)[None], k.repeat_interleave(2, 1).transpose(0, 1)[None],
        v.repeat_interleave(2, 1).transpose(0, 1)[None], is_causal=True, scale=D**-0.5,
    )[0].transpose(0, 1)
    # no mask, and a budget >= T, are both exactly dense
    for budget in (None, T):
        got = oracle_attention(q, k, v, scale=D**-0.5, budget=budget, chunk=64)
        assert torch.allclose(got, ref, atol=1e-5), (budget, (got - ref).abs().max())
    # rows with <= budget visible keys stay dense; later rows do not
    got = oracle_attention(q, k, v, scale=D**-0.5, budget=50, chunk=64)
    assert torch.allclose(got[:50], ref[:50], atol=1e-5)
    assert not torch.allclose(got[100:], ref[100:], atol=1e-3)
    # budget 1 with one dominant key per row returns that key's value
    k1 = torch.zeros(T, Hkv, D, device=dev)
    k1[:, :, 0] = torch.arange(T, device=dev).float()[:, None]  # later keys score higher
    q1 = torch.zeros(T, H, D, device=dev)
    q1[:, :, 0] = 10.0
    got = oracle_attention(q1, k1, v, scale=1.0, budget=1, chunk=64)
    want = v.repeat_interleave(2, 1)  # row t picks key t, the highest visible
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    # the same pick made by an indexer whose scores favour later keys, on plain q/k
    qi = torch.ones(T, 4, device=dev)
    ki = torch.zeros(T, 4, device=dev)
    ki[:, 0] = torch.arange(T, device=dev).float()
    got = oracle_attention(q, k, v, scale=D**-0.5, budget=1, chunk=64, index=(qi, ki))
    assert torch.allclose(got, want, atol=1e-5), (got - want).abs().max()
    # an indexer budget covering the document is dense, whatever it scores
    got = oracle_attention(q, k, v, scale=D**-0.5, budget=T, chunk=64, index=(qi, ki))
    assert torch.allclose(got, ref, atol=1e-5)
    # a KV cache: the last rows alone, against every key, match the full document
    qi = torch.randn(T, 4, device=dev)
    ki = torch.randn(T, 4, device=dev)
    for index in (None, (qi, ki)):
        full = oracle_attention(q, k, v, scale=D**-0.5, budget=50, chunk=64, index=index)
        tail = oracle_attention(
            q[-37:], k, v, scale=D**-0.5, budget=50, chunk=16,
            index=None if index is None else (qi[-37:], ki),
        )
        assert torch.allclose(tail, full[-37:], atol=1e-5), (tail - full[-37:]).abs().max()
    print("selftest ok")


def load_indexers(path: str, device: str = "cuda") -> dict:
    """{layer: DSAIndexer} from a train_indexer.py / dcp_to_hf.py `indexer.pt`, bf16."""
    from models.common.dsa.inner_attention import DSAIndexer

    ckpt = torch.load(path, map_location=device)
    indexers = {}
    for key, w in ckpt["state_dict"].items():
        i, leaf = int(key.split(".")[1]), key.split("indexer.")[1]
        if i not in indexers:
            indexers[i] = DSAIndexer.Config(dim=w.shape[1], **ckpt["config"]).build().to(device)
            indexers[i]._init_self_buffers(buffer_device=torch.device(device))
        indexers[i].get_parameter(leaf).data.copy_(w)
    return {i: m.to(torch.bfloat16) for i, m in indexers.items()}


def _per_token_ce(model, hidden: torch.Tensor, labels: torch.Tensor, chunk: int = 8192) -> torch.Tensor:
    """CE at each position against the loader's shifted `labels` (the target of
    position t, -100 = not scored, which yields 0), fp32, chunked over the vocab."""
    h = hidden.reshape(-1, hidden.shape[-1])
    out = torch.empty(labels.shape[0], dtype=torch.float32, device=h.device)
    for a in range(0, labels.shape[0], chunk):
        logits = model.lm_head(h[a : a + chunk]).float()
        out[a : a + chunk] = F.cross_entropy(logits, labels[a : a + chunk], reduction="none", ignore_index=-100)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir")
    ap.add_argument("--data", help=".jsonl (one doc per line), a .txt file, or a dir of .txt")
    ap.add_argument(
        "--metadataset",
        help="energon metadataset instead of --data: documents through the production "
        "loader (one per row); loss over the loader's labels, i.e. assistant tokens only",
    )
    ap.add_argument("--text-key", default="text")
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--min-tokens", type=int, default=0, help="skip shorter docs (default: --seq-len)")
    ap.add_argument("--budgets", default="1024,2048,4096")
    ap.add_argument("--min-pos", type=int, default=8192, help="headline loss is over targets at >= this position")
    ap.add_argument("--buckets", default="0,8192,16384,32768,65536,131072")
    ap.add_argument("--no-per-layer", action="store_true", help="only dense and all@K")
    ap.add_argument("--chunk", type=int, default=256, help="query rows per fp32 score tile")
    ap.add_argument("--max-docs", type=int, default=0)
    ap.add_argument(
        "--indexer",
        help="checkpoint from utils/train_indexer.py; adds idx_* configs that select "
        "with the trained indexer instead of the oracle",
    )
    ap.add_argument("--no-oracle", action="store_true", help="skip the oracle all@/L@ configs")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.model_dir or bool(args.data) == bool(args.metadataset):
        ap.error("--model-dir and exactly one of --data / --metadataset (or use --selftest)")

    from transformers import AutoTokenizer

    from models.qwen3_5.checkpoint import build_meta, load_hf, materialize
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `utils` is shadowed by an installed package
    from oracle_sparse_recall import _documents

    budgets = [int(b) for b in args.budgets.split(",")]
    edges = [int(b) for b in args.buckets.split(",")]
    # text is truncated to seq_len, so "long enough" means filling the row; the loader
    # skips samples over seq_len instead, so a multimodal row is never full.
    min_tokens = args.min_tokens or (args.seq_len if args.data else 16384)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = build_meta(args.model_dir, seq_len=args.seq_len, with_vision=True)
    materialize(model, "cpu")
    load_hf(model, args.model_dir)
    model = model.to(torch.bfloat16).to("cuda").eval()
    model._skip_lm_head = True

    indexers = load_indexers(args.indexer) if args.indexer else {}
    if indexers:
        from models.common.dsa.dsa_min_memory import (
            _project_simplified_k_index_block,
            _project_simplified_q_index_tile,
        )

    def project(i: int, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The indexer's rope'd q/k for one document, exactly as the kernels build them."""
        m, h = indexers[i], h.unsqueeze(1)
        common = (m.index_head_dim, m.index_rotary_dim, m.rotary_pos_emb, False, True)
        qi = _project_simplified_q_index_tile(h, 0, h.shape[0], m.linear_q.weight, *common)
        ki = _project_simplified_k_index_block(h, 0, h.shape[0], m.linear_k.weight, *common)
        return qi[:, 0, 0], ki[:, 0, 0]

    # Which layers are sparse, at what budget, and who selects, for the current forward.
    state: dict = {"layers": set(), "budget": None, "indexer": False}
    full_layers = []
    for name, block in model.layers.items():
        if not getattr(block, "full_attn", False):
            continue
        full_layers.append(int(name))
        inner = block.attn.inner_attention
        dense_forward = inner.forward

        def forward(q, k, v, *, attention_masks, scale=None, _name=int(name), _dense=dense_forward, **kw):
            if _name not in state["layers"]:
                return _dense(q, k, v, attention_masks=attention_masks, scale=scale, **kw)
            bounds = attention_masks.cu_seq_q.tolist()
            out = torch.empty(q.shape[0], q.shape[1], v.shape[-1], dtype=q.dtype, device=q.device)
            for a, b in zip(bounds[:-1], bounds[1:]):
                index = project(_name, kw["hidden_states"][a:b]) if state["indexer"] else None
                out[a:b] = oracle_attention(
                    q[a:b], k[a:b], v[a:b], scale=scale, budget=state["budget"], chunk=args.chunk,
                    index=index,
                )
            return out

        inner.forward = forward
    print(f"full-attention layers {full_layers}, budgets {budgets}", flush=True)

    configs = [("dense", set(), None, False), ("dense_fp32", set(full_layers), None, False)]
    for pre, use_idx in (("", False), ("idx_", True)):
        if (use_idx and not indexers) or (not use_idx and args.no_oracle):
            continue
        configs += [(f"{pre}all@{K}", set(full_layers), K, use_idx) for K in budgets]
        if not args.no_per_layer:
            configs += [(f"{pre}L{i}@{K}", {i}, K, use_idx) for K in budgets for i in full_layers]
    headline = [c[0] for c in configs if "all@" in c[0]]

    def text_rows():
        for name, text in _documents(Path(args.data), args.text_key):
            ids = tok(text, add_special_tokens=False)["input_ids"][: args.seq_len]
            if len(ids) < min_tokens:
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device="cuda")
            labels = torch.full_like(ids_t, -100)
            labels[:-1] = ids_t[1:]
            yield name, {"input": ids_t, "labels": labels, "positions": torch.arange(len(ids), device="cuda")}, len(ids), {}

    def mm_rows():
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from oracle_sparse_recall import _metadataset_batches
        from transformers import AutoConfig

        ns = argparse.Namespace(
            model_dir=args.model_dir, max_pixels=1048576, chat_template="assets/chat_template_sft.jinja",
            seq_len=args.seq_len, metadataset=args.metadataset, shuffle_buffer_size=100,
            max_samples_per_sequence=100, seed=0, packing_buffer_size=1, min_tokens=min_tokens,
        )
        extra = {"special_tokens": {"image_id": AutoConfig.from_pretrained(args.model_dir).image_token_id}}
        for name, batch, ntok, _nimg, _seg in _metadataset_batches(ns, "cuda"):
            yield name, batch, ntok, extra

    docs = []
    for name, batch, ntok, extra in (text_rows() if args.data else mm_rows()):
        labels = batch["labels"]
        # the predicted token sits at t+1; -100 (prompt, images, padding) is not scored
        tpos = torch.arange(1, labels.shape[0] + 1, device="cuda")
        masks = [(labels != -100) & (tpos >= lo) & (tpos < hi) for lo, hi in zip(edges[:-1], edges[1:])]
        masks.append((labels != -100) & (tpos >= args.min_pos))
        counts = [int(m.sum()) for m in masks]
        if counts[-1] == 0:
            continue
        inputs, _labels, rest = model.preprocess_inputs(batch)
        t0 = time.time()
        sums: dict[str, list[float]] = {}
        for cname, layers, budget, use_idx in configs:
            state["layers"], state["budget"], state["indexer"] = layers, budget, use_idx
            with torch.no_grad():
                ce = _per_token_ce(model, model(inputs, **rest, **extra), labels)
            sums[cname] = [float(ce[m].sum()) for m in masks]
        docs.append({"name": name, "tokens": ntok, "counts": counts, "ce_sum": sums})
        n = counts[-1]
        ref = sums["dense_fp32"][-1] / n
        line = " ".join(f"{c}={sums[c][-1] / n - ref:+.4f}" for c in headline)
        print(
            f"{name}: {ntok} tok, {time.time() - t0:.0f}s, dense {sums['dense'][-1] / n:.4f} "
            f"fp32-dense {ref - sums['dense'][-1] / n:+.4f} | {line}",
            flush=True,
        )
        if args.out:
            Path(args.out).write_text(json.dumps(_summary(docs, configs, edges, args), indent=2))
        if args.max_docs and len(docs) >= args.max_docs:
            break
    if not docs:
        raise RuntimeError(f"no document reached {min_tokens} tokens")
    _print(_summary(docs, configs, edges, args))


def _summary(docs, configs, edges, args) -> dict:
    """Token-weighted mean CE per config and bucket, and the delta vs dense_fp32 with
    a standard error over documents (documents, not tokens, are the independent units)."""
    labels = [f"{lo}-{hi}" for lo, hi in zip(edges[:-1], edges[1:])] + [f">={args.min_pos}"]
    out = {"seq_len": args.seq_len, "min_pos": args.min_pos, "documents": len(docs),
           "doc_tokens": [d["tokens"] for d in docs], "buckets": labels, "configs": {}, "docs": docs}
    for cname, *_ in configs:
        rows = {}
        for j, lab in enumerate(labels):
            n = sum(d["counts"][j] for d in docs)
            if n == 0:
                continue
            per_doc = [
                (d["ce_sum"][cname][j] - d["ce_sum"]["dense_fp32"][j]) / d["counts"][j]
                for d in docs if d["counts"][j]
            ]
            m = sum(per_doc) / len(per_doc)
            se = (sum((x - m) ** 2 for x in per_doc) / max(len(per_doc) - 1, 1) / len(per_doc)) ** 0.5
            rows[lab] = {
                "ce": sum(d["ce_sum"][cname][j] for d in docs) / n,
                "delta": (sum(d["ce_sum"][cname][j] for d in docs)
                          - sum(d["ce_sum"]["dense_fp32"][j] for d in docs)) / n,
                "delta_doc_mean": m,
                "delta_doc_se": se,
                "tokens": n,
            }
        out["configs"][cname] = rows
    return out


def _print(s: dict) -> None:
    labs = [b for b in s["buckets"] if b in s["configs"]["dense"]]
    print(f"\n{s['documents']} documents, seq_len {s['seq_len']}; delta = CE - CE(dense_fp32), nats")
    print(f"{'config':>12}" + "".join(f"{b:>16}" for b in labs) + f"{'doc se (head)':>15}")
    for cname, rows in s["configs"].items():
        cells = "".join(
            f"{rows[b]['ce']:>16.4f}" if cname == "dense_fp32" else f"{rows[b]['delta']:>+16.4f}" for b in labs
        )
        print(f"{cname:>12}{cells}{rows[labs[-1]]['delta_doc_se']:>15.4f}")


if __name__ == "__main__":
    main()
