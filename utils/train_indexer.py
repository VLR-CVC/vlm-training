"""Stage 1 (SPARSE.md 10.1, tier A): train DSA indexers against a frozen dense model.

The backbone runs plain dense attention and is never updated -- it is the
teacher. At each full-attention layer a forward pre-hook hands the layer's q/k
and block input to the vendored `dsa_dense_indexer_loss` (KL from the
head-averaged dense attention to the indexer's single-head scores, tiled, no
`[T, T]` materialized) and backpropagates it on the spot. Only the indexer's two
linears get gradients, so nothing of the backbone's activations is kept and a
64K document costs one no-grad forward plus the KL tiles.

Indexers are standalone modules here, saved under the parameter names the
`attn_backend = "dsa"` model uses (`layers.<i>.attn.inner_attention.indexer.*`),
so the checkpoint drops into that model and into `oracle_sparse_loss.py --indexer`.

Data: every document of the longctx corpora with at least `--min-tokens` tokens,
one document per row, truncated to `--seq-len`. The documents measured in
SPARSE.md 11.7 (the first `--exclude-top` lines of `--exclude`) are held out of
training; the first `--eval-docs` of them give the held-out KL.

`--mm-metadataset` mixes in multimodal documents (a `--mm-frac` share of steps)
through the production loader -- energon, cooker, `PackedBatchEncoder`, one
sample per row -- exactly as `oracle_sparse_recall.py --metadataset` reads them.
`--mm-eval-metadataset` adds a held-out multimodal KL.

The KL covers each row's first `tokens // 4096 * 4096` tokens. The forward still
runs on the whole row, so nothing is cut mid-image; only the KL's length is
gridded, because every new length costs the KL kernels a Triton compile (~13 s).

Usage:
    CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python utils/train_indexer.py \
        --model-dir /path/to/qwen35_2b --out-dir /data/.../indexer_2b --steps 1000
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import queue
import random
import sys
import threading
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CORPORA = ("govreport", "booksum", "longalpaca", "qasper")


def indexer_kl(q_THK, k_THK, h_TD, indexer, scale: float, chunk: int = 2048, backward: bool = True) -> float:
    """`dsa_dense_indexer_loss` for one document, ~10x+ faster, backpropagated on the spot.

    Same loss: KL(head-averaged causal softmax of q.k || softmax of the indexer's rope'd
    single-head scores), with the vendored 1e-10 floors, averaged over all T rows. The
    vendored path is built to fit in little memory: unfused fp32, three passes over the
    keys (one recomputing a teacher norm that is always the head count), and every key
    tile computed even above the causal diagonal. Here each query chunk takes its full
    causal row in one softmax, so the cost is one GEMM pass plus a few elementwise ones.
    The q/k GEMM runs on fp32 copies of bf16 values under TF32, which represents them
    exactly and accumulates in fp32. `--selftest` checks loss and gradients against
    the vendored function.
    """
    from models.common.dsa.dsa_min_memory import (
        _project_simplified_k_index_block,
        _project_simplified_q_index_tile,
    )

    T, H, _ = q_THK.shape
    common = (indexer.index_head_dim, indexer.index_rotary_dim, indexer.rotary_pos_emb, False, True)
    h = h_TD.unsqueeze(1)
    with torch.enable_grad():
        ki = _project_simplified_k_index_block(h, 0, T, indexer.linear_k.weight, *common)[:, 0, 0]
    ki_leaf = ki.detach().float().requires_grad_(True)
    kf = k_THK.float().repeat_interleave(H // k_THK.shape[1], dim=1).permute(1, 2, 0)  # (H, D, T)
    keys = torch.arange(T, device=q_THK.device)
    total = 0.0
    for a in range(0, T, chunk):
        b = min(a + chunk, T)
        future = keys[None, :b] > keys[a:b, None]  # (c, b)
        with torch.no_grad():
            s = torch.bmm(q_THK[a:b].float().transpose(0, 1), kf[:, :, :b]).mul_(scale)  # (H, c, b)
            s.masked_fill_(future, float("-inf"))
            p = s.softmax(dim=-1).mean(dim=0)  # (c, b)
            del s
            plogp = (p * torch.log(p + 1e-10)).sum()
        with torch.enable_grad():
            qi = _project_simplified_q_index_tile(h, a, b, indexer.linear_q.weight, *common)[:, 0, 0]
            z = (qi.float() @ ki_leaf[:b].T).mul(indexer.softmax_scale).masked_fill(future, float("-inf"))
            cross = (p * torch.log(z.softmax(dim=-1) + 1e-10)).sum()
            kl = (plogp - cross) / T
            if backward:
                kl.backward()
        total += float(kl.detach())
    if backward:
        ki.backward(ki_leaf.grad.to(ki.dtype))
    return total


def selftest() -> None:
    """The fast KL against the vendored one: loss and both weight gradients."""
    from models.common.dsa.dsa_min_memory import dsa_dense_indexer_loss
    from models.common.dsa.inner_attention import DSAIndexer

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    T, H, Hkv, D, dim = 1536, 8, 2, 64, 256
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16)
    h = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16)
    idx = DSAIndexer.Config(dim=dim).build().to("cuda")
    idx._init_self_buffers(buffer_device=torch.device("cuda"))
    idx = idx.to(torch.bfloat16)
    ref = dsa_dense_indexer_loss(
        q[:, None], k[:, None], h[:, None], idx, softmax_scale=D**-0.5, loss_coeff=1.0, use_indexer_rope=True
    )
    ref.backward()
    want = [p.grad.float().clone() for p in (idx.linear_q.weight, idx.linear_k.weight)]
    for p in idx.parameters():
        p.grad = None
    got_loss = indexer_kl(q, k, h, idx, D**-0.5, chunk=512)
    got = [p.grad.float() for p in (idx.linear_q.weight, idx.linear_k.weight)]
    assert abs(got_loss - float(ref)) < 1e-3 * abs(float(ref)), (got_loss, float(ref))
    for g, w in zip(got, want):
        rel = (g - w).norm() / w.norm()
        assert rel < 2e-2, rel
    print(f"selftest ok: loss {got_loss:.5f} vs {float(ref):.5f}, grad rel err "
          f"{[round(float((g - w).norm() / w.norm()), 5) for g, w in zip(got, want)]}")


def _digest(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()


def load_documents(root: str, exclude: str, exclude_top: int) -> tuple[list[str], list[str]]:
    """(train, held_out) texts. Held-out = the first `exclude_top` lines of `exclude`."""
    import pyarrow.parquet as pq

    held_out = []
    with open(exclude) as fh:
        for _, line in zip(range(exclude_top), fh):
            held_out.append(json.loads(line)["text"])
    banned = {_digest(t) for t in held_out}
    seen, train = set(), []
    for corpus in CORPORA:
        for split in ("train", "val", "test"):
            path = Path(root, corpus, split, "data.parquet")
            if not path.exists():
                continue
            for text in pq.read_table(path, columns=["context"]).column("context").to_pylist():
                d = _digest(text) if text else None
                if d and d not in banned and d not in seen:
                    seen.add(d)
                    train.append(text)
    return train, held_out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir")
    ap.add_argument("--out-dir")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--kl", choices=("fast", "vendored"), default="fast")
    ap.add_argument("--data-root", default="/data/151-1/datasets/longctx")
    ap.add_argument("--exclude", default="/data/151-2/users/tockier/oracle_loss/longctx_top.jsonl")
    ap.add_argument("--exclude-top", type=int, default=30)
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--min-tokens", type=int, default=16384)
    ap.add_argument("--steps", type=int, default=1000, help="one document per step")
    ap.add_argument("--lr", type=float, default=1e-3, help="the report's stage-1 value")
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--clip", type=float, default=1.0)
    ap.add_argument("--index-head-dim", type=int, default=128)
    ap.add_argument("--index-rotary-dim", type=int, default=64)
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-docs", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", help="start from this indexer checkpoint instead of random")
    ap.add_argument("--mm-metadataset", help="energon metadataset of multimodal training documents")
    ap.add_argument("--mm-eval-metadataset", help="energon metadataset of held-out multimodal documents")
    ap.add_argument("--mm-frac", type=float, default=0.5, help="share of steps drawn from --mm-metadataset")
    ap.add_argument("--max-pixels", type=int, default=1048576)
    args = ap.parse_args()
    if args.selftest:
        selftest()
        return
    if not (args.model_dir and args.out_dir):
        ap.error("--model-dir and --out-dir are required (or use --selftest)")
    torch.backends.cuda.matmul.allow_tf32 = True  # see indexer_kl

    from transformers import AutoTokenizer

    from models.common.dsa.dsa_min_memory import dsa_dense_indexer_loss
    from models.common.dsa.inner_attention import DSAIndexer
    from models.qwen3_5.checkpoint import build_meta, load_hf, materialize

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    model = build_meta(args.model_dir, seq_len=args.seq_len, with_vision=True)
    materialize(model, "cpu")
    load_hf(model, args.model_dir)
    model = model.to(torch.bfloat16).to("cuda").eval().requires_grad_(False)
    model._skip_lm_head = True

    full_layers = [int(n) for n, b in model.layers.items() if getattr(b, "full_attn", False)]
    dim = model.layers[str(full_layers[0])].attn.wq.in_features
    indexers, masters = {}, {}
    for i in full_layers:
        idx = DSAIndexer.Config(
            dim=dim, index_head_dim=args.index_head_dim, index_rotary_dim=args.index_rotary_dim
        ).build().to("cuda")
        idx._init_self_buffers(buffer_device=torch.device("cuda"))
        # bf16 working weights, as under FSDP mixed precision; fp32 masters take the step.
        for name, p in idx.named_parameters():
            masters[f"layers.{i}.attn.inner_attention.indexer.{name}"] = p.detach().float().clone().requires_grad_(True)
        indexers[i] = idx.to(torch.bfloat16)
    params = {f"layers.{i}.attn.inner_attention.indexer.{n}": p for i, m in indexers.items() for n, p in m.named_parameters()}
    if args.init:
        for key, w in torch.load(args.init, map_location="cuda")["state_dict"].items():
            masters[key].data.copy_(w)
            params[key].data.copy_(w)
    opt = torch.optim.AdamW(masters.values(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.0)

    def lr_at(step: int) -> float:
        if step < args.warmup:
            return args.lr * (step + 1) / args.warmup
        t = (step - args.warmup) / max(args.steps - args.warmup, 1)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t)))

    state = {"train": True, "kl": {}, "kl_len": 0}

    def hook(i):
        def pre(_module, a, kw):
            n = state["kl_len"]
            q, k, h = a[0][:n], a[1][:n], kw["hidden_states"][:n]
            if args.kl == "fast":
                state["kl"][i] = indexer_kl(q, k, h, indexers[i], kw["scale"], backward=state["train"])
                return
            with torch.enable_grad():
                kl = dsa_dense_indexer_loss(
                    q.unsqueeze(1), k.unsqueeze(1), h.unsqueeze(1), indexers[i],
                    softmax_scale=kw["scale"], loss_coeff=1.0, use_indexer_rope=True,
                )
                if state["train"]:
                    kl.backward()
            state["kl"][i] = float(kl)
        return pre

    for i in full_layers:
        model.layers[str(i)].attn.inner_attention.register_forward_pre_hook(hook(i), with_kwargs=True)

    def forward(batch: dict, ntok: int, extra: dict) -> tuple[int, dict[int, float]]:
        state["kl_len"] = ntok // 4096 * 4096
        state["kl"] = {}
        inputs, _labels, rest = model.preprocess_inputs(batch)
        with torch.no_grad():
            model(inputs, **rest, **extra)
        return state["kl_len"], dict(state["kl"])

    def run(text: str) -> tuple[int, dict[int, float]]:
        ids = tok(text, add_special_tokens=False)["input_ids"][: args.seq_len]
        ids = ids[: len(ids) // 4096 * 4096]  # the forward too: text has no image runs to cut
        if len(ids) < args.min_tokens:  # the chars screen is loose; skip before paying a forward
            return len(ids), {}
        ids_t = torch.tensor(ids, dtype=torch.long, device="cuda")
        batch = {"input": ids_t, "labels": ids_t, "positions": torch.arange(len(ids), device="cuda")}
        return forward(batch, len(ids), {})

    def mm_rows(metadataset: str):
        """(batch, real tokens, extra) per multimodal document, via the recall probe's loader."""
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # `utils` is shadowed
        from oracle_sparse_recall import _metadataset_batches
        from transformers import AutoConfig

        ns = argparse.Namespace(
            model_dir=args.model_dir, max_pixels=args.max_pixels,
            chat_template="assets/chat_template_sft.jinja", seq_len=args.seq_len,
            metadataset=metadataset, shuffle_buffer_size=100, max_samples_per_sequence=100,
            seed=args.seed, packing_buffer_size=1, min_tokens=args.min_tokens,
        )
        extra = {"special_tokens": {"image_id": AutoConfig.from_pretrained(args.model_dir).image_token_id}}
        # One thread decodes and encodes ahead while the GPU works. Not loader processes:
        # they fork after CUDA is initialized here and die with "initialization error".
        rows: queue.Queue = queue.Queue(maxsize=3)

        def fill():
            for _name, batch, ntok, _nimg, _segments in _metadataset_batches(ns, "cuda"):
                rows.put((batch, ntok, extra))
            rows.put(None)

        threading.Thread(target=fill, daemon=True).start()
        while (row := rows.get()) is not None:
            yield row

    train, held_out = load_documents(args.data_root, args.exclude, args.exclude_top)
    held_out = held_out[: args.eval_docs]
    # Cheap length screen before tokenizing: ~4 chars/token on this English text.
    train = [t for t in train if len(t) >= 3 * args.min_tokens]
    random.Random(args.seed).shuffle(train)
    print(f"layers {full_layers}, {len(train)} candidate train docs, {len(held_out)} held out", flush=True)
    (out_dir / "args.json").write_text(json.dumps(vars(args), indent=2))

    mm_eval = []
    if args.mm_eval_metadataset:
        for row in mm_rows(args.mm_eval_metadataset):
            mm_eval.append(row)
            if len(mm_eval) == args.eval_docs:
                break
        print(f"{len(mm_eval)} held-out multimodal docs, tokens {[r[1] for r in mm_eval]}", flush=True)

    def evaluate(step: int) -> None:
        state["train"] = False
        sets = {"text": [run(t)[1] for t in held_out], "mm": [forward(*r)[1] for r in mm_eval]}
        state["train"] = True
        rec = {"step": step}
        for name, kls in sets.items():
            if kls:
                rec[name] = {i: sum(k[i] for k in kls) / len(kls) for i in full_layers}
                print(f"EVAL {name} step {step}: " + " ".join(f"L{i}={v:.4f}" for i, v in rec[name].items()), flush=True)
        with (out_dir / "eval.jsonl").open("a") as fh:
            fh.write(json.dumps(rec) + "\n")

    def save(step: int) -> None:
        torch.save(
            {"state_dict": {k: v.detach().to(torch.bfloat16).cpu() for k, v in masters.items()},
             "config": {"index_head_dim": args.index_head_dim, "index_rotary_dim": args.index_rotary_dim},
             "step": step},
            out_dir / f"indexer_step{step}.pt",
        )

    evaluate(0)
    step, docs = 0, iter(train)
    mm = mm_rows(args.mm_metadataset) if args.mm_metadataset else None
    pick = random.Random(args.seed + 1)
    log = (out_dir / "train.jsonl").open("a")
    while step < args.steps:
        t0 = time.time()
        source = "mm" if mm is not None and pick.random() < args.mm_frac else "text"
        # Redraw within the source, not the source: redrawing the source on a short text
        # doc tilts the mix toward mm (26 of the first 33 steps at --mm-frac 0.5).
        n = 0
        while n < args.min_tokens:
            n, kl = forward(*next(mm)) if source == "mm" else run(next(docs))
        for name, p in params.items():
            masters[name].grad = p.grad.float()
            p.grad = None
        torch.nn.utils.clip_grad_norm_(masters.values(), args.clip)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        with torch.no_grad():
            for name, p in params.items():
                p.copy_(masters[name])
        step += 1
        rec = {"step": step, "source": source, "tokens": n, "lr": lr_at(step - 1), "sec": round(time.time() - t0, 2), "kl": kl}
        log.write(json.dumps(rec) + "\n")
        log.flush()
        if step % 10 == 0 or step == 1:
            print(f"step {step} {source} {n} tok {rec['sec']}s lr {rec['lr']:.2e} " + " ".join(f"L{i}={v:.4f}" for i, v in kl.items()), flush=True)
        if step % args.eval_every == 0 or step == args.steps:
            evaluate(step)
            save(step)


if __name__ == "__main__":
    main()
