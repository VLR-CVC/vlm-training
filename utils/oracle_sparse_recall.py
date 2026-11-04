"""Oracle recall of a sparse-attention budget, measured on real documents.

Answers one question before any sparse-attention training is written: if a query
could pick its keys *perfectly*, how much attention mass does a budget of `K`
tokens capture? That upper-bounds QSA, DSA and every other top-k scheme, so a
poor number here kills the idea without training an indexer.

Three oracles per budget `K`:

  tok      top-`K` individual tokens.           DSA's ceiling.
  blk_sum  top-`K/r` blocks ranked by block mass. QSA's ceiling at `r`-token
           granularity -- the cost of coarseness is `tok - blk_sum`.
  blk_max  top-`K/r` blocks ranked by block MAX, which is what QSA's Eq. 17
           teacher actually ranks by. `blk_sum - blk_max` is what the max-pool
           ranking leaves on the table.

The incomplete tail block is always counted as captured, matching Eq. 19. That
makes `blk_*` attend `K + r - 1` tokens against `tok`'s `K`, so `blk_sum` can
edge above `tok` on a row whose tail carries real mass -- an asymmetry of the
methods themselves (DSA has no tail rule), not of the measurement. The
granularity cost is therefore read as `tok - (blk_sum - tail)`.

Teacher distribution is the one from the report: per-head softmax, summed over
heads, L1-normalized (which is just `/H`, since each head's row already sums to
one). Computed in fp32 from the q/k the layer actually used, caught with a
forward pre-hook on `inner_attention`, so no model code changes.

Only rows at position >= `--min-pos` are reported: a row with fewer than `K`
visible tokens captures everything trivially and would wash out the mean.

Usage:
    python utils/oracle_sparse_recall.py \
        --model-dir /path/to/hf_snapshot \
        --data long_docs.jsonl \
        --seq-len 65536 --budgets 512,1024,2048,4096

    python utils/oracle_sparse_recall.py --selftest   # no GPU, no model
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_NEG = float("-inf")


def _metrics_from_p(
    p: torch.Tensor, pos: torch.Tensor, r: int, budgets: list[int]
) -> dict[str, torch.Tensor]:
    """Oracle masses for rows `pos` of a teacher distribution `p` of shape (c, T).

    `p` rows sum to 1 over `j <= pos`. Split out so `--selftest` can hit it
    without a model.
    """
    c, T = p.shape
    nb = T // r
    dev = p.device

    pb = p[:, : nb * r].reshape(c, nb, r)
    bsum = pb.sum(-1)
    bmax = pb.amax(-1)

    # Eq. 15: a block is selectable only if it lies entirely in the past.
    block_end = (torch.arange(nb, device=dev) + 1) * r - 1
    complete = block_end[None, :] <= pos[:, None]
    bsum_c = bsum * complete
    # Everything not in a complete block is the tail, and the tail is free.
    tail = (1.0 - bsum_c.sum(-1)).clamp_min(0.0)

    out: dict[str, torch.Tensor] = {"tail": tail}

    ent = -(p * p.clamp_min(1e-20).log()).sum(-1)
    out["eff_support_tok"] = ent.exp()
    bb = torch.cat([bsum_c, tail[:, None]], dim=-1)
    out["eff_support_blk"] = (-(bb * bb.clamp_min(1e-20).log()).sum(-1)).exp()

    for K in budgets:
        kb = min(max(K // r, 1), nb)
        for name, score in (("blk_sum", bsum), ("blk_max", bmax)):
            idx = score.masked_fill(~complete, _NEG).topk(kb, dim=-1).indices
            # topk still returns padding picks when a row has < kb complete
            # blocks; mask their mass out instead of counting it twice.
            got = (bsum.gather(-1, idx) * complete.gather(-1, idx)).sum(-1)
            out[f"{name}@{K}"] = (got + tail).clamp(max=1.0)
        out[f"tok@{K}"] = p.topk(min(K, T), dim=-1).values.sum(-1).clamp(max=1.0)
    return out


def document_segments(
    positions: torch.Tensor, num_tokens: int | None = None
) -> list[tuple[int, int]]:
    """`[start, end)` of every document in a packed row, from the 1D positions.

    Same rule as `Qwen35Model.get_attention_masks`: a document starts at a 0
    followed by a 1, so a run of right padding (every padded token is position 0)
    is not split into one-token documents. That rule also means trailing padding
    is absorbed into the last document, which is why `num_tokens` trims it.
    """
    n = positions.shape[0]
    dev = positions.device
    followed_by_one = torch.cat(
        [positions[1:] == 1, torch.zeros(1, dtype=torch.bool, device=dev)]
    )
    first = torch.arange(n, device=dev) == 0
    starts = ((positions == 0) & followed_by_one) | first
    bounds = starts.nonzero(as_tuple=True)[0].tolist() + [min(num_tokens or n, n)]
    return [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if b > a]


@torch.no_grad()
def _segment_metrics(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    *,
    scale: float,
    r: int,
    budgets: list[int],
    min_pos: int,
    chunk: int,
    row_stride: int,
) -> dict[str, list[torch.Tensor]]:
    """One document. Positions are document-relative, so causality and the block
    grid both start at 0 -- which is the point of slicing per document.

    The (chunk, H, T) score tile is the memory ceiling: fp32, so 128 queries x
    16 heads x 64K keys is ~0.5 GB, doubled during the softmax.
    """
    T, H, _ = q_THK.shape
    hkv = k_THK.shape[1]
    k_full = k_THK.repeat_interleave(H // hkv, dim=1) if hkv != H else k_THK
    keys = torch.arange(T, device=q_THK.device)
    rows = keys[min_pos::row_stride]

    acc: dict[str, list[torch.Tensor]] = {}
    for start in range(0, rows.numel(), chunk):
        pos = rows[start : start + chunk]
        if pos.numel() == 0:
            continue
        qc = q_THK[pos].float()
        s = torch.einsum("chd,thd->cht", qc, k_full.float()) * scale
        s.masked_fill_(keys[None, None, :] > pos[:, None, None], _NEG)
        p = s.softmax(dim=-1).sum(dim=1).div_(H)
        del s
        for key, val in _metrics_from_p(p, pos, r, budgets).items():
            acc.setdefault(key, []).append(val.cpu())
        del p
    return acc


def layer_metrics(
    q_THK: torch.Tensor,
    k_THK: torch.Tensor,
    *,
    scale: float,
    r: int,
    budgets: list[int],
    min_pos: int,
    chunk: int,
    row_stride: int = 1,
    segments: list[tuple[int, int]] | None = None,
) -> dict[str, torch.Tensor]:
    """Oracle masses for one layer of one packed row, document by document.

    Without `segments` the row is treated as a single document, which is what the
    text mode produces. With the real loader a row holds several packed documents
    and a padding tail, and a plain causal mask over the whole row would let
    queries attend across document boundaries -- so each document is measured on
    its own slice.
    """
    T = q_THK.shape[0]
    acc: dict[str, list[torch.Tensor]] = {}
    for a, b in segments or [(0, T)]:
        if b - a <= min_pos:
            continue
        part = _segment_metrics(
            q_THK[a:b],
            k_THK[a:b],
            scale=scale,
            r=r,
            budgets=budgets,
            min_pos=min_pos,
            chunk=chunk,
            row_stride=row_stride,
        )
        for key, vals in part.items():
            acc.setdefault(key, []).extend(vals)
    return {k: torch.cat(v) for k, v in acc.items()}


def _hook(store: dict, name: str, cfg: dict, current: dict):
    def pre(_module, args, kwargs):
        q, k = args[0], args[1]
        m = layer_metrics(
            q, k, scale=kwargs["scale"], segments=current.get("segments"), **cfg
        )
        for key, val in m.items():
            store.setdefault(name, {}).setdefault(key, []).append(val)
        return None

    return pre


def _documents(path: Path, text_key: str):
    if path.is_dir():
        for f in sorted(path.glob("*.txt")):
            yield f.name, f.read_text()
        return
    if path.suffix == ".jsonl":
        with path.open() as fh:
            for i, line in enumerate(fh):
                line = line.strip()
                if line:
                    yield f"doc{i}", json.loads(line)[text_key]
        return
    yield path.name, path.read_text()


def _metadataset_batches(args, device):
    """Documents through the production path: energon metadataset -> cooker ->
    `PackedBatchEncoder`. Nothing here reimplements the loader, so the measurement
    is on what training actually sees, including the chat template, placeholder
    expansion and MRoPE.

    `--packing-buffer-size 1` keeps it to one document per row: packing several
    into a row is correct for training but only complicates the measurement, and
    the long documents this is aimed at do not leave room for a second anyway.
    """
    from transformers import AutoConfig, AutoProcessor

    from data.energon_dataloader import PackedBatchEncoder
    from megatron.energon import WorkerConfig, get_loader, get_train_dataset

    processor = AutoProcessor.from_pretrained(args.model_dir, max_pixels=args.max_pixels)
    if args.chat_template and args.chat_template != "NULL":
        processor.chat_template = Path(args.chat_template).read_text()
    cfg = AutoConfig.from_pretrained(args.model_dir)
    image_token_id = getattr(cfg, "image_token_id", None)
    video_token_id = getattr(cfg, "video_token_id", None)
    vision_cfg = getattr(cfg, "vision_config", None)
    encoder = PackedBatchEncoder(
        processor,
        args.seq_len,
        args.seq_len,
        image_token_id=-1 if image_token_id is None else image_token_id,
        video_token_id=-1 if video_token_id is None else video_token_id,
        spatial_merge_size=getattr(vision_cfg, "spatial_merge_size", None) or 1,
    )
    dataset = get_train_dataset(
        args.metadataset,
        batch_size=1,
        repeat=False,
        shuffle_buffer_size=args.shuffle_buffer_size,
        max_samples_per_sequence=args.max_samples_per_sequence,
        task_encoder=encoder,
        worker_config=WorkerConfig(rank=0, world_size=1, num_workers=0, seed_offset=args.seed),
        packing_buffer_size=args.packing_buffer_size,
    )
    for i, raw in enumerate(get_loader(dataset)):
        batch = {}
        for key, val in raw.items():
            if isinstance(val, torch.Tensor):
                if val.ndim > 1 and val.shape[0] == 1 and key in ("input", "labels", "positions"):
                    val = val.squeeze(0)
                batch[key] = val.to(device)
            else:
                batch[key] = val
        num_tokens = int(batch.pop("num_tokens", batch["input"].shape[0]))
        if num_tokens < args.min_tokens:
            continue
        batch.pop("num_valid_tokens", None)
        batch.pop("num_samples", None)
        segments = document_segments(batch["positions"], num_tokens)
        n_img = 0 if "grid_thw" not in batch else batch["grid_thw"].shape[0]
        yield f"row{i}", batch, num_tokens, n_img, segments


def _report(store: dict, budgets: list[int], r: int) -> dict:
    rows = []
    for layer, metrics in store.items():
        agg = {k: torch.cat(v) for k, v in metrics.items()}
        n = int(agg["tail"].numel())
        row = {"layer": layer, "rows": n}
        for key, val in agg.items():
            row[key] = float(val.mean())
            row[f"{key}_p5"] = float(val.quantile(0.05)) if n else float("nan")
        rows.append(row)

    width = 11 * len(budgets)
    print(f"\n{'layer':>18} {'rows':>9} {'eff_sup_tok':>12} {'eff_sup_blk':>12} {'tail':>7}")
    for row in rows:
        print(
            f"{row['layer']:>18} {row['rows']:>9} "
            f"{row['eff_support_tok']:>12.1f} {row['eff_support_blk']:>12.1f} "
            f"{row['tail']:>7.4f}"
        )
    for name in ("tok", "blk_sum", "blk_max"):
        head = "".join(f"{'K=' + str(b):>11}" for b in budgets)
        print(f"\n{name} oracle mass (mean / p5), r={r}")
        print(f"{'layer':>18}{head}")
        for row in rows:
            mean = "".join(f"{row[f'{name}@{b}']:>11.4f}" for b in budgets)
            p5 = "".join(f"{row[f'{name}@{b}_p5']:>11.4f}" for b in budgets)
            print(f"{row['layer']:>18}{mean}")
            print(f"{'  (p5)':>18}{p5}")
    if rows:
        K = budgets[-1]
        # `blk_*` attends K + r - 1 tokens (the free tail) against `tok`'s K, so this
        # slightly flatters the block oracle. It is still the right order of magnitude
        # for what r-token granularity costs, and it grows with context.
        gran = [(r["layer"], r[f"tok@{K}"] - r[f"blk_sum@{K}"]) for r in rows]
        rank = [(r["layer"], r[f"blk_sum@{K}"] - r[f"blk_max@{K}"]) for r in rows]
        print(f"\nat K={K}, per layer:")
        print(f"  block granularity costs {min(g for _, g in gran):.4f}-"
              f"{max(g for _, g in gran):.4f} of mass (tok - blk_sum)")
        print(f"  max-pool ranking costs  {min(g for _, g in rank):.4f}-"
              f"{max(g for _, g in rank):.4f} of mass (blk_sum - blk_max)")
        worst = min(rows, key=lambda x: x[f"blk_sum@{K}"])
        print(f"  worst layer {worst['layer']} at {worst[f'blk_sum@{K}']:.4f} "
              f"mean, {worst[f'blk_sum@{K}_p5']:.4f} p5")
    return {"budgets": budgets, "r": r, "layers": rows}


def selftest() -> None:
    r, budgets = 4, [8, 16]

    # one-hot teacher: any budget captures everything
    p = torch.zeros(3, 64)
    pos = torch.tensor([40, 50, 63])
    p[0, 7] = p[1, 33] = p[2, 2] = 1.0
    m = _metrics_from_p(p, pos, r, budgets)
    for b in budgets:
        for name in ("tok", "blk_sum", "blk_max"):
            assert torch.allclose(m[f"{name}@{b}"], torch.ones(3), atol=1e-6), (name, b)
    assert torch.allclose(m["eff_support_tok"], torch.ones(3), atol=1e-4)

    # uniform teacher over j <= pos: top-K tokens capture K/(pos+1)
    T = 64
    pos = torch.tensor([39, 63])
    p = torch.zeros(2, T)
    for i, pp in enumerate(pos.tolist()):
        p[i, : pp + 1] = 1.0 / (pp + 1)
    m = _metrics_from_p(p, pos, r, budgets)
    for b in budgets:
        want = torch.tensor([min(b / (pp + 1), 1.0) for pp in pos.tolist()])
        assert torch.allclose(m[f"tok@{b}"], want, atol=1e-5), (b, m[f"tok@{b}"], want)
    # blocks are equal mass under a uniform teacher, so b/r blocks + tail
    for b in budgets:
        kb = b // r
        want = torch.tensor(
            [min((kb * r + (pp + 1) % r) / (pp + 1), 1.0) for pp in pos.tolist()]
        )
        assert torch.allclose(m[f"blk_sum@{b}"], want, atol=1e-5), (b, m[f"blk_sum@{b}"], want)
    assert torch.allclose(m["eff_support_tok"], (pos + 1).float(), rtol=1e-3)

    # mass outside complete blocks is tail, and tail is free
    p = torch.zeros(1, T)
    pos = torch.tensor([9])  # blocks 0,1 complete (0-7); tokens 8,9 are tail
    p[0, 8] = p[0, 9] = 0.5
    m = _metrics_from_p(p, pos, r, [4])
    assert torch.allclose(m["tail"], torch.tensor([1.0]), atol=1e-6), m["tail"]
    assert torch.allclose(m["blk_sum@4"], torch.tensor([1.0]), atol=1e-6)

    # a row with fewer complete blocks than the budget must not double-count
    p = torch.zeros(1, T)
    pos = torch.tensor([7])
    p[0, :8] = 1.0 / 8
    m = _metrics_from_p(p, pos, r, [64])
    assert torch.allclose(m["blk_sum@64"], torch.tensor([1.0]), atol=1e-6), m["blk_sum@64"]

    # invariants on a real score tile: more budget never captures less, the
    # token oracle dominates the block oracle once the free tail is removed,
    # and sum-ranking dominates max-ranking.
    torch.manual_seed(0)
    T, H, Hkv, D = 512, 8, 2, 64
    m = layer_metrics(
        torch.randn(T, H, D),
        torch.randn(T, Hkv, D),
        scale=D**-0.5,
        r=4,
        budgets=[16, 64, 256],
        min_pos=128,
        chunk=64,
    )
    assert m["tail"].numel() == T - 128
    for name in ("tok", "blk_sum", "blk_max"):
        assert (m[f"{name}@256"] >= m[f"{name}@64"] - 1e-6).all(), name
        assert (m[f"{name}@256"] <= 1 + 1e-5).all(), name
    assert (m["blk_sum@256"] >= m["blk_max@256"] - 1e-5).all()
    assert (m["tok@256"] >= m["blk_sum@256"] - m["tail"] - 1e-5).all()
    assert (m["eff_support_blk"] <= m["eff_support_tok"] + 1e-3).all()

    # packed rows: documents must be measured separately, and right padding
    # (every padded token is position 0) must not become attendable keys.
    pos = torch.cat(
        [torch.arange(300), torch.arange(500), torch.zeros(224, dtype=torch.long)]
    )
    assert document_segments(pos, 800) == [(0, 300), (300, 800)]
    assert document_segments(pos) == [(0, 300), (300, 1024)]  # untrimmed: padding absorbed
    T, H, Hkv, D = 1024, 4, 1, 32
    seg = layer_metrics(
        torch.randn(T, H, D), torch.randn(T, Hkv, D),
        scale=D**-0.5, r=4, budgets=[64], min_pos=128, chunk=64,
        segments=document_segments(pos, 800),
    )
    assert seg["tail"].numel() == (300 - 128) + (500 - 128), seg["tail"].numel()

    print("selftest ok")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model-dir", help="HF snapshot directory holding the weights")
    ap.add_argument(
        "--model-config",
        help="architecture json; defaults to <model-dir>/config.json. Rarely needed: "
        "the snapshots are self-describing. If a load fails on a missing key like "
        "`model.embed_tokens.weight`, the cause is almost certainly --no-vision on a "
        "multimodal snapshot (without the tower the adapter drops the "
        "`language_model.` prefix), not the config.",
    )
    ap.add_argument("--no-vision", action="store_true", help="build without the vision tower")
    ap.add_argument("--data", help=".jsonl (one doc per line), a .txt file, or a dir of .txt")
    ap.add_argument("--text-key", default="text")
    ap.add_argument(
        "--metadataset",
        help="energon metadataset yaml; runs documents through the production "
        "cooker + PackedBatchEncoder path. Replaces --data.",
    )
    ap.add_argument("--shuffle-buffer-size", type=int, default=100)
    ap.add_argument("--max-samples-per-sequence", type=int, default=100)
    ap.add_argument("--packing-buffer-size", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chat-template", default="assets/chat_template_sft.jinja")
    ap.add_argument("--max-pixels", type=int, default=1048576)
    ap.add_argument("--seq-len", type=int, default=65536)
    ap.add_argument("--budgets", default="512,1024,2048,4096", help="token budgets K")
    ap.add_argument("--block-size", type=int, default=4, help="r, the QSA compression ratio")
    ap.add_argument("--min-pos", type=int, default=0, help="skip rows below this position (default: 2x max budget)")
    ap.add_argument("--chunk", type=int, default=128, help="query rows per score tile")
    ap.add_argument(
        "--row-stride",
        type=int,
        default=1,
        help="measure every Nth query row; the cost is linear in the row count, "
        "so 16 turns an hour into minutes and barely moves the means",
    )
    ap.add_argument("--max-docs", type=int, default=0, help="0 = all")
    ap.add_argument(
        "--min-tokens",
        type=int,
        default=8192,
        help="skip documents shorter than this. Note the two input modes differ on "
        "over-long documents: --data truncates to --seq-len, while --metadataset "
        "skips them (PackedBatchEncoder rejects length > row_length), so the "
        "measured document lengths are not the same. The report records them.",
    )
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--out", help="write the per-layer table as json")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.model_dir:
        ap.error("--model-dir is required (or use --selftest)")
    if bool(args.data) == bool(args.metadataset):
        ap.error("pass exactly one of --data (raw text) or --metadataset (the real loader)")

    budgets = [int(b) for b in args.budgets.split(",")]
    min_pos = args.min_pos or 2 * max(budgets)
    if min_pos >= args.seq_len:
        ap.error(f"--min-pos {min_pos} leaves no rows at --seq-len {args.seq_len}")

    from transformers import AutoTokenizer

    from models.qwen3_5.checkpoint import build_meta, load_hf, materialize

    tok = AutoTokenizer.from_pretrained(args.model_dir)
    # build_meta and load_hf take separate paths on purpose: the architecture may
    # have to come from the repo's json while the weights come from the snapshot.
    model = build_meta(
        args.model_config or args.model_dir,
        seq_len=args.seq_len,
        with_vision=bool(args.metadataset) or not args.no_vision,
    )
    # Materialize on the host, not the GPU: build_meta's dtype is fp32, so a 35B
    # would want ~140 GB of HBM before the cast. Load, cast, then move.
    materialize(model, "cpu")
    load_hf(model, args.model_dir)
    model = model.to(getattr(torch, args.dtype)).to("cuda")
    model.eval()
    # The hooks fire inside the attention layers, so the logits are never read --
    # and at 64K x 248k vocab they are a 32 GB allocation that OOMs a 35B.
    model._skip_lm_head = True

    store: dict = {}
    current: dict = {}
    cfg = {
        "r": args.block_size,
        "budgets": budgets,
        "min_pos": min_pos,
        "chunk": args.chunk,
        "row_stride": args.row_stride,
    }
    handles = []
    for name, block in model.layers.items():
        if getattr(block, "full_attn", False):
            handles.append(
                block.attn.inner_attention.register_forward_pre_hook(
                    _hook(store, f"layer{name}", cfg, current), with_kwargs=True
                )
            )
    if not handles:
        raise RuntimeError("no full-attention layers found; nothing to measure")
    print(f"hooked {len(handles)} full-attention layers, min_pos={min_pos}")

    device = next(model.parameters()).device

    def text_batches():
        for name, text in _documents(Path(args.data), args.text_key):
            ids = tok(text, add_special_tokens=False)["input_ids"][: args.seq_len]
            if len(ids) < max(args.min_tokens, min_pos + 1):
                print(f"skip {name}: {len(ids)} tokens")
                continue
            ids_t = torch.tensor(ids, dtype=torch.long, device=device)
            yield name, {
                "input": ids_t,
                "labels": ids_t,
                "positions": torch.arange(len(ids), device=device),
            }, len(ids), 0, None

    if args.metadataset:
        from transformers import AutoConfig

        image_token_id = getattr(AutoConfig.from_pretrained(args.model_dir), "image_token_id", None)
        if image_token_id is None:
            raise RuntimeError("the snapshot config has no image_token_id; wrong model for images?")
        source = _metadataset_batches(args, device)
        extra = {"special_tokens": {"image_id": image_token_id}}
    else:
        source = text_batches()
        extra = {}

    seen = 0
    doc_lengths: list[int] = []
    for name, batch, ntok, nimg, segments in source:
        if ntok <= min_pos:
            continue
        doc_lengths.append(ntok)
        # the hook needs the document bounds of THIS row, and preprocess_inputs
        # overwrites batch["positions"] with the MRoPE ones, so they are taken first.
        current["segments"] = segments
        inputs, _labels, rest = model.preprocess_inputs(batch)
        with torch.no_grad():
            model(inputs, **rest, **extra)
        seen += 1
        print(f"{name}: {ntok} tokens, {nimg} images")
        if args.max_docs and seen >= args.max_docs:
            break
    for h in handles:
        h.remove()
    if seen == 0:
        raise RuntimeError("no document was long enough; lower --min-tokens or check --data")

    report = _report(store, budgets, args.block_size)
    report["documents"] = seen
    report["seq_len"] = args.seq_len
    report["min_pos"] = min_pos
    report["row_stride"] = args.row_stride
    if doc_lengths:
        ordered = sorted(doc_lengths)
        report["doc_tokens_mean"] = sum(ordered) / len(ordered)
        report["doc_tokens_median"] = ordered[len(ordered) // 2]
        report["doc_tokens_min"] = ordered[0]
        report["doc_tokens_max"] = ordered[-1]
        print(
            f"\ndocuments measured: n={len(ordered)} tokens "
            f"mean={report['doc_tokens_mean']:.0f} median={report['doc_tokens_median']} "
            f"range={ordered[0]}-{ordered[-1]}"
        )
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
