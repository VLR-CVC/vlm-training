"""Verify decode-speed gain from the MTP head via self-speculative decoding.

Compares plain greedy autoregressive decoding ("baseline") against depth-1 MTP
self-speculation ("spec") on the native models, and reports the draft acceptance
rate, tokens per trunk forward, and wall-clock tokens/sec.

How depth-1 MTP speculation works here (greedy, exact):
  - The trunk predicts the next token ``t`` (and gives hidden states).
  - The MTP head drafts ``d`` = the token *after* ``t`` (from the trunk hidden +
    embed(t), run over the full prefix so it attends to context).
  - One trunk forward on ``[ids, t, d]`` verifies both positions in parallel:
      * position of ``t`` predicts the true token after ``t`` (``r1``);
      * position of ``d`` is a bonus predicting the token after ``d`` (``r2``).
    If ``r1 == d`` the draft is accepted -> 2 tokens from 1 trunk forward;
    otherwise 1 token (and ``r1`` is the corrected continuation).
  Greedy spec output is identical to greedy baseline output, so the script also
  asserts the two token streams match (a correctness check on the head).

Caveats:
  - No KV cache: every forward reprocesses the whole prefix (training-shaped
    varlen forward). The trunk (many layers) dominates wall-clock, so the
    speedup tracks "tokens per trunk forward". A KV-cache deployment would shift
    absolute numbers but the acceptance rate is the portable metric.
  - Run uncompiled (eager) to avoid recompilation noise across lengths.
  - A randomly-initialized MTP head (e.g. a fresh Qwen3-VL head) drafts garbage
    -> ~0 acceptance -> spec is slower. Use a checkpoint whose mtp.* weights are
    trained (e.g. Qwen3.5) to see the real gain.

Run:
    CUDA_VISIBLE_DEVICES=7 python eval/benchmark_mtp_decode.py \
        --model_dir /data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b \
        --model_type qwen3_5 --num_tokens 128
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

PROMPTS = [
    "The history of the Roman Empire began when",
    "Here is a simple recipe for chocolate chip cookies. First,",
    "In the field of machine learning, a transformer is",
    "Once upon a time, in a small village by the sea, there lived",
]


def load_model(model_dir: str, model_type: str, device: str):
    if model_type == "qwen3_5":
        from models.qwen3_5.model import Qwen3_5ForCausalLM as M
    elif model_type == "qwen3_vl":
        from models.qwen3_vl.model import Qwen3VLForCausalLM as M
    else:
        raise ValueError(f"unknown model_type {model_type}")
    model, cfg = M.from_pretrained(model_dir, dtype=torch.bfloat16, device=device, load_vision=False)
    model.eval()
    if model.mtp is None:
        raise RuntimeError(
            "model has no MTP head (mtp_num_hidden_layers=0 in config.json); "
            "nothing to speculate with."
        )
    return model, cfg


def _text_rope(model, length: int, device, dtype):
    pos = torch.arange(length, device=device).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = model._compute_cos_sin(pos)
    return cos.to(dtype), sin.to(dtype)


def _trunk(model, seq: torch.Tensor):
    """Return (hidden, logits) for a packed single-segment text row (1, M)."""
    device = seq.device
    dtype = model.lm_head.weight.dtype
    M = seq.shape[1]
    cos, sin = _text_rope(model, M, device, dtype)
    cu = torch.tensor([0, M], dtype=torch.int32, device=device)
    embeds = model.model.language_model.embed_tokens(seq)
    h = model.model.language_model(embeds, cos, sin, cu, M)
    return h, model.lm_head(h)


def _mtp_draft(model, seq: torch.Tensor, h_seq: torch.Tensor, next_tok: torch.Tensor):
    """Draft the token after ``next_tok`` using the MTP head over the full prefix.

    ``h_seq`` is the trunk hidden over ``seq`` (its last position predicts
    ``next_tok``). MTP consumes hidden_i + embed(token_{i+1}); at the last
    position token_{i+1} is the just-predicted ``next_tok``.
    """
    device = seq.device
    dtype = model.lm_head.weight.dtype
    M = seq.shape[1]
    cos, sin = _text_rope(model, M, device, dtype)
    cu = torch.tensor([0, M], dtype=torch.int32, device=device)
    shifted = torch.cat([seq[:, 1:], next_tok.view(1, 1)], dim=1)
    next_embeds = model.model.language_model.embed_tokens(shifted)
    mtp_h = model.mtp(h_seq, next_embeds, cos, sin, cu, M)
    return model.lm_head(mtp_h[:, -1]).argmax(-1)[0]  # 0-dim token id


@torch.no_grad()
def baseline_decode(model, ids: torch.Tensor, n_tokens: int):
    out = []
    n_trunk = 0
    while len(out) < n_tokens:
        _, logits = _trunk(model, ids)
        n_trunk += 1
        t = logits[0, -1].argmax()
        ids = torch.cat([ids, t.view(1, 1)], dim=1)
        out.append(int(t))
    return out, {"trunk_forwards": n_trunk}


@torch.no_grad()
def spec_decode(model, ids: torch.Tensor, n_tokens: int):
    out = []
    n_trunk = 0
    n_steps = 0
    n_accept = 0

    h, logits = _trunk(model, ids)
    n_trunk += 1
    t = logits[0, -1].argmax()
    d = _mtp_draft(model, ids, h, t)

    while len(out) < n_tokens:
        verify = torch.cat([ids, t.view(1, 1), d.view(1, 1)], dim=1)
        h, logits = _trunk(model, verify)
        n_trunk += 1
        n_steps += 1
        r1 = logits[0, -2].argmax()  # true token after t
        r2 = logits[0, -1].argmax()  # bonus: token after d
        if torch.equal(r1, d):
            n_accept += 1
            ids = verify
            out.append(int(t)); out.append(int(d))
            t = r2
            d = _mtp_draft(model, ids, h, t)
        else:
            ids = torch.cat([ids, t.view(1, 1)], dim=1)
            out.append(int(t))
            t = r1
            d = _mtp_draft(model, ids, h[:, : ids.shape[1]], t)

    stats = {
        "trunk_forwards": n_trunk,
        "steps": n_steps,
        "accept_rate": n_accept / max(n_steps, 1),
    }
    return out[:n_tokens], stats


def _timed(fn, *args):
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    res = fn(*args)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return res, time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--model_type", default="qwen3_5", choices=["qwen3_5", "qwen3_vl"])
    ap.add_argument("--num_tokens", type=int, default=128)
    ap.add_argument("--num_prompts", type=int, default=len(PROMPTS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=None,
                    help="write a presentation-ready Markdown report to this path")
    ap.add_argument("--show_chars", type=int, default=320,
                    help="max chars of generated text to show in the console")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    device = args.device
    model, cfg = load_model(args.model_dir, args.model_type, device)
    tok = AutoTokenizer.from_pretrained(args.model_dir, use_fast=False)
    print(f"model={args.model_type}  mtp_num_hidden_layers={cfg.text.mtp_num_hidden_layers}")

    # warmup (compile-free, but warms kernels / caches)
    warm = tok(PROMPTS[0], return_tensors="pt").input_ids.to(device)
    baseline_decode(model, warm, 8)
    spec_decode(model, warm, 8)

    results = []  # one dict per prompt
    tot_base_tok = tot_base_t = tot_spec_tok = tot_spec_t = 0.0
    spec_trunk = base_trunk = 0
    for prompt in PROMPTS[: args.num_prompts]:
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)

        (base_out, base_stats), base_t = _timed(baseline_decode, model, ids, args.num_tokens)
        (spec_out, spec_stats), spec_t = _timed(spec_decode, model, ids, args.num_tokens)

        match = base_out == spec_out  # greedy spec is exact -> identical text
        text = tok.decode(spec_out, skip_special_tokens=True)

        base_tps = args.num_tokens / base_t
        spec_tps = args.num_tokens / spec_t
        spec_trunk += spec_stats["trunk_forwards"]
        base_trunk += base_stats["trunk_forwards"]
        tot_base_tok += args.num_tokens; tot_base_t += base_t
        tot_spec_tok += args.num_tokens; tot_spec_t += spec_t
        results.append({
            "prompt": prompt, "text": text,
            "base_tps": base_tps, "spec_tps": spec_tps, "speedup": spec_tps / base_tps,
            "accept": spec_stats["accept_rate"], "exact": match,
        })

    overall = {
        "model": args.model_type,
        "num_tokens": args.num_tokens,
        "dtype": str(model.lm_head.weight.dtype).replace("torch.", ""),
        "base_tps": tot_base_tok / tot_base_t,
        "spec_tps": tot_spec_tok / tot_spec_t,
        "speedup": (tot_spec_tok / tot_spec_t) / (tot_base_tok / tot_base_t),
        "accept": sum(r["accept"] for r in results) / len(results),
        "tok_per_fwd_base": tot_base_tok / base_trunk,
        "tok_per_fwd_spec": tot_spec_tok / spec_trunk,
        "all_exact": all(r["exact"] for r in results),
    }
    _print_console(results, overall, args.show_chars)
    if args.out:
        _write_markdown(args.out, results, overall)
        print(f"\nMarkdown report written to {args.out}")


def _print_console(results, overall, show_chars):
    BOLD, DIM, GRN, CYN, RST = "\033[1m", "\033[2m", "\033[32m", "\033[36m", "\033[0m"
    bar = "=" * 70
    print(f"\n{bar}")
    print(f"{BOLD}  MTP Self-Speculative Decoding — {overall['model']}{RST}")
    print(bar)
    print(f"  {DIM}greedy · no KV-cache (eager) · {overall['dtype']} · "
          f"{overall['num_tokens']} tokens/prompt{RST}")
    exact_note = "lossless (identical output)" if overall["all_exact"] else "OUTPUT DIFFERS!"
    print(f"\n  {BOLD}{GRN}▶ {overall['speedup']:.2f}× faster decoding{RST}  "
          f"({overall['base_tps']:.1f} → {overall['spec_tps']:.1f} tok/s)  ·  "
          f"{exact_note}")
    print(f"  {BOLD}▶ draft acceptance {overall['accept']:.0%}{RST}  ·  "
          f"{overall['tok_per_fwd_spec']:.2f} tokens per model forward "
          f"(vs {overall['tok_per_fwd_base']:.2f} baseline)")

    print(f"\n  {BOLD}Sample generations{RST} {DIM}(prompt → continuation; "
          f"baseline and MTP produce the same text){RST}")
    print("  " + "-" * 66)
    for i, r in enumerate(results, 1):
        gen = r["text"].replace("\n", " ").strip()
        if len(gen) > show_chars:
            gen = gen[:show_chars].rstrip() + " …"
        print(f"  {BOLD}[{i}]{RST} {CYN}{r['prompt']}{RST}")
        print(f"      {gen}")
        print(f"      {DIM}{r['speedup']:.2f}× · accept {r['accept']:.0%} · "
              f"{'✓ identical' if r['exact'] else '✗ DIFFERS'}{RST}")

    print(f"\n  {BOLD}Per-prompt metrics{RST}")
    print(f"  {'prompt':<34}{'base t/s':>10}{'spec t/s':>10}{'speedup':>9}{'accept':>8}")
    print("  " + "-" * 66)
    for r in results:
        print(f"  {r['prompt'][:32]:<34}{r['base_tps']:>10.1f}{r['spec_tps']:>10.1f}"
              f"{r['speedup']:>8.2f}×{r['accept']:>8.0%}")
    print(f"  {BOLD}{'OVERALL':<34}{overall['base_tps']:>10.1f}{overall['spec_tps']:>10.1f}"
          f"{overall['speedup']:>8.2f}×{overall['accept']:>8.0%}{RST}")
    print(bar + "\n")


def _write_markdown(path, results, overall):
    L = []
    L.append(f"# MTP Self-Speculative Decoding — {overall['model']}\n")
    L.append(f"*greedy · no KV-cache (eager) · {overall['dtype']} · "
             f"{overall['num_tokens']} tokens/prompt*\n")
    L.append(f"## Headline\n")
    exact = "**lossless** — identical output to standard decoding" if overall["all_exact"] \
        else "**WARNING: output differs from baseline**"
    L.append(f"- 🚀 **{overall['speedup']:.2f}× faster decoding** "
             f"({overall['base_tps']:.1f} → {overall['spec_tps']:.1f} tok/s)")
    L.append(f"- ✅ {exact}")
    L.append(f"- 🎯 **{overall['accept']:.0%} draft acceptance** — "
             f"{overall['tok_per_fwd_spec']:.2f} tokens per model forward "
             f"(vs {overall['tok_per_fwd_base']:.2f} baseline)\n")

    L.append("## Throughput\n")
    L.append("| Prompt | Baseline tok/s | MTP tok/s | Speedup | Accept |")
    L.append("|---|--:|--:|--:|--:|")
    for r in results:
        L.append(f"| {r['prompt']} | {r['base_tps']:.1f} | {r['spec_tps']:.1f} "
                 f"| {r['speedup']:.2f}× | {r['accept']:.0%} |")
    L.append(f"| **Overall** | **{overall['base_tps']:.1f}** | **{overall['spec_tps']:.1f}** "
             f"| **{overall['speedup']:.2f}×** | **{overall['accept']:.0%}** |\n")

    L.append("## Sample generations\n")
    L.append("_Baseline and MTP speculative decoding produce the **same** text "
             "(greedy is exact); MTP just gets there faster._\n")
    for i, r in enumerate(results, 1):
        L.append(f"**{i}. {r['prompt']}**\n")
        L.append(f"> {r['text'].strip()}\n")
        L.append(f"<sub>{r['speedup']:.2f}× faster · {r['accept']:.0%} draft acceptance · "
                 f"{'identical to baseline ✓' if r['exact'] else 'DIFFERS ✗'}</sub>\n")

    with open(path, "w") as f:
        f.write("\n".join(L))


if __name__ == "__main__":
    main()
