from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

MAX_CONSECUTIVE_REPEAT = 8     # >= this many identical tokens in a row -> degenerate
MIN_DISTINCT_2 = 0.25          # unique bigrams / bigrams
MIN_DISTINCT_3 = 0.30          # unique trigrams / trigrams
MIN_UNIQUE_TOKEN_RATIO = 0.35  # unique tokens / tokens
CYCLE_MAX_PERIOD = 50          # search short repeating cycles up to this period
CYCLE_COVERAGE = 0.60          # fraction of tokens explained by a cycle -> degenerate
MIN_TOKENS = 3                 # shorter than this -> degenerate (empty/near-empty)

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp")

DEFAULT_TEXT_PROMPTS = [
    "What is a transformer?",
    "Explain the difference between a list and a tuple in Python.",
    "Write a short poem about the ocean.",
    "What is the capital of France, and what is it famous for?",
    "Give me three tips for staying focused while working.",
]

DEFAULT_COMPLETION_PROMPTS = [
    "The capital of France is",
    "The three primary colors are",
    "def fibonacci(n):\n    ",
    "Q: What is the capital of Japan?\nA: Tokyo\n\nQ: What is a transformer (in machine learning)?\nA:",
]


def max_consecutive_repeat(tokens: list[int]) -> int:
    best = cur = 0
    prev = None
    for t in tokens:
        cur = cur + 1 if t == prev else 1
        prev = t
        best = max(best, cur)
    return best


def distinct_n(tokens: list[int], n: int) -> float:
    if len(tokens) < n:
        return 1.0
    grams = [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]
    return len(set(grams)) / max(1, len(grams))


def dominant_cycle(tokens: list[int]) -> tuple[int, float]:
    """Smallest period p whose repetition best explains the tail of the output.

    Returns (period, coverage). coverage = fraction of positions i (i>=p) where
    tokens[i] == tokens[i-p]; a looping model has coverage ~1.0 at small p.
    """
    n = len(tokens)
    best_p, best_cov = 0, 0.0
    for p in range(1, min(CYCLE_MAX_PERIOD, n // 2) + 1):
        matches = sum(1 for i in range(p, n) if tokens[i] == tokens[i - p])
        cov = matches / max(1, n - p)
        if cov > best_cov:
            best_p, best_cov = p, cov
    return best_p, best_cov


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def rouge_l_f1(pred: str, ref: str) -> float:
    a, b = _words(pred), _words(ref)
    if not a or not b:
        return 0.0
    # LCS length via DP
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    lcs = dp[len(a)][len(b)]
    prec, rec = lcs / len(a), lcs / len(b)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def wordset_f1(pred: str, ref: str) -> float:
    a, b = set(_words(pred)), set(_words(ref))
    if not a or not b:
        return 0.0
    inter = len(a & b)
    prec, rec = inter / len(a), inter / len(b)
    return 0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec)


def score_generation(token_ids: list[int]) -> dict:
    n = len(token_ids)
    d2, d3 = distinct_n(token_ids, 2), distinct_n(token_ids, 3)
    uniq_ratio = len(set(token_ids)) / max(1, n)
    max_run = max_consecutive_repeat(token_ids)
    period, coverage = dominant_cycle(token_ids)

    reasons = []
    if n < MIN_TOKENS:
        reasons.append(f"too short ({n} tokens)")
    if max_run >= MAX_CONSECUTIVE_REPEAT:
        reasons.append(f"{max_run} identical tokens in a row")
    if d2 < MIN_DISTINCT_2:
        reasons.append(f"distinct-2={d2:.2f}<{MIN_DISTINCT_2}")
    if d3 < MIN_DISTINCT_3:
        reasons.append(f"distinct-3={d3:.2f}<{MIN_DISTINCT_3}")
    if uniq_ratio < MIN_UNIQUE_TOKEN_RATIO:
        reasons.append(f"unique-token-ratio={uniq_ratio:.2f}<{MIN_UNIQUE_TOKEN_RATIO}")
    if period and coverage >= CYCLE_COVERAGE and n >= 2 * CYCLE_MAX_PERIOD:
        reasons.append(f"loops with period {period} (coverage {coverage:.2f})")

    return {
        "n_tokens": n,
        "max_consecutive_repeat": max_run,
        "distinct_1": round(len(set(token_ids)) / max(1, n), 3),
        "distinct_2": round(d2, 3),
        "distinct_3": round(d3, 3),
        "cycle_period": period,
        "cycle_coverage": round(coverage, 3),
        "degenerate": bool(reasons),
        "reasons": reasons,
    }


def resolve_model_path(args) -> str:
    if args.model_path:
        return args.model_path
    if not (args.checkpoint_dir and args.base_model):
        raise SystemExit("provide --model_path OR (--checkpoint_dir AND --base_model)")

    # already-converted snapshots live under <checkpoint_dir>/models/step-*
    snaps = sorted(glob.glob(os.path.join(args.checkpoint_dir, "models", "step-*")),
                   key=lambda p: int(re.search(r"(\d+)$", p).group(1)))
    ckpts = sorted(glob.glob(os.path.join(args.checkpoint_dir, "checkpoint-step-*")),
                   key=lambda p: int(re.search(r"(\d+)$", p).group(1)))
    latest_ckpt_step = int(re.search(r"(\d+)$", ckpts[-1]).group(1)) if ckpts else None
    latest_snap_step = int(re.search(r"(\d+)$", snaps[-1]).group(1)) if snaps else None

    if latest_snap_step is not None and latest_snap_step == latest_ckpt_step:
        print(f"[convert] latest snapshot step-{latest_snap_step} already present, using it")
        return snaps[-1]

    if latest_ckpt_step is None:
        raise SystemExit(f"no checkpoint-step-* found in {args.checkpoint_dir}")

    print(f"[convert] converting latest DCP checkpoint step-{latest_ckpt_step} -> HF")
    from utils.convertion_script import convert_nested_dcp_batch
    convert_nested_dcp_batch(args.base_model, args.checkpoint_dir)
    out = os.path.join(args.checkpoint_dir, "models", f"step-{latest_ckpt_step}")
    if not os.path.isdir(out):
        raise SystemExit(f"conversion did not produce {out}")
    return out


@torch.no_grad()
def generate(model, processor, device, content, image, max_new_tokens):
    """Greedy-decode one turn. `content` is the user message content list;
    `image` is a PIL image or None (text-only). Returns (response_text, gen_ids)."""
    messages = [{"role": "user", "content": content}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    kwargs = {"text": [text], "return_tensors": "pt"}
    if image is not None:
        kwargs["images"] = [image]
    inputs = processor(**kwargs).to(device)
    prompt_len = inputs.input_ids.shape[1]
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = gen[0][prompt_len:].tolist()
    response = processor.batch_decode(
        [gen[0][prompt_len:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0].strip()
    return response, gen_ids


@torch.no_grad()
def generate_raw(model, processor, device, prompt_text, max_new_tokens):
    """Greedy-decode a completion: feed `prompt_text` VERBATIM (no chat template).
    This is how a base LM is meant to be prompted. Returns (continuation, gen_ids)."""
    inputs = processor(text=[prompt_text], return_tensors="pt").to(device)
    prompt_len = inputs.input_ids.shape[1]
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    gen_ids = gen[0][prompt_len:].tolist()
    continuation = processor.batch_decode(
        [gen[0][prompt_len:]], skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]
    return continuation, gen_ids


@torch.no_grad()
def run(args) -> int:
    model_path = resolve_model_path(args)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Loading model from {model_path} on {device} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, trust_remote_code=True,
    ).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    images = sorted(
        p for p in glob.glob(os.path.join(args.images_dir, "*"))
        if p.lower().endswith(IMAGE_EXTS)
    )
    if not images and not args.no_images:
        raise SystemExit(f"no images found in {args.images_dir}")

    results, n_degenerate = [], 0

    if not args.no_images:
        print("\n" + "#" * 60 + "\n# IMAGE CAPTIONING (image-only prompt, as in training)\n" + "#" * 60)
    for img_path in ([] if args.no_images else images):
        name = os.path.splitext(os.path.basename(img_path))[0]
        img = Image.open(img_path).convert("RGB")

        # SAME prompt shape as llava_recap training: image-only user turn.
        response, gen_ids = generate(
            model, processor, device, [{"type": "image"}], img, args.max_new_tokens
        )
        metrics = score_generation(gen_ids)

        ref_path = os.path.join(args.images_dir, f"{name}.txt")
        ref_scores = None
        if os.path.exists(ref_path):
            with open(ref_path) as f:
                ref = f.read().strip()
            ref_scores = {
                "rouge_l_f1": round(rouge_l_f1(response, ref), 3),
                "wordset_f1": round(wordset_f1(response, ref), 3),
            }

        if metrics["degenerate"]:
            n_degenerate += 1

        results.append({
            "kind": "image", "prompt": name, "response": response,
            "metrics": metrics, "reference": ref_scores,
        })

        status = "DEGENERATE" if metrics["degenerate"] else "ok"
        print(f"\n[image: {name}] {status}")
        print(f"  metrics: n={metrics['n_tokens']} max_run={metrics['max_consecutive_repeat']} "
              f"d1={metrics['distinct_1']} d2={metrics['distinct_2']} d3={metrics['distinct_3']} "
              f"cycle(p={metrics['cycle_period']},cov={metrics['cycle_coverage']})")
        if metrics["reasons"]:
            print(f"  reasons: {'; '.join(metrics['reasons'])}")
        if ref_scores:
            print(f"  vs reference: rougeL={ref_scores['rouge_l_f1']} wordsetF1={ref_scores['wordset_f1']}")
        print(f"  RESPONSE:\n{response}\n")

    if args.text_prompts:
        print("\n" + "#" * 60 + "\n# TEXT-ONLY QUESTIONS (chat template, no image)\n" + "#" * 60)
    for i, q in enumerate(args.text_prompts):
        response, gen_ids = generate(
            model, processor, device, [{"type": "text", "text": q}], None, args.max_new_tokens
        )
        metrics = score_generation(gen_ids)
        if metrics["degenerate"]:
            n_degenerate += 1
        results.append({
            "kind": "text", "prompt": q, "response": response,
            "metrics": metrics, "reference": None,
        })
        status = "DEGENERATE" if metrics["degenerate"] else "ok"
        print(f"\n[text {i}] {status}  Q: {q}")
        print(f"  metrics: n={metrics['n_tokens']} max_run={metrics['max_consecutive_repeat']} "
              f"d1={metrics['distinct_1']} d2={metrics['distinct_2']} d3={metrics['distinct_3']} "
              f"cycle(p={metrics['cycle_period']},cov={metrics['cycle_coverage']})")
        if metrics["reasons"]:
            print(f"  reasons: {'; '.join(metrics['reasons'])}")
        print(f"  RESPONSE:\n{response}\n")

    # Completion / few-shot probes: raw prompts, no chat template. Informational
    # only (greedy repetition is expected), so they do NOT gate the verdict.
    completion_degenerate = 0
    if args.completion_prompts:
        print("\n" + "#" * 60 +
              "\n# COMPLETION / FEW-SHOT PROMPTS (raw, no chat template — informational)\n" +
              "#" * 60)
    for i, q in enumerate(args.completion_prompts):
        continuation, gen_ids = generate_raw(model, processor, device, q, args.max_new_tokens)
        metrics = score_generation(gen_ids)
        if metrics["degenerate"]:
            completion_degenerate += 1
        results.append({
            "kind": "completion", "prompt": q, "response": continuation,
            "metrics": metrics, "reference": None, "gates_verdict": False,
        })
        status = "loops(greedy)" if metrics["degenerate"] else "ok"
        print(f"\n[completion {i}] {status}  PROMPT: {q!r}")
        print(f"  metrics: n={metrics['n_tokens']} max_run={metrics['max_consecutive_repeat']} "
              f"d1={metrics['distinct_1']} d2={metrics['distinct_2']} d3={metrics['distinct_3']} "
              f"cycle(p={metrics['cycle_period']},cov={metrics['cycle_coverage']})")
        print(f"  CONTINUATION:\n{continuation}\n")

    # Verdict is gated ONLY by image + chat-text checks (n_degenerate).
    n_gating = sum(1 for r in results if r.get("gates_verdict", True))
    passed = n_gating - n_degenerate
    verdict = "PASS" if n_degenerate == 0 else "FAIL"
    print("\n" + "=" * 60)
    print(f"TEST SUITE {verdict}: {passed}/{n_gating} gating checks healthy, {n_degenerate} degenerate")
    if args.completion_prompts:
        print(f"(completion probes: {len(args.completion_prompts) - completion_degenerate}/"
              f"{len(args.completion_prompts)} coherent, {completion_degenerate} greedy-looped — informational)")
    refd = [r for r in results if r["reference"]]
    if refd:
        avg_rouge = sum(r["reference"]["rouge_l_f1"] for r in refd) / len(refd)
        print(f"mean ROUGE-L over {len(refd)} referenced images: {avg_rouge:.3f}")
    print("=" * 60)

    if args.output:
        with open(args.output, "w") as f:
            json.dump({"verdict": verdict, "n_degenerate": n_degenerate,
                       "completion_degenerate": completion_degenerate, "results": results}, f, indent=2)
        print(f"report written to {args.output}")

    return 1 if n_degenerate > 0 else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", default=None, help="HF snapshot dir (e.g. .../models/step-500)")
    ap.add_argument("--checkpoint_dir", default=None, help="trainer output_dir with checkpoint-step-*")
    ap.add_argument("--base_model", default=None, help="HF base model dir (for DCP->HF conversion)")
    ap.add_argument("--images_dir", default="test_images")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--text_prompt", action="append", dest="text_prompts", default=None,
                    help="chat-template text question (repeatable); overrides the defaults")
    ap.add_argument("--completion_prompt", action="append", dest="completion_prompts", default=None,
                    help="raw completion/few-shot prompt, no chat template (repeatable); overrides defaults")
    ap.add_argument("--no_text", action="store_true", help="skip the chat-template text questions")
    ap.add_argument("--no_completion", action="store_true", help="skip the raw completion prompts")
    ap.add_argument("--no_images", action="store_true", help="skip the image captioning")
    ap.add_argument("--output", default=None, help="optional JSON report path")
    args = ap.parse_args()
    if args.no_text:
        args.text_prompts = []
    elif args.text_prompts is None:
        args.text_prompts = DEFAULT_TEXT_PROMPTS
    if args.no_completion:
        args.completion_prompts = []
    elif args.completion_prompts is None:
        args.completion_prompts = DEFAULT_COMPLETION_PROMPTS
    sys.exit(run(args))
