"""Per-layer cost of Qwen4-Exp against Qwen3.5, at their released configs.

The two architectures are not the same shape -- Qwen4 carries `hc_count`
residual streams, routes through 512 experts and runs sparse attention, while
Qwen3.5 is a plain hidden state with a dense MLP -- so a like-for-like layer
comparison would be meaningless. What is comparable is what you would actually
pay to train each: one decoder layer of each type at the config the checkpoint
ships, scaled by how many of those layers the model has.

Weights are random (only `config.json` is read), which is fine: latency and
memory do not depend on the values.

Usage:
    python -m models.tests.bench_qwen4_vs_qwen3_5
    python -m models.tests.bench_qwen4_vs_qwen3_5 --seq-lens 2048,4096 --iters 20
    python -m models.tests.bench_qwen4_vs_qwen3_5 --no-backward
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import torch

from models.qwen3_5.config import Qwen3_5Config
from models.qwen3_5.model import DecoderLayer as Qwen35Layer
from models.qwen4.config import Qwen4Config
from models.qwen4.model import DecoderLayer as Qwen4Layer
from models.qwen4 import model as qwen4_model

QWEN4_SNAPSHOT = Path(os.environ.get("QWEN4_SNAPSHOT", "/data/151-2/users/tockier/models/qwen4"))
QWEN35_SNAPSHOT = Path(
    os.environ.get("QWEN3_5_SNAPSHOT", "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_27b")
)


def _cuda_time(fn, iters: int, warmup: int) -> float:
    """Median per-iteration latency in ms.

    Timed per iteration rather than as a block, and reported as a median: these
    GPUs are shared, and Triton autotunes on the first call at each new shape,
    so a mean over a single timed block picks up whatever else was running.
    """
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iters):
        start, end = torch.cuda.Event(True), torch.cuda.Event(True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def _build(cls, *args, device, dtype):
    with torch.device("meta"):
        layer = cls(*args)
    layer = layer.to_empty(device=device).to(dtype)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(0.0, 0.02)
        for b in layer.buffers():
            if b.is_floating_point():
                b.normal_(0.0, 0.02)
    return layer


NA = float("nan")


def _measure(layer, make_inputs, iters, warmup, backward):
    device = next(layer.parameters()).device
    params = sum(p.numel() for p in layer.parameters())
    out = dict(fwd_ms=NA, bwd_ms=NA, fwd_peak_mib=NA, step_peak_mib=NA,
               params=params, note="")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base = torch.cuda.memory_allocated(device)

    try:
        args, kwargs = make_inputs()
        with torch.no_grad():
            out["fwd_ms"] = _cuda_time(lambda: layer(*args, **kwargs), iters, warmup)
        out["fwd_peak_mib"] = (torch.cuda.max_memory_allocated(device) - base) / 2**20
        del args, kwargs
    except torch.OutOfMemoryError:
        out["note"] = "OOM (fwd)"
        torch.cuda.empty_cache()
        return out

    if backward:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

        def step():
            layer.zero_grad(set_to_none=True)
            a, kw = make_inputs()
            layer(*a, **kw).square().mean().backward()

        try:
            step_ms = _cuda_time(step, iters, warmup)
            out["step_peak_mib"] = (torch.cuda.max_memory_allocated(device) - base) / 2**20
            out["bwd_ms"] = step_ms - out["fwd_ms"]
        except torch.OutOfMemoryError:
            out["note"] = "OOM (bwd)"
        finally:
            layer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()

    return out


def bench_qwen35(cfg, seq_len, device, dtype, iters, warmup, backward):
    text = cfg.text
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    rope_dim = int(text.head_dim * text.rope_parameters.get("partial_rotary_factor", 1.0))
    cos = torch.randn(1, seq_len, rope_dim, device=device, dtype=dtype)
    sin = torch.randn_like(cos)

    out = {}
    for layer_type in ("linear_attention", "full_attention"):
        layer = _build(Qwen35Layer, text, layer_type, device=device, dtype=dtype)

        def make():
            x = torch.randn(1, seq_len, text.hidden_size, device=device, dtype=dtype)
            return (x, cos, sin, cu, seq_len), {}

        out[layer_type] = _measure(layer, make, iters, warmup, backward)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    return out


def bench_qwen4(cfg, seq_len, device, dtype, iters, warmup, backward):
    text = cfg.text
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=device)
    seg = torch.zeros(seq_len, dtype=torch.long, device=device)
    rope_dim = int(text.head_dim * text.rope_parameters.get("partial_rotary_factor", 1.0))
    cos = torch.randn(1, seq_len, rope_dim, device=device, dtype=dtype)
    sin = torch.randn_like(cos)
    width = text.hc_count * text.hidden_size

    out = {}
    for layer_type in ("linear_attention", "qwen_sparse_attention"):
        idx = text.layer_types.index(layer_type)
        layer = _build(Qwen4Layer, text, idx, None, device=device, dtype=dtype)

        def make():
            x = torch.randn(1, seq_len, width, device=device, dtype=dtype)
            return (x, cos, sin, cu, seq_len, seg, None), {}

        out[layer_type] = _measure(layer, make, iters, warmup, backward)
        del layer
        gc.collect()
        torch.cuda.empty_cache()
    return out


def _counts(layer_types):
    return {t: layer_types.count(t) for t in set(layer_types)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq-lens", default="2048,4096,8192")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--no-backward", action="store_true")
    ap.add_argument("--qwen4", default=str(QWEN4_SNAPSHOT))
    ap.add_argument("--qwen35", default=str(QWEN35_SNAPSHOT))
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this benchmark needs a GPU"
    device = torch.device("cuda")
    dtype = torch.bfloat16
    backward = not args.no_backward

    cfg4 = Qwen4Config.from_json(Path(args.qwen4) / "config.json")
    cfg35 = Qwen3_5Config.from_json(Path(args.qwen35) / "config.json")

    # Qwen3.5's model recomputes its schedule from the interval rather than
    # reading `layer_types`, so mirror that here.
    q35_types = [
        "full_attention" if (i + 1) % cfg35.text.full_attention_interval == 0
        else "linear_attention"
        for i in range(cfg35.text.num_hidden_layers)
    ]

    print(f"device      : {torch.cuda.get_device_name(0)}  ({dtype})")
    print(f"Qwen4       : {args.qwen4}")
    print(f"              hidden {cfg4.text.hidden_size} x hc {cfg4.text.hc_count}"
          f" | {cfg4.text.num_hidden_layers} layers {_counts(cfg4.text.layer_types)}"
          f" | experts {cfg4.text.num_experts} top-{cfg4.text.num_experts_per_tok}")
    print(f"Qwen3.5     : {args.qwen35}")
    print(f"              hidden {cfg35.text.hidden_size}"
          f" | {cfg35.text.num_hidden_layers} layers {_counts(q35_types)}"
          f" | dense mlp {cfg35.text.intermediate_size}")
    print()

    for seq_len in [int(s) for s in args.seq_lens.split(",")]:
        r35 = bench_qwen35(cfg35, seq_len, device, dtype, args.iters, args.warmup, backward)
        r4 = bench_qwen4(cfg4, seq_len, device, dtype, args.iters, args.warmup, backward)

        print(f"=== seq_len {seq_len} (one packed document) ===")
        head = f"{'layer':<34}{'params':>12}{'fwd ms':>10}{'bwd ms':>10}{'fwd MiB':>10}{'step MiB':>10}"
        print(head)
        print("-" * len(head))
        rows = [
            (f"qwen3.5  linear_attention", r35["linear_attention"]),
            (f"qwen3.5  full_attention", r35["full_attention"]),
            (f"qwen4    linear_attention", r4["linear_attention"]),
            (f"qwen4    qwen_sparse_attention", r4["qwen_sparse_attention"]),
        ]
        for name, m in rows:
            print(f"{name:<34}{m['params']/1e6:>11.1f}M{m['fwd_ms']:>10.2f}"
                  f"{m['bwd_ms']:>10.2f}{m['fwd_peak_mib']:>10.0f}"
                  f"{m['step_peak_mib']:>10.0f}  {m['note']}")

        qsa_note = qwen4_model._FLEX_ATTENTION.get("compile_failed")
        if qsa_note:
            print("\n  ! QSA ran on the EAGER flex-attention path on this GPU, which\n"
                  "    materializes the full score matrix, so its latency and especially\n"
                  "    its memory are an artifact of this card, not the architecture.\n"
                  "    At head_dim=256 the compiled kernel needs ~156 KiB of shared memory;\n"
                  "    Hopper has 228 KiB, SM89/SM120 have 100 KiB.")

        incomplete = [n for n, m in rows if m["note"]]
        if incomplete:
            print(f"\n  (no whole-model figures: {', '.join(incomplete)})\n")
            continue

        # whole-model extrapolation: layer cost x how many of that layer exist
        def total(counts, res, key):
            return sum(counts.get(t, 0) * res[t][key] for t in res)

        c35, c4 = _counts(q35_types), _counts(cfg4.text.layer_types)
        f35, f4 = total(c35, r35, "fwd_ms"), total(c4, r4, "fwd_ms")
        print()
        print(f"{'all decoder layers, forward':<34}"
              f"qwen3.5 {f35:>8.1f} ms   qwen4 {f4:>8.1f} ms   "
              f"({f4 / f35:.2f}x)")
        if backward:
            s35 = f35 + total(c35, r35, "bwd_ms")
            s4 = f4 + total(c4, r4, "bwd_ms")
            print(f"{'all decoder layers, fwd+bwd':<34}"
                  f"qwen3.5 {s35:>8.1f} ms   qwen4 {s4:>8.1f} ms   "
                  f"({s4 / s35:.2f}x)")
            print(f"{'per token, fwd+bwd':<34}"
                  f"qwen3.5 {s35 * 1e3 / seq_len:>8.1f} us   "
                  f"qwen4 {s4 * 1e3 / seq_len:>8.1f} us")
        p35 = sum(c35.get(t, 0) * r35[t]["params"] for t in r35)
        p4 = sum(c4.get(t, 0) * r4[t]["params"] for t in r4)
        print(f"{'decoder params':<34}"
              f"qwen3.5 {p35/1e9:>8.1f}B    qwen4 {p4/1e9:>8.1f}B")
        print()


if __name__ == "__main__":
    main()
