"""Per-module microbenchmark for Qwen3.5-9B.

Decomposes `perf/fwd_bwd_time` without a profiler: time one linear-attention
DecoderLayer, one full-attention DecoderLayer and the raw gated-delta-rule
kernel at the real shapes, then multiply by the layer counts.

The question it exists to answer: TP shards GDN heads, so at tp=4 each rank runs
the delta rule with `linear_num_value_heads / 4` heads. If the kernel is
occupancy-bound on that dimension, per-layer time will barely fall from tp=1 to
tp=4 -- the work per rank drops 4x while the time does not. If it is
compute-bound it will fall close to 4x.

Run under srun with one GPU:
  PYTHONPATH=. python models/tests/bench_layers.py --model-dir <dir>
"""
import argparse
import dataclasses
import json

import torch

from models.qwen3_5.config import Qwen3_5Config
from models.qwen3_5.model import DecoderLayer
import models.qwen3_5.compile_ops as _ops

DEV = "cuda"
DT = torch.bfloat16


def timeit(fn, warmup=5, iters=20):
    """Median ms over `iters`, CUDA events, median so a stray page fault does
    not set the number."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        s, e = torch.cuda.Event(True), torch.cuda.Event(True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def local_cfg(text_cfg, tp):
    """The config a single rank sees under tensor parallelism: heads and the MLP
    intermediate divided, hidden_size unchanged (activations are replicated)."""
    return dataclasses.replace(
        text_cfg,
        num_attention_heads=text_cfg.num_attention_heads // tp,
        num_key_value_heads=text_cfg.num_key_value_heads // tp,
        intermediate_size=text_cfg.intermediate_size // tp,
        linear_num_key_heads=text_cfg.linear_num_key_heads // tp,
        linear_num_value_heads=text_cfg.linear_num_value_heads // tp,
    )


def make_inputs(cfg, L, ndocs):
    x = torch.randn(1, L, cfg.hidden_size, device=DEV, dtype=DT, requires_grad=True)
    bounds = [round(i * L / ndocs) for i in range(ndocs + 1)]
    cu = torch.tensor(bounds, device=DEV, dtype=torch.int32)
    rot = getattr(cfg, "partial_rotary_factor", 1.0) or 1.0
    R = int(cfg.head_dim * rot)
    cos = torch.randn(1, L, R, device=DEV, dtype=torch.float32)
    sin = torch.randn(1, L, R, device=DEV, dtype=torch.float32)
    max_seqlen = max(b - a for a, b in zip(bounds, bounds[1:]))
    return x, cos, sin, cu, max_seqlen


def bench_layer(layer, x, cos, sin, cu, max_seqlen):
    grad = torch.randn_like(x)

    def fwd():
        with torch.autocast("cuda", DT):
            layer(x, cos, sin, cu, max_seqlen)

    def fwd_bwd():
        layer.zero_grad(set_to_none=True)
        if x.grad is not None:
            x.grad = None
        with torch.autocast("cuda", DT):
            out = layer(x, cos, sin, cu, max_seqlen)
        out.backward(grad)

    return timeit(fwd), timeit(fwd_bwd)


def bench_raw_gdr(cfg, tp, L, cu, chunk_sizes):
    """The delta-rule kernel alone, at the shapes the layer feeds it. q and k
    arrive already expanded to the value-head count by `repeat_interleave`."""
    hv = cfg.linear_num_value_heads // tp
    dk, dv = cfg.linear_key_head_dim, cfg.linear_value_head_dim
    q = torch.randn(1, L, hv, dk, device=DEV, dtype=DT, requires_grad=True)
    k = torch.randn(1, L, hv, dk, device=DEV, dtype=DT, requires_grad=True)
    v = torch.randn(1, L, hv, dv, device=DEV, dtype=DT, requires_grad=True)
    g = torch.randn(1, L, hv, device=DEV, dtype=torch.float32, requires_grad=True)
    beta = torch.rand(1, L, hv, device=DEV, dtype=DT, requires_grad=True)
    cu64 = cu.to(torch.int64)

    out = {}
    for cs in chunk_sizes:
        kw = {} if cs is None else {"chunk_size": cs}
        try:
            o, _ = _ops._fla_chunk_gated_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu64, **kw
            )
        except TypeError as exc:
            out[cs] = f"unsupported ({exc.__class__.__name__})"
            continue
        go = torch.randn_like(o)

        def fwd():
            _ops._fla_chunk_gated_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu64, **kw
            )

        def fwd_bwd():
            for t in (q, k, v, g, beta):
                t.grad = None
            o2, _ = _ops._fla_chunk_gated_delta_rule(
                q, k, v, g, beta, use_qk_l2norm_in_kernel=True, cu_seqlens=cu64, **kw
            )
            o2.backward(go)

        out[cs] = (timeit(fwd), timeit(fwd_bwd))
    return out


def bench_head_loss(cfg, tp, L):
    """`lm_head` and `causal_lm_loss`, the largest single tensors in the step.

    Under this TP plan `lm_head` is ColwiseParallel with `output_layouts=
    Replicate()`, so each rank computes `vocab/tp` columns and the full vocab is
    then all-gathered. The gather is a collective and is NOT measured here --
    only the local matmul, and the loss on the replicated full-vocab logits.
    """
    from models.qwen3_5.utils import causal_lm_loss

    h = torch.randn(1, L, cfg.hidden_size, device=DEV, dtype=DT, requires_grad=True)
    head = torch.nn.Linear(cfg.hidden_size, cfg.vocab_size // tp, bias=False).to(DEV, DT)
    gh = torch.randn(1, L, cfg.vocab_size // tp, device=DEV, dtype=DT)

    def head_fwd():
        with torch.autocast("cuda", DT):
            head(h)

    def head_fwd_bwd():
        head.zero_grad(set_to_none=True)
        h.grad = None
        with torch.autocast("cuda", DT):
            out = head(h)
        out.backward(gh)

    hf, hfb = timeit(head_fwd), timeit(head_fwd_bwd)
    del gh
    torch.cuda.empty_cache()

    # the loss sees the full vocabulary on every rank, whatever tp is
    logits = torch.randn(1, L, cfg.vocab_size, device=DEV, dtype=DT, requires_grad=True)
    labels = torch.randint(0, cfg.vocab_size, (1, L), device=DEV, dtype=torch.int64)
    labels[:, ::3] = -100

    def loss_fwd():
        causal_lm_loss(logits, labels)

    def loss_fwd_bwd():
        logits.grad = None
        causal_lm_loss(logits, labels).backward()

    lf, lfb = timeit(loss_fwd, warmup=3, iters=10), timeit(loss_fwd_bwd, warmup=3, iters=10)
    peak = torch.cuda.max_memory_allocated() / 1024 ** 3
    del logits, labels, h, head
    torch.cuda.empty_cache()
    return hf, hfb, lf, lfb, peak


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/e/project1/reformo/ockier1/qwen_models/qwen3_5_9b")
    ap.add_argument("--seq-len", type=int, default=6144)
    ap.add_argument("--ndocs", type=int, default=4)
    ap.add_argument("--tp", type=int, nargs="+", default=[1, 2, 4])
    args = ap.parse_args()

    cfg = Qwen3_5Config.from_json(f"{args.model_dir}/config.json").text
    L = args.seq_len
    n_layers = cfg.num_hidden_layers
    n_full = n_layers // cfg.full_attention_interval
    n_lin = n_layers - n_full

    print(f"model: {n_layers} layers = {n_lin} linear + {n_full} full, "
          f"hidden {cfg.hidden_size}, seq_len {L}, {args.ndocs} packed docs")
    print(f"GDN heads: {cfg.linear_num_key_heads} key / {cfg.linear_num_value_heads} value, "
          f"attn heads: {cfg.num_attention_heads} q / {cfg.num_key_value_heads} kv")
    print(f"device: {torch.cuda.get_device_name(0)}, "
          f"{torch.cuda.get_device_properties(0).multi_processor_count} SMs\n")

    rows = []
    for tp in args.tp:
        lcfg = local_cfg(cfg, tp)
        x, cos, sin, cu, max_seqlen = make_inputs(lcfg, L, args.ndocs)
        for kind, count in (("linear_attention", n_lin), ("full_attention", n_full)):
            layer = DecoderLayer(lcfg, kind).to(DEV, DT)
            f, fb = bench_layer(layer, x, cos, sin, cu, max_seqlen)
            rows.append((tp, kind, count, f, fb))
            del layer
            torch.cuda.empty_cache()

    print(f"{'tp':>3} {'layer':<18}{'n':>4}{'fwd ms':>9}{'fwd+bwd':>9}"
          f"{'bwd':>8}{'x n (ms)':>11}")
    for tp, kind, count, f, fb in rows:
        print(f"{tp:>3} {kind:<18}{count:>4}{f:>9.2f}{fb:>9.2f}{fb-f:>8.2f}{fb*count:>11.1f}")

    print()
    for tp in args.tp:
        tot = sum(fb * c for t, _, c, _, fb in rows if t == tp)
        print(f"tp={tp}: all layers fwd+bwd = {tot/1000:.3f} s "
              f"(measured step fwd_bwd_time at tp=4 was 1.154 s)")

    print("\nlm_head (vocab/tp columns) and loss (full vocab, replicated)")
    print(f"{'tp':>3}{'head fwd':>10}{'head f+b':>10}{'loss fwd':>10}"
          f"{'loss f+b':>10}{'head+loss':>11}{'peak GiB':>10}")
    head_loss = {}
    for tp in args.tp:
        torch.cuda.reset_peak_memory_stats()
        hf, hfb, lf, lfb, peak = bench_head_loss(cfg, tp, L)
        head_loss[tp] = hfb + lfb
        print(f"{tp:>3}{hf:>10.2f}{hfb:>10.2f}{lf:>10.2f}{lfb:>10.2f}"
              f"{hfb+lfb:>11.2f}{peak:>10.1f}")

    print()
    for tp in args.tp:
        layers = sum(fb * c for t, _, c, _, fb in rows if t == tp)
        tot = layers + head_loss[tp]
        print(f"tp={tp}: layers {layers/1000:.3f}s + head/loss {head_loss[tp]/1000:.3f}s "
              f"= {tot/1000:.3f}s   (measured fwd_bwd_time at tp=4: 1.154 s)")

    print("\nraw gated_delta_rule kernel, chunk_size sweep")
    print(f"{'tp':>3}{'heads':>7}{'chunk':>8}{'fwd ms':>9}{'fwd+bwd':>9}{'x n_lin':>10}")
    for tp in args.tp:
        _, _, _, cu, _ = make_inputs(local_cfg(cfg, tp), L, args.ndocs)
        res = bench_raw_gdr(cfg, tp, L, cu, [None, 64, 128, 256])
        hv = cfg.linear_num_value_heads // tp
        for cs, val in res.items():
            label = "default" if cs is None else str(cs)
            if isinstance(val, str):
                print(f"{tp:>3}{hv:>7}{label:>8}{val:>19}")
            else:
                f, fb = val
                print(f"{tp:>3}{hv:>7}{label:>8}{f:>9.2f}{fb:>9.2f}{fb*n_lin:>10.1f}")


if __name__ == "__main__":
    main()
