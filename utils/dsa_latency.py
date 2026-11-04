"""Latency of DSA vs dense attention on the Qwen3.5-2B shapes, and what it buys the model.

Three measurements, forward only, batch 1 unless `--decode-batch`:

  prefill  one full-attention layer over a T-token document:
             dense  `varlen_attn` (the flash kernel production trains with)
             dsa    `dsa_min_memory_gqa_forward_only` (indexer scores, top-K,
                    sparse attention; the Triton kernels stage 2 trained with)
  decode   one new token against a T-token cache, one layer:
             dense  SDPA over the whole cache
             dsa    indexer scores over the cached index keys, top-K, SDPA
                    over the K gathered keys (plain torch ops: there is no
                    fused DSA decode kernel here, so this is an upper bound)
  model    `--model-dir`: the HF model's whole prefill and per-token decode
           at T (SDPA in its 6 full-attention layers), and the DSA model
           projected from it: total - 6 * (dense layer - dsa layer). The
           18 linear-attention layers, MLPs and lm_head are unchanged by DSA.

Weights are random: kernel time does not depend on which keys get picked.
Times are CUDA-event medians in ms. Needs an idle GPU -- a shared one makes
every number here meaningless.

    CUDA_VISIBLE_DEVICES=2 python utils/dsa_latency.py --model-dir HF_DIR --out lat.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def timed(fn, warmup: int = 3, iters: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        fn()
        b.record()
        torch.cuda.synchronize()
        times.append(a.elapsed_time(b))
    return sorted(times)[len(times) // 2]


def make_indexer(dim: int, topk: int, head_dim: int):
    from models.common.dsa.inner_attention import DSAIndexer

    m = DSAIndexer.Config(dim=dim, index_topk=topk, index_head_dim=head_dim).build().cuda()
    m._init_self_buffers(buffer_device=torch.device("cuda"))
    for w in (m.linear_q.weight, m.linear_k.weight):
        torch.nn.init.normal_(w, std=dim**-0.5)
    return m.to(torch.bfloat16)


def prefill(T: int, H: int, Hkv: int, D: int, dim: int, indexer) -> dict:
    from models.common.attention import varlen_attn
    from models.common.dsa.dsa_min_memory import dsa_min_memory_gqa_forward_only

    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(T, H, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(T, Hkv, D, device="cuda", dtype=torch.bfloat16, generator=g)
    h = torch.randn(T, dim, device="cuda", dtype=torch.bfloat16, generator=g)
    cu = torch.tensor([0, T], device="cuda", dtype=torch.int32)
    scale = D**-0.5

    def dense():
        varlen_attn(q, k, v, cu, cu, T, T, scale=scale, window_size=(-1, 0), enable_gqa=True)

    def dsa():
        dsa_min_memory_gqa_forward_only(
            q[:, None], k[:, None], v[:, None], h[:, None], indexer, softmax_scale=scale, use_indexer_rope=True,
        )

    with torch.no_grad():
        return {"dense_ms": timed(dense), "dsa_ms": timed(dsa, warmup=2, iters=5)}


def prefill_sdpa(T: int, H: int, Hkv: int, D: int) -> dict:
    """The same dense layer through each PyTorch SDPA backend; None = refused or OOM.
    On sm_120 with GQA, `varlen_attn` itself runs aten's flash kernel (FA2), so
    `FLASH_ATTENTION` here should match `dense_ms`."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(1, H, T, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(1, Hkv, T, D, device="cuda", dtype=torch.bfloat16)
    out = {}
    for b in (SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH):
        try:
            with sdpa_kernel([b]), torch.no_grad():
                out[b.name] = timed(lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True), warmup=1, iters=3)
        except (RuntimeError, torch.OutOfMemoryError) as e:
            out[b.name] = None
            print(f"  T={T} {b.name}: {str(e).splitlines()[0][:120]}", flush=True)
        torch.cuda.empty_cache()
    return out


def decode(T: int, B: int, H: int, Hkv: int, D: int, dim: int, topk: int, idx_dim: int) -> dict:
    g = torch.Generator(device="cuda").manual_seed(0)
    q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.bfloat16, generator=g)
    k = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16, generator=g)
    v = torch.randn(B, Hkv, T, D, device="cuda", dtype=torch.bfloat16, generator=g)
    h = torch.randn(B, dim, device="cuda", dtype=torch.bfloat16, generator=g)
    wq = torch.randn(idx_dim, dim, device="cuda", dtype=torch.bfloat16, generator=g) * dim**-0.5
    ki = torch.randn(B, T, idx_dim, device="cuda", dtype=torch.bfloat16, generator=g)
    K = min(topk, T)

    def dense():
        F.scaled_dot_product_attention(q, k, v, enable_gqa=True)

    def dsa():
        qi = h @ wq.T  # [B, idx]; the new key's projection is the same size, folded in here
        top = torch.bmm(ki, qi[:, :, None])[:, :, 0].topk(K, dim=-1).indices  # [B, K]
        sel = top[:, None, :, None].expand(B, Hkv, K, D)
        F.scaled_dot_product_attention(q, k.gather(2, sel), v.gather(2, sel), enable_gqa=True)

    with torch.no_grad():
        return {"dense_ms": timed(dense, iters=50), "dsa_ms": timed(dsa, iters=50)}


def model_times(model_dir: str, T: int, steps: int = 16) -> dict:
    """Whole HF model: prefill of T tokens, then ms per decoded token at ~T."""
    from transformers import AutoModelForImageTextToText

    global _model
    if "_model" not in globals():
        _model = AutoModelForImageTextToText.from_pretrained(model_dir, dtype=torch.bfloat16).to("cuda").eval()
    ids = torch.randint(1000, 100000, (1, T), device="cuda")
    with torch.no_grad():
        prefill_ms = timed(lambda: _model(input_ids=ids, logits_to_keep=1), warmup=1, iters=3)
        out = _model(input_ids=ids, use_cache=True, logits_to_keep=1)
        cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        for _ in range(3):  # warm the decode path
            out = _model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(steps):
            out = _model(input_ids=nxt, past_key_values=cache, use_cache=True)
            cache, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
        b.record()
        torch.cuda.synchronize()
    del cache, out
    torch.cuda.empty_cache()
    return {"prefill_ms": prefill_ms, "decode_ms_per_token": a.elapsed_time(b) / steps}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lengths", default="8192,16384,32768,65536,131072")
    ap.add_argument("--topk", type=int, default=2048)
    ap.add_argument("--decode-batch", type=int, default=1)
    ap.add_argument("--model-dir", help="HF snapshot: adds whole-model timings and the projection")
    ap.add_argument("--sdpa", action="store_true", help="also time the dense layer per SDPA backend")
    ap.add_argument("--skip-dsa", action="store_true", help="with --sdpa: only the dense backends")
    ap.add_argument("--out")
    args = ap.parse_args()

    # Qwen3.5-2B full-attention layers; 6 of its 24 layers
    H, Hkv, D, dim, idx_dim, n_full = 8, 2, 256, 2048, 128, 6
    indexer = make_indexer(dim, args.topk, idx_dim)
    rows = []
    for T in [int(x) for x in args.lengths.split(",")]:
        r = {"T": T, "topk": args.topk}
        if args.sdpa:
            r["prefill_sdpa"] = prefill_sdpa(T, H, Hkv, D)
            print(f"T={T:>6}  sdpa prefill/layer " + "  ".join(
                f"{k} {v:.2f}" if v is not None else f"{k} -" for k, v in r["prefill_sdpa"].items()) + " ms", flush=True)
            if args.skip_dsa:
                rows.append(r)
                continue
        r["prefill"] = prefill(T, H, Hkv, D, dim, indexer)
        r["decode"] = decode(T, args.decode_batch, H, Hkv, D, dim, args.topk, idx_dim)
        line = (
            f"T={T:>6}  prefill/layer dense {r['prefill']['dense_ms']:8.2f}  dsa {r['prefill']['dsa_ms']:8.2f} ms"
            f" | decode/layer dense {r['decode']['dense_ms']:.3f}  dsa {r['decode']['dsa_ms']:.3f} ms"
        )
        if args.model_dir:
            m = model_times(args.model_dir, T)
            proj_p = m["prefill_ms"] - n_full * (r["prefill"]["dense_ms"] - r["prefill"]["dsa_ms"])
            proj_d = m["decode_ms_per_token"] - n_full * (r["decode"]["dense_ms"] - r["decode"]["dsa_ms"])
            r["model"] = {**m, "dsa_prefill_ms_projected": proj_p, "dsa_decode_ms_per_token_projected": proj_d}
            line += (
                f" | model prefill {m['prefill_ms']:.0f} -> dsa {proj_p:.0f} ms"
                f", decode {m['decode_ms_per_token']:.1f} -> dsa {proj_d:.1f} ms/tok"
            )
        print(line, flush=True)
        rows.append(r)
        if args.out:
            with open(args.out, "w") as f:
                json.dump({"gpu": torch.cuda.get_device_name(), "rows": rows}, f, indent=2)


if __name__ == "__main__":
    main()
