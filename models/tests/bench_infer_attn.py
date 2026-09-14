"""Attention at inference shapes: dense vs QSA, decode and prefill.

The repo has no KV cache, no generate, no decode path -- these are training-only
varlen implementations. So this measures the one thing that differs between the
four benchmark variants, the per-layer attention op, at real inference shapes.
MLP, projections and norms are identical across variants and are left out.

Why decode may go the other way from training: the whole training result is that
a BlockMask is sparse over (query-tile, key-tile) pairs and a 128-query tile must
visit the union of what all 128 queries select -- an oracle indexer still unions
to 53.8% of tiles. Decode has ONE query per step. There is no tile and no union;
per-query sparsity is exactly what the kernel sees.

Decode QSA is therefore not flex_attention at all. `run_flex_attention` requires
`total % 128 == 0` and would fall back to the eager path at Q_LEN=1. The real
decode op is: score pooled keys, top-k, gather the selected KV, attend over them.
That is what is measured here.

Decode is memory-bandwidth-bound, so bytes moved is reported alongside time:
dense reads the whole KV cache every step, QSA reads the pooled-key index plus
`budget` gathered entries.
"""
import sys, time
import torch
from flash_attn import flash_attn_func

H, HKV, D = 16, 4, 256          # Qwen3.5-9B full-attention geometry
IDX_H, IDX_D, RATIO, BUDGET = 4, 128, 4, 512
DEV, DT = "cuda", torch.bfloat16
KEEP = BUDGET // RATIO


def timeit(fn, iters=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def decode(S):
    """One query against a KV cache of S tokens."""
    q = torch.randn(1, 1, H, D, device=DEV, dtype=DT)
    k = torch.randn(1, S, HKV, D, device=DEV, dtype=DT)
    v = torch.randn(1, S, HKV, D, device=DEV, dtype=DT)

    def dense():
        flash_attn_func(q, k, v, causal=False)

    nblk = S // RATIO
    pooled_k = torch.randn(nblk, IDX_D, device=DEV, dtype=DT)   # index cache
    q_idx = torch.randn(IDX_H, IDX_D, device=DEV, dtype=DT)
    off = torch.arange(RATIO, device=DEV)

    def qsa():
        scores = torch.relu(q_idx @ pooled_k.T).sum(0)          # (nblk,)
        blk = scores.topk(min(KEEP, nblk)).indices
        tok = (blk.unsqueeze(1) * RATIO + off).reshape(-1).clamp_(max=S - 1)
        ks = k[:, tok]
        vs = v[:, tok]
        flash_attn_func(q, ks, vs, causal=False)

    d, s = timeit(dense), timeit(qsa)
    el = k.element_size()
    dense_b = 2 * S * HKV * D * el
    qsa_b = nblk * IDX_D * el + 2 * min(BUDGET, S) * HKV * D * el
    print(f"  S={S:7d}  dense {d:7.3f} ms ({dense_b/2**20:8.1f} MiB)   "
          f"qsa {s:7.3f} ms ({qsa_b/2**20:7.2f} MiB)   "
          f"speedup {d/s:6.2f}x   traffic {dense_b/qsa_b:6.1f}x", flush=True)
    return d, s


def prefill(S):
    """All S queries at once -- the training regime, tile union and all."""
    sys.path.insert(0, "/e/project1/open-sci-mm/ockier1/vlm-training")
    from models.qwen3_5.model import _create_block_mask, run_flex_attention
    import torch._functorch.config
    torch._functorch.config.donated_buffer = False

    q = torch.randn(1, S, H, D, device=DEV, dtype=DT)
    k = torch.randn(1, S, HKV, D, device=DEV, dtype=DT)
    v = torch.randn(1, S, HKV, D, device=DEV, dtype=DT)

    def dense():
        flash_attn_func(q, k, v, causal=True)

    # oracle-like selection: 53.8% tile union is what a trained indexer gives
    # (oracle_mask.py, job 1791719). Approximated here by the locality window
    # that reproduced ~45-75% union in bench_union_curve.py.
    pos = torch.arange(S, device=DEV)
    blk_of = pos // RATIO
    span = min(8192, S) // RATIO
    sel_idx = (blk_of.unsqueeze(1) - (torch.rand(S, KEEP, device=DEV) * span).long()).clamp_(min=0)
    sel = torch.zeros(S, S // RATIO, dtype=torch.bool, device=DEV)
    sel.scatter_(1, sel_idx, True)
    tail = blk_of * RATIO

    def mask_mod(b, h, qi, ki):
        return (ki <= qi) & (sel[qi, blk_of[ki]] | (ki >= tail[qi]))

    cbm = _create_block_mask()
    bm = cbm(mask_mod, B=None, H=None, Q_LEN=S, KV_LEN=S, device=DEV)
    qf = q.transpose(1, 2).contiguous()
    kf = k.transpose(1, 2).contiguous()
    vf = v.transpose(1, 2).contiguous()

    def qsa():
        run_flex_attention(qf, kf, vf, bm, D ** -0.5)

    d, s = timeit(dense, iters=8), timeit(qsa, iters=8)
    print(f"  S={S:7d}  dense {d:8.3f} ms   qsa {s:8.3f} ms   "
          f"{'speedup' if s < d else 'slowdown'} {max(d/s, s/d):6.2f}x", flush=True)
    return d, s


print("DECODE  (1 query vs KV cache of S) -- no query tile, no union")
dec = {S: decode(S) for S in (4096, 8192, 16384, 32768, 65536, 131072)}

print("\nPREFILL (S queries) -- the training regime")
pre = {}
for S in (4096, 8192, 16384, 32768):
    try:
        pre[S] = prefill(S)
    except Exception as e:
        print(f"  S={S}: FAILED {type(e).__name__}: {str(e)[:90]}")

# model-level attention totals: the four variants differ only in how many
# full-attention layers they have and whether those use QSA. The 24 GDN layers
# in the hybrid variants are O(1) in S and identical within each compared pair.
print("\nMODEL-LEVEL attention time per decode step (full-attention layers only)")
print(f"{'S':>8s} {'hybrid(8 dense)':>16s} {'qsa512(8 QSA)':>14s} {'fullattn(32)':>13s} {'allqsa512(32)':>14s}")
for S, (d, s) in dec.items():
    print(f"{S:8d} {8*d:14.2f}ms {8*s:12.2f}ms {32*d:11.2f}ms {32*s:12.2f}ms")
