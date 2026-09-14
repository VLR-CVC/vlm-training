"""Kernel cost as a function of the BlockMask union, with the oracle marked.

oracle_mask.py measured what a *perfectly trained* indexer would produce on real
Qwen3.5-9B weights and real clevr batches: the union over a 128-query tile
covers 53.8% of causal tiles (per-layer 51.0-58.7), against 7.6% for a pure
locality prior and 100% for the random indexer the ladder actually ran.

bench_qsa_overhead.py timed only the two endpoints -- 0.31x dense at 7.6% union,
5.03x at ~98%. Reading 53.8% off a straight line between them is a guess, and it
is the guess the whole verdict rests on. This measures the curve instead.

`window` controls locality: each query draws its `budget/ratio` blocks uniformly
from the W tokens preceding it. W = budget is perfectly local; W = L is uniform
random. Sweeping W walks the union from ~7% to ~100%.
"""
import sys, time
import torch
import torch._functorch.config
torch._functorch.config.donated_buffer = False

sys.path.insert(0, "/e/project1/open-sci-mm/ockier1/vlm-training")
from models.qwen3_5.model import _create_block_mask, _segment_ids, run_flex_attention
from flash_attn import flash_attn_varlen_func

H, HKV, D = 16, 4, 256
DEV, DT = "cuda", torch.bfloat16
BUDGET, RATIO, TILE = 512, 4, 128
KEEP = BUDGET // RATIO
ORACLE_UNION = 53.8          # measured, oracle_mask.py job 1791719


def timeit(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run(T, L, window):
    n = max(1, T // L)
    cu = torch.arange(0, n + 1, device=DEV, dtype=torch.int32) * L
    if cu[-1] != T:
        cu = torch.cat([cu, torch.tensor([T], device=DEV, dtype=torch.int32)])
    q = torch.randn(T, H, D, device=DEV, dtype=DT, requires_grad=True)
    k = torch.randn(T, HKV, D, device=DEV, dtype=DT, requires_grad=True)
    v = torch.randn(T, HKV, D, device=DEV, dtype=DT, requires_grad=True)

    def dense():
        flash_attn_varlen_func(q, k, v, cu, cu, L, L, causal=True).sum().backward()

    seg = _segment_ids(cu, T)
    nblk = (T + RATIO - 1) // RATIO
    pos = torch.arange(T, device=DEV)
    blk_of = pos // RATIO
    tail = blk_of * RATIO

    # draw KEEP blocks uniformly from the `window` tokens before each query
    span = min(window, T) // RATIO
    off = (torch.rand(T, KEEP, device=DEV) * span).long()
    sel_idx = (blk_of.unsqueeze(1) - off).clamp_(min=0)
    sel = torch.zeros(T, nblk, dtype=torch.bool, device=DEV)
    sel.scatter_(1, sel_idx, True)

    ntile = T // TILE
    kt = (sel_idx * RATIO) // TILE
    qt = (pos // TILE).unsqueeze(1).expand_as(kt)
    grid = torch.zeros(ntile, ntile, dtype=torch.bool, device=DEV)
    grid[qt.reshape(-1), kt.reshape(-1)] = True
    causal = torch.tril(torch.ones(ntile, ntile, dtype=torch.bool, device=DEV))
    union = 100.0 * (grid & causal).sum().item() / causal.sum().item()

    def mask_mod(b, h, qi, ki):
        return (seg[qi] == seg[ki]) & (ki <= qi) & (sel[qi, blk_of[ki]] | (ki >= tail[qi]))

    cbm = _create_block_mask()
    bm = cbm(mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=DEV)
    qf = q.detach().view(1, T, H, D).transpose(1, 2).clone().requires_grad_(True)
    kf = k.detach().view(1, T, HKV, D).transpose(1, 2).clone().requires_grad_(True)
    vf = v.detach().view(1, T, HKV, D).transpose(1, 2).clone().requires_grad_(True)

    def flex():
        run_flex_attention(qf, kf, vf, bm, D ** -0.5).sum().backward()

    d, m, f = timeit(dense), timeit(lambda: cbm(mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=DEV)), timeit(flex)
    mark = "  <-- oracle" if abs(union - ORACLE_UNION) < 6 else ""
    print(f"  window {window:6d}  union {union:5.1f}%  dense {d:6.2f}ms  "
          f"mask {m:5.2f} + flex {f:7.2f} = {m+f:7.2f}ms  {(m+f)/d:5.2f}x dense{mark}",
          flush=True)


# clevr at seq 16384 packs 2-3 documents per row (oracle_mask.py: docs=2, docs=3)
for T, L in ((16384, 6144), (16384, 16384)):
    print(f"T={T} L={L}  ({max(1, T//L)} docs/row)")
    for w in (512, 1024, 2048, 4096, 8192, 16384):
        if w > L:
            continue
        try:
            run(T, L, w)
        except Exception as e:
            print(f"  window {w}: FAILED {type(e).__name__}: {str(e)[:80]}")
    print()
