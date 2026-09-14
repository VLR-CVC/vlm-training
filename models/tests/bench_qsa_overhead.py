"""Why is QSA ~5x slower than dense varlen when it does ~1/15th the arithmetic?

Round 1 (T=10240) established: `create_block_mask` is flat at 0.41ms and ~1% of
the cost. All of the overhead is the flex-attention kernel, at ~4.8x dense
regardless of L -- it scales as if dense even though QSA attends a fixed 512
tokens per query.

The suspect is granularity. QSA picks `budget/ratio` blocks of `ratio` tokens;
flex_attention's BlockMask tiles are `_FLEX_BLOCK_SIZE` (128) wide. With
ratio=4, 512 attended tokens arrive as 128 four-token granules that can land in
128 distinct tiles -- and a 10240-token document only has 80 tiles, so none can
be skipped. The kernel visits every tile and masks within it.

This run separates that from an artifact of round 1's uniformly random mask:

  scatter=random      128 granules placed anywhere   (round 1's mask)
  scatter=local       the 512 tokens nearest the query, contiguous
  ratio sweep         4 / 32 / 128, at fixed 512-token budget

If `local` and ratio=128 are fast, the sparsity is real and the loss is a
tile-alignment problem. If every arm is ~5x, flex is simply slow at head_dim
256 and QSA cannot win through it at any setting.
"""
import sys, time
import torch

# Round 1's T=16384 half died on retain_graph=True against inductor's donated
# buffers. Rebuilding the graph per iteration would time the rebuild too, so
# disable the donation instead -- it changes memory reuse, not kernel cost.
import torch._functorch.config
torch._functorch.config.donated_buffer = False

sys.path.insert(0, "/e/project1/open-sci-mm/ockier1/vlm-training")
from models.qwen3_5.model import _create_block_mask, _segment_ids, run_flex_attention
from flash_attn import flash_attn_varlen_func

H, HKV, D = 16, 4, 256
DEV, DT = "cuda", torch.bfloat16
BUDGET = 512


def timeit(fn, iters=8, warmup=3):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def run(T, L, ratio, scatter):
    n = T // L
    cu = torch.arange(0, n + 1, device=DEV, dtype=torch.int32) * L
    if cu[-1] != T:
        cu = torch.cat([cu, torch.tensor([T], device=DEV, dtype=torch.int32)])
    q = torch.randn(T, H, D, device=DEV, dtype=DT, requires_grad=True)
    k = torch.randn(T, HKV, D, device=DEV, dtype=DT, requires_grad=True)
    v = torch.randn(T, HKV, D, device=DEV, dtype=DT, requires_grad=True)

    def dense():
        flash_attn_varlen_func(q, k, v, cu, cu, L, L, causal=True).sum().backward()

    seg = _segment_ids(cu, T)
    nblk = (T + ratio - 1) // ratio
    keep = max(1, BUDGET // ratio)
    pos = torch.arange(T, device=DEV)
    blk_of = pos // ratio
    tail = blk_of * ratio

    sel = torch.zeros(T, nblk, dtype=torch.bool, device=DEV)
    if scatter == "random":
        sel.scatter_(1, torch.randint(0, nblk, (T, keep), device=DEV), True)
    else:  # local: the `keep` blocks immediately preceding each query
        base = (blk_of.unsqueeze(1) - torch.arange(keep, device=DEV).unsqueeze(0))
        sel.scatter_(1, base.clamp_(min=0), True)

    def mask_mod(b, h, qi, ki):
        return (seg[qi] == seg[ki]) & (ki <= qi) & (sel[qi, blk_of[ki]] | (ki >= tail[qi]))

    cbm = _create_block_mask()
    bm = cbm(mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=DEV)
    qf = q.detach().view(1, T, H, D).transpose(1, 2).clone().requires_grad_(True)
    kf = k.detach().view(1, T, HKV, D).transpose(1, 2).clone().requires_grad_(True)
    vf = v.detach().view(1, T, HKV, D).transpose(1, 2).clone().requires_grad_(True)

    def flex():
        run_flex_attention(qf, kf, vf, bm, D ** -0.5).sum().backward()

    d = timeit(dense)
    m = timeit(lambda: cbm(mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=DEV))
    f = timeit(flex)
    # how many 128-wide tiles a query's selection actually touches
    tiles = (sel[T // 2].nonzero().flatten() * ratio // 128).unique().numel()
    print(f"  T={T} L={L:6d} ratio={ratio:4d} {scatter:6s} | dense {d:7.2f} | "
          f"mask {m:5.2f} + flex {f:7.2f} = {m+f:7.2f} | {(m+f)/d:5.2f}x | "
          f"tiles touched {tiles:3d}/{max(1,L//128):3d}")


for T, L in ((10240, 10240), (16384, 16384), (16384, 5120)):
    for scatter in ("random", "local"):
        for ratio in (4, 32, 128):
            try:
                run(T, L, ratio, scatter)
            except Exception as e:
                print(f"  T={T} L={L} ratio={ratio} {scatter}: FAILED {type(e).__name__}: {str(e)[:90]}")
    print()
