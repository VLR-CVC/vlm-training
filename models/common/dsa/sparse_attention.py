"""A faster forward for the vendored DSA sparse attention.

`dsa_min_memory_triton._dsa_sparse_attention_forward_kernel` runs one program per
(query, head) in two passes: it gathers every selected key twice (softmax stats,
then output) for each of the GQA heads that share them, in fp32, and on sm_100+
without tensor cores. At 64K on the 2B shapes that kernel is 82% of a DSA prefill
(311 of 369 ms).

The kernel here runs one program per (query, KV group): the DSA selection is
shared by all heads, so the group's query heads attend over the same gathered keys,
loaded once. The head axis, padded to 16, makes q @ k^T and p @ v tensor-core
dots, and one pass with an online softmax replaces the two. bf16/fp16 operands
with fp32 accumulation, as in FlashAttention.

The vendored module stays untouched (it is re-vendored by `scripts/vendor_dsa.py`);
`install()` swaps its `_triton_sparse_attention_forward`, which its callers look up
as a module global, for a dispatcher that falls back to the original for fp32,
for sm_100/sm_103 (the vendored comment reports tl.dot lowering bugs there), and
when DSA_GROUPED_SPARSE_ATTN=0. Only the forward changes; the backward kernels
recompute their own probabilities.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

from models.common.dsa import dsa_min_memory as _vendored_mm
from models.common.dsa import dsa_min_memory_triton as _vendored


def _configs():
    # head_dim 256: a 64-key bf16 tile is 32 KB for k and again for v, so stages
    # stay at 2 to fit consumer Blackwell's ~100 KB of shared memory per block.
    return [
        triton.Config({"BLOCK_K": 32}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_K": 128}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_configs(), key=["topk", "HEAD_DIM", "VALUE_DIM", "REPEAT"])
@triton.jit
def _grouped_forward_kernel(
    query_ptr,
    key_ptr,
    value_ptr,
    topk_indices_ptr,
    output_ptr,
    softmax_scale,
    q_start,
    topk: tl.constexpr,
    REPEAT: tl.constexpr,
    q_stride_m,
    q_stride_b,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_b,
    k_stride_g,
    k_stride_d,
    v_stride_s,
    v_stride_b,
    v_stride_g,
    v_stride_d,
    ti_stride_b,
    ti_stride_m,
    ti_stride_k,
    out_stride_m,
    out_stride_b,
    out_stride_h,
    out_stride_d,
    HEAD_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_DV: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    query_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    group_idx = tl.program_id(2)
    offs_h = tl.arange(0, BLOCK_H)
    offs_d = tl.arange(0, BLOCK_D)
    offs_dv = tl.arange(0, BLOCK_DV)
    heads = group_idx * REPEAT + offs_h
    head_mask = offs_h < REPEAT
    q = tl.load(
        query_ptr
        + query_idx * q_stride_m
        + batch_idx * q_stride_b
        + heads[:, None] * q_stride_h
        + offs_d[None, :] * q_stride_d,
        mask=head_mask[:, None] & (offs_d[None, :] < HEAD_DIM),
        other=0.0,
    )
    position = q_start + query_idx
    row_max = tl.full((BLOCK_H,), -float("inf"), dtype=tl.float32)
    row_sum = tl.zeros((BLOCK_H,), dtype=tl.float32)
    acc = tl.zeros((BLOCK_H, BLOCK_DV), dtype=tl.float32)
    for support_start in range(0, topk, BLOCK_K):
        offs_k = support_start + tl.arange(0, BLOCK_K)
        support_mask = offs_k < topk
        selected = tl.load(
            topk_indices_ptr + batch_idx * ti_stride_b + query_idx * ti_stride_m + offs_k * ti_stride_k,
            mask=support_mask,
            other=0,
        )
        valid = support_mask & (selected <= position)
        k = tl.load(
            key_ptr
            + selected[:, None] * k_stride_s
            + batch_idx * k_stride_b
            + group_idx * k_stride_g
            + offs_d[None, :] * k_stride_d,
            mask=valid[:, None] & (offs_d[None, :] < HEAD_DIM),
            other=0.0,
        )
        scores = tl.dot(q, tl.trans(k)) * softmax_scale
        scores = tl.where(valid[None, :], scores, -float("inf"))
        new_max = tl.maximum(row_max, tl.max(scores, axis=1))
        # A block with no visible key keeps the max at -inf, and exp(-inf - -inf) is nan.
        safe_max = tl.where(new_max == -float("inf"), 0.0, new_max)
        alpha = tl.exp(row_max - safe_max)
        probs = tl.exp(scores - safe_max[:, None])
        row_sum = row_sum * alpha + tl.sum(probs, axis=1)
        v = tl.load(
            value_ptr
            + selected[:, None] * v_stride_s
            + batch_idx * v_stride_b
            + group_idx * v_stride_g
            + offs_dv[None, :] * v_stride_d,
            mask=valid[:, None] & (offs_dv[None, :] < VALUE_DIM),
            other=0.0,
        )
        acc = acc * alpha[:, None] + tl.dot(probs.to(v.dtype), v)
        row_max = new_max
    acc = acc / row_sum[:, None]
    tl.store(
        output_ptr
        + query_idx * out_stride_m
        + batch_idx * out_stride_b
        + heads[:, None] * out_stride_h
        + offs_dv[None, :] * out_stride_d,
        acc,
        mask=head_mask[:, None] & (offs_dv[None, :] < VALUE_DIM),
    )


def grouped_forward(query, key, value, topk_indices, softmax_scale: float, q_start: int) -> torch.Tensor:
    """query [M, B, H, D], key/value [S, B, G, D], topk_indices [B, M, K] -> [M, B, H, Dv]."""
    query_len, batch_size, num_heads, head_dim = query.shape
    num_groups = key.size(2)
    repeat = num_heads // num_groups
    value_dim = value.size(-1)
    output = value.new_empty((query_len, batch_size, num_heads, value_dim))
    _grouped_forward_kernel[(query_len, batch_size, num_groups)](
        query, key, value, topk_indices, output, softmax_scale, q_start,
        topk_indices.size(-1), repeat,
        *query.stride(), *key.stride(), *value.stride(), *topk_indices.stride(), *output.stride(),
        HEAD_DIM=head_dim,
        VALUE_DIM=value_dim,
        BLOCK_H=max(16, triton.next_power_of_2(repeat)),
        BLOCK_D=triton.next_power_of_2(head_dim),
        BLOCK_DV=max(32, triton.next_power_of_2(value_dim)),
    )
    return output


def _use_grouped(query, key, value) -> bool:
    flag = os.environ.get("DSA_GROUPED_SPARSE_ATTN")
    if flag == "0":
        return False
    if not (query.dtype == key.dtype == value.dtype and query.dtype in (torch.bfloat16, torch.float16)):
        return False
    if query.size(2) // key.size(2) > 64:
        return False
    return flag == "1" or torch.cuda.get_device_capability(query.device)[0] != 10


_original_forward = _vendored._triton_sparse_attention_forward


def _dispatch(query, key, value, topk_indices, softmax_scale, q_start):
    if _use_grouped(query, key, value):
        return grouped_forward(query, key, value, topk_indices, softmax_scale, q_start)
    return _original_forward(query, key, value, topk_indices, softmax_scale, q_start)


# ---------------------------------------------------------------------------
# Fused top-K routing
#
# The vendored `_simplified_topk_index_tile` scores 16K-key blocks, runs
# `torch.topk` on each, merges the running top-K with another topk, then sorts the
# support by position: ~40 of the 121 ms of a 64K DSA layer after the grouped
# attention. Here each query sub-chunk is scored against all its visible keys
# once, and `_row_topk_kernel` selects the exact top-K of every row with a radix
# select (11 + 11 + 10 bits, one histogram pass each) and an emit pass that writes
# the survivors in position order -- the order the vendored code sorts into, so
# no merge and no sort. Rows with <= K visible keys keep all of them and pad with
# an index past the row (masked by the attention, as the vendored -inf picks are).
# ---------------------------------------------------------------------------

_SCORE_BUDGET_BYTES = 1 << 30


@triton.jit
def _orderable(x):
    """fp32 -> uint32 whose unsigned order is the float order."""
    u = x.to(tl.uint32, bitcast=True)
    flip = tl.where((u >> 31) == 1, tl.full(u.shape, 0xFFFFFFFF, tl.uint32), tl.full(u.shape, 0x80000000, tl.uint32))
    return u ^ flip


@triton.jit
def _radix_pass(scores_ptr, row_off, length, prefix, k_rem, PREFIX_SHIFT: tl.constexpr, SHIFT: tl.constexpr,
                BITS: tl.constexpr, BLOCK: tl.constexpr):
    """Histogram of the BITS-wide digit at SHIFT among keys whose bits above
    PREFIX_SHIFT equal `prefix`; returns the digit holding the k_rem-th largest
    and how many keys sit strictly above it."""
    NB: tl.constexpr = 1 << BITS
    hist = tl.zeros((NB,), dtype=tl.int32)
    for start in range(0, length, BLOCK):
        offs = start + tl.arange(0, BLOCK)
        m = offs < length
        key = _orderable(tl.load(scores_ptr + row_off + offs, mask=m, other=0.0))
        if PREFIX_SHIFT < 32:
            m = m & ((key >> PREFIX_SHIFT) == prefix)
        digit = ((key >> SHIFT) & (NB - 1)).to(tl.int32)
        hist += tl.histogram(digit, NB, mask=m)
    bins = tl.arange(0, NB)
    above_or_at = tl.sum(hist) - tl.cumsum(hist, axis=0) + hist  # keys with digit >= b
    b = tl.max(tl.where(above_or_at >= k_rem, bins, -1))
    above = tl.sum(tl.where(bins > b, hist, 0))
    return b, above


@triton.jit
def _row_topk_kernel(scores_ptr, out_idx_ptr, out_score_ptr, q_start, s_stride_row, o_stride_row,
                     key_len, TOPK: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    position = q_start + row
    length = tl.minimum(position + 1, key_len)
    row_off = row.to(tl.int64) * s_stride_row
    out_off = row.to(tl.int64) * o_stride_row
    if length <= TOPK:
        for start in range(0, TOPK, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < TOPK
            keep = offs < length
            sc = tl.load(scores_ptr + row_off + offs, mask=keep & m, other=-float("inf"))
            tl.store(out_idx_ptr + out_off + offs, tl.where(keep, offs, position + 1).to(tl.int64), mask=m)
            tl.store(out_score_ptr + out_off + offs, sc, mask=m)
    else:
        b1, a1 = _radix_pass(scores_ptr, row_off, length, 0, TOPK, 32, 21, 11, BLOCK)
        k_rem = TOPK - a1
        prefix = b1.to(tl.uint32)
        b2, a2 = _radix_pass(scores_ptr, row_off, length, prefix, k_rem, 21, 10, 11, BLOCK)
        k_rem = k_rem - a2
        prefix = (prefix << 11) | b2.to(tl.uint32)
        b3, a3 = _radix_pass(scores_ptr, row_off, length, prefix, k_rem, 10, 0, 10, BLOCK)
        ties = k_rem - a3  # how many keys equal to the threshold to take
        threshold = (prefix << 10) | b3.to(tl.uint32)
        written = 0
        ties_seen = 0
        for start in range(0, length, BLOCK):
            offs = start + tl.arange(0, BLOCK)
            m = offs < length
            sc = tl.load(scores_ptr + row_off + offs, mask=m, other=0.0)
            key = _orderable(sc)
            eq = m & (key == threshold)
            tie_rank = ties_seen + tl.cumsum(eq.to(tl.int32), axis=0) - eq.to(tl.int32)
            take = m & ((key > threshold) | (eq & (tie_rank < ties)))
            slot = written + tl.cumsum(take.to(tl.int32), axis=0) - take.to(tl.int32)
            tl.store(out_idx_ptr + out_off + slot, offs.to(tl.int64), mask=take)
            tl.store(out_score_ptr + out_off + slot, sc, mask=take)
            written += tl.sum(take.to(tl.int32))
            ties_seen += tl.sum(eq.to(tl.int32))


def row_topk(scores: torch.Tensor, q_start: int, topk: int) -> tuple[torch.Tensor, torch.Tensor]:
    """scores [M, N] fp32, row m = query q_start + m, keys > the query ignored ->
    (fp32 scores, int64 indices) [M, topk], indices ascending."""
    M, N = scores.shape
    idx = torch.empty((M, topk), device=scores.device, dtype=torch.int64)
    val = torch.empty((M, topk), device=scores.device, dtype=torch.float32)
    _row_topk_kernel[(M,)](scores, idx, val, q_start, scores.stride(0), idx.stride(0), N,
                           TOPK=topk, BLOCK=1024, num_warps=8)
    return val, idx


def fused_topk_index_tile(
    hidden_states, key, q_start, q_end, linear_q_weight, index_topk, index_head_dim, index_rotary_dim,
    rotary_pos_emb, rotary_interleaved, use_indexer_rope, score_scale, key_chunk_size,
    simplified_input_norm=None, linear_k_weight=None, full_k_index=None,
):
    """Drop-in for the vendored `_simplified_topk_index_tile` (same arguments and
    returns: scores and indices [B, M, K] in position order, and q_index)."""
    q_index = _vendored_mm._project_simplified_q_index_tile(
        hidden_states, q_start, q_end, linear_q_weight, index_head_dim, index_rotary_dim,
        rotary_pos_emb, rotary_interleaved, use_indexer_rope, simplified_input_norm,
    )
    limit = min(q_end, key.size(0))
    topk = min(index_topk, limit)
    if linear_k_weight is None:
        keys = key[:limit]
    elif full_k_index is not None:
        keys = full_k_index[:limit]
    else:
        keys = _vendored_mm._project_simplified_k_index_block(
            hidden_states, 0, limit, linear_k_weight, index_head_dim, index_rotary_dim,
            rotary_pos_emb, rotary_interleaved, use_indexer_rope, simplified_input_norm,
        )
    rows = q_end - q_start
    batch = q_index.size(1)
    sub = max(256, min(rows, _SCORE_BUDGET_BYTES // (4 * limit)))
    scores_out = torch.empty((batch, rows, topk), device=q_index.device, dtype=torch.float32)
    idx_out = torch.empty((batch, rows, topk), device=q_index.device, dtype=torch.int64)
    for a in range(0, rows, sub):
        b = min(a + sub, rows)
        # keys beyond this sub-chunk's last query are never visible to it
        visible = min(limit, q_start + b)
        scores = _vendored_mm._simplified_index_scores_block(
            q_index[a:b], keys[:visible], score_scale, q_start + a, 0
        )  # [B, m, visible]
        for bi in range(batch):
            v, i = row_topk(scores[bi], q_start + a, topk)
            scores_out[bi, a:b], idx_out[bi, a:b] = v, i
    return scores_out, idx_out, q_index


_original_topk_tile = _vendored_mm._simplified_topk_index_tile


def _topk_dispatch(*args, **kwargs):
    if os.environ.get("DSA_FUSED_TOPK") == "0":
        return _original_topk_tile(*args, **kwargs)
    hidden_states = args[0] if args else kwargs["hidden_states"]
    if not hidden_states.is_cuda:
        return _original_topk_tile(*args, **kwargs)
    return fused_topk_index_tile(*args, **kwargs)


def install() -> None:
    _vendored._triton_sparse_attention_forward = _dispatch
    _vendored_mm._simplified_topk_index_tile = _topk_dispatch


def selftest() -> None:
    """Grouped vs the vendored per-head kernel vs an fp32 reference, invalid indices included."""
    torch.manual_seed(0)
    M, S, H, G, D, K, q_start = 300, 700, 8, 2, 256, 128, 400
    q = torch.randn(M, 1, H, D, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(S, 1, G, D, device="cuda", dtype=torch.bfloat16)
    v = torch.randn(S, 1, G, D, device="cuda", dtype=torch.bfloat16)
    # rows see q_start + m + 1 keys; some picks are past the row (masked), one row sees only key 0
    idx = torch.randint(0, S, (1, M, K), device="cuda")
    idx[0, :, 0] = 0
    scale = D**-0.5
    got = grouped_forward(q, k, v, idx, scale, q_start)
    old = _original_forward(q, k, v, idx, scale, q_start)
    # fp32 reference
    pos = q_start + torch.arange(M, device="cuda")
    kk = k[:, 0].float().repeat_interleave(H // G, 1)[idx[0]]  # [M, K, H, D]
    vv = v[:, 0].float().repeat_interleave(H // G, 1)[idx[0]]
    s = torch.einsum("mhd,mkhd->mhk", q[:, 0].float(), kk) * scale
    s = s.masked_fill((idx[0] > pos[:, None])[:, None, :], float("-inf"))
    ref = torch.einsum("mhk,mkhd->mhd", s.softmax(-1), vv)
    e_new = (got[:, 0].float() - ref).abs().max().item()
    e_old = (old[:, 0].float() - ref).abs().max().item()
    assert e_new < 2e-2, (e_new, e_old)
    print(f"selftest ok: max |grouped - fp32 ref| {e_new:.2e}, |vendored - fp32 ref| {e_old:.2e}")


def selftest_topk() -> None:
    """row_topk picks exactly torch.topk's set, in position order, short rows padded."""
    torch.manual_seed(0)
    M, N, K, q_start = 200, 5000, 512, 4800
    scores = torch.randn(M, N, device="cuda")
    scores[:, ::7] = scores[:, 3:4]  # ties at many values
    pos = q_start + torch.arange(M, device="cuda")
    masked = scores.masked_fill(torch.arange(N, device="cuda")[None] > pos[:, None], float("-inf"))
    val, idx = row_topk(scores, q_start, K)
    ref_val = masked.topk(K, dim=-1).values
    assert torch.equal(val.sort(-1).values, ref_val.sort(-1).values), "selected values differ"
    assert bool((idx.diff(dim=-1) > 0).all()), "not in position order"
    assert bool((idx <= pos[:, None]).all()), "picked a future key"
    assert torch.equal(torch.gather(scores, 1, idx), val)
    # short rows: all visible keys, then padding past the row
    v2, i2 = row_topk(scores[:10], 100, K)  # rows see 101..110 keys
    for r in range(10):
        n = 101 + r
        assert torch.equal(i2[r, :n].cpu(), torch.arange(n)) and bool((i2[r, n:] == 100 + r + 1).all())
    print("selftest_topk ok")


if __name__ == "__main__":
    selftest()
    selftest_topk()
