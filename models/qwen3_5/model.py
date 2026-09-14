from __future__ import annotations

from pathlib import Path
import inspect
import math
import os
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn

_SECTION_TIMING = os.environ.get("QWEN_SECTION_TIMING") == "1"
_SECTION_MS: dict[str, float] = {}


class _SectionTimer:
    """Attribute forward time to named regions with CUDA events.

    `mark(name)` closes the region that began at the previous mark. Six
    single-line calls instead of six nested context managers, so the forward
    keeps its indentation.

    Enabled by `QWEN_SECTION_TIMING=1`; a no-op otherwise, cheap enough to leave
    in the hot path. It synchronizes on construction and on every mark, so it is
    a diagnostic, not something to run a real job with.

    Forward only. Backward is one fused `.backward()` call and cannot be split
    this way -- that needs a profiler trace.
    """

    __slots__ = ("_prev",)

    def __init__(self):
        if not _SECTION_TIMING:
            self._prev = None
            return
        # Drain first. Even with the forward's own `.item()` calls removed,
        # anything upstream that has not finished would otherwise be charged to
        # whichever region happens to run first.
        torch.cuda.synchronize()
        self._prev = torch.cuda.Event(enable_timing=True)
        self._prev.record()

    def mark(self, name: str) -> None:
        if self._prev is None:
            return
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        e.synchronize()
        _SECTION_MS[name] = _SECTION_MS.get(name, 0.0) + self._prev.elapsed_time(e)
        self._prev = e

def _round_max_seqlen(n: int) -> int:
    """Power-of-two upper bound; see `train.utils.round_max_seqlen`. Duplicated
    rather than imported so the model package stays independent of `train`."""
    return 1 if n <= 1 else 1 << (n - 1).bit_length()

def packed_positions(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Position ids for a packed row: `arange` within each document, restarting
    at every cu_seqlens boundary.

    Each token's absolute index minus its segment's start. `output_size` is what
    keeps `repeat_interleave` from syncing to discover the output length.
    """
    starts = cu_seqlens[:-1].to(torch.int64)
    lens = (cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int64)
    seg_start = torch.repeat_interleave(starts, lens, output_size=total)
    return torch.arange(total, device=cu_seqlens.device) - seg_start


def pop_section_ms() -> dict[str, float]:
    """Accumulated per-region forward milliseconds since the last call."""
    out = dict(_SECTION_MS)
    _SECTION_MS.clear()
    return out

from models.qwen3_5.config import (
    Qwen3_5Config, Qwen3_5TextConfig, Qwen3_5VisionConfig
)
from models.qwen3_5.utils import (
    _dtensor_unwrap,
    _dtensor_rewrap,
    _local,
    CausalLMOutput,
    causal_lm_loss,
    apply_rope,
    mrope_cos_sin,
    apply_rope_vision,
    load_safetensors_into,
)
from models.qwen3_5 import compile_ops as _ops

_VARLEN_HAS_GQA = "enable_gqa" in inspect.signature(varlen_attn).parameters

def _gqa(q, k) -> dict:
    """`enable_gqa=True` when q and k disagree on head count, else nothing."""
    if _VARLEN_HAS_GQA and q.shape[-2] != k.shape[-2]:
        return {"enable_gqa": True}
    return {}

def _varlen_sdpa(q, k, v, cu_seqlens, causal: bool):
    """torch-native block-diagonal SDPA, for dtypes the flash kernels refuse."""
    total = q.shape[0]
    idx = torch.arange(total, device=q.device)
    # `right=True` puts a token that lands exactly on a boundary in the segment
    # that starts there, which is what cu_seqlens means.
    seg = torch.bucketize(idx, cu_seqlens[1:-1].to(idx.dtype), right=True)
    mask = seg[:, None] == seg[None, :]
    if causal:
        mask = mask & (idx[:, None] >= idx[None, :])
    out = F.scaled_dot_product_attention(
        q.transpose(0, 1).unsqueeze(0),
        k.transpose(0, 1).unsqueeze(0),
        v.transpose(0, 1).unsqueeze(0),
        attn_mask=mask[None, None],
        enable_gqa=q.shape[1] != k.shape[1],
    )
    return out.squeeze(0).transpose(0, 1)  # (total, num_heads, head_dim)

class RMSNormGated(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @staticmethod
    def _run_fla_rms_norm_gated(hs, gate, weight, eps):
        return _ops.dispatch_rms_norm_gated(hs, gate, weight, eps)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        D = orig_shape[-1]
        (hs_local, gate_local), wrap = _dtensor_unwrap(hidden_states, gate)
        out = RMSNormGated._run_fla_rms_norm_gated(
            hs_local.reshape(-1, D),
            gate_local.reshape(-1, D),
            _local(self.weight),
            self.eps,
        )
        return _dtensor_rewrap(out.reshape(orig_shape), wrap)

class OffsetRMSNorm(nn.Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``, weight init to zeros.
    Taken from Torchtitan - shares impl w/ transformers
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (1 + weight) offset matches HF Qwen3_5 vs plain RMSNorm.
        # F.rms_norm handles the fp32 upcast internally and lets inductor fuse.
        # See https://github.com/huggingface/transformers/pull/29402
        #
        # `.to(x.dtype)`: `F.rms_norm` is not on autocast's cast list, so with
        # `master_dtype = "float32"` (the default) the fp32 `weight` promotes a
        # bf16 input back to fp32. That leaked all the way into the attention
        # kernel -- `RuntimeError: FlashAttention only support fp16 and bf16 data
        # type` -- and silently made every layer output fp32. HF's RMSNorm casts
        # back for the same reason.
        return F.rms_norm(x, self.weight.shape, 1.0 + self.weight, self.eps).to(x.dtype)

# ---------------------------------------------------------------------------
# Qwen Sparse Attention (QSA), ported verbatim from `models/qwen4/model.py`.
#
# The only qwen4 piece this variant needs: Qwen3.5's `SelfAttention` is already
# identical to Qwen4's apart from the indexer, so swapping the dense varlen
# kernel for a QSA-masked flex kernel isolates the attention mechanism with
# every other parameter and module held fixed.
#
# Kept byte-identical to the qwen4 source so the two stay diffable.
# ---------------------------------------------------------------------------

_CREATE_BLOCK_MASK: dict = {}

_FLEX_ATTENTION: dict = {}

def _segment_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Per-token document index for a packed row. Graph-traceable, no host sync."""
    return torch.bucketize(
        torch.arange(total, device=cu_seqlens.device), cu_seqlens[1:-1], right=True
    )

# Transient bytes the selection loop allocates per query row: the (chunk, NB)
# score matrix in fp32 plus its bool eligibility mask, and -- the dominant term
# -- the int64 argsort permutation over all NB blocks.
_QSA_CHUNK_BYTES_PER_ROW = 4 + 1 + 8

# Budget for those transients. Chunking exists to bound them, but a fixed 1024
# leaves most of the win on the table at ordinary training shapes: at T=4096,
# NB=1024 one chunk runs the indexer in 1.24 ms against 2.23 ms for four, since
# every chunk repeats the same handful of launch-bound elementwise kernels.
_QSA_CHUNK_BUDGET = 256 * 2**20

def _qsa_query_chunk(total: int, num_blocks: int) -> int:
    """Query rows per selection chunk: as many as the budget allows.

    Never below 1024, which is what this was pinned to before, so a very wide
    block grid degrades to the old behaviour instead of to something slower.
    """
    per_row = max(1, num_blocks * _QSA_CHUNK_BYTES_PER_ROW)
    return max(1024, min(total, _QSA_CHUNK_BUDGET // per_row))

class QSAIndexer(nn.Module):
    """Qwen Sparse Attention token selector.

    HF's reference implementation loops over every (batch, query) pair. This is
    the same algorithm expressed over a packed varlen row.

    The partition is fixed per document: for a document spanning ``[d0, d1)``,
    block ``b`` is tokens ``[d0 + b*ratio, d0 + (b+1)*ratio)``. A query at
    position ``t`` therefore sees exactly the first ``(t - d0 + 1) // ratio``
    complete blocks, plus a tail of ``(t - d0 + 1) % ratio`` tokens ending at
    ``t`` which are always attended. So there is no per-query re-blocking: only
    a per-query prefix length and a top-k over that prefix.

    Returns the pieces the caller needs to build a flex-attention mask.
    """

    def __init__(self, cfg: Qwen3_5TextConfig):
        super().__init__()
        self.n_heads = cfg.indexer_n_heads
        self.kv_heads = cfg.indexer_kv_heads
        self.head_dim = cfg.indexer_head_dim
        self.budget = cfg.indexer_budget
        self.compress_ratio = cfg.indexer_compress_ratio
        self.block_topk = self.budget // self.compress_ratio

        self.index_qk_proj = nn.Linear(
            cfg.hidden_size, (self.n_heads + self.kv_heads) * self.head_dim, bias=False
        )
        self.q_layernorm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_layernorm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

    @torch.no_grad()
    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        seg_id: torch.Tensor,
        query_chunk: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ``(selected, block_of_token, tail_start)``.

        ``selected``        (T, NB) bool - block b chosen for query t
        ``block_of_token``  (T,)    long - global block slot owning token t
        ``tail_start``      (T,)    long - first always-visible token for query t
        """
        # Under TP `x` arrives as a DTensor (the attention module's
        # `PrepareModuleInput`) while the indexer's own weights are left
        # unsharded and plain -- it is frozen, produces no gradient and returns
        # index tensors, so it runs entirely on local tensors.
        (x,), _ = _dtensor_unwrap(x)

        total = x.shape[1]
        device = x.device
        ratio = self.compress_ratio

        doc_start = cu_seqlens[:-1].long()[seg_id]              # (T,)
        pos_in_doc = torch.arange(total, device=device) - doc_start

        doc_len = (cu_seqlens[1:] - cu_seqlens[:-1]).long()     # (D,)
        blocks_per_doc = (doc_len + ratio - 1) // ratio
        block_offset = torch.cat([
            blocks_per_doc.new_zeros(1), blocks_per_doc.cumsum(0)[:-1]
        ])
        num_blocks = int(blocks_per_doc.sum().item())
        block_of_token = block_offset[seg_id] + pos_in_doc // ratio

        n_complete = (pos_in_doc + 1) // ratio
        tail_start = doc_start + n_complete * ratio

        qk = self.index_qk_proj(x)
        q, token_k = torch.split(
            qk,
            [self.n_heads * self.head_dim, self.kv_heads * self.head_dim],
            dim=-1,
        )
        q = q.reshape(total, self.n_heads, self.head_dim)
        token_k = token_k.reshape(total, self.head_dim)
        q = self.q_layernorm(q)
        # apply_rope wants (B, H, S, D)
        q = apply_rope(q.transpose(0, 1).unsqueeze(0), None, cos, sin)
        q = q[0].transpose(0, 1)                                # (T, H, D)

        # mean-pool token keys into their block slot
        acc = token_k.new_zeros((num_blocks, self.head_dim), dtype=torch.float32)
        acc.index_add_(0, block_of_token, token_k.float())
        cnt = token_k.new_zeros((num_blocks,), dtype=torch.float32)
        cnt.index_add_(0, block_of_token, torch.ones_like(cnt[:1]).expand(total))
        pooled = (acc / cnt.clamp_min(1).unsqueeze(-1)).to(token_k.dtype)
        complete = cnt == ratio                                 # (NB,)
        pooled = self.k_layernorm(pooled)

        # rope each pooled key at the position of its first token
        block_first = torch.zeros(num_blocks, dtype=torch.long, device=device)
        block_first.scatter_reduce_(
            0, block_of_token, torch.arange(total, device=device), reduce="amin",
            include_self=False,
        )
        cos_b, sin_b = cos[0][block_first], sin[0][block_first]
        pooled = apply_rope(
            pooled.unsqueeze(0).unsqueeze(0), None, cos_b.unsqueeze(0), sin_b.unsqueeze(0)
        )[0, 0]                                                 # (NB, D)

        block_doc = torch.zeros(num_blocks, dtype=torch.long, device=device)
        block_doc.scatter_(0, block_of_token, seg_id)
        block_local = torch.zeros(num_blocks, dtype=torch.long, device=device)
        block_local.scatter_(0, block_of_token, pos_in_doc // ratio)

        selected = torch.zeros((total, num_blocks), dtype=torch.bool, device=device)
        scale = 1.0 / math.sqrt(self.head_dim)
        k_take = min(self.block_topk, num_blocks)

        if query_chunk is None:
            query_chunk = _qsa_query_chunk(total, num_blocks)

        for lo in range(0, total, query_chunk):
            hi = min(lo + query_chunk, total)
            scores = torch.einsum(
                "thd,bd->thb", q[lo:hi].float(), pooled.float()
            ).relu().sum(dim=1) * scale                          # (chunk, NB)
            eligible = (
                complete.unsqueeze(0)
                & (block_doc.unsqueeze(0) == seg_id[lo:hi].unsqueeze(1))
                & (block_local.unsqueeze(0) < n_complete[lo:hi].unsqueeze(1))
            )
            scores = scores.masked_fill(~eligible, float("-inf"))
            # Ties are common (a query with no positive affinity scores 0 against
            # several blocks). `topk` leaves the winner unspecified, so rank with a
            # stable descending sort instead: equal scores keep ascending block
            # order, which is what the reference implementation ends up picking.
            order = torch.argsort(scores, dim=-1, descending=True, stable=True)
            top_idx = order[:, :k_take]
            # Mark the whole top-k, then clear whatever was not eligible using
            # the mask already in hand. The direct form,
            #     keep = torch.isfinite(scores.gather(-1, top_idx))
            #     selected[rows[keep], top_idx[keep]] = True
            # is equivalent, but indexing by a bool mask has to count its True
            # entries on the host, so it synchronizes the device once per chunk
            # -- 48 stalls a step at 12 QSA layers and 4 chunks. `top_idx` holds
            # a slice of a permutation, so it never repeats an index and the
            # scatter cannot race with itself.
            row_view = selected[lo:hi]
            row_view.scatter_(1, top_idx, True)
            row_view &= eligible

        return selected, block_of_token, tail_start

def _create_block_mask():
    """`create_block_mask`, compiled once per process.

    Compiling it is worth 10x -- 1.28 ms against 0.13 ms at T=4096 -- and the
    result is bit-identical, block index for block index. The deprecated
    `_compile=True` argument is *not* the same thing and gets nowhere near it.

    Left on auto-dynamic on purpose. `dynamic=False` is a trap here: the block
    grid width follows the packed document lengths, so it changes most steps,
    and each new width recompiles for about 2 s. Auto-dynamic settles into two
    graphs and stays at 0.13 ms.
    """
    if "fn" not in _CREATE_BLOCK_MASK:
        from torch.nn.attention.flex_attention import create_block_mask

        _CREATE_BLOCK_MASK["fn"] = torch.compile(create_block_mask)
    return _CREATE_BLOCK_MASK["fn"]

def _flex_attention(compiled: bool = True):
    """`flex_attention`, compiled once per process.

    The eager path materializes the full (q, kv) score matrix -- torch warns
    about exactly this -- which at training sequence lengths costs more memory
    than the dense attention QSA exists to avoid. Compiling generates the fused
    kernel that actually consumes the BlockMask's sparsity.

    Inductor has no template for rows shorter than the 128-token block size, so
    `run_flex_attention` falls back to eager for those. Packed training rows are
    always `seq_len`, so that only affects small inputs and tests.
    """
    key = "compiled" if compiled else "eager"
    if key not in _FLEX_ATTENTION:
        from torch.nn.attention.flex_attention import flex_attention

        _FLEX_ATTENTION[key] = (
            torch.compile(flex_attention, dynamic=False) if compiled else flex_attention
        )
    return _FLEX_ATTENTION[key]

_FLEX_BLOCK_SIZE = 128

# Backward-kernel block sizes to fall back through when inductor's own choice
# does not fit in shared memory. `None` means "let inductor autotune", which is
# what we want wherever it works -- the forward kernel is never the one that
# runs out, so shrinking is only ever applied to the backward blocks.
#
# 32x32 is not a compromise: measured on SM120 at head_dim=256, T=4096, it is
# bit-identical to the larger tiles and *faster* than either config that also
# fits (7.9 ms vs 19.9 ms for 32x64 and 20.2 ms for 16x64). Large head_dim
# wants small tiles here.
_FLEX_BWD_BLOCKS = (
    None,
    {"BLOCK_M1": 32, "BLOCK_N1": 32, "BLOCK_M2": 32, "BLOCK_N2": 32},
)

def _resolve_flex_kernel_options(q, k, v, block_mask, scaling):
    """First entry of `_FLEX_BWD_BLOCKS` whose backward kernel compiles here.

    Returns the chosen dict (possibly `None`) or raises the last failure. Only
    called once per process; the answer is cached by `run_flex_attention`.

    The probe has to run the *backward*, not just the forward: the forward
    template fits everywhere, and the out-of-shared-memory failure is raised by
    `triton_tem_fused_flex_attention_backward`. Autograd is what pulls that
    kernel in, so a forward-only probe reports success and the real failure
    lands mid-training-step.

    Hence `enable_grad`, which is load-bearing rather than defensive. The first
    call into this module can land inside a `no_grad` region -- under pipeline
    parallelism it always does, because `PipelineStage` runs a shape-inference
    forward before training starts. Without it the probe's `backward()` raises
    "element 0 of tensors does not require grad", that gets recorded as a
    compile failure, and every QSA layer silently runs the eager path for the
    rest of the run.
    """
    fa = _flex_attention(compiled=True)
    # A probe on the real tensors would consume their grads; run it on small
    # detached clones with the same dtype/head_dim, which is all the kernel's
    # shared-memory footprint depends on.
    probe = [
        t.detach()[..., : 2 * _FLEX_BLOCK_SIZE, :].clone().requires_grad_(True)
        for t in (q, k, v)
    ]
    from torch.nn.attention.flex_attention import create_block_mask

    probe_mask = create_block_mask(
        lambda b, h, q_idx, kv_idx: kv_idx <= q_idx,
        B=None, H=None,
        Q_LEN=2 * _FLEX_BLOCK_SIZE, KV_LEN=2 * _FLEX_BLOCK_SIZE,
        device=q.device,
    )

    last = None
    for options in _FLEX_BWD_BLOCKS:
        try:
            with torch.enable_grad():
                out = fa(
                    *probe, block_mask=probe_mask, scale=scaling, enable_gqa=True,
                    **({"kernel_options": options} if options else {}),
                )
                out.sum().backward()
            return options
        except Exception as exc:  # noqa: BLE001 - probing for a working config
            last = exc
    raise last

def run_flex_attention(q, k, v, block_mask, scaling):
    """Compiled flex-attention where it works, eager where it does not.

    Two things make the compiled path unavailable:

    * rows shorter than the 128-token block size, which inductor has no
      template for (packed training rows are always `seq_len`, so this is
      really just small inputs and tests);
    * not enough shared memory for the generated backward kernel. At Qwen4's
      `head_dim=256` inductor's default tiles want 112 KiB, which Hopper has
      (228 KiB) and SM89/SM120 (100 KiB) do not.

    The second one is recoverable, and `_FLEX_BWD_BLOCKS` recovers it: smaller
    backward tiles fit in 100 KiB and give the same numbers. That matters a lot
    more than it sounds, because the eager path materializes the full score
    matrix -- at head_dim=256, T=8192 it is 285 ms and 39.6 GiB for a *single*
    layer against 28 ms and 0.33 GiB compiled, and the real config has 12 QSA
    layers. Eager is a correctness fallback, not a performance one.

    `_FLEX_ATTENTION["compile_failed"]` records a genuine failure so it is
    visible rather than silent.
    """
    total = q.shape[-2]
    if total >= _FLEX_BLOCK_SIZE and total % _FLEX_BLOCK_SIZE == 0:
        if not _FLEX_ATTENTION.get("compile_failed"):
            if "kernel_options" not in _FLEX_ATTENTION:
                try:
                    _FLEX_ATTENTION["kernel_options"] = _resolve_flex_kernel_options(
                        q, k, v, block_mask, scaling
                    )
                except Exception as exc:
                    _FLEX_ATTENTION["compile_failed"] = repr(exc)
                    warnings.warn(
                        "QSA: compiled flex-attention unavailable on this device "
                        f"({type(exc).__name__}), falling back to the eager path, "
                        "which materializes the full score matrix. Expect high "
                        "memory use at long sequence lengths.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
            options = _FLEX_ATTENTION.get("kernel_options")
            if not _FLEX_ATTENTION.get("compile_failed"):
                return _flex_attention(compiled=True)(
                    q, k, v, block_mask=block_mask, scale=scaling, enable_gqa=True,
                    **({"kernel_options": options} if options else {}),
                )
    return _flex_attention(compiled=False)(
        q, k, v, block_mask=block_mask, scale=scaling, enable_gqa=True
    )


class SelfAttention(nn.Module):
    def __init__(self, cfg: Qwen3_5TextConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.scaling = self.head_dim ** -0.5
        self.indexer = QSAIndexer(cfg) if cfg.use_qsa else None

    @staticmethod
    def _run_varlen_attn(q, k, v, cu_seqlens, max_seqlen):
        # the flash kernels behind `varlen_attn` take fp16/bf16 only. `bf16_compute
        # = false` runs the whole model in fp32, so fall back the way
        # `models/qwen3_vl/model.py:dispatch_varlen_attention` does.
        if q.dtype not in (torch.float16, torch.bfloat16):
            return _varlen_sdpa(q, k, v, cu_seqlens, causal=True)
        return varlen_attn(
            q, k, v,
            cu_seq_q=cu_seqlens, cu_seq_k=cu_seqlens,
            max_q=max_seqlen, max_k=max_seqlen,
            window_size=(-1, 0),  # causal
            **_gqa(q, k),
        )

    @staticmethod
    def _run_flex_attn(q, k, v, block_mask, scaling):
        return run_flex_attention(q, k, v, block_mask, scaling)

    def _qsa_block_mask(self, x, cos, sin, cu_seqlens, seg_id, total):
        selected, block_of_token, tail_start = self.indexer(
            x, cos, sin, cu_seqlens, seg_id
        )

        def mask_mod(b, h, q_idx, kv_idx):
            same_doc = seg_id[q_idx] == seg_id[kv_idx]
            causal = kv_idx <= q_idx
            chosen = selected[q_idx, block_of_token[kv_idx]] | (kv_idx >= tail_start[q_idx])
            return same_doc & causal & chosen

        return _create_block_mask()(
            mask_mod, B=None, H=None, Q_LEN=total, KV_LEN=total, device=x.device,
        )

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
    ) -> torch.Tensor:
        total = x.shape[1]
        input_shape = x.shape[:-1]

        block_mask = None
        if self.indexer is not None:
            # ponytail: `seg_id` is recomputed here rather than threaded down
            # from `LanguageModel.forward`. It is a bucketize over cu_seqlens,
            # far cheaper than the plumbing through both decoder-layer classes,
            # and it keeps this an attention-local change. Thread it if the
            # per-layer cost ever shows up in a profile.
            seg_id = _segment_ids(cu_seqlens, total)
            block_mask = self._qsa_block_mask(x, cos, sin, cu_seqlens, seg_id, total)

        q, gate = torch.chunk(self.q_proj(x).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(*input_shape, -1)

        q = q.view(1, total, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(1, total, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(1, total, self.num_kv_heads, self.head_dim)

        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        # Unwrap DTensors before RoPE so apply_rope receives plain tensors and
        # avoids the DTensor.from_local path that can't run in compiled code.
        (q, k, v), wrap = _dtensor_unwrap(q, k, v)
        q, k = apply_rope(q, k, cos, sin)

        # `-1`, not `self.num_heads`: `_dtensor_unwrap` above hands back the LOCAL
        # shard, which under TP holds `num_heads // tp_size` heads. Naming the
        # global count here made every TP run die with
        # `shape '[10240, 16, 256]' is invalid for input of size 10485760`.
        # Same idiom as the `q_proj` view further up.
        if block_mask is not None:
            # flex wants (B, H, S, D), which is the layout `apply_rope` left
            # behind -- no reshape to varlen's (S, H, D) and back.
            out = SelfAttention._run_flex_attn(q, k, v, block_mask, self.scaling)
            out = _dtensor_rewrap(out, wrap)
            # `-1` for the same reason as the varlen branch below: under TP this
            # is the local shard's head count, not the global one.
            out = out.transpose(1, 2).reshape(1, total, -1)
        else:
            q = q.transpose(1, 2).reshape(total, -1, self.head_dim).contiguous()
            k = k.transpose(1, 2).reshape(total, -1, self.head_dim).contiguous()
            v = v.transpose(1, 2).reshape(total, -1, self.head_dim).contiguous()

            out = SelfAttention._run_varlen_attn(q, k, v, cu_seqlens, max_seqlen)
            out = _dtensor_rewrap(out, wrap)

            out = out.reshape(1, total, self.num_heads * self.head_dim)

        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen3_5TextConfig, **kwargs):
        super().__init__()
        self.n_key_heads = cfg.linear_num_key_heads
        self.n_value_heads = cfg.linear_num_value_heads
        self.key_head_dim = cfg.linear_key_head_dim
        self.value_head_dim = cfg.linear_value_head_dim
        self.conv_kernel_size = cfg.linear_conv_kernel_dim

        dim = cfg.hidden_size

        key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
        value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
        conv_dim = key_dim * 2 + value_dim

        self.in_proj_qkv = nn.Linear(dim, conv_dim, bias=False)
        self.in_proj_z = nn.Linear(dim, value_dim, bias=False)
        self.in_proj_a = nn.Linear(dim, cfg.linear_num_value_heads, bias=False)
        self.in_proj_b = nn.Linear(dim, cfg.linear_num_value_heads, bias=False)

        self.conv1d = nn.Conv1d(
            in_channels=conv_dim,
            out_channels=conv_dim,
            bias=False,
            kernel_size=cfg.linear_conv_kernel_dim,
            groups=conv_dim,  # depthwise
            padding=0,  # causal padding applied manually in forward
        )

        self.A_log = nn.Parameter(torch.zeros(cfg.linear_num_value_heads))
        self.dt_bias = nn.Parameter(torch.ones(cfg.linear_num_value_heads))

        self.norm = RMSNormGated(cfg.linear_value_head_dim, eps=cfg.rms_norm_eps)
        self.out_proj = nn.Linear(value_dim, dim, bias=False)

    @staticmethod
    def _run_conv1d(x, weight, bias, seq_idx):
        return _ops.dispatch_causal_conv1d(x, weight, bias, seq_idx)

    @staticmethod
    def _run_gated_delta_rule(q, k, v, g, beta, cu_seqlens):
        return _ops.dispatch_gated_delta_rule(q, k, v, g, beta, cu_seqlens)

    def forward(self, x: torch.Tensor, cu_seqlens, **kwargs) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.in_proj_qkv(x)  # (B, L, conv_dim) — channel-last in memory
        z = self.in_proj_z(x)
        a = self.in_proj_a(x)
        b = self.in_proj_b(x)
        (qkv, z, a, b), wrap = _dtensor_unwrap(qkv, z, a, b)

        # Per-token segment index for causal_conv1d_fn's packed mode. Using
        # bucketize keeps this graph-traceable with no host sync.
        seq_idx = torch.bucketize(
            torch.arange(L, device=qkv.device), cu_seqlens[1:-1], right=True
        ).to(torch.int32).unsqueeze(0).expand(B, -1).contiguous()

        # Fused causal-conv1d + SiLU. Triton kernel → isolated behind disable.
        mixed_qkv = GatedDeltaNet._run_conv1d(
            qkv.transpose(1, 2),
            _local(self.conv1d.weight).squeeze(1),
            _local(self.conv1d.bias),
            seq_idx,
        ).transpose(1, 2)  # (B, L, conv_dim)

        # Split into q, k, v and reshape to (B, L, H, D)
        key_dim = self.n_key_heads * self.key_head_dim
        value_dim = self.n_value_heads * self.value_head_dim
        q, k, v = mixed_qkv.split([key_dim, key_dim, value_dim], dim=-1)
        q = q.view(B, L, self.n_key_heads, self.key_head_dim)
        k = k.view(B, L, self.n_key_heads, self.key_head_dim)
        v = v.view(B, L, self.n_value_heads, self.value_head_dim)

        # Grouped heads: repeat q, k to match n_value_heads.
        repeat = self.n_value_heads // self.n_key_heads
        if repeat > 1:
            q = q.repeat_interleave(repeat, dim=2)
            k = k.repeat_interleave(repeat, dim=2)

        # Log-decay (g) and update weight (beta). A_log/dt_bias may be Replicate
        # DTensors under TP — _local() is now compile-friendly (no-op for plain tensors).
        g = -torch.exp(_local(self.A_log).float()) * F.softplus(a.float() + _local(self.dt_bias))
        beta = torch.sigmoid(b)

        # Gated delta rule in (B, L, H, D) layout. Triton kernel → isolated behind disable.
        output = GatedDeltaNet._run_gated_delta_rule(q, k, v, g, beta, cu_seqlens)

        # Gated norm (Triton inside RMSNormGated, already DTensor-safe).
        z = z.view(B, L, self.n_value_heads, self.value_head_dim)
        output = self.norm(output, z)
        return self.out_proj(_dtensor_rewrap(output.reshape(B, L, -1), wrap))

class MLP(nn.Module):
    def __init__(self, cfg: Qwen3_5TextConfig):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class _DecoderLayerBase(nn.Module):
    """Construction shared by both layer types.

    `forward` lives on the subclasses, not here, and that is the whole point.
    Dynamo keys its cache on the *code object*: one shared `forward` gives both
    layer types a single cache, and the `self.self_attn if ... else
    self.linear_attn` branch then guards on `self._modules['self_attn']` --
    a KeyError for every linear-attention layer, so every alternation between
    the two types is a cache miss.

    With 24 linear and 8 full-attention layers that was the last remaining
    recompile driver after the vision tower was fixed: tlparse on jobs 1781695 /
    1781696 reported `models/qwen3_5/model.py:352`,
    `last reason: KeyError on self._modules['self_attn']`, 32 recompiles and
    then eager fallback. Two code objects means two caches, each with a stable
    guard set, and no branch to guard at all.
    """

    def __init__(self, cfg: Qwen3_5TextConfig, layer_type: str):
        super().__init__()
        self.layer_type = layer_type
        if self.layer_type == "full_attention":
            self.self_attn = SelfAttention(cfg)
        else:
            self.linear_attn = GatedDeltaNet(cfg)

        self.mlp = MLP(cfg)
        self.input_layernorm = OffsetRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = OffsetRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def _mlp_block(self, x):
        return x + self.mlp(self.post_attention_layernorm(x))

class FullAttentionDecoderLayer(_DecoderLayerBase):
    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        x = x + self.self_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return self._mlp_block(x)

class LinearAttentionDecoderLayer(_DecoderLayerBase):
    def forward(self, x, cos, sin, cu_seqlens, max_seqlen):
        # GatedDeltaNet ignores cos/sin; they are passed for a uniform signature
        x = x + self.linear_attn(
            self.input_layernorm(x),
            cos=cos,
            sin=sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        return self._mlp_block(x)

def DecoderLayer(cfg: Qwen3_5TextConfig, layer_type: str) -> _DecoderLayerBase:
    """Pick the layer class for `layer_type`.

    A factory rather than a class so the existing call sites and tests are
    unchanged. State-dict keys are unaffected: the submodule names are identical
    either way, and nothing in the repo does `isinstance(x, DecoderLayer)` --
    `apply_tp` and `compile_model` both branch on
    `hasattr(block, "self_attn")`, which still holds.
    """
    cls = (
        FullAttentionDecoderLayer
        if layer_type == "full_attention"
        else LinearAttentionDecoderLayer
    )
    return cls(cfg, layer_type)

class LanguageModel(nn.Module):
    """HF name: `model.language_model`."""

    def __init__(self, cfg: Qwen3_5TextConfig):
        super().__init__()
        self.cfg = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        layers = []
        for layer_id in range(cfg.num_hidden_layers):
            is_full = (layer_id + 1) % cfg.full_attention_interval == 0
            layer_type = "full_attention" if is_full else "linear_attention"
            layers.append(DecoderLayer(cfg, layer_type, ))
        self.layers = nn.ModuleList(layers)

        self.norm = OffsetRMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        visual_pos_masks: torch.Tensor | None = None,
        deepstack_visual_embeds: list[torch.Tensor] | None = None,
    ) -> torch.Tensor:
        x = inputs_embeds
        for i, layer in enumerate(self.layers):
            x = layer(x, cos, sin, cu_seqlens, max_seqlen)
            if deepstack_visual_embeds is not None and i < len(deepstack_visual_embeds):
                x = x.clone()
                x[visual_pos_masks] = (
                    x[visual_pos_masks] + deepstack_visual_embeds[i].to(x.dtype)
                )
        return self.norm(x)

class VisionPatchEmbed(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.patch_size = cfg.patch_size
        self.temporal_patch_size = cfg.temporal_patch_size
        self.in_channels = cfg.in_channels
        self.embed_dim = cfg.hidden_size
        kernel = [self.temporal_patch_size, self.patch_size, self.patch_size]
        self.proj = nn.Conv3d(
            self.in_channels, self.embed_dim, kernel_size=kernel, stride=kernel, bias=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        target_dtype = self.proj.weight.dtype
        x = x.view(-1, self.in_channels, self.temporal_patch_size, self.patch_size, self.patch_size)
        return self.proj(x.to(dtype=target_dtype)).view(-1, self.embed_dim)

class VisionMLP(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.linear_fc1 = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=True)
        self.linear_fc2 = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=True)
        if cfg.hidden_act == "gelu_pytorch_tanh":
            self.act_fn = nn.GELU(approximate="tanh")
        elif cfg.hidden_act == "gelu":
            self.act_fn = nn.GELU()
        elif cfg.hidden_act == "silu":
            self.act_fn = nn.SiLU()
        else:
            raise ValueError(f"Unsupported vision hidden_act: {cfg.hidden_act}")

    def forward(self, x):
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

class VisionRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seqlen: int) -> torch.Tensor:
        seq = torch.arange(seqlen, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        return torch.outer(seq, self.inv_freq)  # (seqlen, dim/2)

class VisionAttention(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.dim = cfg.hidden_size
        self.num_heads = cfg.num_heads
        self.head_dim = self.dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(self.dim, self.dim * 3, bias=True)
        self.proj = nn.Linear(self.dim, self.dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        S = hidden_states.shape[0]
        q, k, v = (
            self.qkv(hidden_states)
            .reshape(S, 3, self.num_heads, self.head_dim)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )  # each (S, H, D) — already the layout varlen_attn wants
        cos, sin = position_embeddings
        q, k = apply_rope_vision(q, k, cos, sin)

        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

        (q, k, v), wrap = _dtensor_unwrap(q, k, v)
        out = varlen_attn(
            q, k, v,
            cu_seq_q=cu_seqlens, cu_seq_k=cu_seqlens,
            max_q=max_seqlen, max_k=max_seqlen,
            window_size=(-1, -1),  # non-causal
            **_gqa(q, k),
        )
        out = _dtensor_rewrap(out, wrap)
        return self.proj(out.reshape(S, self.dim))

class VisionBlock(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.norm2 = nn.LayerNorm(cfg.hidden_size, eps=1e-6)
        self.attn = VisionAttention(cfg)
        self.mlp = VisionMLP(cfg)

    def forward(self, x, cu_seqlens, max_seqlen, position_embeddings):
        x = x + self.attn(self.norm1(x), cu_seqlens, max_seqlen, position_embeddings)
        x = x + self.mlp(self.norm2(x))
        return x

class VisionPatchMerger(nn.Module):
    def __init__(self, cfg: Qwen3_5VisionConfig, use_postshuffle_norm: bool = False):
        super().__init__()
        self.hidden_size = cfg.hidden_size * (cfg.spatial_merge_size ** 2)
        self.use_postshuffle_norm = use_postshuffle_norm
        norm_dim = self.hidden_size if use_postshuffle_norm else cfg.hidden_size
        self.norm = nn.LayerNorm(norm_dim, eps=1e-6)
        self.linear_fc1 = nn.Linear(self.hidden_size, self.hidden_size)
        self.act_fn = nn.GELU()
        self.linear_fc2 = nn.Linear(self.hidden_size, cfg.out_hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_postshuffle_norm:
            x = self.norm(x.view(-1, self.hidden_size))
        else:
            x = self.norm(x).view(-1, self.hidden_size)
        return self.linear_fc2(self.act_fn(self.linear_fc1(x)))

class VisionModel(nn.Module):
    """HF name: `model.visual`. Mirrors `Qwen3VLVisionModel` exactly."""

    def __init__(self, cfg: Qwen3_5VisionConfig):
        super().__init__()
        self.cfg = cfg
        self.spatial_merge_size = cfg.spatial_merge_size
        self.patch_size = cfg.patch_size
        self.num_grid_per_side = int(cfg.num_position_embeddings ** 0.5)

        self.patch_embed = VisionPatchEmbed(cfg)
        self.pos_embed = nn.Embedding(cfg.num_position_embeddings, cfg.hidden_size)

        head_dim = cfg.hidden_size // cfg.num_heads
        self.rotary_pos_emb = VisionRotaryEmbedding(head_dim // 2)

        self.blocks = nn.ModuleList([VisionBlock(cfg) for _ in range(cfg.depth)])
        self.merger = VisionPatchMerger(cfg, use_postshuffle_norm=False)
        self.deepstack_visual_indexes = list(cfg.deepstack_visual_indexes)
        self.deepstack_merger_list = nn.ModuleList(
            [VisionPatchMerger(cfg, use_postshuffle_norm=True)
             for _ in range(len(self.deepstack_visual_indexes))]
        )

    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge = self.spatial_merge_size
        grid_list = grid_thw.tolist()
        max_hw = max(max(h, w) for _, h, w in grid_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim/2)
        device = freq_table.device

        total = sum(t * h * w for t, h, w in grid_list)
        pos_ids = torch.empty((total, 2), dtype=torch.long, device=device)
        offset = 0
        for t, h, w in grid_list:
            mh, mw = h // merge, w // merge
            block_rows = torch.arange(mh, device=device)
            block_cols = torch.arange(mw, device=device)
            intra_r = torch.arange(merge, device=device)
            intra_c = torch.arange(merge, device=device)
            row_idx = block_rows[:, None, None, None] * merge + intra_r[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge + intra_c[None, None, None, :]
            row_idx = row_idx.expand(mh, mw, merge, merge).reshape(-1)
            col_idx = col_idx.expand(mh, mw, merge, merge).reshape(-1)
            coords = torch.stack((row_idx, col_idx), dim=-1)
            if t > 1:
                coords = coords.repeat(t, 1)
            n = coords.shape[0]
            pos_ids[offset : offset + n] = coords
            offset += n

        emb = freq_table[pos_ids]  # (total, 2, dim/2)
        return emb.flatten(1)  # (total, dim)

    def fast_pos_embed_interpolate(self, grid_thw: torch.Tensor) -> torch.Tensor:
        grid_list = grid_thw.tolist()
        grid_ts = [r[0] for r in grid_list]
        grid_hs = [r[1] for r in grid_list]
        grid_ws = [r[2] for r in grid_list]
        device = self.pos_embed.weight.device

        idx_list: list[list[int]] = [[], [], [], []]
        weight_list: list[list[float]] = [[], [], [], []]

        for _t, h, w in grid_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w)
            h_floor = h_idxs.int()
            w_floor = w_idxs.int()
            h_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            dh = h_idxs - h_floor
            dw = w_idxs - w_floor
            base_h = h_floor * self.num_grid_per_side
            base_h_ceil = h_ceil * self.num_grid_per_side
            indices = [
                (base_h[None].T + w_floor[None]).flatten(),
                (base_h[None].T + w_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_floor[None]).flatten(),
                (base_h_ceil[None].T + w_ceil[None]).flatten(),
            ]
            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]
            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_t = torch.tensor(idx_list, dtype=torch.long, device=device)
        wt = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pe = self.pos_embed(idx_t) * wt[:, :, None]
        patch_pe = pe[0] + pe[1] + pe[2] + pe[3]
        chunks = patch_pe.split([h * w for h, w in zip(grid_hs, grid_ws)])

        merge = self.spatial_merge_size
        out = []
        for pe_chunk, t, h, w in zip(chunks, grid_ts, grid_hs, grid_ws):
            pe_chunk = pe_chunk.repeat(t, 1)
            pe_chunk = (
                pe_chunk.view(t, h // merge, merge, w // merge, merge, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            out.append(pe_chunk)
        return torch.cat(out)

    def forward(
        self,
        hidden_states: torch.Tensor,
        grid_thw: torch.Tensor,
        max_seqlen: int | None = None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Returns (merged_hidden_states, deepstack_features)."""
        torch._dynamo.maybe_mark_dynamic(hidden_states, 0)
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seg_lens = torch.repeat_interleave(
            grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]
        )
        cu = F.pad(seg_lens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
        torch._dynamo.maybe_mark_dynamic(cu, 0)  # one entry per image in the batch
        if max_seqlen is None:
            # Same story as the text tower: varlen attention needs a Python int,
            # so the only way to avoid a sync is to compute it off the GPU. The
            # trainer passes `vision_max_seqlen` from the host-side grid.
            max_seqlen = _round_max_seqlen(int(seg_lens.max().item()))

        deepstack: list[torch.Tensor] = []
        for i, blk in enumerate(self.blocks):
            hidden_states = blk(hidden_states, cu, max_seqlen, position_embeddings)
            if i in self.deepstack_visual_indexes:
                merger = self.deepstack_merger_list[self.deepstack_visual_indexes.index(i)]
                deepstack.append(merger(hidden_states))

        merged = self.merger(hidden_states)
        return merged, deepstack

class Qwen3_5Inner(nn.Module):
    """HF name: `model`. Groups `language_model` and `visual`.
    This is only used to match the state keys. """

    def __init__(self, cfg: Qwen3_5Config):
        super().__init__()
        self.language_model = LanguageModel(cfg.text)
        self.visual = VisionModel(cfg.vision)

class Qwen3_5ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen3_5Config, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen3_5Inner(cfg)
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

        # Text rope: store only inv_freq; cos/sin are computed per-forward via
        # MRoPE (3D position ids). For text-only inputs the 3 axes share the
        # same arange, which collapses to plain 1D rope.
        head_dim = cfg.text.head_dim
        partial = cfg.text.rope_parameters.get('partial_rotary_factor', 1.0)
        rope_dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            cfg.text.rope_parameters['rope_theta'] ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        self.register_buffer("text_inv_freq", inv_freq, persistent=False)

        cfg_rope_section = cfg.text.rope_parameters['mrope_section']
        self.mrope_section = list(cfg_rope_section)

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 3D MRoPE position_ids over a packed row.

        Args:
            input_ids: (1, total) packed token ids.
            cu_seqlens: (N+1,) int32 cumulative offsets (starts at 0).
            image_grid_thw, video_grid_thw: per-image/video (T,H,W) grids.

        Positions reset to 0 at each packed-sample boundary. Within a segment,
        text runs are `arange`, image/video runs are 3D per HF's algorithm.
        """
        image_id = self.cfg.image_token_id
        video_id = self.cfg.video_token_id
        spatial = self.cfg.vision.spatial_merge_size

        if video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(video_grid_thw, video_grid_thw[:, 0], dim=0)
            video_grid_thw[:, 0] = 1

        _, S = input_ids.shape
        device = input_ids.device
        mm_type = torch.zeros(S, dtype=torch.int64, device=device)
        mm_type[input_ids[0] == image_id] = 1
        mm_type[input_ids[0] == video_id] = 2
        types_all = mm_type.tolist()

        image_iter = iter(image_grid_thw) if image_grid_thw is not None else None
        video_iter = iter(video_grid_thw) if video_grid_thw is not None else None

        bounds = cu_seqlens.tolist()
        out = torch.zeros(3, 1, S, dtype=torch.int64, device=device)

        for start, end in zip(bounds[:-1], bounds[1:]):
            if start == end:
                continue
            types_seg = types_all[start:end]
            pos_list: list[torch.Tensor] = []
            current = 0
            j = 0
            while j < len(types_seg):
                k = j
                while k < len(types_seg) and types_seg[k] == types_seg[j]:
                    k += 1
                key = types_seg[j]
                length = k - j
                if key == 0:
                    p = torch.arange(length, device=device).view(1, -1).expand(3, -1) + current
                    current += length
                else:
                    grid = next(image_iter if key == 1 else video_iter)
                    t, h, w = int(grid[0]), int(grid[1]), int(grid[2])
                    llm_h, llm_w, llm_t = h // spatial, w // spatial, t
                    n = llm_t * llm_h * llm_w
                    pw = torch.arange(current, current + llm_w, device=device).repeat(llm_h * llm_t)
                    ph = torch.arange(current, current + llm_h, device=device).repeat_interleave(
                        llm_w * llm_t
                    )
                    pt = torch.full((n,), current, device=device, dtype=torch.int64)
                    p = torch.stack([pt, ph, pw], dim=0)
                    current += max(llm_h, llm_w)
                pos_list.append(p)
                j = k
            out[:, 0, start:end] = torch.cat(pos_list, dim=1)
        return out

    def _compute_cos_sin(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """position_ids: (B, S) or (3, B, S) → cos/sin of shape (B, S, D)."""
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        return mrope_cos_sin(self.text_inv_freq, position_ids, self.mrope_section)

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        *,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        **kwargs,
    ) -> "CausalLMOutput | torch.Tensor":
        """Varlen-only forward.

        Expected layout: `input_ids` / `inputs_embeds` is a single packed row
        `(1, total)` / `(1, total, H)`. `attention_mask` is interpreted as
        `cu_seqlens`: a 1D int32 tensor of cumulative offsets starting at 0
        (same tensor consumed by `torch.nn.attention.varlen.varlen_attn`).
        If `attention_mask` is None, the whole row is treated as one sample.
        """
        assert (input_ids is None) ^ (inputs_embeds is None)
        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids is not None:
            assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
                f"varlen expects packed (1, total), got {tuple(input_ids.shape)}"
            )

        _t = _SectionTimer()

        if inputs_embeds is None:
            inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1
        total = inputs_embeds.shape[1]
        device = inputs_embeds.device

        if attention_mask is None:
            cu_seqlens = torch.tensor([0, total], device=device, dtype=torch.int32)
        else:
            assert attention_mask.dim() == 1, (
                "attention_mask must be cu_seqlens: 1D int32, starts at 0, "
                f"ends at total; got {attention_mask.dim()}D"
            )
            torch._assert_async(attention_mask[0] == 0)
            torch._assert_async(attention_mask[-1] == total)
            cu_seqlens = attention_mask.to(torch.int32)

        if max_seqlen is None:
            max_seqlen = _round_max_seqlen(
                int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())
            )

        # avoids the first compilation with a static shape
        torch._dynamo.maybe_mark_dynamic(cu_seqlens, 0)

        _t.mark("prologue")

        visual_pos_masks: torch.Tensor | None = None
        deepstack_visual_embeds: list[torch.Tensor] | None = None

        if pixel_values is not None:
            assert image_grid_thw is not None
            torch._dynamo.maybe_mark_dynamic(pixel_values, 0)
            merged, deepstack = self.model.visual(
                pixel_values, image_grid_thw, max_seqlen=kwargs.get("vision_max_seqlen")
            )
            merged = merged.to(inputs_embeds.dtype)
            image_mask = input_ids == self.cfg.image_token_id

            torch._assert_async(image_mask.sum() == merged.shape[0])
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), merged
            )
            visual_pos_masks = image_mask
            deepstack_visual_embeds = deepstack

        if pixel_values_videos is not None:
            assert video_grid_thw is not None
            merged_v, deepstack_v = self.model.visual(pixel_values_videos, video_grid_thw)
            merged_v = merged_v.to(inputs_embeds.dtype)
            video_mask = input_ids == self.cfg.video_token_id
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.unsqueeze(-1).expand_as(inputs_embeds), merged_v
            )
            if visual_pos_masks is None:
                visual_pos_masks = video_mask
                deepstack_visual_embeds = deepstack_v
            else:
                combined = visual_pos_masks | video_mask
                image_only = visual_pos_masks[combined]
                video_only = video_mask[combined]
                merged_ds = []
                for img_ds, vid_ds in zip(deepstack_visual_embeds, deepstack_v):
                    e = img_ds.new_zeros(combined.sum().item(), img_ds.shape[-1])
                    e[image_only] = img_ds
                    e[video_only] = vid_ds
                    merged_ds.append(e)
                visual_pos_masks = combined
                deepstack_visual_embeds = merged_ds

        _t.mark("visual")

        if position_ids is None:
            if image_grid_thw is not None or video_grid_thw is not None:
                assert input_ids is not None, "need input_ids to compute 3D MRoPE positions"
                position_ids = self.get_rope_index(
                    input_ids,
                    cu_seqlens=cu_seqlens,
                    image_grid_thw=image_grid_thw,
                    video_grid_thw=video_grid_thw,
                )
            else:
                pos = packed_positions(cu_seqlens, total)
                position_ids = pos.view(1, 1, -1).expand(3, 1, -1)

        cos, sin = self._compute_cos_sin(position_ids)
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        _t.mark("rope")

        h = self.model.language_model(
            inputs_embeds,
            cos,
            sin,
            cu_seqlens,
            max_seqlen,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        _t.mark("layers")

        logits = self.lm_head(h)
        _t.mark("lm_head")

        if labels is None:
            return logits
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        loss = causal_lm_loss(logits, labels)
        _t.mark("loss")
        return CausalLMOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(
        cls,
        snapshot_dir: str | Path,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        *,
        load_vision: bool = True,
        load_weights: bool = True,
    ) -> "Qwen3_5ForCausalLM":
        snapshot_dir = Path(snapshot_dir)
        cfg = Qwen3_5Config.from_json(snapshot_dir / "config.json")
        if dtype is None:
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[cfg.torch_dtype]

        with torch.device("meta"):
            model = cls(cfg)
        model = model.to_empty(device=device).to(dtype=dtype)

        if load_weights:
            load_safetensors_into(
                model,
                snapshot_dir,
                device=device,
                dtype=dtype,
                load_vision=load_vision,
            )
        else:
            # `to_empty` leaves uninitialised memory, which is NaN often enough
            # to look like a training bug. The trainer's `init_qwen35` overwrites
            # the decoder and projector, but not every buffer, so zero first.
            # No logging here: this module has no logger on purpose (it stays
            # independent of `train`); the caller reports the no-weights path.
            for p in model.parameters():
                p.detach().zero_()
            for b in model.buffers():
                if b.is_floating_point():
                    b.detach().zero_()

        # `to_empty` above re-materializes every parameter and breaks the
        # tie established in `__init__`. Re-tie here so `lm_head` (absent
        # from checkpoints when tied) shares storage with the embedding.
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.language_model.embed_tokens.weight

        # `to_empty` also wipes non-persistent buffers. Recompute the vision
        # rotary `inv_freq` (it's not in the safetensors).
        if load_vision:
            head_dim_v = cfg.vision.hidden_size // cfg.vision.num_heads
            rdim = head_dim_v // 2
            inv_freq_v = 1.0 / (
                10000.0 ** (torch.arange(0, rdim, 2, dtype=torch.float32, device=device) / rdim)
            )
            model.model.visual.rotary_pos_emb.inv_freq = inv_freq_v

        # Recompute text inv_freq (non-persistent buffer wiped by `to_empty`).
        head_dim = cfg.text.head_dim
        partial = cfg.text.rope_parameters.get('partial_rotary_factor', 1.0)
        rope_dim = int(head_dim * partial)
        text_inv = 1.0 / (
            cfg.text.rope_parameters['rope_theta']
            ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
        )
        model.text_inv_freq = text_inv
        return model, cfg
