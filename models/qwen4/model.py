from __future__ import annotations

import math
import warnings
from pathlib import Path

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn

from models.qwen4.config import (
    Qwen4Config, Qwen4TextConfig, Qwen4VisionConfig
)
from models.qwen4.utils import (
    _dtensor_unwrap,
    _dtensor_rewrap,
    _local,
    iter_layers,
    CausalLMOutput,
    causal_lm_loss,
    apply_rope,
    mrope_cos_sin,
    apply_rope_vision,
    load_safetensors_into,
)
from models.qwen4 import compile_ops as _ops

# torch 2.11's `varlen_attn` silently accepted a q/k head-count mismatch. 2.14
# added an `enable_gqa` parameter and raises `ValueError` unless it is set, so
# every GQA model broke on the upgrade. Pass it only when the installed torch
# knows the keyword, so the same source still runs on 2.11.
_VARLEN_HAS_GQA = "enable_gqa" in inspect.signature(varlen_attn).parameters


def _gqa(q, k) -> dict:
    """`enable_gqa=True` when q and k disagree on head count, else nothing."""
    if _VARLEN_HAS_GQA and q.shape[-2] != k.shape[-2]:
        return {"enable_gqa": True}
    return {}


# --------------------------------------------------------------------------
# norms
# --------------------------------------------------------------------------

class RMSNormGated(nn.Module):
    """Gated RMSNorm: ``weight * norm(x) * act(gate)``.

    ``activation`` is Qwen4's ``output_gate_type`` — the released checkpoint
    uses ``sigmoid``, where Qwen3.5 used ``silu``.
    """

    def __init__(self, dim: int, eps: float = 1e-6, activation: str = "silu"):
        super().__init__()
        self.eps = eps
        self.activation = "swish" if activation == "silu" else activation
        self.weight = nn.Parameter(torch.ones(dim))

    @staticmethod
    def _run_fla_rms_norm_gated(hs, gate, weight, eps, activation):
        return _ops.rms_norm_gated(hs, gate, weight, eps, activation)

    def _torch_ref(self, hs, gate, weight):
        dtype = hs.dtype
        h = hs.float()
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = weight * h.to(dtype)
        act = torch.sigmoid if self.activation == "sigmoid" else F.silu
        return (h * act(gate.float())).to(dtype)

    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        orig_shape = hidden_states.shape
        D = orig_shape[-1]
        (hs_local, gate_local), wrap = _dtensor_unwrap(hidden_states, gate)
        if hs_local.is_cuda:
            out = RMSNormGated._run_fla_rms_norm_gated(
                hs_local.reshape(-1, D),
                gate_local.reshape(-1, D),
                _local(self.weight),
                self.eps,
                self.activation,
            )
        else:
            # FLA is a Triton kernel; CPU parity tests take the torch path.
            out = self._torch_ref(
                hs_local.reshape(-1, D), gate_local.reshape(-1, D), _local(self.weight)
            )
        return _dtensor_rewrap(out.reshape(orig_shape), wrap)

class OffsetRMSNorm(nn.Module):
    """RMSNorm with offset: ``(1 + weight) * norm(x)``, weight init to zeros.

    ``group_size`` reproduces HF ``Qwen4ExpTextRMSNorm``: the statistic is taken
    over groups of ``group_size`` channels (used on the ``hc_count * hidden``
    hyper-connection vectors), while ``weight`` spans the full width.
    """

    def __init__(self, dim: int, group_size: int | None = None, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.group_size = group_size
        if group_size is not None and dim % group_size != 0:
            raise ValueError(f"dim ({dim}) must be divisible by group_size ({group_size}).")
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group_size is None:
            # F.rms_norm handles the fp32 upcast internally and lets inductor fuse.
            return F.rms_norm(x, self.weight.shape, 1.0 + self.weight, self.eps)
        h = x.float().unflatten(-1, (-1, self.group_size))
        h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.eps)
        h = h.flatten(-2) * (1.0 + self.weight.float())
        return h.type_as(x)

# --------------------------------------------------------------------------
# packed-sequence helpers
# --------------------------------------------------------------------------

class _AllReduceTP(torch.autograd.Function):
    """Sum-reduce a partial activation across the TP group.

    The routed-expert forward runs on local tensors (`grouped_mm`, `index_add`),
    which DTensor cannot see through, so the reduction the rowwise-sharded
    `down_proj` implies has to be made explicit. The gradient of an all-reduce
    sum is the identity, since every rank already receives the full upstream
    gradient.
    """

    @staticmethod
    def forward(ctx, x, group):
        import torch.distributed as dist

        # Reduce into a fresh buffer rather than in place: mutating the input
        # of a custom Function upsets autograd's view tracking once the MoE
        # block is compiled ("its base ... has been modified inplace").
        out = x.clone(memory_format=torch.contiguous_format)
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        return grad_out, None


def _all_reduce_tp(x: torch.Tensor, group) -> torch.Tensor:
    return _AllReduceTP.apply(x, group)


class _ReduceGradTP(torch.autograd.Function):
    """Identity forward, sum-reduce the gradient across the TP group.

    Counterpart of :class:`_AllReduceTP` on the input side. Every rank runs
    every expert over the *same* replicated tokens but only owns a slice of the
    intermediate dimension, so each rank's gradient w.r.t. the expert input is a
    partial sum. Without this the modules feeding the MoE block (the router and
    the mlp hyper-connection) see 1/tp_size of their gradient.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        # A clone, not `x` or a view of it: anything aliasing the input makes
        # autograd track a view created inside a custom Function, which then
        # errors once the MoE block is compiled.
        return x.clone()

    @staticmethod
    def backward(ctx, grad_out):
        import torch.distributed as dist

        grad_out = grad_out.contiguous()
        dist.all_reduce(grad_out, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad_out, None


def _reduce_grad_tp(x: torch.Tensor, group) -> torch.Tensor:
    return _ReduceGradTP.apply(x, group)


def _segment_ids(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """Per-token document index for a packed row. Graph-traceable, no host sync."""
    return torch.bucketize(
        torch.arange(total, device=cu_seqlens.device), cu_seqlens[1:-1], right=True
    )

# --------------------------------------------------------------------------
# attention
# --------------------------------------------------------------------------

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

    def __init__(self, cfg: Qwen4TextConfig):
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

class SelfAttention(nn.Module):
    """Gated full attention. Sparse (QSA) when the config carries ``indexer_*``."""

    def __init__(self, cfg: Qwen4TextConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.n_rep = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim * 2, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

        self.q_norm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)
        self.k_norm = OffsetRMSNorm(self.head_dim, eps=cfg.rms_norm_eps)

        self.indexer = QSAIndexer(cfg) if cfg.use_qsa else None

    @staticmethod
    def _run_varlen_attn(q, k, v, cu_seqlens, max_seqlen):
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
        seg_id: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = x.shape[1]
        input_shape = x.shape[:-1]

        block_mask = None
        if self.indexer is not None:
            assert seg_id is not None, "QSA needs per-token document ids"
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

        if block_mask is not None:
            out = SelfAttention._run_flex_attn(q, k, v, block_mask, self.scaling)
            out = _dtensor_rewrap(out, wrap)
            out = out.transpose(1, 2).reshape(1, total, self.num_heads * self.head_dim)
        else:
            q = q.transpose(1, 2).reshape(total, self.num_heads, self.head_dim).contiguous()
            k = k.transpose(1, 2).reshape(total, self.num_kv_heads, self.head_dim).contiguous()
            v = v.transpose(1, 2).reshape(total, self.num_kv_heads, self.head_dim).contiguous()
            out = SelfAttention._run_varlen_attn(q, k, v, cu_seqlens, max_seqlen)
            out = _dtensor_rewrap(out, wrap)
            out = out.reshape(1, total, self.num_heads * self.head_dim)

        out = out * torch.sigmoid(gate)
        return self.o_proj(out)

class GatedDeltaNet(nn.Module):
    def __init__(self, cfg: Qwen4TextConfig, **kwargs):
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

        self.norm = RMSNormGated(
            cfg.linear_value_head_dim, eps=cfg.rms_norm_eps, activation=cfg.output_gate_type
        )
        self.out_proj = nn.Linear(value_dim, dim, bias=False)

    @staticmethod
    def _run_conv1d(x, weight, bias, seq_idx):
        # The fused kernel is CUDA-only and ships as a separate extension that
        # may not be installed; `causal_conv1d_torch` is the plain-torch
        # equivalent. Both branches are Python-level constants by trace time.
        if x.is_cuda and _ops.causal_conv1d_available():
            return _ops.causal_conv1d(x, weight, bias, seq_idx)
        return _ops.causal_conv1d_torch(x, weight, bias, seq_idx)

    # Rebound by `set_gdn_backend`. Kept as a plain staticmethod so the choice
    # is a Python-level constant by the time `torch.compile(fullgraph=True)`
    # traces the layer, rather than a branch inside the graph.
    @staticmethod
    def _run_gated_delta_rule(q, k, v, g, beta, cu_seqlens):
        return _ops.gated_delta_rule(q, k, v, g, beta, cu_seqlens)

    def forward(self, x: torch.Tensor, cu_seqlens, seg_id=None, **kwargs) -> torch.Tensor:
        B, L, _ = x.shape
        qkv = self.in_proj_qkv(x)  # (B, L, conv_dim) - channel-last in memory
        z = self.in_proj_z(x)
        a = self.in_proj_a(x)
        b = self.in_proj_b(x)
        (qkv, z, a, b), wrap = _dtensor_unwrap(qkv, z, a, b)

        if seg_id is None:
            seg_id = _segment_ids(cu_seqlens, L)
        seq_idx = seg_id.to(torch.int32).unsqueeze(0).expand(B, -1).contiguous()

        # Fused causal-conv1d + SiLU. Triton kernel -> isolated behind disable.
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
        # DTensors under TP - _local() is compile-friendly (no-op for plain tensors).
        g = -torch.exp(_local(self.A_log).float()) * F.softplus(a.float() + _local(self.dt_bias))
        beta = torch.sigmoid(b)

        output = GatedDeltaNet._run_gated_delta_rule(q, k, v, g, beta, cu_seqlens)

        z = z.view(B, L, self.n_value_heads, self.value_head_dim)
        output = self.norm(output, z)
        return self.out_proj(_dtensor_rewrap(output.reshape(B, L, -1), wrap))

_CREATE_BLOCK_MASK: dict = {}


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


_FLEX_ATTENTION: dict = {}


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


def _run_gdn_fla(q, k, v, g, beta, cu_seqlens):
    return _ops.gated_delta_rule(q, k, v, g, beta, cu_seqlens)


def _run_gdn_flashqla(q, k, v, g, beta, cu_seqlens):
    return _ops.gated_delta_rule_qla(q, k, v, g, beta, cu_seqlens)


def set_gdn_backend(name: str = "auto") -> str:
    """Pick the kernel behind `GatedDeltaNet`, returning what was selected.

    ``flashqla``  FlashQLA (github.com/QwenLM/FlashQLA), the Qwen team's TileLang
                  kernels: 2-3x forward / 2x backward over FLA on Hopper. Needs
                  SM90/SM100/SM103 for training; SM120/121 have no backward
                  kernel and SM89 and below are unsupported entirely.
    ``fla``       the flash-linear-attention Triton kernels (the default).
    ``auto``      FlashQLA when it is importable on this device *and* ships a
                  backward kernel, otherwise FLA.

    Call this after the model is on its device: FlashQLA's architecture check
    runs against the current CUDA device at import time.
    """
    if name not in ("auto", "fla", "flashqla"):
        raise ValueError(f"unknown GDN backend: {name!r}")

    if name == "flashqla" or (name == "auto" and _ops.flashqla_has_backward()):
        if _ops.flashqla() is None:
            raise RuntimeError(
                f"FlashQLA requested but not usable here: {_ops.flashqla_unavailable_reason()}"
            )
        GatedDeltaNet._run_gated_delta_rule = staticmethod(_run_gdn_flashqla)
        return "flashqla"

    GatedDeltaNet._run_gated_delta_rule = staticmethod(_run_gdn_fla)
    return "fla"


def gdn_backend() -> str:
    fn = GatedDeltaNet._run_gated_delta_rule
    return "flashqla" if fn is _run_gdn_flashqla else "fla"


# --------------------------------------------------------------------------
# feed-forward: mixture of experts
# --------------------------------------------------------------------------

class MLP(nn.Module):
    def __init__(self, cfg: Qwen4TextConfig, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

class Experts(nn.Module):
    """Routed experts stored as stacked 3D tensors (HF layout)."""

    def __init__(self, cfg: Qwen4TextConfig):
        super().__init__()
        self.num_experts = cfg.num_experts
        self.hidden_dim = cfg.hidden_size
        self.intermediate_dim = cfg.moe_intermediate_size
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )
        # set by the TP pass when the intermediate dim is sharded, in which case
        # each rank's `down_proj` yields a partial sum over hidden
        self.tp_group = None

    def _forward_grouped(self, x, top_k_index, top_k_weights):
        tokens, k = top_k_index.shape
        flat_expert = top_k_index.reshape(-1)
        order = torch.argsort(flat_expert, stable=True)
        token_of = torch.div(order, k, rounding_mode="floor")

        counts = torch.bincount(flat_expert, minlength=self.num_experts)
        offs = counts.cumsum(0).to(torch.int32)

        # `_grouped_mm` is bf16/fp16 only. The trainer keeps master weights in
        # fp32 and relies on autocast, so cast here the way autocast casts an
        # `nn.Linear`.
        dtype = x.dtype
        xg = x[token_of]
        gu = torch._grouped_mm(
            xg, _local(self.gate_up_proj).to(dtype).transpose(-2, -1), offs=offs
        )
        gate, up = gu.chunk(2, dim=-1)
        h = F.silu(gate) * up
        out = torch._grouped_mm(
            h, _local(self.down_proj).to(dtype).transpose(-2, -1), offs=offs
        )
        out = out * top_k_weights.reshape(-1)[order].unsqueeze(-1).to(out.dtype)

        y = torch.zeros_like(x)
        return y.index_add_(0, token_of, out.to(y.dtype))

    def _forward_loop(self, x, top_k_index, top_k_weights):
        """Reference path (HF-equivalent). Any device/dtype, slow with 512 experts."""
        final = torch.zeros_like(x)
        gate_up = _local(self.gate_up_proj)
        down = _local(self.down_proj)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()
        for expert_idx in hit:
            e = expert_idx[0]
            top_k_pos, token_idx = torch.where(mask[e])
            current = x[token_idx]
            gate, up = F.linear(current, gate_up[e]).chunk(2, dim=-1)
            h = F.silu(gate) * up
            h = F.linear(h, down[e])
            h = h * top_k_weights[token_idx, top_k_pos, None]
            final.index_add_(0, token_idx, h.to(final.dtype))
        return final

    def forward(self, x, top_k_index, top_k_weights):
        if self.tp_group is not None:
            # both the token input and the router weights are replicated inputs
            # to a computation each rank only owns a slice of
            x = _reduce_grad_tp(x, self.tp_group)
            top_k_weights = _reduce_grad_tp(top_k_weights, self.tp_group)
        # Under autocast the hidden states still arrive in fp32 (the previous op
        # is an elementwise add), so dispatching on `x.dtype` alone sends every
        # training step down the Python reference loop -- slow, and not
        # `fullgraph` compilable because it iterates over a data-dependent
        # number of hit experts.
        compute_dtype = x.dtype
        if torch.is_autocast_enabled("cuda"):
            compute_dtype = torch.get_autocast_dtype("cuda")
        if x.is_cuda and compute_dtype in (torch.bfloat16, torch.float16):
            out = self._forward_grouped(
                x.to(compute_dtype), top_k_index, top_k_weights
            )
        else:
            out = self._forward_loop(x, top_k_index, top_k_weights)
        if self.tp_group is not None:
            out = _all_reduce_tp(out, self.tp_group)
        return out

class TopKRouter(nn.Module):
    def __init__(self, cfg: Qwen4TextConfig):
        super().__init__()
        self.top_k = cfg.num_experts_per_tok
        self.num_experts = cfg.num_experts
        self.norm_topk_prob = cfg.norm_topk_prob
        self.weight = nn.Parameter(torch.zeros(cfg.num_experts, cfg.hidden_size))

    def forward(self, x):
        logits = F.linear(x, self.weight)
        probs = F.softmax(logits, dtype=torch.float, dim=-1)
        top_value, top_index = torch.topk(probs, self.top_k, dim=-1)
        if self.norm_topk_prob:
            top_value = top_value / top_value.sum(dim=-1, keepdim=True)
        return logits, top_value.to(logits.dtype), top_index

class SparseMoeBlock(nn.Module):
    def __init__(self, cfg: Qwen4TextConfig):
        super().__init__()
        self.gate = TopKRouter(cfg)
        self.experts = Experts(cfg)
        self.shared_expert = MLP(cfg, cfg.shared_expert_intermediate_size)
        self.shared_expert_gate = nn.Linear(cfg.hidden_size, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, H = x.shape
        flat = x.reshape(-1, H)
        shared = self.shared_expert(flat)
        _logits, weights, index = self.gate(flat)
        out = self.experts(flat, index, weights)
        shared = torch.sigmoid(self.shared_expert_gate(flat)) * shared
        return (out + shared).reshape(B, L, H)

# --------------------------------------------------------------------------
# hyper-connections
# --------------------------------------------------------------------------

class GatedResidual(nn.Module):
    """Mixes the ``hc_count`` residual streams down to one, and (optionally)
    returns the per-stream weights used to inject the block output back."""

    def __init__(self, cfg: Qwen4TextConfig, use_combine: bool = True):
        super().__init__()
        self.hc_count = cfg.hc_count
        self.hidden_size = cfg.hidden_size
        hc_hidden = self.hc_count * self.hidden_size
        self.hc_norm = OffsetRMSNorm(hc_hidden, group_size=self.hidden_size, eps=cfg.rms_norm_eps)
        self.input_mix_weight_down = nn.Linear(hc_hidden, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, hc_hidden, bias=False)
        self.block_inject_weight = (
            nn.Linear(hc_hidden, self.hc_count, bias=False) if use_combine else None
        )

    def forward(self, hyper_input: torch.Tensor):
        if hyper_input.shape[-1] != self.hc_count * self.hidden_size:
            raise ValueError(
                f"Expected {self.hc_count * self.hidden_size} hyper-connection features, "
                f"got {hyper_input.shape[-1]}."
            )
        normed = self.hc_norm(hyper_input)
        mix = F.silu(self.input_mix_weight_down(normed) / self.hc_count)
        mix = torch.sigmoid(self.input_mix_weight_up(mix))
        mix = mix.unflatten(-1, (self.hc_count, self.hidden_size))
        mixed = (mix * normed.unflatten(-1, (self.hc_count, self.hidden_size))).mean(dim=-2)
        if self.block_inject_weight is None:
            return mixed
        inject = 2 * torch.sigmoid(self.block_inject_weight(normed) / self.hc_count)
        return mixed, hyper_input, inject

# --------------------------------------------------------------------------
# per-layer embeddings (PLE)
# --------------------------------------------------------------------------

_MASK64 = (1 << 64) - 1
_SPLITMIX_GAMMA = 0x9E3779B97F4A7C15
_SPLITMIX_M1 = 0xBF58476D1CE4E5B9
_SPLITMIX_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007

def _splitmix64(value: int) -> int:
    value = (value + _SPLITMIX_GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _SPLITMIX_M1) & _MASK64
    value = ((value ^ (value >> 27)) * _SPLITMIX_M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64

def _build_layer_multipliers(unigram_vocab_size, ngram_size, ple_layer_index, seed):
    multipliers = []
    for position in range(ngram_size):
        raw = _splitmix64(seed + _PRIME_1 * ple_layer_index + position)
        multipliers.append(raw % unigram_vocab_size)
    return torch.tensor(multipliers, dtype=torch.long)

def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    factor = 3
    while factor * factor <= value:
        if value % factor == 0:
            return False
        factor += 2
    return True

def _find_nth_prime_after(start: int, count: int) -> int:
    value = start
    found = 0
    while found < count:
        value += 1
        if _is_prime(value):
            found += 1
    return value

class NGramEmbedding(nn.Module):
    """Hashed n-gram lookup, optionally sharded across TP ranks by n-gram head.

    The table is block-diagonal in the n-gram head: head ``h`` only ever emits
    rows from its own vocabulary slice, and its embedding lands in exactly
    columns ``[h*head_dim, (h+1)*head_dim)`` of the flattened output. Row block
    and column block are the same block, so a shard on the flat head index is
    row-parallel and column-parallel at once -- no id masking, no all-reduce on
    the gather, and the per-rank outputs concatenate into the unsharded result
    with no permutation.

    Two layout notes, both forced by that shard:

    * Runtime rows are padded *per head* to a common ``rows_per_head``, so head
      ``j`` sits at ``j * rows_per_head``. The checkpoint packs the heads back
      to back at their true (prime) sizes, which would give the ranks unequal
      row counts; ``DTensor``'s ``Shard(0)`` and DCP both need every rank to
      hold the same shape. The padding costs a few thousand rows out of ~320M
      and is never indexed. :func:`load_safetensors_into` maps the packed
      checkpoint rows onto the padded runtime rows head by head.
    * The per-head sizes, offsets and n-gram orders are non-persistent buffers.
      They are derived from the config, they differ per rank once sharded, and
      a plain (non-DTensor) buffer that differs per rank is written to a DCP
      checkpoint by whichever rank wins the dedup. :meth:`materialize` rebuilds
      them after ``to_empty``.
    """

    def __init__(self, cfg: Qwen4TextConfig, embedding_dim: int, ple_layer_index: int):
        super().__init__()
        self.ngram_size = cfg.ngram_size
        self.context_len = self.ngram_size - 1
        self.heads_per_ngram = cfg.heads_per_ngram
        self.ngram_heads = (self.ngram_size - 1) * self.heads_per_ngram
        self.ple_layer_index = ple_layer_index
        self.eos_token_id = cfg.eos_token_id
        head_dim_per_ngram = embedding_dim // self.ngram_heads

        self.tp_size = max(1, getattr(cfg, "ple_tp_size", 1))
        self.tp_rank = getattr(cfg, "ple_tp_rank", 0)
        assert self.ngram_heads % self.tp_size == 0, (
            f"n-gram head count {self.ngram_heads} is not divisible by "
            f"ple_tp_size={self.tp_size}"
        )
        self.heads_per_rank = self.ngram_heads // self.tp_size

        # Every head's vocabulary is the next prime after `ngram_vocab_size_base`,
        # so the sizes differ head to head. `checkpoint_offsets` are where those
        # slices sit in the packed on-disk table.
        sizes, offsets, total = [], [], 0
        for head_idx in range(self.ngram_heads):
            global_head_idx = ple_layer_index * self.ngram_heads + head_idx
            size = _find_nth_prime_after(cfg.ngram_vocab_size_base - 1, global_head_idx + 1)
            sizes.append(size)
            offsets.append(total)
            total += size
        self.total_vocab_size = total

        divisor = cfg.make_ngram_vocab_size_divisible_by
        self.rows_per_head = math.ceil(max(sizes) / divisor) * divisor
        # rows of the packed (checkpoint / HF) table, for the converters below
        self.packed_rows = math.ceil(total / divisor) * divisor

        head_start = self.tp_rank * self.heads_per_rank
        self._local_heads = list(range(head_start, head_start + self.heads_per_rank))
        # (packed checkpoint row, row count) per local head, in local order
        self.checkpoint_windows = [(offsets[h], sizes[h]) for h in self._local_heads]

        self.register_buffer(
            "layer_multipliers",
            _build_layer_multipliers(cfg.vocab_size, cfg.ngram_size, ple_layer_index, cfg.seed),
            persistent=True,
        )
        for name, tensor in self._derived_buffers().items():
            self.register_buffer(name, tensor, persistent=False)

        # Build the table without advancing the global RNG. Its height depends
        # on the TP shard, so a shifted stream would leave a TP=1 and a TP=2 run
        # with different random init for every module constructed after it --
        # `init_qwen4` re-draws most of them, but not the bare `conv1d`,
        # `A_log` and `dt_bias` parameters.
        devices = [torch.cuda.current_device()] if torch.cuda.is_initialized() else []
        with torch.random.fork_rng(devices=devices):
            self.ngram_embedding = nn.Embedding(
                self.heads_per_rank * self.rows_per_head, head_dim_per_ngram
            )

    def _derived_buffers(self) -> dict[str, torch.Tensor]:
        sizes, offsets, orders = [], [], []
        for local_idx, head in enumerate(self._local_heads):
            sizes.append(self.checkpoint_windows[local_idx][1])
            offsets.append(local_idx * self.rows_per_head)
            # which n-gram order this head hashes: 0 is the bigram
            orders.append(head // self.heads_per_ngram)
        return {
            "ngram_heads_vocab_sizes": torch.tensor(sizes, dtype=torch.long),
            "ngram_heads_offsets": torch.tensor(offsets, dtype=torch.long),
            "ngram_head_orders": torch.tensor(orders, dtype=torch.long),
        }

    @torch.no_grad()
    def packed_table(self) -> torch.Tensor:
        """This rank's rows in the packed layout HF and the checkpoint use.

        Only meaningful unsharded; a shard holds a subset of the rows and the
        rest of the returned table is zero.
        """
        weight = _local(self.ngram_embedding.weight)
        out = weight.new_zeros((self.packed_rows, weight.shape[1]))
        for local_idx, (start, size) in enumerate(self.checkpoint_windows):
            base = local_idx * self.rows_per_head
            out[start : start + size] = weight[base : base + size]
        return out

    @torch.no_grad()
    def load_packed_table(self, packed: torch.Tensor) -> None:
        """Copy a packed-layout table into this rank's padded runtime rows."""
        weight = _local(self.ngram_embedding.weight)
        weight.zero_()
        for local_idx, (start, size) in enumerate(self.checkpoint_windows):
            base = local_idx * self.rows_per_head
            weight[base : base + size] = packed[start : start + size]

    @torch.no_grad()
    def materialize(self) -> None:
        """Recompute the non-persistent head tables (``to_empty`` wipes them)."""
        device = self.ngram_embedding.weight.device
        for name, tensor in self._derived_buffers().items():
            setattr(self, name, tensor.to(device))

    def _shift_right_ignore_eos(self, token_ids: torch.Tensor, shift: int) -> torch.Tensor:
        if shift == 0:
            return token_ids
        batch_size, seq_len = token_ids.shape
        positions = torch.arange(seq_len, device=token_ids.device, dtype=torch.long)
        eos_positions = torch.where(token_ids == self.eos_token_id, positions, -1)
        previous_eos_inclusive = torch.cummax(eos_positions, dim=1).values
        previous_eos = torch.cat(
            [eos_positions.new_full((batch_size, 1), -1), previous_eos_inclusive[:, :-1]], dim=1
        )
        segment_start = previous_eos + 1
        position_in_segment = positions.unsqueeze(0) - segment_start
        source_positions = positions - shift
        gather_positions = source_positions.clamp_min(0).unsqueeze(0).expand(batch_size, -1)
        shifted = token_ids.gather(dim=1, index=gather_positions)
        valid = (position_in_segment >= shift) & (source_positions.unsqueeze(0) >= 0)
        return torch.where(valid, shifted, token_ids.new_full((), self.eos_token_id))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.long()
        previous_context = input_ids.new_full(
            (input_ids.shape[0], self.context_len), self.eos_token_id
        )
        token_history = torch.cat([previous_context, input_ids], dim=-1)
        shifted = [self._shift_right_ignore_eos(token_history, s) for s in range(self.ngram_size)]

        # One hash per n-gram order, built incrementally: order n+1 is order n
        # xored with one more shifted token. Indexing per head afterwards (in
        # place of a loop over orders) is what lets a rank own an arbitrary
        # contiguous head range rather than a whole order's worth of heads.
        mixed = shifted[0] * self.layer_multipliers[0]
        per_order = []
        for position in range(1, self.ngram_size):
            mixed = torch.bitwise_xor(mixed, shifted[position] * self.layer_multipliers[position])
            per_order.append(mixed)
        hashes = torch.stack(per_order, dim=-1)                  # (B, T, ngram_size - 1)

        ngram_ids = hashes.index_select(-1, self.ngram_head_orders)
        ngram_ids = torch.remainder(ngram_ids, self.ngram_heads_vocab_sizes)
        ngram_ids = ngram_ids + self.ngram_heads_offsets
        ngram_ids = ngram_ids[:, -input_ids.shape[1]:]
        # The weight is a `Shard(0)` DTensor under TP while the ids are plain,
        # so the gather runs on the local rows -- which are exactly this rank's
        # heads.
        return F.embedding(ngram_ids, _local(self.ngram_embedding.weight)).flatten(-2)

class PLELayer(nn.Module):
    """Inject hashed n-gram features into every hyper-connection stream."""

    def __init__(self, cfg: Qwen4TextConfig, ple_layer_index: int):
        super().__init__()
        self.hidden_size = cfg.hidden_size
        self.hc_count = cfg.hc_count
        hc_hidden = self.hidden_size * self.hc_count
        ple_embed_dim = cfg.ple_embed_dim
        self.ple_embedding = NGramEmbedding(cfg, ple_embed_dim, ple_layer_index)

        conv_kernel_size = cfg.ple_conv_kernel_size
        conv_dilation = cfg.ngram_size
        self.short_conv_state_len = (conv_kernel_size - 1) * conv_dilation
        self.key_proj = nn.Linear(ple_embed_dim, hc_hidden, bias=False)
        self.value_proj = nn.Linear(ple_embed_dim, self.hidden_size, bias=False)
        self.norm_key = OffsetRMSNorm(hc_hidden, group_size=self.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_query = OffsetRMSNorm(hc_hidden, group_size=self.hidden_size, eps=cfg.rms_norm_eps)
        self.norm_conv = OffsetRMSNorm(hc_hidden, group_size=self.hidden_size, eps=cfg.rms_norm_eps)
        self.conv1d = nn.Conv1d(
            hc_hidden, hc_hidden,
            kernel_size=conv_kernel_size,
            groups=hc_hidden,
            dilation=conv_dilation,
            bias=False,
        )

    def _short_conv(self, hidden_states: torch.Tensor, cu_seqlens=None) -> torch.Tensor:
        """Causal dilated depthwise conv, zero-padded at each document start.

        The dilation rules out the `causal_conv1d` kernel (and its `seq_idx`
        packing support), so this is a plain `nn.Conv1d` over the whole packed
        row. That is correct everywhere except the first `short_conv_state_len`
        tokens of each packed document, which would otherwise convolve over the
        tail of the previous one; those few positions are recomputed below with
        the out-of-document taps zeroed, matching the zero left-padding HF
        applies at the start of a sequence.
        """
        seq_len = hidden_states.shape[1]
        state_len = self.short_conv_state_len
        # The conv itself wants (B, C, T), but everything around it -- the
        # document-start fixup included -- stays in (B, T, C). Writing the fixup
        # in channel-major order leaves a `copy_` into a permuted `empty` in the
        # joint graph (the transposed `index_copy` backward) that AOTAutograd
        # refuses to functionalize, and the layer then fails to compile with
        # `fullgraph=True`.
        h = hidden_states.transpose(1, 2).contiguous()          # (B, C, T)
        padded = F.pad(h, (state_len, 0))
        padded = padded[..., -(state_len + seq_len):]
        out = self.conv1d(padded).transpose(1, 2).contiguous()  # (B, T, C)

        if cu_seqlens is not None and cu_seqlens.numel() > 2 and state_len > 0:
            out = self._fix_document_starts(hidden_states, out, cu_seqlens, state_len)

        return F.silu(out)

    def _fix_document_starts(self, hidden_states, out, cu_seqlens, state_len):
        """Recompute the first `state_len` positions of each packed document.

        ``hidden_states`` and ``out`` are both (B, T, C).
        """
        device = hidden_states.device
        weight = self.conv1d.weight.squeeze(1)                  # (C, K)
        kernel = weight.shape[-1]
        dilation = self.conv1d.dilation[0]

        starts = cu_seqlens[1:-1].long()                        # every document but the first
        ends = cu_seqlens[2:].long()
        offsets = torch.arange(state_len, device=device)
        pos = starts[:, None] + offsets[None, :]                # (D-1, state_len)
        doc_start = starts[:, None].expand_as(pos)
        keep = pos < ends[:, None]
        pos, doc_start = pos[keep], doc_start[keep]
        # No early return on an empty `pos`: the mask makes its length
        # data-dependent, and branching on it breaks `fullgraph=True` compiles
        # ("Could not guard on data-dependent expression"). Every op below is a
        # no-op at length zero anyway.

        fixed = torch.zeros(
            hidden_states.shape[0], pos.numel(), hidden_states.shape[2],
            device=device, dtype=out.dtype,
        )
        for j in range(kernel):
            src = pos - (kernel - 1 - j) * dilation
            valid = (src >= doc_start).to(hidden_states.dtype)
            taps = hidden_states.index_select(1, src.clamp_min(0))   # (B, n, C)
            fixed = fixed + weight[:, j] * taps * valid[None, :, None]

        # `fixed` follows the conv's dtype (bf16 under autocast) while `out` may
        # not; index_copy requires both to match.
        return out.index_copy(1, pos, fixed.to(out.dtype))

    def forward(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        embeddings = self.ple_embedding(input_ids)
        key = self.norm_key(self.key_proj(embeddings)).unflatten(-1, (self.hc_count, self.hidden_size))
        value = self.value_proj(embeddings)
        query = self.norm_query(hidden_states).unflatten(-1, (self.hc_count, self.hidden_size))
        gate = (key * query).sum(dim=-1, keepdim=True) / math.sqrt(self.hidden_size)
        gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
        gated = torch.sigmoid(gate) * value.unsqueeze(-2)
        gated_normed = self.norm_conv(gated.flatten(-2))
        gated = gated.flatten(-2)
        return gated + self._short_conv(gated_normed, cu_seqlens)

# --------------------------------------------------------------------------
# decoder
# --------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen4TextConfig, layer_idx: int, ple_layer_index: int | None):
        super().__init__()
        self.layer_type = cfg.layer_types[layer_idx]
        if self.layer_type == "linear_attention":
            self.linear_attn = GatedDeltaNet(cfg)
        else:
            self.self_attn = SelfAttention(cfg)
        self.mlp = SparseMoeBlock(cfg)
        self.ple = PLELayer(cfg, ple_layer_index) if ple_layer_index is not None else None
        self.attn_hyper_connection = GatedResidual(cfg)
        self.mlp_hyper_connection = GatedResidual(cfg)

    def forward(self, x, cos, sin, cu_seqlens, max_seqlen, seg_id, ple_input_ids):
        if self.ple is not None:
            x = x + self.ple(x, ple_input_ids, cu_seqlens)

        x, hyper, inject = self.attn_hyper_connection(x)
        if self.layer_type == "linear_attention":
            x = self.linear_attn(x, cu_seqlens=cu_seqlens, seg_id=seg_id)
        else:
            x = self.self_attn(
                x, cos=cos, sin=sin, cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen, seg_id=seg_id,
            )
        x = hyper + (x.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)

        x, hyper, inject = self.mlp_hyper_connection(x)
        x = self.mlp(x)
        x = hyper + (x.unsqueeze(-2) * inject.unsqueeze(-1)).flatten(-2)
        return x

class LanguageModel(nn.Module):
    """HF name: `model.language_model`."""

    def __init__(self, cfg: Qwen4TextConfig):
        super().__init__()
        self.cfg = cfg
        self.hc_count = cfg.hc_count
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)

        ple_ids = list(cfg.ple_layer_ids)  # one-indexed
        layers = []
        for layer_idx in range(cfg.num_hidden_layers):
            ple_index = ple_ids.index(layer_idx + 1) if (layer_idx + 1) in ple_ids else None
            layers.append(DecoderLayer(cfg, layer_idx, ple_index))
        self.layers = nn.ModuleList(layers)

        # Qwen4 has no final RMSNorm: the mixer is the last op.
        self.hyper_connection_mixer = GatedResidual(cfg, use_combine=False)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        seg_id: torch.Tensor | None = None,
        ple_input_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        total = inputs_embeds.shape[1]
        if seg_id is None:
            seg_id = _segment_ids(cu_seqlens, total)
        # Under PP the hyper-connection stream is what crosses a stage
        # boundary, so only the stage that owns `embed_tokens` expands
        # (1, T, hidden) into the hc_count copies -- later stages already
        # receive (1, T, hc_count * hidden).
        if self.embed_tokens is not None:
            x = inputs_embeds.repeat(1, 1, self.hc_count)
        else:
            x = inputs_embeds
        for layer in iter_layers(self.layers):
            x = layer(x, cos, sin, cu_seqlens, max_seqlen, seg_id, ple_input_ids)
        # `hyper_connection_mixer` is None on every stage but the last.
        if self.hyper_connection_mixer is None:
            return x
        return self.hyper_connection_mixer(x)

# --------------------------------------------------------------------------
# vision
# --------------------------------------------------------------------------

class VisionPatchEmbed(nn.Module):
    def __init__(self, cfg: Qwen4VisionConfig):
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
    def __init__(self, cfg: Qwen4VisionConfig):
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
    def __init__(self, cfg: Qwen4VisionConfig):
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
        )  # each (S, H, D) - already the layout varlen_attn wants
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
    def __init__(self, cfg: Qwen4VisionConfig):
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
    def __init__(self, cfg: Qwen4VisionConfig, use_postshuffle_norm: bool = False):
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
    """HF name: `model.visual`. Same tower as Qwen3.5 minus deepstack."""

    def __init__(self, cfg: Qwen4VisionConfig):
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

    def forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        hidden_states = self.patch_embed(hidden_states)
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds

        rotary = self.rot_pos_emb(grid_thw)
        emb = torch.cat((rotary, rotary), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        seg_lens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
        cu = F.pad(seg_lens.cumsum(dim=0, dtype=torch.int32), (1, 0), value=0)
        max_seqlen = int(seg_lens.max().item())

        for blk in self.blocks:
            hidden_states = blk(hidden_states, cu, max_seqlen, position_embeddings)

        return self.merger(hidden_states)

# --------------------------------------------------------------------------
# top level
# --------------------------------------------------------------------------

class Qwen4Inner(nn.Module):
    """HF name: `model`. Groups `language_model` and `visual`.
    This is only used to match the state keys."""

    def __init__(self, cfg: Qwen4Config):
        super().__init__()
        self.language_model = LanguageModel(cfg.text)
        self.visual = VisionModel(cfg.vision)

class Qwen4ForCausalLM(nn.Module):
    def __init__(self, cfg: Qwen4Config, **kwargs):
        super().__init__()
        self.cfg = cfg
        self.model = Qwen4Inner(cfg)
        self.lm_head = nn.Linear(cfg.text.hidden_size, cfg.text.vocab_size, bias=False)
        if cfg.tie_word_embeddings:
            self.lm_head.weight = self.model.language_model.embed_tokens.weight

        head_dim = cfg.text.head_dim
        partial = cfg.text.rope_parameters.get('partial_rotary_factor', 1.0)
        rope_dim = int(head_dim * partial)
        inv_freq = 1.0 / (
            cfg.text.rope_parameters['rope_theta']
            ** (torch.arange(0, rope_dim, 2, dtype=torch.float32) / rope_dim)
        )
        self.register_buffer("text_inv_freq", inv_freq, persistent=False)
        self.mrope_section = list(cfg.text.rope_parameters['mrope_section'])

    def get_rope_index(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        image_grid_thw: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute 3D MRoPE position_ids over a packed row.

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
        """position_ids: (B, S) or (3, B, S) -> cos/sin of shape (B, S, D)."""
        if position_ids.dim() == 2:
            position_ids = position_ids.unsqueeze(0).expand(3, -1, -1)
        return mrope_cos_sin(self.text_inv_freq, position_ids, self.mrope_section)

    def forward(
        self,
        hidden_states: torch.Tensor | None = None,
        *,
        input_ids: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
        pixel_values_videos: torch.Tensor | None = None,
        video_grid_thw: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        **kwargs,
    ) -> "CausalLMOutput | torch.Tensor":
        """Varlen-only forward.

        Expected layout: `input_ids` / `inputs_embeds` is a single packed row
        `(1, total)` / `(1, total, H)`. `attention_mask` is interpreted as
        `cu_seqlens`: a 1D int32 tensor of cumulative offsets starting at 0.

        Under pipeline parallelism this is also the stage forward, and the
        `None`s left by `apply_pp_qwen4` say which stage it is:

        - `embed_tokens` present -> first stage. `PipelineStage` passes the
          stage input positionally, which for stage 0 is the token ids, so
          `input_ids` doubles as the positional slot.
        - `embed_tokens` absent -> `hidden_states` carries the incoming
          `(1, total, hc_count * hidden_size)` hyper-connection stream.
          `input_ids` still arrives as a replicated kwarg: MRoPE, the image
          scatter and the PLE n-gram lookup all need it on every stage.
        - `lm_head` present -> last stage, returns logits. Otherwise the raw
          stream is returned for the next stage.
        """
        is_first_stage = self.model.language_model.embed_tokens is not None

        if is_first_stage and input_ids is None:
            # `hidden_states` is the first positional parameter so that
            # `PipelineStage` can hand a stage its input positionally. For
            # stage 0 that input *is* the token ids, and every ordinary
            # (non-PP) caller passing `model(input_ids, ...)` positionally
            # lands here too.
            input_ids = hidden_states

        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if input_ids is not None:
            assert input_ids.dim() == 2 and input_ids.shape[0] == 1, (
                f"varlen expects packed (1, total), got {tuple(input_ids.shape)}"
            )

        if is_first_stage:
            assert (input_ids is None) ^ (inputs_embeds is None)
            if self.cfg.text.ple_layer_ids and input_ids is None:
                raise ValueError(
                    "PLE layers need `input_ids`; pass them alongside `inputs_embeds`."
                )
            if inputs_embeds is None:
                inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        else:
            if hidden_states is None:
                raise ValueError(
                    "a non-first pipeline stage needs `hidden_states` from the "
                    "stage before it"
                )
            if input_ids is None:
                raise ValueError(
                    "`input_ids` must reach every pipeline stage: MRoPE, the "
                    "image scatter and PLE all index by token id"
                )
            inputs_embeds = hidden_states

        assert inputs_embeds.dim() == 3 and inputs_embeds.shape[0] == 1
        total = inputs_embeds.shape[1]
        device = inputs_embeds.device

        if attention_mask is None:
            cu_seqlens = torch.tensor([0, total], device=device, dtype=torch.int32)
        else:
            assert (
                attention_mask.dim() == 1
                and attention_mask[0].item() == 0
                and attention_mask[-1].item() == total
            ), "attention_mask must be cu_seqlens: 1D int32, starts at 0, ends at total"
            cu_seqlens = attention_mask.to(torch.int32)

        max_seqlen = int((cu_seqlens[1:] - cu_seqlens[:-1]).max().item())

        # The vision tower lives on stage 0 only; later stages still receive
        # the pixel kwargs (replicated) and must ignore them.
        if pixel_values is not None and self.model.visual is not None:
            assert image_grid_thw is not None
            merged = self.model.visual(pixel_values, image_grid_thw).to(inputs_embeds.dtype)
            image_mask = input_ids == self.cfg.image_token_id
            assert image_mask.sum().item() == merged.shape[0], (
                f"image tokens={image_mask.sum().item()} vs features={merged.shape[0]}"
            )
            inputs_embeds = inputs_embeds.masked_scatter(
                image_mask.unsqueeze(-1).expand_as(inputs_embeds), merged
            )

        if pixel_values_videos is not None and self.model.visual is not None:
            assert video_grid_thw is not None
            merged_v = self.model.visual(pixel_values_videos, video_grid_thw).to(inputs_embeds.dtype)
            video_mask = input_ids == self.cfg.video_token_id
            inputs_embeds = inputs_embeds.masked_scatter(
                video_mask.unsqueeze(-1).expand_as(inputs_embeds), merged_v
            )

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
                pos = torch.zeros(total, device=device, dtype=torch.int64)
                for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
                    pos[start:end] = torch.arange(end - start, device=device)
                position_ids = pos.view(1, 1, -1).expand(3, 1, -1)

        cos, sin = self._compute_cos_sin(position_ids)
        cos = cos.to(inputs_embeds.dtype)
        sin = sin.to(inputs_embeds.dtype)

        h = self.model.language_model(
            inputs_embeds,
            cos,
            sin,
            cu_seqlens,
            max_seqlen,
            ple_input_ids=input_ids,
        )
        # Not the last stage: hand the hyper-connection stream on unchanged.
        if self.lm_head is None:
            return h

        logits = self.lm_head(h)

        if labels is None:
            return logits
        if labels.dim() == 1:
            labels = labels.unsqueeze(0)
        loss = causal_lm_loss(logits, labels)
        return CausalLMOutput(loss=loss, logits=logits)

    @classmethod
    def from_pretrained(
        cls,
        snapshot_dir: str | Path,
        dtype: torch.dtype | None = None,
        device: str | torch.device = "cpu",
        *,
        load_vision: bool = True,
        load_ple: bool = True,
        ple_tp_size: int = 1,
        ple_tp_rank: int = 0,
    ) -> "Qwen4ForCausalLM":
        """``ple_tp_size``/``ple_tp_rank`` head-shard the n-gram table at build
        time. Every other weight is built whole and sharded later by
        `apply_tp`; the PLE table cannot be, because at the released
        ``ngram_vocab_size_base`` it is ~95 GiB on its own and would have to be
        materialized (and then upcast to fp32) before anything could split it.
        """
        snapshot_dir = Path(snapshot_dir)
        cfg = Qwen4Config.from_json(snapshot_dir / "config.json")
        cfg.text.ple_tp_size = ple_tp_size
        cfg.text.ple_tp_rank = ple_tp_rank
        cfg.text.validate()
        if dtype is None:
            dtype = {
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
                "float32": torch.float32,
            }[cfg.torch_dtype]

        with torch.device("meta"):
            model = cls(cfg)
        model = model.to_empty(device=device).to(dtype=dtype)

        load_safetensors_into(
            model,
            snapshot_dir,
            device=device,
            dtype=dtype,
            load_vision=load_vision,
            load_ple=load_ple,
        )

        # `to_empty` re-materializes every parameter and breaks the tie
        # established in `__init__`. Re-tie here.
        if cfg.tie_word_embeddings:
            model.lm_head.weight = model.model.language_model.embed_tokens.weight

        # `to_empty` also wipes non-persistent buffers.
        for module in model.modules():
            if isinstance(module, NGramEmbedding):
                module.materialize()
        if load_vision:
            head_dim_v = cfg.vision.hidden_size // cfg.vision.num_heads
            rdim = head_dim_v // 2
            inv_freq_v = 1.0 / (
                10000.0 ** (torch.arange(0, rdim, 2, dtype=torch.float32, device=device) / rdim)
            )
            model.model.visual.rotary_pos_emb.inv_freq = inv_freq_v

        head_dim = cfg.text.head_dim
        partial = cfg.text.rope_parameters.get('partial_rotary_factor', 1.0)
        rope_dim = int(head_dim * partial)
        model.text_inv_freq = 1.0 / (
            cfg.text.rope_parameters['rope_theta']
            ** (torch.arange(0, rope_dim, 2, dtype=torch.float32, device=device) / rope_dim)
        )
        return model, cfg
