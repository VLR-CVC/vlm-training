"""DSA as a swappable `inner_attention`, over our packed-document rows.

The vendored kernels are Megatron-shaped: `[s, b, h, d]`, one sequence per batch
entry, and a causal mask built from `torch.arange(q_start, q_end)` rather than
from any `cu_seqlens` -- the min-memory backend has no varlen support at all
(that lives in Megatron's other DSA backend). Our rows pack many documents into
one token axis, so handing a whole row over would let queries attend across
document boundaries.

Rather than teach the kernels varlen, this walks the document boundaries
`VarlenMetadata` already carries and hands the kernels one `[L, B, h, d]` batch
at a time. The positions the kernels invent are then 0..L within each batch
entry, which is exactly right, and no Triton is touched.

Documents at or below `min_sparse_len` take dense SDPA instead: at a budget that
already covers the whole document DSA *is* dense attention, so selecting is
wasted work. They still pay `dsa_dense_indexer_loss`, so the indexer trains on
short documents too -- without it a blend of mostly-short documents would leave
the indexer barely trained. Raising `min_sparse_len` past the longest document
turns the layer into dense attention that trains an indexer, which is the
warmup schedule.

Those short documents are grouped by length and run as one padded batch per
group (`_group_segments`), because the short-document call is launch-bound
rather than compute-bound. Long documents still go one per call, where the
batch axis buys nothing. The indexer loss is a single mean over every real
query row in the packed row, so it does not move with the packing.

ponytail: the boundary list costs a device-to-host sync per forward, and a
group holding one document still pays for a full padded length. Both are
cheap next to a kernel launch; revisit if a profile says otherwise.
"""

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torch import nn

from models.common.attention import InnerAttention, VarlenMetadata
from models.common.dsa._shim import ProcessGroupCollection
from models.common.dsa.dsa_min_memory import dsa_dense_indexer_loss, dsa_min_memory_gqa
from models.common.module import Module
from models.common.nn_modules import Linear

__all__ = ["DSAInnerAttention", "DSAIndexer"]


@spmd.register_local_autograd_function
class _SideLoss(torch.autograd.Function):
    """Carry a side loss on the main graph without returning it.

    The indexer loss has to be in the backward graph, but it cannot ride out
    through the return value: `inner_attention` returns one tensor, and under
    activation checkpointing a tensor stashed on the module would belong to a
    graph that was thrown away. Seeding its gradient here keeps it tied to the
    activation it was computed beside, so it survives recomputation.
    """

    @staticmethod
    def forward(ctx, out: torch.Tensor, side_loss: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(side_loss)
        return out.view_as(out)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (side_loss,) = ctx.saved_tensors
        # d(total)/d(side_loss) = 1: `loss_coeff` is already baked into the value
        # the kernels return, so the weight lives in the config, not here.
        return grad_out, torch.ones_like(side_loss)


class DSAIndexer(Module):
    """The nine attributes the kernels read off `indexer`, as a real module.

    They read `linear_q.weight` directly instead of calling the module, so these
    two linears must not be sharded by TP -- hence no `sharding_config`. FSDP2
    is fine: it publishes the unsharded parameter in its pre-forward hook, and
    this runs inside the wrapped block's forward.

    `rotary_pos_emb` is deliberately not the trunk's `MRoPE`. The kernels index
    positions themselves and want frequencies, not a cos/sin table, and the
    indexer only has to *rank* keys -- it does not have to agree with the
    trunk's multimodal position scheme.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        index_topk: int = 2048
        index_head_dim: int = 128
        index_rotary_dim: int = 64
        rope_theta: float = 10000.0

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.index_topk = config.index_topk
        self.index_head_dim = config.index_head_dim
        # The kernels split head_dim into [nope, rope] halves, so an odd rotary
        # width has no pair to rotate.
        self.index_rotary_dim = config.index_rotary_dim - (config.index_rotary_dim % 2)
        self.softmax_scale = config.index_head_dim**-0.5
        self.rope_theta = config.rope_theta
        self.linear_q = Linear.Config(
            in_features=config.dim, out_features=config.index_head_dim, bias=False
        ).build()
        self.linear_k = Linear.Config(
            in_features=config.dim, out_features=config.index_head_dim, bias=False
        ).build()
        self.pg_collection = ProcessGroupCollection()
        # `config` is the attribute name the kernels getattr into, not this
        # module's own Config.
        self.config = _KernelConfig()
        self.register_buffer("inv_freq", None, persistent=False)

    def _init_self_buffers(self, *, buffer_device: torch.device | None = None) -> None:
        width = self.index_rotary_dim
        self.inv_freq = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, width, 2, device=buffer_device, dtype=torch.float32)
                / width
            )
        )

    @property
    def rotary_pos_emb(self) -> "DSAIndexer":
        """The kernels only ever read `.inv_freq` off this, so hand back self."""
        return self


@dataclass
class _KernelConfig:
    """`getattr(indexer.config, ...)` is read for exactly these two."""

    dsa_indexer_mode: str = "simplified"
    rotary_interleaved: bool = False


def _segments(cu_seq: torch.Tensor) -> list[tuple[int, int]]:
    """Document `[start, end)` pairs, dropping the padding.

    `create_varlen_metadata_for_document` pads `cu_seqlens` up to a bucket with
    repeats of the row length, so the tail of the list is zero-length segments.
    """
    bounds = cu_seq.tolist()
    return [(s, e) for s, e in zip(bounds, bounds[1:]) if e > s]


# Padded lengths are rounded to this, which bounds the number of distinct shapes
# the kernels see per layer at `min_sparse_len / _PAD_MULTIPLE`. That matters more
# than the wasted rows: a new shape costs a Triton compile, and documents whose
# lengths are all different would pay one per document per step.
_PAD_MULTIPLE = 256


def _group_segments(
    segments: list[tuple[int, int]], min_sparse_len: int
) -> tuple[dict[int, list[tuple[int, int]]], list[tuple[int, int]]]:
    """Short documents grouped by padded length; long ones left on their own.

    Batching only pays where the call is launch-bound. Measured on one card:
    25 documents of 320 tokens cost 2.42 ms each alone and 12.35 ms as one
    batch (4.9x), while `dsa_min_memory_gqa` at 2048 tokens is already
    compute-bound and scales linearly in the batch (15.9 ms at B=1, 63.8 at
    B=4), so long documents gain nothing from being stacked.
    """
    dense: dict[int, list[tuple[int, int]]] = {}
    sparse: list[tuple[int, int]] = []
    for start, end in segments:
        length = end - start
        if length > min_sparse_len:
            sparse.append((start, end))
            continue
        padded = -(-length // _PAD_MULTIPLE) * _PAD_MULTIPLE
        dense.setdefault(padded, []).append((start, end))
    return dense, sparse


class DSAInnerAttention(InnerAttention):
    """Megatron's min-memory DSA, one call per packed document."""

    # The segment loop reads document boundaries off the device and the kernels
    # are large autograd.Functions with Python tile loops, data-dependent
    # branches and Triton launches inside. Dynamo cannot trace that, so the
    # block holding this compiles around it instead of through it; see
    # `train.parallel.compile.apply_compile`.
    supports_fullgraph = False

    @dataclass(kw_only=True, slots=True)
    class Config(InnerAttention.Config):
        indexer: DSAIndexer.Config
        loss_coeff: float = 1.0
        use_indexer_rope: bool = True
        use_triton: bool = True
        # Documents this short go dense. 0 means "when the budget already
        # covers the document"; a very large value is the dense warmup.
        min_sparse_len: int = 0

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.indexer = config.indexer.build()
        self.loss_coeff = config.loss_coeff
        self.use_indexer_rope = config.use_indexer_rope
        self.use_triton = config.use_triton
        self.min_sparse_len = config.min_sparse_len or self.indexer.index_topk
        # Detached, for logging only; the trained copy rides _SideLoss.
        self.last_indexer_loss: float = 0.0

    # Break here rather than letting Dynamo try and fail: tracing the loop
    # raises on the unbacked symints `.tolist()` produces, and the message is
    # about data-dependent guards rather than about this kernel.
    @torch._dynamo.disable
    def forward(
        self,
        q_THK: torch.Tensor,
        k_THK: torch.Tensor,
        v_THV: torch.Tensor,
        *,
        attention_masks: VarlenMetadata,
        scale: float | None = None,
        enable_gqa: bool = False,
        hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert isinstance(attention_masks, VarlenMetadata), type(attention_masks)
        assert hidden_states is not None, (
            "DSA routes on hidden states, not on q/k; the outer attention must "
            "pass hidden_states=x_TD."
        )
        if scale is None:
            scale = q_THK.shape[-1] ** -0.5

        # cu_seq_q only: this is self-attention, so queries and keys share one
        # segmentation, which is what makes the per-document view causal-correct.
        segments = _segments(attention_masks.cu_seq_q)
        # `out` starts uninitialized, so a gap in the boundaries would ship
        # garbage rows rather than fail. The boundary list is already on the
        # host, so this costs nothing.
        assert segments and segments[0][0] == 0 and segments[-1][1] == q_THK.shape[0], (
            f"document boundaries {segments[:2]}..{segments[-1:]} do not cover "
            f"all {q_THK.shape[0]} tokens"
        )
        dense, sparse = _group_segments(segments, self.min_sparse_len)
        # Query heads, value width: GQA means this is neither q's nor v's shape.
        out = v_THV.new_empty(q_THK.shape[0], q_THK.shape[1], v_THV.shape[-1])
        # The kernels return the KL averaged over the query rows they were given,
        # so a mean is recovered by weighting each call by its row count. Summing
        # the per-call means instead would scale the loss with how the row happens
        # to be packed: 25 short documents would weigh 25x one long one.
        kl_sum = q_THK.new_zeros((), dtype=torch.float32)
        rows = 0

        with spmd.no_typecheck():
            for padded, group in dense.items():
                # Zero rows contribute no KL: the pad query projects to zero
                # under both the main and the indexer weights, so teacher and
                # student are the same uniform distribution. Verified against
                # the unpadded call in `test_padding_is_free`.
                qb = q_THK.new_zeros(padded, len(group), *q_THK.shape[1:])
                kb = k_THK.new_zeros(padded, len(group), *k_THK.shape[1:])
                vb = v_THV.new_zeros(padded, len(group), *v_THV.shape[1:])
                hb = hidden_states.new_zeros(padded, len(group), hidden_states.shape[-1])
                for i, (start, end) in enumerate(group):
                    n = end - start
                    qb[:n, i] = q_THK[start:end]
                    kb[:n, i] = k_THK[start:end]
                    vb[:n, i] = v_THV[start:end]
                    hb[:n, i] = hidden_states[start:end]
                # Padding sits after every real token, so causality already
                # hides it from the rows that are kept.
                ob = _dense_batch(qb, kb, vb, scale, enable_gqa)
                for i, (start, end) in enumerate(group):
                    out[start:end] = ob[: end - start, i]
                    rows += end - start
                if self.loss_coeff > 0.0:
                    kl_sum = kl_sum + dsa_dense_indexer_loss(
                        qb, kb, hb, self.indexer,
                        softmax_scale=scale,
                        loss_coeff=self.loss_coeff,
                        use_indexer_rope=self.use_indexer_rope,
                        use_triton=self.use_triton,
                    ) * (padded * len(group))

            for start, end in sparse:
                q = q_THK[start:end].unsqueeze(1)
                k = k_THK[start:end].unsqueeze(1)
                v = v_THV[start:end].unsqueeze(1)
                h = hidden_states[start:end].unsqueeze(1)
                seg_out, seg_loss = dsa_min_memory_gqa(
                    q, k, v, h, self.indexer,
                    softmax_scale=scale,
                    loss_coeff=self.loss_coeff,
                    use_indexer_rope=self.use_indexer_rope,
                    use_triton=self.use_triton,
                )
                out[start:end] = seg_out.view(end - start, *out.shape[1:])
                kl_sum = kl_sum + seg_loss * (end - start)
                rows += end - start

        loss = kl_sum / max(rows, 1)
        self.last_indexer_loss = float(loss.detach())
        out = _SideLoss.apply(out, loss)
        if spmd.is_type_checking():
            spmd.assert_type(
                out, spmd.get_local_type(q_THK), spmd.get_partition_spec(q_THK)
            )
        return out


def _dense_batch(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, enable_gqa: bool
) -> torch.Tensor:
    """Causal SDPA over a `[L, B, h, d]` batch of equal-length documents.

    SDPA rather than our `varlen_attn`: the batch axis already separates the
    documents, so no `cu_seqlens` has to be built per layer.
    """
    qt, kt, vt = (t.permute(1, 2, 0, 3) for t in (q, k, v))
    o = F.scaled_dot_product_attention(
        qt, kt, vt, is_causal=True, scale=scale, enable_gqa=enable_gqa
    )
    return o.permute(2, 0, 1, 3)


def indexer_losses(model: nn.Module) -> dict[str, float]:
    """Last step's per-layer indexer loss, for logging."""
    return {
        name: module.last_indexer_loss
        for name, module in model.named_modules()
        if isinstance(module, DSAInnerAttention)
    }
