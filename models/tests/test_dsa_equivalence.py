"""Equivalence harness for the vendored min-memory DSA kernels.

Before DSA goes anywhere near a model, this checks the four things the port can
get wrong independently of sparsity: the vendoring and shim, the `[s, b, h, d]`
layout, GQA head grouping, and the causal mask. The lever is the budget -- at
`index_topk == seq_len` the indexer selects every causal key, so DSA must
reproduce dense causal attention exactly, and any mismatch is one of those four
and not a property of the algorithm.

The reference is fp32 SDPA, not our own kernel: `varlen_attn` runs in bf16 and
would turn a layout bug and kernel noise into the same number. It is measured
against the same reference so the two errors can be compared.

The remaining tests cover what the dense case cannot reach: that Triton and the
PyTorch oracle agree under real sparsity, that a real budget actually bites, and
that the fused indexer loss produces gradients.

    pytest models/tests/test_dsa_equivalence.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F

from models.common.attention import (
    create_varlen_metadata_for_document,
    VarlenInnerAttention,
    VarlenMetadata,
)
from models.common.dsa.inner_attention import DSAIndexer, DSAInnerAttention
from models.common.dsa.inner_attention import _group_segments, _segments
from models.common.dsa._shim import ProcessGroupCollection
from models.common.dsa.dsa_min_memory import dsa_dense_indexer_loss, dsa_min_memory_gqa

# Qwen3.5-shaped GQA. Nothing here depends on the exact counts, but a head count
# that is not a multiple of the group count would hide a grouping bug.
N_HEADS, N_GROUPS, HEAD_DIM = 16, 2, 128
INDEX_HEAD_DIM, INDEX_ROTARY_DIM = 128, 64
HIDDEN = 2048

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="DSA kernels are CUDA-only")


class _RopeCarrier(torch.nn.Module):
    """What the kernels actually read off `rotary_pos_emb`: `inv_freq`, nothing else.

    They index positions themselves rather than taking a cos/sin table, so the
    real `MRoPE` cannot be handed over as-is; the frequencies are the whole
    contract.
    """

    def __init__(self, dim: int, base: float = 10000.0, device=None) -> None:
        super().__init__()
        self.inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, device=device, dtype=torch.float32) / dim)
        )
        self.rotary_interleaved = False


class _Indexer(torch.nn.Module):
    """The nine attributes `dsa_min_memory_gqa` reads. No Megatron module needed."""

    def __init__(self, index_topk: int, device, dtype) -> None:
        super().__init__()
        self.linear_q = torch.nn.Linear(HIDDEN, INDEX_HEAD_DIM, bias=False, device=device, dtype=dtype)
        self.linear_k = torch.nn.Linear(HIDDEN, INDEX_HEAD_DIM, bias=False, device=device, dtype=dtype)
        self.index_topk = index_topk
        self.index_head_dim = INDEX_HEAD_DIM
        self.index_rotary_dim = INDEX_ROTARY_DIM
        self.rotary_pos_emb = _RopeCarrier(INDEX_ROTARY_DIM, device=device)
        self.softmax_scale = INDEX_HEAD_DIM**-0.5
        self.pg_collection = ProcessGroupCollection()
        # `config` is read for two getattrs only.
        self.config = type("Cfg", (), {"dsa_indexer_mode": "simplified", "rotary_interleaved": False})()


def _inputs(seq_len: int, index_topk: int, device="cuda", dtype=torch.bfloat16, seed=0):
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape, grad=True):
        t = torch.randn(*shape, generator=g, device=device, dtype=dtype)
        return t.requires_grad_(grad)

    query = rnd(seq_len, 1, N_HEADS, HEAD_DIM)
    key = rnd(seq_len, 1, N_GROUPS, HEAD_DIM)
    value = rnd(seq_len, 1, N_GROUPS, HEAD_DIM)
    hidden = rnd(seq_len, 1, HIDDEN)
    return query, key, value, hidden, _Indexer(index_topk, device, dtype)


def _dense_reference(query, key, value, scale):
    """fp32 causal SDPA, [s, b, h, d] in and out."""
    q, k, v = (t.detach().float().permute(1, 2, 0, 3) for t in (query, key, value))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=scale, enable_gqa=True)
    return out.permute(2, 0, 1, 3)


def _varlen_reference(query, key, value, scale):
    """Our production kernel on a single document, reshaped to its packed [T, H, D]."""
    seq_len = query.size(0)
    cu = torch.tensor([0, seq_len], dtype=torch.int32, device=query.device)
    inner = VarlenInnerAttention(VarlenInnerAttention.Config())
    out = inner(
        query.detach().squeeze(1),
        key.detach().squeeze(1),
        value.detach().squeeze(1),
        attention_masks=VarlenMetadata(cu, cu, seq_len, seq_len),
        scale=scale,
        enable_gqa=True,
    )
    return out.unsqueeze(1)


@pytest.mark.parametrize("use_triton", [True, False])
def test_full_budget_matches_dense(use_triton):
    """`index_topk == seq_len` selects every causal key, so DSA is dense attention."""
    seq_len = 512
    scale = HEAD_DIM**-0.5
    query, key, value, hidden, indexer = _inputs(seq_len, index_topk=seq_len)

    out, loss = dsa_min_memory_gqa(
        query, key, value, hidden, indexer,
        softmax_scale=scale, loss_coeff=0.0, use_indexer_rope=True, use_triton=use_triton,
    )
    out = out.view(seq_len, 1, N_HEADS, HEAD_DIM)

    ref = _dense_reference(query, key, value, scale)
    dsa_err = (out.float() - ref).abs().max().item()
    varlen_err = (_varlen_reference(query, key, value, scale).float() - ref).abs().max().item()
    print(f"\n  triton={use_triton}  max|dsa-ref|={dsa_err:.4f}  max|varlen-ref|={varlen_err:.4f}")

    assert loss.item() == 0.0, "loss_coeff=0 must not produce a loss"
    # bf16 inputs against an fp32 reference: both kernels sit at the same order of
    # error, so the bound is bf16's, and a layout or masking bug blows straight past it.
    assert dsa_err < 5e-2, f"DSA at full budget is not dense attention: {dsa_err}"
    assert varlen_err < 5e-2, f"varlen baseline drifted, so the bound is wrong: {varlen_err}"


def test_triton_matches_torch_oracle_when_sparse():
    """Under a real budget the two backends must still agree.

    `dsa_min_memory.py` calls its PyTorch path the correctness oracle, and the
    Triton path a dispatch for supported tiles. Only a sparse budget exercises
    the merge across key blocks, where the two differ in how they chunk.
    """
    seq_len, scale = 1024, HEAD_DIM**-0.5
    args = _inputs(seq_len, index_topk=128)
    common = dict(softmax_scale=scale, loss_coeff=0.0, use_indexer_rope=True)
    triton_out, _ = dsa_min_memory_gqa(*args, use_triton=True, **common)
    torch_out, _ = dsa_min_memory_gqa(*args, use_triton=False, **common)
    err = (triton_out.float() - torch_out.float()).abs().max().item()
    print(f"\n  max|triton-torch|={err:.5f}")
    assert err < 2e-2, f"Triton and the PyTorch oracle disagree: {err}"


def test_budget_bites():
    """A budget below the sequence length must change the answer.

    Without this a silently-dense kernel would pass every other test here.
    """
    seq_len, scale = 1024, HEAD_DIM**-0.5
    query, key, value, hidden, indexer = _inputs(seq_len, index_topk=64)
    out, _ = dsa_min_memory_gqa(
        query, key, value, hidden, indexer,
        softmax_scale=scale, loss_coeff=0.0, use_indexer_rope=True,
    )
    out = out.view(seq_len, 1, N_HEADS, HEAD_DIM)
    ref = _dense_reference(query, key, value, scale)
    # Early rows are below the budget and must stay exact; late rows must not.
    head_err = (out[:64].float() - ref[:64]).abs().max().item()
    tail_err = (out[-64:].float() - ref[-64:]).abs().max().item()
    print(f"\n  rows<budget err={head_err:.4f}  rows>>budget err={tail_err:.4f}")
    assert head_err < 5e-2, f"rows shorter than the budget must be dense: {head_err}"
    assert tail_err > 1e-1, f"budget had no effect, kernel may be dense: {tail_err}"


def test_backward_and_indexer_loss():
    """Gradients reach every input, and the fused KL produces indexer gradients."""
    seq_len, scale = 512, HEAD_DIM**-0.5
    query, key, value, hidden, indexer = _inputs(seq_len, index_topk=128)
    out, loss = dsa_min_memory_gqa(
        query, key, value, hidden, indexer,
        softmax_scale=scale, loss_coeff=1.0, use_indexer_rope=True,
    )
    assert torch.isfinite(loss) and loss.item() > 0.0, f"indexer loss is {loss.item()}"
    (out.float().square().mean() + loss).backward()
    print(f"\n  indexer loss={loss.item():.4f}")
    for name, t in (
        ("query", query), ("key", key), ("value", value),
        ("linear_q.weight", indexer.linear_q.weight),
        ("linear_k.weight", indexer.linear_k.weight),
    ):
        assert t.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(t.grad).all(), f"{name} gradient is not finite"
        assert t.grad.abs().max().item() > 0.0, f"{name} gradient is all zero"

    # The indexer is cut off from the trunk on purpose: `DSASimplifiedMinMemoryGQAFn`
    # returns None for the `hidden_states` gradient, so the KL trains the two indexer
    # linears and nothing else. Two consequences for the integration: the indexer
    # parameters have to reach the optimizer on their own (no main-loss path will
    # find them), and `hidden_states` may be handed over detached. If this assert
    # ever fails upstream changed its mind and the loss now perturbs the trunk.
    assert hidden.grad is None, "upstream now back-propagates the indexer loss into the trunk"


# ---------------------------------------------------------------------------
# The varlen segment loop: DSA over a row that packs several documents.
# ---------------------------------------------------------------------------

def _build_inner(index_topk: int, min_sparse_len=0, device="cuda", dtype=torch.bfloat16):
    """Built the way FSDP2 leaves it: fp32 buffers, `param_dtype` params."""
    inner = DSAInnerAttention(
        DSAInnerAttention.Config(
            indexer=DSAIndexer.Config(
                dim=HIDDEN, index_topk=index_topk,
                index_head_dim=INDEX_HEAD_DIM, index_rotary_dim=INDEX_ROTARY_DIM,
            ),
            min_sparse_len=min_sparse_len,
        )
    )
    inner.init_states(buffer_device=torch.device(device))
    inner.to(device=device)
    for linear in (inner.indexer.linear_q, inner.indexer.linear_k):
        linear.to(dtype)
    return inner


def _packed_row(doc_lens, device="cuda", dtype=torch.bfloat16, seed=1):
    """Packed `[T, H, D]` tensors plus the metadata the trainer would build."""
    total = sum(doc_lens)
    g = torch.Generator(device=device).manual_seed(seed)

    def rnd(*shape):
        return torch.randn(*shape, generator=g, device=device, dtype=dtype).requires_grad_(True)

    positions = torch.cat(
        [torch.arange(n, device=device, dtype=torch.int32) for n in doc_lens]
    )
    masks = create_varlen_metadata_for_document(positions)
    return rnd(total, N_HEADS, HEAD_DIM), rnd(total, N_GROUPS, HEAD_DIM), \
        rnd(total, N_GROUPS, HEAD_DIM), rnd(total, HIDDEN), masks


def _per_document_reference(q, k, v, doc_lens, scale):
    """fp32 causal SDPA per document -- block-diagonal by construction.

    This is the thing the segment loop has to reproduce: anything that attends
    across a document boundary cannot match it.
    """
    out, start = torch.empty_like(q, dtype=torch.float32), 0
    for n in doc_lens:
        sl = slice(start, start + n)
        qs, ks, vs = (t[sl].detach().float().unsqueeze(0).transpose(1, 2) for t in (q, k, v))
        o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=scale, enable_gqa=True)
        out[sl] = o.transpose(1, 2).squeeze(0)
        start += n
    return out


def test_segment_loop_is_block_diagonal():
    """Dense warmup over a packed row must equal per-document attention.

    `min_sparse_len` above every document length sends all of them down the
    dense branch, which isolates the segmentation from the sparsity: if the loop
    mis-slices, or lets one document see another, this is where it shows.
    """
    doc_lens, scale = [1024, 256, 128], HEAD_DIM**-0.5
    q, k, v, hidden, masks = _packed_row(doc_lens)
    inner = _build_inner(index_topk=128, min_sparse_len=10**9)

    out = inner(q, k, v, attention_masks=masks, scale=scale, enable_gqa=True, hidden_states=hidden)
    ref = _per_document_reference(q, k, v, doc_lens, scale)
    varlen = VarlenInnerAttention(VarlenInnerAttention.Config())(
        q.detach(), k.detach(), v.detach(),
        attention_masks=masks, scale=scale, enable_gqa=True,
    )
    dsa_err = (out.float() - ref).abs().max().item()
    varlen_err = (varlen.float() - ref).abs().max().item()
    print(f"\n  max|dsa-ref|={dsa_err:.4f}  max|varlen-ref|={varlen_err:.4f}")
    assert dsa_err < 5e-2, f"segment loop is not block-diagonal causal: {dsa_err}"
    assert varlen_err < 5e-2, f"varlen baseline drifted, so the bound is wrong: {varlen_err}"
    assert inner.last_indexer_loss > 0.0, "dense branch did not train the indexer"


def test_short_documents_stay_dense_long_one_goes_sparse():
    """The mixed row: the budget bites only the document that exceeds it."""
    doc_lens, topk, scale = [2048, 256, 128], 256, HEAD_DIM**-0.5
    q, k, v, hidden, masks = _packed_row(doc_lens)
    inner = _build_inner(index_topk=topk)  # min_sparse_len defaults to the budget

    out = inner(q, k, v, attention_masks=masks, scale=scale, enable_gqa=True, hidden_states=hidden)
    ref = _per_document_reference(q, k, v, doc_lens, scale)
    long_tail = (out[2048 - 128 : 2048].float() - ref[2048 - 128 : 2048]).abs().max().item()
    short = (out[2048:].float() - ref[2048:]).abs().max().item()
    print(f"\n  long-doc tail err={long_tail:.4f}  short-doc err={short:.4f}")
    assert short < 5e-2, f"documents within the budget must stay dense: {short}"
    assert long_tail > 1e-1, f"the long document was not routed to DSA: {long_tail}"


def test_segment_loop_backward_reaches_the_indexer():
    """The side loss has to carry gradient without being returned."""
    doc_lens, scale = [2048, 256], HEAD_DIM**-0.5
    q, k, v, hidden, masks = _packed_row(doc_lens)
    inner = _build_inner(index_topk=256)

    out = inner(q, k, v, attention_masks=masks, scale=scale, enable_gqa=True, hidden_states=hidden)
    out.float().square().mean().backward()
    print(f"\n  indexer loss={inner.last_indexer_loss:.4f}")
    for name, t in (
        ("query", q), ("key", k), ("value", v),
        ("linear_q.weight", inner.indexer.linear_q.weight),
        ("linear_k.weight", inner.indexer.linear_k.weight),
    ):
        assert t.grad is not None, f"{name} got no gradient"
        assert torch.isfinite(t.grad).all(), f"{name} gradient is not finite"
        assert t.grad.abs().max().item() > 0.0, f"{name} gradient is all zero"
    # Same reason as the single-document case: the indexer is cut off from the trunk.
    assert hidden.grad is None, "the indexer loss now reaches the trunk"


def test_uncovered_boundaries_are_rejected():
    """A metadata bug must fail, not ship uninitialized rows."""
    q, k, v, hidden, _ = _packed_row([256, 128])
    cu = torch.tensor([0, 256], dtype=torch.int32, device=q.device)  # 128 tokens unaccounted
    inner = _build_inner(index_topk=64, min_sparse_len=10**9)
    with pytest.raises(AssertionError, match="do not cover"):
        inner(q, k, v, attention_masks=VarlenMetadata(cu, cu, 256, 256),
              scale=HEAD_DIM**-0.5, enable_gqa=True, hidden_states=hidden)


class _Block(torch.nn.Module):
    """The shape of a decoder block around the kernel: traceable work, then DSA."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.norm = torch.nn.RMSNorm(HEAD_DIM, device="cuda", dtype=torch.bfloat16)

    def forward(self, q, k, v, hidden, masks):
        out = self.inner(
            self.norm(q), self.norm(k), v, attention_masks=masks,
            scale=HEAD_DIM**-0.5, enable_gqa=True, hidden_states=hidden,
        )
        return out.float().square().mean()


def test_compiled_block_breaks_at_the_kernel_instead_of_failing():
    """`apply_compile` drops `fullgraph` for DSA blocks; this is why.

    `forward` is `torch._dynamo.disable`d, so with the break allowed the block
    compiles around the kernel and matches eager bit for bit. Demanded whole it
    raises, which is what `supports_fullgraph = False` exists to avoid.
    """
    q, k, v, hidden, masks = _packed_row([1024, 512])
    block = _Block(_build_inner(index_topk=256))

    eager = block(q, k, v, hidden, masks)
    eager.backward()
    eager_grad = q.grad.clone()
    q.grad = None

    torch._dynamo.reset()
    with pytest.raises(Exception, match="(?i)graph break|fullgraph|skip"):
        torch.compile(block, fullgraph=True)(q, k, v, hidden, masks)

    torch._dynamo.reset()
    compiled = torch.compile(block, fullgraph=False)(q, k, v, hidden, masks)
    compiled.backward()
    print(f"\n  eager={eager.item():.6f} compiled={compiled.item():.6f}")
    # The kernel itself is untouched -- the difference is Inductor's fused
    # RMSNorm reassociating, which lands around 1e-9 and varies with whatever
    # kernel autotuning picks on the card the test runs on. Nowhere near a real
    # divergence, so the tolerance is tight rather than zero.
    torch.testing.assert_close(compiled, eager, rtol=0, atol=1e-8)
    torch.testing.assert_close(q.grad, eager_grad, rtol=0, atol=1e-8)


def test_padding_is_free():
    """Zero query rows must add no KL, or batching unequal documents is wrong.

    A padded row projects to zero under both the main and the indexer weights,
    so the teacher and the indexer are the same uniform distribution over the
    same keys and their KL is zero. That is what lets `_group_segments` stack
    documents of different lengths into one call.
    """
    inner = _build_inner(index_topk=1024)
    length, padded = 320, 512
    q, k, _, hidden, _ = _packed_row([length])
    q, k, hidden = (t.unsqueeze(1) for t in (q, k, hidden))

    bare = dsa_dense_indexer_loss(
        q, k, hidden, inner.indexer, softmax_scale=HEAD_DIM**-0.5,
        loss_coeff=1.0, use_indexer_rope=True,
    )
    qp = q.new_zeros(padded, *q.shape[1:]); qp[:length] = q
    kp = k.new_zeros(padded, *k.shape[1:]); kp[:length] = k
    hp = hidden.new_zeros(padded, *hidden.shape[1:]); hp[:length] = hidden
    with_pad = dsa_dense_indexer_loss(
        qp, kp, hp, inner.indexer, softmax_scale=HEAD_DIM**-0.5,
        loss_coeff=1.0, use_indexer_rope=True,
    )
    # The kernel means over every row it was given, so the padded call is the
    # same sum spread over more rows.
    rescaled = with_pad * padded / length
    print(f"\n  bare={bare.item():.3e} padded*{padded / length:.2f}={rescaled.item():.3e}")
    torch.testing.assert_close(rescaled, bare, rtol=2e-2, atol=0)


def test_indexer_loss_does_not_move_with_the_packing():
    """One mean over real query rows, so repacking the same text cannot change it.

    The earlier version summed one mean per document, which weighted a row of
    25 short documents 25x a row holding one long one -- the same corpus would
    train the indexer differently depending on how the packer filled the row.
    """
    scale = HEAD_DIM**-0.5
    inner = _build_inner(index_topk=1024, min_sparse_len=1024)

    def loss_of(doc_lens):
        q, k, v, hidden, masks = _packed_row(doc_lens, seed=3)
        inner(q, k, v, attention_masks=masks, scale=scale,
              enable_gqa=True, hidden_states=hidden)
        return inner.last_indexer_loss

    one, eight = loss_of([512]), loss_of([512] * 8)
    print(f"\n  1 document {one:.4e} | 8 identical {eight:.4e}")
    assert one > 0.0, "nothing was measured"
    assert abs(eight - one) / one < 0.3, (
        f"loss moved {eight / one:.2f}x when the same document was packed 8 times"
    )


def test_grouping_keeps_documents_apart():
    """Unequal short documents share one padded call; they must not mix.

    `_group_segments` pads to a multiple of 256, so these three land in two
    groups, one of which carries two documents of different lengths.
    """
    doc_lens = [200, 300, 700]
    scale = HEAD_DIM**-0.5
    q, k, v, hidden, masks = _packed_row(doc_lens)
    inner = _build_inner(index_topk=1024, min_sparse_len=1024)

    dense, sparse = _group_segments(_segments(masks.cu_seq_q), 1024)
    assert not sparse and sorted(dense) == [256, 512, 768], (dense, sparse)
    assert len(dense[512]) == 1 and len(dense[256]) == 1

    out = inner(q, k, v, attention_masks=masks, scale=scale,
                enable_gqa=True, hidden_states=hidden)
    reference = _per_document_reference(q, k, v, doc_lens, scale)
    err = (out.float() - reference).abs().max().item()
    print(f"\n  max abs error vs per-document reference {err:.4f}")
    assert err < 0.02, err
