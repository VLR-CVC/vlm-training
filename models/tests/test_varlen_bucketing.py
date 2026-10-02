"""`create_varlen_metadata_for_document` buckets the two quantities that reach
the compiled decoder block as guards -- the `cu_seq_q` length and `max_k`.

The bucketing is only sound if the padding documents change nothing. They are
zero-length (`cu_seqlens` repeats `num_tokens`), which every varlen consumer
should skip, but "should" is why the GPU test below exists: it is the one that
fails if a kernel mishandles an empty segment.

CPU tests run anywhere; the equivalence test needs a GPU."""
import pytest
import torch

from models.common.attention import (
    _MIN_DOC_BUCKET,
    _MIN_MAX_SEQLEN,
    _next_pow2,
    create_varlen_metadata_for_document,
)


def _positions(doc_lengths):
    return torch.cat([torch.arange(n) for n in doc_lengths])


def test_padding_is_zero_length_documents():
    """Every padded entry must equal num_tokens, so `diff` sees a 0."""
    pos = _positions([100, 200, 300])
    md = create_varlen_metadata_for_document(pos)
    n = pos.shape[0]

    assert md.cu_seq_q[0] == 0 and md.cu_seq_q[-1] == n
    real = md.cu_seq_q[:4]
    assert real.tolist() == [0, 100, 300, 600]
    assert (md.cu_seq_q[4:] == n).all(), md.cu_seq_q
    lengths = torch.diff(md.cu_seq_q)
    assert lengths[:3].tolist() == [100, 200, 300]
    assert (lengths[3:] == 0).all()


def test_document_count_collapses_to_few_buckets():
    """The MN5 run's 185 distinct counts are what blew dynamo's 64-entry cache."""
    counts = range(90, 470)                       # the range that run actually saw
    lens = set()
    for n in counts:
        md = create_varlen_metadata_for_document(_positions([8] * n))
        lens.add(int(md.cu_seq_q.numel()))
    assert len(lens) <= 4, f"{len(lens)} distinct cu_seq_q lengths: {sorted(lens)}"
    assert lens == {129, 257, 513}, sorted(lens)


def test_max_seqlen_only_ever_rounds_up():
    """Rounding down would under-size the kernel and silently truncate."""
    for lengths in ([1], [5, 5], [1023], [1025], [4096, 7], [8191]):
        pos = _positions(lengths)
        md = create_varlen_metadata_for_document(pos)
        assert md.max_k >= max(lengths), f"{md.max_k} < {max(lengths)} for {lengths}"
        assert md.max_k <= _next_pow2(pos.shape[0])


def test_max_seqlen_collapses_to_few_buckets():
    """Rows long enough that the `<= row length` cap is not the binding limit --
    below ~512 tokens the cap dominates and the floor never applies, which is
    correct but says nothing about bucketing."""
    lens = {create_varlen_metadata_for_document(_positions([n, 16])).max_k
            for n in range(513, 4000)}
    assert lens == {_MIN_MAX_SEQLEN, 2048, 4096}, sorted(lens)


def test_max_seqlen_never_exceeds_the_row():
    """A short row must not be told to expect a 1024-token document."""
    md = create_varlen_metadata_for_document(_positions([40, 24]))
    assert md.max_k == 64, md.max_k


def test_single_document_row():
    """One document filling the row: no padding beyond the bucket floor."""
    md = create_varlen_metadata_for_document(_positions([4096]))
    assert torch.diff(md.cu_seq_q)[0] == 4096
    assert (torch.diff(md.cu_seq_q)[1:] == 0).all()
    assert md.cu_seq_q.numel() == _MIN_DOC_BUCKET + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_padding_does_not_change_varlen_attention():
    """The load-bearing one: zero-length trailing documents must be inert."""
    from models.common.attention import varlen_attn

    torch.manual_seed(0)
    lengths = [96, 160, 48, 208]
    pos = _positions(lengths).cuda()
    n, heads, dim = pos.shape[0], 4, 64

    q, k, v = (torch.randn(n, heads, dim, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))

    padded = create_varlen_metadata_for_document(pos)
    # the same metadata without any padding, which is what shipped before
    cu = torch.tensor([0, 96, 256, 304, 512], dtype=torch.int32, device="cuda")

    out_pad = varlen_attn(q, k, v, padded.cu_seq_q, padded.cu_seq_k,
                          padded.max_q, padded.max_k)
    out_raw = varlen_attn(q, k, v, cu, cu, 256, 256)
    torch.testing.assert_close(out_pad, out_raw, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_padding_does_not_change_gated_delta_net():
    """GDN is 18 of the 24 layers in the hybrid schedule, and it is the mask the
    `cu_seq_q` guard actually fired on. It also *reads* the document count
    (`gdn.py:100`), so the padding is not obviously inert here the way it is for
    `varlen_attn`."""
    from attn_gym.linear import causal_conv1d, chunk_gdn, l2norm

    torch.manual_seed(0)
    lengths = [96, 160, 48, 208]
    pos = _positions(lengths).cuda()
    n, heads, dim = pos.shape[0], 4, 128   # the fused chunk kernel requires K=V=128

    padded = create_varlen_metadata_for_document(pos)
    cu_raw = torch.tensor([0, 96, 256, 304, 512], dtype=torch.int32, device="cuda")
    assert padded.cu_seq_q.numel() > cu_raw.numel(), "test needs padding to exist"

    q, k, v = (torch.randn(1, n, heads, dim, device="cuda", dtype=torch.bfloat16)
               for _ in range(3))
    g = torch.randn(1, n, heads, device="cuda", dtype=torch.float32)
    beta = torch.rand(1, n, heads, device="cuda", dtype=torch.bfloat16)

    def run(cu):
        out, _ = chunk_gdn(
            l2norm(q, cu_seqlens=cu), l2norm(k, cu_seqlens=cu), v, g, beta,
            cu_seqlens=cu, scale=dim ** -0.5, impl="fused",
        )
        return out

    torch.testing.assert_close(run(padded.cu_seq_q), run(cu_raw), rtol=0, atol=0)

    # the per-document conv reset reads cu_seqlens too
    x = torch.randn(1, n, 128, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(128, 4, device="cuda", dtype=torch.bfloat16)
    conv = lambda cu: causal_conv1d(x, w, activation="silu", cu_seqlens=cu)
    torch.testing.assert_close(conv(padded.cu_seq_q), conv(cu_raw), rtol=0, atol=0)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
