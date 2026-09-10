"""`causal_lm_loss` must stay bit-comparable to the `F.cross_entropy` it replaced."""

import pytest
import torch
import torch.nn.functional as F

import models.qwen3_vl.model as vl
from models.qwen3_vl.model import _ce_chunk, causal_lm_loss, set_loss_chunk_mb


@pytest.fixture(params=[0, 1], ids=["whole", "chunked"])
def chunking(request):
    """Run every parity case through both loss paths."""
    before = vl._CE_CHUNK_BYTES
    set_loss_chunk_mb(request.param)
    yield request.param
    vl._CE_CHUNK_BYTES = before


def _reference(logits, labels, ignore_index=-100):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=ignore_index,
    )


def _case(T, V, dtype, n_ignored, seed=0, device="cpu"):
    g = torch.Generator(device="cpu").manual_seed(seed)
    logits = torch.randn(1, T, V, generator=g).to(device=device, dtype=dtype)
    labels = torch.randint(0, V, (1, T), generator=g).to(device)
    if n_ignored:
        labels[0, :n_ignored] = -100
    return logits, labels


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("n_ignored", [0, 7])
def test_matches_cross_entropy_value(dtype, n_ignored, chunking):
    logits, labels = _case(64, 1024, dtype, n_ignored)
    got = causal_lm_loss(logits, labels)
    want = _reference(logits, labels).float()
    # The loss is always fp32 now: the chunk loop upcasts before the logsumexp,
    # which is what `F.cross_entropy` only did when autocast happened to be on.
    assert got.dtype is torch.float32
    tol = 2e-6 if dtype is torch.float32 else 3e-3
    assert torch.allclose(got, want, rtol=tol, atol=tol), (got.item(), want.item())


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_cross_entropy_grad(dtype, chunking):
    logits, labels = _case(64, 1024, dtype, 7)
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    causal_lm_loss(a, labels).backward()
    _reference(b, labels).backward()
    tol = 1e-6 if dtype is torch.float32 else 8e-3
    assert torch.allclose(a.grad, b.grad, rtol=tol, atol=tol)


def test_spans_several_chunks():
    """The chunk loop has to be exercised, not just its first iteration."""
    V = 1024
    set_loss_chunk_mb(1)
    chunk = _ce_chunk(V)
    T = 2 * chunk + 3
    logits, labels = _case(T, V, torch.float32, 5)
    a = logits.clone().requires_grad_(True)
    b = logits.clone().requires_grad_(True)
    assert T - 1 > 2 * chunk
    try:
        causal_lm_loss(a, labels).backward()
    finally:
        set_loss_chunk_mb(0)
    _reference(b, labels).backward()
    assert torch.allclose(a.grad, b.grad, rtol=1e-6, atol=1e-6)


def test_all_ignored_is_zero_with_grad(chunking):
    logits, labels = _case(32, 256, torch.float32, 0)
    labels[:] = -100
    logits = logits.requires_grad_(True)
    loss = causal_lm_loss(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def test_chunk_shrinks_with_vocab():
    set_loss_chunk_mb(512)
    try:
        assert _ce_chunk(151936) < _ce_chunk(1024)
        assert _ce_chunk(10**9) >= 256
    finally:
        set_loss_chunk_mb(0)


def test_chunking_is_off_by_default():
    """The whole-row path is the fast one; chunking is opt-in."""
    assert vl._CE_CHUNK_BYTES == 0
