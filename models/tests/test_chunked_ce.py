"""Chunking the loss trades memory for nothing else: the number and the gradient
must be identical to the unchunked path. It is also on the hot path, so it must
not reintroduce a host sync.

CPU-only."""
import pytest
import torch

import models.qwen3_5.utils as u
from models.qwen3_5.utils import causal_lm_loss, set_loss_chunk_mb


@pytest.fixture(autouse=True)
def _reset_chunking():
    before = u._CE_CHUNK_BYTES
    yield
    u._CE_CHUNK_BYTES = before


def _batch(t=64, v=512, ignore_every=3, seed=0):
    torch.manual_seed(seed)
    logits = torch.randn(1, t, v, dtype=torch.float32, requires_grad=True)
    labels = torch.randint(0, v, (1, t))
    labels[0, ::ignore_every] = -100
    return logits, labels


def _loss_and_grad(mb, logits, labels):
    set_loss_chunk_mb(mb)
    if logits.grad is not None:
        logits.grad = None
    loss = causal_lm_loss(logits, labels)
    loss.backward()
    return loss.detach().clone(), logits.grad.clone()


def test_chunked_matches_unchunked_value_and_gradient():
    logits, labels = _batch()
    # 1 MiB / (512 vocab * 4 B) = 512 rows, but the floor is 256 and there are
    # 63 rows after the shift -- so force several chunks by shrinking the budget
    # through the vocab instead.
    ref_loss, ref_grad = _loss_and_grad(0, logits, labels)
    for mb in (1, 4, 64):
        got_loss, got_grad = _loss_and_grad(mb, logits, labels)
        torch.testing.assert_close(got_loss, ref_loss, rtol=1e-6, atol=1e-6)
        torch.testing.assert_close(got_grad, ref_grad, rtol=1e-5, atol=1e-6)


def test_several_chunks_are_actually_taken():
    """A parity test that silently ran one chunk would prove nothing."""
    set_loss_chunk_mb(1)
    big_vocab = 1 << 12
    assert u._ce_chunk(big_vocab) == 256, u._ce_chunk(big_vocab)  # hits the floor
    logits, labels = _batch(t=1200, v=big_vocab)
    ref_loss, ref_grad = _loss_and_grad(0, logits, labels)
    got_loss, got_grad = _loss_and_grad(1, logits, labels)
    # 1199 rows after the shift, 256 per chunk -> 5 chunks
    torch.testing.assert_close(got_loss, ref_loss, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(got_grad, ref_grad, rtol=1e-5, atol=1e-6)


def test_all_ignored_is_zero_with_finite_gradient():
    """The unchunked path lost its `== 0` branch to remove a sync; the chunked
    path never had one. Both have to survive a fully-masked batch."""
    torch.manual_seed(0)
    logits = torch.randn(1, 16, 32, requires_grad=True)
    labels = torch.full((1, 16), -100)
    for mb in (0, 1):
        loss, grad = _loss_and_grad(mb, logits, labels)
        assert loss.item() == 0.0, mb
        assert torch.isfinite(grad).all(), mb
        assert (grad == 0).all(), mb


def test_bf16_logits_give_a_bf16_gradient():
    """Backward writes the gradient in the logits' dtype -- that is the point,
    it is what keeps the (N, V) gradient out of fp32."""
    set_loss_chunk_mb(1)
    torch.manual_seed(0)
    logits = torch.randn(1, 300, 4096, dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 4096, (1, 300))
    causal_lm_loss(logits, labels).backward()
    assert logits.grad.dtype is torch.bfloat16


def test_set_loss_chunk_mb_round_trips():
    set_loss_chunk_mb(0)
    assert u._CE_CHUNK_BYTES == 0
    set_loss_chunk_mb(512)
    assert u._CE_CHUNK_BYTES == 512 * 1024 * 1024
    set_loss_chunk_mb(-5)  # clamped, not an error
    assert u._CE_CHUNK_BYTES == 0


def test_qwen3_vl_setter_exists():
    """`train/train_qwen.py:202` imports this unconditionally for Qwen3-VL; it
    was never defined, so every VL run died at startup with ImportError."""
    from models.qwen3_vl.model import set_loss_chunk_mb as vl_set

    import models.qwen3_vl.model as vl

    before = vl._CE_CHUNK_BYTES
    try:
        vl_set(256)
        assert vl._CE_CHUNK_BYTES == 256 * 1024 * 1024
    finally:
        vl._CE_CHUNK_BYTES = before


if __name__ == "__main__":
    import sys

    fns = [(k, v) for k, v in sorted(globals().items()) if k.startswith("test_")]
    for name, fn in fns:
        saved = u._CE_CHUNK_BYTES
        try:
            fn()
            print(f"  ok   {name}")
        except Exception as exc:
            print(f"  FAIL {name}: {type(exc).__name__}: {exc}")
            sys.exit(1)
        finally:
            u._CE_CHUNK_BYTES = saved
    print(f"chunked CE: {len(fns)} checks passed")
