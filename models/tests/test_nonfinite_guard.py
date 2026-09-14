"""The non-finite gradient guard.

The failure it exists for: one `nan` gradient is all-reduced to every rank,
`clip_grads_with_norm_` scales every gradient by `nan`, and the optimizer
writes `nan` into the parameters *and* the moments -- after which every
subsequent step is `nan`. Measured at 16, 64 and 256 nodes on two datasets.
"""

from __future__ import annotations

import pytest
import torch

from train.utils import zero_grads_if_nonfinite_


def _params(n=3, numel=4, grad_fill=1.0):
    ps = []
    for _ in range(n):
        p = torch.nn.Parameter(torch.zeros(numel))
        p.grad = torch.full((numel,), grad_fill)
        ps.append(p)
    return ps


@pytest.mark.parametrize("bad_norm", [float("nan"), float("inf"), float("-inf")])
def test_nonfinite_norm_zeroes_every_gradient(bad_norm):
    ps = _params()
    hit = zero_grads_if_nonfinite_(ps, torch.tensor(bad_norm))
    assert hit.item() == 1.0
    for p in ps:
        assert torch.all(p.grad == 0), p.grad


def test_finite_norm_leaves_gradients_alone():
    ps = _params(grad_fill=2.5)
    hit = zero_grads_if_nonfinite_(ps, torch.tensor(3.7))
    assert hit.item() == 0.0
    for p in ps:
        assert torch.all(p.grad == 2.5)


def test_nan_gradients_are_actually_cleared_not_scaled():
    """`nan * 0.0` is `nan`. Scaling by `isfinite(norm)` would look like a fix
    and leave the poison in place, so the guard must select, not multiply."""
    ps = _params()
    ps[0].grad[1] = float("nan")
    ps[2].grad[:] = float("inf")

    zero_grads_if_nonfinite_(ps, torch.tensor(float("nan")))

    for p in ps:
        assert torch.all(torch.isfinite(p.grad)), p.grad
        assert torch.all(p.grad == 0), p.grad


def test_params_without_grads_are_skipped():
    ps = _params()
    ps[1].grad = None
    zero_grads_if_nonfinite_(ps, torch.tensor(float("nan")))
    assert ps[1].grad is None
    assert torch.all(ps[0].grad == 0)


@pytest.mark.parametrize("shape", [(), (1,)])
def test_accepts_scalar_or_one_element_norm(shape):
    ps = _params()
    zero_grads_if_nonfinite_(ps, torch.full(shape, float("nan")))
    assert torch.all(ps[0].grad == 0)


def test_returned_flag_accumulates_as_a_counter():
    """The trainer does `self.nonfinite_skips += ...` on a device tensor."""
    total = torch.zeros(())
    for norm in [1.0, float("nan"), 2.0, float("inf")]:
        total += zero_grads_if_nonfinite_(_params(), torch.tensor(norm))
    assert total.item() == 2.0


def test_moments_survive_a_skipped_step():
    """Zeroed gradients must leave an Adam state that still carries history --
    that is the difference between a skipped step and a corrupted optimizer."""
    p = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([p], lr=1e-3)

    p.grad = torch.full((4,), 0.5)
    opt.step()
    good_exp_avg = opt.state[p]["exp_avg"].clone()
    assert torch.all(good_exp_avg != 0)

    p.grad = torch.full((4,), float("nan"))
    zero_grads_if_nonfinite_([p], torch.tensor(float("nan")))
    opt.step()

    st = opt.state[p]
    assert torch.all(torch.isfinite(st["exp_avg"]))
    assert torch.all(torch.isfinite(st["exp_avg_sq"]))
    assert torch.all(torch.isfinite(p))
    # decayed, not destroyed
    assert torch.allclose(st["exp_avg"], good_exp_avg * 0.9)
