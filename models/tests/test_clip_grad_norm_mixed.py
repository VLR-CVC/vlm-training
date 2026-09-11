"""`clip_grad_norm_mixed` has to report a real norm and actually clip.

Both failure modes here are silent: a wrong norm shows up only as a suspicious
`gnorm` in the logs, and a clip that does nothing shows up only as training
that diverges later than it should.
"""

import pytest
import torch

from train.utils import clip_grad_norm_mixed


def _params(scale=1.0, n=4, seed=0):
    g = torch.Generator().manual_seed(seed)
    ps = []
    for _ in range(n):
        p = torch.nn.Parameter(torch.zeros(8, 8))
        p.grad = torch.randn(8, 8, generator=g) * scale
        ps.append(p)
    return ps


def _norm(ps):
    return torch.linalg.vector_norm(
        torch.stack([g.norm() for g in (p.grad for p in ps)])
    )


def test_reports_the_pre_clip_norm():
    ps = _params()
    want = _norm(ps)
    got = clip_grad_norm_mixed(ps, max_norm=1e9)
    assert torch.allclose(torch.as_tensor(float(got)), want, rtol=1e-5)
    assert float(got) > 0


def test_actually_clips():
    ps = _params(scale=10.0)
    before = float(_norm(ps))
    assert before > 1.0
    clip_grad_norm_mixed(ps, max_norm=1.0)
    assert float(_norm(ps)) == pytest.approx(1.0, rel=1e-4)


def test_leaves_small_grads_alone():
    ps = _params(scale=1e-3)
    before = [p.grad.clone() for p in ps]
    clip_grad_norm_mixed(ps, max_norm=1.0)
    for b, p in zip(before, ps):
        assert torch.equal(b, p.grad)


def test_matches_torch_clip_grad_norm():
    a = _params(scale=5.0, seed=1)
    b = _params(scale=5.0, seed=1)
    got = float(clip_grad_norm_mixed(a, max_norm=0.7))
    want = float(torch.nn.utils.clip_grad_norm_(b, 0.7))
    assert got == pytest.approx(want, rel=1e-6)
    for pa, pb in zip(a, b):
        assert torch.allclose(pa.grad, pb.grad, rtol=1e-6, atol=1e-8)


def test_accepts_a_generator():
    ps = _params(scale=10.0)
    got = clip_grad_norm_mixed((p for p in ps), max_norm=1.0)
    assert float(got) > 1.0
    assert float(_norm(ps)) == pytest.approx(1.0, rel=1e-4)


def test_no_grads_is_zero():
    p = torch.nn.Parameter(torch.zeros(4))
    assert float(clip_grad_norm_mixed([p], max_norm=1.0)) == 0.0
