"""Distillation loss: `chunked_loss` with a teacher (train/train_opd.py, docs/OPD.md).

    CUDA_VISIBLE_DEVICES=3 pytest models/tests/test_distill_loss.py -q
"""

from __future__ import annotations

import pytest
import torch
from torch.distributions import Categorical, kl_divergence

from train.loss import IGNORE_INDEX, chunked_loss, cross_entropy_sum, distill_sums

T, D, V = 32, 16, 64
CHUNKS = 4


def _setup():
    torch.manual_seed(0)
    head = torch.nn.Linear(D, V, bias=False)
    t_head = torch.nn.Linear(D, V, bias=False).requires_grad_(False)
    hidden = torch.randn(T, D, requires_grad=True)
    t_hidden = torch.randn(T, D)
    labels = torch.randint(0, V, (T,))
    labels[::3] = IGNORE_INDEX
    return head, t_head, hidden, t_hidden, labels


@pytest.mark.parametrize("reverse", [False, True])
def test_distill_sums_matches_torch_distributions(reverse):
    _, _, _, _, labels = _setup()
    logits, t_logits = torch.randn(T, V), torch.randn(T, V)
    ce, kl = distill_sums(logits, t_logits, labels, reverse)

    p, q = Categorical(logits=logits), Categorical(logits=t_logits)
    per_tok = kl_divergence(p, q) if reverse else kl_divergence(q, p)
    mask = labels != IGNORE_INDEX
    assert torch.allclose(kl, per_tok[mask].sum(), atol=1e-4)
    assert torch.allclose(ce, cross_entropy_sum(logits, labels), atol=1e-4)


@pytest.mark.parametrize("reverse", [False, True])
def test_chunked_equals_unchunked_value_and_grad(reverse):
    head, t_head, hidden, t_hidden, labels = _setup()
    kw, ce_w = 0.7, 0.3
    gt = torch.tensor(float((labels != IGNORE_INDEX).sum()))
    metrics = {}

    loss = chunked_loss(
        hidden, labels, head, gt, CHUNKS, metrics=metrics,
        teacher_hidden=t_hidden, teacher_lm_head=t_head,
        kl_weight=kw, ce_weight=ce_w, reverse_kl=reverse,
    )
    loss.backward()
    got_grad = hidden.grad.clone()

    ref_h = hidden.detach().clone().requires_grad_(True)
    ce, kl = distill_sums(head(ref_h), t_head(t_hidden), labels, reverse)
    want = (ce_w * ce + kw * kl) / gt
    want.backward()

    assert torch.allclose(loss, want, atol=1e-5)
    assert torch.allclose(got_grad, ref_h.grad, atol=1e-5)
    assert torch.allclose(metrics["kl_sum"], kl.detach(), atol=1e-4)
    assert torch.allclose(metrics["ce_sum"], ce.detach(), atol=1e-4)


def test_self_distillation_has_zero_kl():
    """Teacher = student: KL 0, gradient 0 at kl_weight 1 / ce_weight 0."""
    head, _, hidden, _, labels = _setup()
    gt = torch.tensor(float(T))
    metrics = {}
    loss = chunked_loss(
        hidden, labels, head, gt, CHUNKS, metrics=metrics,
        teacher_hidden=hidden.detach(), teacher_lm_head=head,
    )
    loss.backward()
    assert loss.abs() < 1e-6
    assert hidden.grad.abs().max() < 1e-6
    assert torch.allclose(metrics["ce_sum"], cross_entropy_sum(head(hidden), labels), atol=1e-4)
