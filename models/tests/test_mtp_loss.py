"""MTP objective: `loss = main + mtp_scale * mean(depth_losses)`.

    CUDA_VISIBLE_DEVICES=1 pytest models/tests/test_mtp_loss.py -q

torchtitan b21f7d43e puts this in `MTPLoss` inside `ChunkedLossWrapper`
(`models/deepseek_v3/mtp.py`). We have one `chunked_loss` function instead of
that class hierarchy, so the weighting lives there and the tuple path is an
extension of the single-tensor one.

The three things that go wrong silently rather than loudly:
  * the weight applied to the REPORTED loss but not to the gradient, leaving the
    auxiliary objective at full strength in the backward;
  * a depth's gradient injected into the wrong output, or summed into the main
    hidden state (the depths chain, so the graph reconverges);
  * the single-output path drifting while only the MTP path is exercised.
"""

from __future__ import annotations

import pytest
import torch

from train.loss import chunked_loss, cross_entropy_sum, IGNORE_INDEX

T, D, V = 32, 16, 64
CHUNKS = 4


def _head():
    torch.manual_seed(0)
    return torch.nn.Linear(D, V, bias=False)


def _rows(n_outputs, device="cpu"):
    torch.manual_seed(1)
    hiddens = tuple(
        torch.randn(T, D, device=device, requires_grad=True) for _ in range(n_outputs)
    )
    labels = tuple(
        torch.randint(0, V, (T,), device=device) for _ in range(n_outputs)
    )
    return hiddens, labels


def test_single_tensor_path_is_unchanged():
    """Everything else here is new behaviour; this is the old contract."""
    head = _head()
    (hidden,), (labels,) = _rows(1)
    denom = torch.tensor(float(T))

    loss = chunked_loss(hidden, labels, head, denom, CHUNKS)
    want = cross_entropy_sum(head(hidden), labels) / denom
    assert torch.allclose(loss, want, atol=1e-5)

    loss.backward()
    assert hidden.grad is not None and hidden.grad.abs().sum() > 0


def test_tuple_of_one_equals_the_bare_tensor():
    """A 1-tuple has no depths, so it must reduce to the main loss exactly --
    and must not divide by zero computing `mtp_scale / num_depths`."""
    head = _head()
    (hidden,), (labels,) = _rows(1)
    denom = torch.tensor(float(T))
    assert torch.equal(
        chunked_loss(hidden, labels, head, denom, CHUNKS),
        chunked_loss((hidden,), (labels,), head, denom, CHUNKS),
    )


@pytest.mark.parametrize("num_depths", [1, 2, 3])
def test_weighted_sum_matches_the_formula(num_depths):
    head = _head()
    hiddens, labels = _rows(num_depths + 1)
    denom = torch.tensor(float(T))
    scale = 0.3

    got = chunked_loss(hiddens, labels, head, denom, CHUNKS, mtp_scale=scale)

    with torch.no_grad():
        terms = [cross_entropy_sum(head(h), lab) / denom
                 for h, lab in zip(hiddens, labels, strict=True)]
    want = terms[0] + scale * sum(terms[1:]) / num_depths
    assert torch.allclose(got, want, atol=1e-5), (got.item(), want.item())


def test_the_weight_reaches_the_gradient_not_just_the_report():
    """A depth's hidden-state gradient must be scaled by `mtp_scale /
    num_depths`. Weighting only the reported scalar would train the auxiliary
    objective at full strength -- and the loss curve would look right."""
    head = _head()
    hiddens, labels = _rows(3)
    denom = torch.tensor(float(T))
    scale = 0.3
    num_depths = 2

    chunked_loss(hiddens, labels, head, denom, CHUNKS, mtp_scale=scale).backward()
    got = [h.grad.clone() for h in hiddens]

    # the same depth on its own, unweighted
    for h in hiddens:
        h.grad = None
    solo = chunked_loss(hiddens[1], labels[1], head, denom, CHUNKS)
    solo.backward()
    unweighted = hiddens[1].grad.clone()

    assert torch.allclose(got[1], unweighted * (scale / num_depths), atol=1e-6)
    # the main path keeps weight 1
    for h in hiddens:
        h.grad = None
    chunked_loss(hiddens[0], labels[0], head, denom, CHUNKS).backward()
    assert torch.allclose(got[0], hiddens[0].grad, atol=1e-6)


def test_every_output_gets_its_own_gradient():
    """Gradients must land on the output they came from. Summing them into the
    main hidden state would still train, just the wrong thing."""
    head = _head()
    hiddens, labels = _rows(3)
    denom = torch.tensor(float(T))

    chunked_loss(hiddens, labels, head, denom, CHUNKS).backward()
    for i, h in enumerate(hiddens):
        assert h.grad is not None, i
        assert h.grad.abs().sum() > 0, i
    # and they are genuinely different tensors, not one broadcast
    assert not torch.allclose(hiddens[0].grad, hiddens[1].grad)


def test_ignored_labels_contribute_nothing():
    """Shifted rows are IGNORE_INDEX wherever the shift ran off the row or
    crossed a document boundary, which is most of the tail at higher depths."""
    head = _head()
    hiddens, labels = _rows(2)
    denom = torch.tensor(float(T))

    all_ignored = (labels[0], torch.full_like(labels[1], IGNORE_INDEX))
    got = chunked_loss(hiddens, all_ignored, head, denom, CHUNKS)
    main_only = chunked_loss(hiddens[0], labels[0], head, denom, CHUNKS)
    assert torch.allclose(got, main_only, atol=1e-6)

    got.backward()
    assert hiddens[1].grad.abs().sum() == 0, "a fully ignored depth has no gradient"


def test_metrics_report_unweighted_depth_losses():
    """The breakdown that reaches wandb. Depths are reported unweighted so they
    are comparable across `mtp_scale` settings."""
    head = _head()
    hiddens, labels = _rows(3)
    denom = torch.tensor(float(T))

    metrics: dict = {}
    chunked_loss(hiddens, labels, head, denom, CHUNKS, mtp_scale=0.3, metrics=metrics)

    assert set(metrics) == {"loss_main", "loss_mtp_1", "loss_mtp_2"}
    with torch.no_grad():
        for i, key in enumerate(("loss_main", "loss_mtp_1", "loss_mtp_2")):
            want = cross_entropy_sum(head(hiddens[i]), labels[i]) / denom
            assert torch.allclose(metrics[key], want, atol=1e-5), key


def test_mismatched_tuple_lengths_raise():
    head = _head()
    hiddens, labels = _rows(3)
    with pytest.raises(ValueError, match="one labels row per output"):
        chunked_loss(hiddens, labels[:2], head, torch.tensor(float(T)), CHUNKS)


def test_chunking_does_not_change_the_result():
    """The whole point of the chunked path is that it is exact."""
    head = _head()
    hiddens, labels = _rows(3)
    denom = torch.tensor(float(T))
    a = chunked_loss(hiddens, labels, head, denom, 1)
    b = chunked_loss(hiddens, labels, head, denom, 8)
    assert torch.allclose(a, b, atol=1e-5), (a.item(), b.item())


def _clip_fixture(main_grad: float, depth_grad: float):
    """A model shaped like the trainer sees it: `mtp_layers` plus a trunk."""
    model = torch.nn.Module()
    model.trunk = torch.nn.Linear(4, 4, bias=False)
    model.mtp_layers = torch.nn.ModuleList([torch.nn.Linear(4, 4, bias=False)])
    for module, value in ((model.trunk, main_grad), (model.mtp_layers[0], depth_grad)):
        module.weight.grad = torch.full_like(module.weight, value)
    return model


def test_depth_gradients_do_not_shrink_the_main_path():
    """The bug this fixes: one global norm is dominated by the freshly
    initialised depths, so `max_grad_norm` scales the MAIN gradients down far
    harder than the same run without MTP. Each group must see the same bound it
    would see alone."""
    from train.utils import clip_grad_norm_mixed, clip_grad_norm_split_mtp

    # main norm is already 1.0 exactly, so a correct clip leaves it untouched
    main_grad = 0.5  # 4x4 entries of 0.5 -> norm 2.0
    model = _clip_fixture(main_grad, depth_grad=20.0)
    main_norm, mtp_norm = clip_grad_norm_split_mtp(model, max_norm=2.0)

    assert torch.allclose(main_norm, torch.tensor(2.0))
    assert torch.allclose(mtp_norm, torch.tensor(80.0))
    assert torch.allclose(
        model.trunk.weight.grad, torch.full((4, 4), main_grad)
    ), "main gradients were scaled by the depths' norm"
    assert torch.allclose(
        model.mtp_layers[0].weight.grad, torch.full((4, 4), 20.0 * 2.0 / 80.0)
    )

    # what the single global clip did instead: main grads cut ~40x
    global_model = _clip_fixture(main_grad, depth_grad=20.0)
    clip_grad_norm_mixed(list(global_model.parameters()), 2.0)
    assert global_model.trunk.weight.grad.abs().max() < main_grad / 20


def test_frozen_depths_report_no_mtp_norm():
    """`mtp_layers` with no grads is the no-MTP computation, and the caller uses
    `None` to decide whether to log and finiteness-check a second norm."""
    from train.utils import clip_grad_norm_split_mtp

    model = _clip_fixture(0.5, depth_grad=20.0)
    model.mtp_layers[0].weight.grad = None
    _, mtp_norm = clip_grad_norm_split_mtp(model, max_norm=2.0)
    assert mtp_norm is None
