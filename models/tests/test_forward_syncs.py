"""Two rewrites done to remove host syncs from the forward changed real logic,
so pin the behaviour they are supposed to preserve.

CPU-only."""
import torch
import torch.nn.functional as F

from models.qwen3_5.model import packed_positions
from models.qwen3_5.utils import causal_lm_loss


def _reference_positions(cu, total):
    """What the loop version produced, verbatim."""
    pos = torch.zeros(total, dtype=torch.int64)
    for start, end in zip(cu[:-1].tolist(), cu[1:].tolist()):
        pos[start:end] = torch.arange(end - start)
    return pos


def test_packed_positions_matches_the_loop():
    for bounds in (
        [0, 6],            # one document
        [0, 2, 6],         # uneven split
        [0, 3, 6, 9, 12],  # four equal
        [0, 1, 2, 3],      # length-1 documents
    ):
        cu = torch.tensor(bounds, dtype=torch.int32)
        total = bounds[-1]
        got = packed_positions(cu, total)
        want = _reference_positions(cu, total)
        assert torch.equal(got, want), f"{bounds}: {got.tolist()} != {want.tolist()}"


def test_packed_positions_restarts_at_boundaries():
    cu = torch.tensor([0, 3, 7], dtype=torch.int32)
    assert packed_positions(cu, 7).tolist() == [0, 1, 2, 0, 1, 2, 3]


def _reference_loss(logits, labels, ignore_index=-100):
    """The branch-on-a-device-value version this replaced."""
    shift_logits = logits[..., :-1, :].contiguous().float()
    shift_labels = labels[..., 1:].contiguous()
    flat_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_labels = shift_labels.view(-1)
    if (flat_labels != ignore_index).sum() == 0:
        return flat_logits.sum() * 0.0
    return F.cross_entropy(flat_logits, flat_labels, ignore_index=ignore_index)


def test_loss_matches_the_branching_version():
    torch.manual_seed(0)
    logits = torch.randn(1, 12, 50)
    labels = torch.randint(0, 50, (1, 12))
    labels[0, ::3] = -100
    got, want = causal_lm_loss(logits, labels), _reference_loss(logits, labels)
    assert torch.allclose(got, want, atol=1e-6), f"{got} vs {want}"


def test_loss_is_zero_when_every_label_is_ignored():
    """The whole point of the branch that was removed. `reduction='sum'` over a
    clamped count has to reproduce it, including a finite backward."""
    torch.manual_seed(0)
    logits = torch.randn(1, 8, 20, requires_grad=True)
    labels = torch.full((1, 8), -100)
    loss = causal_lm_loss(logits, labels)
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(logits.grad).all()
    assert (logits.grad == 0).all()


def test_loss_ignores_only_the_ignored():
    """A single supervised token: the loss must equal that token's CE exactly,
    not an average diluted by the ignored ones."""
    torch.manual_seed(0)
    logits = torch.randn(1, 5, 7)
    labels = torch.full((1, 5), -100)
    labels[0, 3] = 2  # predicted from position 2 after the shift
    got = causal_lm_loss(logits, labels)
    want = F.cross_entropy(logits[0, 2].float().unsqueeze(0), torch.tensor([2]))
    assert torch.allclose(got, want, atol=1e-6), f"{got} vs {want}"


if __name__ == "__main__":
    test_packed_positions_matches_the_loop()
    test_packed_positions_restarts_at_boundaries()
    test_loss_matches_the_branching_version()
    test_loss_is_zero_when_every_label_is_ignored()
    test_loss_ignores_only_the_ignored()
    print("forward syncs: 5 checks passed")
