"""Row-sparse AdamW for Engram tables (`train/engram_optim.py`): touched rows get
exactly AdamW's update, untouched rows do not move, other params are AdamW's."""

import torch

from train.engram_optim import register_lazy_adamw

ROWS, DIM = 20, 4


def _setup(seed=0):
    g = torch.Generator().manual_seed(seed)
    table = torch.nn.Parameter(torch.randn(ROWS, DIM, generator=g))
    other = torch.nn.Parameter(torch.randn(3, 3, generator=g))
    return table, other, g


def test_first_step_equals_dense_adamw():
    # from zero state, rows with zero gradient do not move under dense AdamW either
    table, other, g = _setup()
    ref_t, ref_o = table.detach().clone().requires_grad_(), other.detach().clone().requires_grad_()
    opt = torch.optim.AdamW([{"params": [table, other], "weight_decay": 0.0}], lr=1e-2)
    ref = torch.optim.AdamW([{"params": [ref_t, ref_o], "weight_decay": 0.0}], lr=1e-2)
    register_lazy_adamw(opt, [table])

    rows = torch.tensor([1, 5, 5, 7])
    grad = torch.zeros(ROWS, DIM)
    grad[rows] = torch.randn(len(rows), DIM, generator=g)
    og = torch.randn(3, 3, generator=g)
    table.touched_rows.append(rows)
    table.grad, ref_t.grad = grad.clone(), grad.clone()
    other.grad, ref_o.grad = og.clone(), og.clone()
    opt.step()
    ref.step()
    torch.testing.assert_close(table, ref_t)
    torch.testing.assert_close(other, ref_o)
    assert table.touched_rows == []
    assert set(opt.state[table]) == {"step", "exp_avg", "exp_avg_sq"}


def test_untouched_rows_keep_their_values_and_state():
    table, other, g = _setup(1)
    opt = torch.optim.AdamW([{"params": [table], "weight_decay": 0.0}], lr=1e-2, betas=(0.9, 0.99))
    register_lazy_adamw(opt, [table])

    def step(rows):
        grad = torch.zeros(ROWS, DIM)
        grad[rows] = torch.randn(len(rows), DIM, generator=g)
        table.touched_rows.append(rows)
        table.grad = grad
        opt.step()
        return grad

    step(torch.tensor([0, 1, 2]))
    before = table.detach().clone()
    m_before = opt.state[table]["exp_avg"].clone()
    v_before = opt.state[table]["exp_avg_sq"].clone()
    grad = step(torch.tensor([2, 3]))

    untouched = [r for r in range(ROWS) if r not in (2, 3)]
    torch.testing.assert_close(table[untouched], before[untouched])
    torch.testing.assert_close(opt.state[table]["exp_avg"][untouched], m_before[untouched])
    # touched rows: AdamW's formula at step 2
    b1, b2, eps, lr = 0.9, 0.99, 1e-8, 1e-2
    for r in (2, 3):
        m = b1 * m_before[r] + (1 - b1) * grad[r]
        v = b2 * v_before[r] + (1 - b2) * grad[r] ** 2
        want = before[r] - lr * (m / (1 - b1**2)) / ((v / (1 - b2**2)).sqrt() + eps)
        torch.testing.assert_close(table[r], want)
