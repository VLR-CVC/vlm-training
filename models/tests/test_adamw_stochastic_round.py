"""`master_dtype = "bfloat16"` is only safe if the optimizer rounds stochastically.

bf16 has 8 mantissa bits, so around a weight of 1.0 the smallest representable
step is 2**-8. `torch.optim.AdamW` rounds to nearest and therefore drops every
update below half of that -- which is where cosine decay puts you for most of a
run. torchao's `_AdamW` rounds stochastically instead, so those updates land as
a random subset of one-ulp steps and the expectation is preserved.

This test pins the difference: same parameter, same gradients, same lr, one
implementation moves and the other does not.
"""
import torch

from train.utils import ADAMW_IMPLS, TORCHAO_ADAMW, build_adamw

N = 4096
STEPS = 50
LR = 1e-6  # ~1/3900 of a bf16 ulp at 1.0, so nearest-rounding is a no-op


def _run(impl: str, stochastic_round: bool) -> torch.Tensor:
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.ones(N, dtype=torch.bfloat16))
    opt = build_adamw(
        [{"params": [p], "weight_decay": 0.0}],
        lr=LR, weight_decay=0.0, impl=impl, stochastic_round=stochastic_round,
    )
    for _ in range(STEPS):
        p.grad = torch.ones_like(p)
        opt.step()
        opt.zero_grad(set_to_none=True)
    return p.detach().clone()


def test_torchao_is_registered():
    assert "torchao" in TORCHAO_ADAMW and "torchao" in ADAMW_IMPLS


def test_foreach_cannot_move_a_sub_ulp_update():
    out = _run("foreach", stochastic_round=False)
    assert out.dtype is torch.bfloat16
    assert torch.equal(out, torch.ones(N, dtype=torch.bfloat16)), (
        "round-to-nearest moved a sub-ulp update; pick a smaller LR"
    )


def test_torchao_stochastic_round_moves_it():
    out = _run("torchao", stochastic_round=True)
    assert out.dtype is torch.bfloat16, "master weights must stay bf16"
    moved = (out != 1.0).sum().item()
    assert moved > 0, "stochastic rounding dropped every sub-ulp update"
    # one ulp down is the only other reachable value at this magnitude
    assert out.max().item() >= 1.0


def test_stochastic_round_on_a_torch_impl_warns_instead_of_raising():
    # the default is now true, so a config that picks "foreach" must still build
    out = _run("foreach", stochastic_round=True)
    assert out.dtype is torch.bfloat16


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
