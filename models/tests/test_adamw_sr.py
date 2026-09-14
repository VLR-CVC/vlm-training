"""`AdamWSR` has to clear three bars: the rounding must be unbiased, the update
must match the implementation it replaces, and -- the whole reason it exists --
a bf16 weight must actually move at the production learning rate.

CPU-only."""
import math

import pytest
import torch

from train.adamw_sr import AdamWSR, stochastic_round_

# 1/sqrt(4096) = 2**-6: the initialisation scale of a Qwen3.5-9B hidden-size
# weight. bf16 keeps 7 explicit mantissa bits, so one ULP here is 2**-13 =
# 1.22e-4 and half a ULP is 6.1e-5 -- three times the production lr below.
W_SCALE = 1.0 / math.sqrt(4096)
BF16_MANTISSA_BITS = 7
PROD_LR = 2e-5  # lr_llm in configs/jupiter/scaling/qwen3_5_9b.toml


def test_rounding_is_unbiased():
    """Round the same value many times; the mean must land on the input, not on
    either neighbour."""
    torch.manual_seed(0)
    # a value 1/4 of the way from one bf16 neighbour to the next
    lo = torch.tensor([W_SCALE], dtype=torch.bfloat16).float()
    ulp = lo.item() * 2**-BF16_MANTISSA_BITS
    target = lo.item() + 0.25 * ulp

    n = 200_000
    x = torch.full((n,), target, dtype=torch.float32)
    rand = torch.randint(0, 1 << 16, (n,), dtype=torch.int32)
    stochastic_round_(x, rand)

    vals = x.unique()
    assert len(vals) == 2, f"expected two neighbours, got {vals}"
    frac_up = (x > lo.item()).float().mean().item()
    assert abs(frac_up - 0.25) < 0.01, f"rounded up {frac_up:.3f} of the time, want 0.25"
    assert abs(x.mean().item() - target) < ulp * 0.01


def test_rounding_is_unbiased_for_negatives():
    """The bit trick rounds away from zero, which is downward for negatives."""
    torch.manual_seed(0)
    base = torch.tensor([W_SCALE], dtype=torch.bfloat16).float().item()
    ulp = base * 2**-BF16_MANTISSA_BITS
    target = -(base + 0.75 * ulp)

    x = torch.full((200_000,), target, dtype=torch.float32)
    stochastic_round_(x, torch.randint(0, 1 << 16, (200_000,), dtype=torch.int32))
    assert abs(x.mean().item() - target) < ulp * 0.01


def test_rounded_values_survive_the_bf16_cast():
    """The low mantissa bits must be zero, so the later cast cannot re-round."""
    torch.manual_seed(0)
    x = torch.randn(1000, dtype=torch.float32)
    stochastic_round_(x, torch.randint(0, 1 << 16, (1000,), dtype=torch.int32))
    assert torch.equal(x.bfloat16().float(), x)


def _run(opt_cls, lr, steps, grad_fn, dtype=torch.bfloat16, **kw):
    torch.manual_seed(0)
    p = torch.full((4096,), W_SCALE, dtype=dtype, requires_grad=True)
    opt = opt_cls([p], lr=lr, weight_decay=0.0, **kw)
    for i in range(steps):
        p.grad = grad_fn(i).to(dtype)
        opt.step()
    return p.detach().clone()


def test_bf16_weights_move_at_the_production_lr():
    """The point of the whole file. Round-to-nearest cannot represent a 2e-5
    update to a 0.0156 weight and silently drops it; stochastic rounding must
    not."""
    grad = lambda i: torch.ones(4096)

    frozen = _run(AdamWSR, PROD_LR, 200, grad, stochastic_round=False)
    moved = _run(AdamWSR, PROD_LR, 200, grad, stochastic_round=True)

    start = torch.full((4096,), W_SCALE, dtype=torch.bfloat16)
    n_frozen = (frozen != start).sum().item()
    n_moved = (moved != start).sum().item()

    assert n_frozen == 0, f"round-to-nearest moved {n_frozen} elements; test premise gone"
    assert n_moved > 4000, f"stochastic rounding only moved {n_moved}/4096"

    # and it moves by about the right amount: 200 steps of a ~lr-sized update
    drift = (start.float() - moved.float()).mean().item()
    assert 0.5 * 200 * PROD_LR < drift < 1.5 * 200 * PROD_LR, f"drift {drift:.3g}"


def test_matches_torchao_without_rounding():
    """With rounding off the update must be term-for-term what
    `torchao.optim._AdamW` produces, since that is the implementation this
    replaces. fp32 params so neither side is quantising."""
    ao = pytest.importorskip("torchao.optim")

    torch.manual_seed(0)
    grads = [torch.randn(512) * 0.01 for _ in range(20)]

    def run(make_opt):
        torch.manual_seed(0)
        p = torch.randn(512, dtype=torch.float32, requires_grad=True) * 0.02
        p = p.detach().requires_grad_(True)
        opt = make_opt([p])
        for g in grads:
            p.grad = g.clone()
            opt.step()
        return p.detach()

    mine = run(lambda ps: AdamWSR(ps, lr=1e-3, weight_decay=0.01, stochastic_round=False))
    theirs = run(lambda ps: ao._AdamW(ps, lr=1e-3, weight_decay=0.01))
    torch.testing.assert_close(mine, theirs, rtol=1e-5, atol=1e-7)


def test_bucketing_does_not_change_the_result():
    """Buckets are an implementation detail; one bucket or ten must agree."""
    grads = [torch.randn(1024) * 0.01 for _ in range(10)]

    def run(bucket_mb):
        torch.manual_seed(0)
        ps = [torch.full((1024,), W_SCALE, dtype=torch.float32, requires_grad=True)
              for _ in range(8)]
        opt = AdamWSR(ps, lr=1e-3, weight_decay=0.01, stochastic_round=False,
                      bucket_mb=bucket_mb)
        for g in grads:
            for p in ps:
                p.grad = g.clone()
            opt.step()
        return torch.cat([p.detach() for p in ps])

    torch.testing.assert_close(run(64), run(1), rtol=0, atol=0)


def test_per_group_lr_is_respected():
    """The trainer runs separate rates for llm / mlp / vit, and buckets must not
    span groups."""
    torch.manual_seed(0)
    slow = torch.full((256,), W_SCALE, dtype=torch.float32, requires_grad=True)
    fast = torch.full((256,), W_SCALE, dtype=torch.float32, requires_grad=True)
    opt = AdamWSR(
        [{"params": [slow], "lr": 1e-6}, {"params": [fast], "lr": 1e-3}],
        weight_decay=0.0, stochastic_round=False,
    )
    for _ in range(5):
        slow.grad = torch.ones(256)
        fast.grad = torch.ones(256)
        opt.step()
    d_slow = (W_SCALE - slow.detach()).abs().mean().item()
    d_fast = (W_SCALE - fast.detach()).abs().mean().item()
    assert d_fast > 100 * d_slow, f"slow {d_slow:.3g} fast {d_fast:.3g}"


def test_parameter_larger_than_one_bucket():
    """Regression: the first GPU run died with `scratch too small: 21233664 <
    32219136`. Scratch was sized from the first bucket, and `lm_head` /
    `embed_tokens` are 254M-element shards at tp=4 -- larger than a bucket on
    their own. Parameters are now split across buckets, so a single one can
    never define the bound."""
    big = 1_500_000  # > bucket_elems at bucket_mb=1 (524288)

    def run(bucket_mb):
        torch.manual_seed(0)
        ps = [torch.full((big,), W_SCALE, dtype=torch.float32, requires_grad=True),
              torch.full((999,), W_SCALE, dtype=torch.float32, requires_grad=True)]
        opt = AdamWSR(ps, lr=1e-3, weight_decay=0.01, stochastic_round=False,
                      bucket_mb=bucket_mb)
        for i in range(3):
            for p in ps:
                p.grad = torch.full_like(p, 0.01 * (i + 1))
            opt.step()
        return [p.detach().clone() for p in ps]

    split, whole = run(1), run(64)
    for a, b in zip(split, whole):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_step_count_survives_a_state_dict_round_trip():
    """Bias correction depends on the step number, so a resumed run that
    restarts it at 1 takes a wrong-sized first update. The counter lives in
    `param_groups` precisely so `state_dict()` carries it."""
    torch.manual_seed(0)
    p = torch.full((64,), W_SCALE, dtype=torch.float32, requires_grad=True)
    opt = AdamWSR([p], lr=1e-3, weight_decay=0.0, stochastic_round=False)
    for _ in range(4):
        p.grad = torch.ones(64)
        opt.step()

    sd = opt.state_dict()
    assert sd["param_groups"][0]["step"] == 4

    p2 = torch.full((64,), W_SCALE, dtype=torch.float32, requires_grad=True)
    opt2 = AdamWSR([p2], lr=1e-3, weight_decay=0.0, stochastic_round=False)
    opt2.load_state_dict(sd)
    assert opt2.param_groups[0]["step"] == 4


if __name__ == "__main__":
    import sys

    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        try:
            fn()
            print(f"  ok   {fn.__name__}")
        except Exception as exc:
            print(f"  FAIL {fn.__name__}: {type(exc).__name__}: {exc}")
            sys.exit(1)
    print(f"adamw_sr: {len(fns)} checks passed")
