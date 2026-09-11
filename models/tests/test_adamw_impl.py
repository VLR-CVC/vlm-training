"""`build_adamw`: implementation selection, and what bf16 master weights need.

Everything here runs on CPU except the torchao variants, whose quantized
moments are CUDA-only.
"""

import pytest
import torch

from train.utils import (
    ADAMW_IMPLS,
    MASTER_DTYPES,
    TORCHAO_ADAMW,
    build_adamw,
    cast_master_weights,
)


def _groups(dtype=torch.float32, device="cpu"):
    p = torch.nn.Parameter(torch.randn(64, 64, dtype=dtype, device=device))
    return [{"params": [p], "lr": 1e-3, "weight_decay": 0.0}], p


def _step(optimizer, p, n=3):
    for _ in range(n):
        p.grad = torch.randn_like(p) * 1e-3
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)


def test_torch_impls_select_the_right_kernel():
    for impl, flag in (("foreach", "foreach"), ("fused", "fused")):
        groups, _ = _groups()
        opt = build_adamw(groups, lr=1e-3, weight_decay=0.0, impl=impl)
        assert isinstance(opt, torch.optim.AdamW)
        assert opt.param_groups[0][flag] is True

    groups, _ = _groups()
    opt = build_adamw(groups, lr=1e-3, weight_decay=0.0, impl="forloop")
    assert opt.param_groups[0]["foreach"] is False
    assert opt.param_groups[0]["fused"] is False


def test_unknown_impl_raises():
    groups, _ = _groups()
    with pytest.raises(ValueError, match="adamw_impl must be one of"):
        build_adamw(groups, lr=1e-3, weight_decay=0.0, impl="adafactor")


def test_stochastic_round_rejects_torch_adamw():
    """torch.optim.AdamW rounds to nearest and has no way not to.

    Accepting the flag there would make `master_dtype = "bfloat16"` silently
    the arm that loses ~0.9 loss, which is exactly the failure the flag exists
    to prevent.
    """
    for impl in ("foreach", "fused", "forloop"):
        groups, _ = _groups()
        with pytest.raises(ValueError, match="requires a torchao adamw_impl"):
            build_adamw(
                groups, lr=1e-3, weight_decay=0.0, impl=impl, stochastic_round=True
            )


def test_foreach_is_the_default_impl():
    from train.config import Training

    t = Training()
    assert t.adamw_impl == "foreach"
    assert t.adamw_stochastic_round is False
    assert t.master_dtype == "float32"
    assert t.adamw_impl in ADAMW_IMPLS


def test_torch_impls_step_bf16_params():
    """bf16 master weights work with the torch implementations too -- they just
    round to nearest, which is why `adamw_stochastic_round` is refused there."""
    groups, p = _groups(dtype=torch.bfloat16)
    before = p.detach().clone()
    _step(build_adamw(groups, lr=1e-2, weight_decay=0.0, impl="foreach"), p)
    assert p.dtype is torch.bfloat16
    assert not torch.equal(p.detach(), before)


@pytest.mark.cuda_only
@pytest.mark.parametrize("impl", sorted(TORCHAO_ADAMW))
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_torchao_impls_step(impl, dtype):
    if not torch.cuda.is_available():
        pytest.skip("torchao quantized optimizer states are CUDA-only")
    import torchao.optim as ao_optim

    groups, p = _groups(dtype=dtype, device="cuda")
    before = p.detach().clone()
    opt = build_adamw(groups, lr=1e-2, weight_decay=0.0, impl=impl)
    assert isinstance(opt, getattr(ao_optim, TORCHAO_ADAMW[impl]))
    _step(opt, p)
    assert p.dtype is dtype
    assert torch.isfinite(p).all()
    assert not torch.equal(p.detach(), before)


@pytest.mark.cuda_only
def test_stochastic_round_keeps_tiny_updates():
    """The reason the flag exists.

    An update far below bf16's resolution (8 mantissa bits) is discarded by
    round-to-nearest every step forever, so the weight never moves. Stochastic
    rounding keeps it in expectation, so repeated tiny updates do accumulate.
    """
    if not torch.cuda.is_available():
        pytest.skip("torchao quantized optimizer states are CUDA-only")

    def run(stochastic):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.full((256, 256), 1.0, dtype=torch.bfloat16,
                                          device="cuda"))
        opt = build_adamw(
            [{"params": [p], "lr": 1e-6, "weight_decay": 0.0}],
            lr=1e-6, weight_decay=0.0, impl="fp8", stochastic_round=stochastic,
        )
        for _ in range(50):
            p.grad = torch.ones_like(p)
            opt.step()
            opt.zero_grad(set_to_none=True)
        return (p.detach().float() - 1.0).abs().mean().item()

    to_nearest = run(False)
    stochastic = run(True)
    assert to_nearest == 0.0, "expected round-to-nearest to discard the update"
    assert stochastic > 0.0, "expected stochastic rounding to accumulate it"


class _RopeModel(torch.nn.Module):
    """A parameter and the shape of buffer the rope tables actually are:
    non-persistent and fp32 on purpose."""

    def __init__(self):
        super().__init__()
        self.lin = torch.nn.Linear(8, 8)
        self.embed = torch.nn.Embedding(16, 8)
        self.lm_head = torch.nn.Linear(8, 16, bias=False)
        self.lm_head.weight = self.embed.weight  # tied, as in from_pretrained
        inv_freq = 1.0 / (1e6 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
        self.register_buffer("inv_freq", inv_freq, persistent=False)


@pytest.mark.parametrize("name,dtype", sorted(MASTER_DTYPES.items()))
def test_master_cast_moves_params_only(name, dtype):
    m = _RopeModel()
    before = m.inv_freq.clone()
    assert cast_master_weights(m, name) is dtype
    assert all(p.dtype is dtype for p in m.parameters())
    # `Module.to(dtype)` would take this with it; it must not.
    assert m.inv_freq.dtype is torch.float32
    assert torch.equal(m.inv_freq, before)


def test_master_cast_preserves_weight_tying():
    m = _RopeModel()
    cast_master_weights(m, "bfloat16")
    assert m.lm_head.weight is m.embed.weight


def test_master_cast_rejects_unknown_dtype():
    with pytest.raises(ValueError, match="master_dtype must be one of"):
        cast_master_weights(_RopeModel(), "float16")


def test_bf16_rope_table_would_be_wrong():
    """Why `cast_master_weights` skips buffers.

    bf16 keeps 8 mantissa bits, so rounding `inv_freq` is worth radians of
    angle error by the time it is multiplied by a position a few thousand
    tokens in -- a silent rope corruption, not a rounding nit.
    """
    f32 = 1.0 / (1e6 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    err = (f32.bfloat16().float() - f32).abs()
    assert (err * 4000).max() > 1.0
