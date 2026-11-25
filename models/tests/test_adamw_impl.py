"""`train.optimizer.build_adamw`: implementation selection, and the fp32 master cast.

Everything here runs on CPU except the torchao variants, whose quantized
moments are CUDA-only.
"""

import pytest
import torch

from train.optimizer import ADAMW_IMPLS, TORCHAO_ADAMW, build_adamw
from train.utils import cast_master_weights


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


def test_config_default_impl_is_valid():
    from train.config import Training

    assert Training().adamw_impl in ADAMW_IMPLS


@pytest.mark.cuda_only
@pytest.mark.parametrize("impl", sorted(TORCHAO_ADAMW))
def test_torchao_impls_step(impl):
    dtype = torch.float32
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


def test_master_cast_moves_params_only():
    m = _RopeModel().to(torch.bfloat16)
    m.inv_freq = m.inv_freq.float()
    before = m.inv_freq.clone()
    assert cast_master_weights(m) is torch.float32
    assert all(p.dtype is torch.float32 for p in m.parameters())
    # `Module.to(dtype)` would take this with it; it must not.
    assert m.inv_freq.dtype is torch.float32
    assert torch.equal(m.inv_freq, before)


def test_master_cast_preserves_weight_tying():
    m = _RopeModel()
    cast_master_weights(m)
    assert m.lm_head.weight is m.embed.weight


def test_bf16_rope_table_would_be_wrong():
    """Why `cast_master_weights` skips buffers.

    bf16 keeps 8 mantissa bits, so rounding `inv_freq` is worth radians of
    angle error by the time it is multiplied by a position a few thousand
    tokens in -- a silent rope corruption, not a rounding nit.
    """
    f32 = 1.0 / (1e6 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64))
    err = (f32.bfloat16().float() - f32).abs()
    assert (err * 4000).max() > 1.0
