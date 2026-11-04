"""fp8 grouped GEMM wiring for the MoE experts.

    pytest models/tests/test_fp8_grouped_mm.py -q

`torch._scaled_grouped_mm` runs only on compute capability 9.0 / 10.0 (GH200
yes; the L40S and workstation-Blackwell cards here are sm_89 and sm_120), so the
kernel itself CANNOT be exercised on this machine. What is testable, and what
these cover, is everything around it -- the part that would silently compute the
wrong thing rather than raise:

  * fp8 and the padded dispatcher are switched on together, never separately:
    torchao 0.18.0 refuses to pad internally, and an unaligned token group is a
    wrong answer, not an error;
  * the operand handed to torchao has the layout it documents (`B_t` of shape
    (E, K, N) with strides (K*N, 1, K)) and `offs` is the cumsum of the PADDED
    counts;
  * bf16 remains bit-identical to before.

The kernel numerics are checked on JUPITER by scripts/jup_fp8_check.sbatch.
"""

from __future__ import annotations

import pytest
import torch

from models.common.moe import GroupedExperts, set_moe_grouped_mm
from models.common.token_dispatcher import (
    AllToAllTokenDispatcher,
    LocalTokenDispatcher,
    TorchAOTokenDispatcher,
)


def _experts_config(**kw):
    return GroupedExperts.Config(
        dim=16, hidden_dim=8, num_experts=4, param_init={}, **kw
    )


class _FakeLayer:
    """Just enough of a layer config for the walker: a `moe` with routed experts."""

    class _Routed:
        def __init__(self, dispatcher):
            self.inner_experts = _experts_config()
            self.token_dispatcher = dispatcher

    class _MoE:
        def __init__(self, dispatcher):
            self.routed_experts = _FakeLayer._Routed(dispatcher)

    def __init__(self, dispatcher=None, moe=True):
        self.moe = (
            _FakeLayer._MoE(
                dispatcher or LocalTokenDispatcher.Config(num_experts=4, top_k=2)
            )
            if moe
            else None
        )


@pytest.fixture
def hopper(monkeypatch):
    """Pretend to be sm_90. The walker's capability guard is correct to refuse
    the cards in this box, but the wiring it guards still needs testing."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (9, 0))


class _FakeModelConfig:
    def __init__(self, layers):
        self.layers = layers


def test_fp8_also_installs_the_padding_dispatcher(hopper):
    """The pairing is the whole point: fp8 without padded groups is wrong, not
    slow, and torchao 0.18.0 cannot pad for us."""
    cfg = _FakeModelConfig([_FakeLayer(), _FakeLayer()])
    n = set_moe_grouped_mm(cfg, "fp8_rowwise")
    assert n == 2
    for layer in cfg.layers:
        routed = layer.moe.routed_experts
        assert routed.inner_experts.grouped_mm == "fp8_rowwise"
        assert isinstance(routed.token_dispatcher, TorchAOTokenDispatcher.Config)
        assert routed.token_dispatcher.pad_multiple == 16


def test_an_existing_ao_dispatcher_keeps_its_identity(hopper):
    """EP already swapped in a TorchAO dispatcher: re-pad it, do not replace it
    (replacing would drop whatever else EP configured)."""
    ao = TorchAOTokenDispatcher.Config(num_experts=4, top_k=2, pad_multiple=32)
    cfg = _FakeModelConfig([_FakeLayer(dispatcher=ao)])
    set_moe_grouped_mm(cfg, "fp8_rowwise", pad_multiple=16)
    td = cfg.layers[0].moe.routed_experts.token_dispatcher
    assert td is ao and td.pad_multiple == 16


def test_bf16_is_a_no_op_and_bad_kinds_raise(hopper):
    cfg = _FakeModelConfig([_FakeLayer()])
    assert set_moe_grouped_mm(cfg, "bf16") == 0
    assert isinstance(
        cfg.layers[0].moe.routed_experts.token_dispatcher, LocalTokenDispatcher.Config
    )
    with pytest.raises(ValueError, match="unknown grouped_mm"):
        set_moe_grouped_mm(cfg, "int4")
    with pytest.raises(ValueError, match="no MoE layers"):
        set_moe_grouped_mm(_FakeModelConfig([_FakeLayer(moe=False)]), "fp8_rowwise")


def test_fp8_hands_torchao_the_documented_layout():
    """torchao requires B_t of shape (E, K, N) with strides (K*N, 1, K) and an
    `offs` cumsum. Getting the transpose wrong computes a plausible wrong number
    rather than raising, so pin it with a spy instead of the real kernel."""
    import models.common.moe as moe_mod

    experts = _experts_config(grouped_mm="fp8_rowwise").build()
    E, O, I = 4, 8, 16
    A = torch.randn(32, I)
    weight_EOI = torch.randn(E, O, I)
    offs = torch.tensor([8, 16, 24, 32], dtype=torch.int32)

    seen = {}

    def spy(A_, B_t, offs_, out_dtype=None, pad_token_groups_for_grouped_mm=None, **kw):
        seen.update(
            shape=tuple(B_t.shape),
            strides=tuple(B_t.stride()),
            out_dtype=out_dtype,
            pad_flag=pad_token_groups_for_grouped_mm,
            offs=offs_.tolist(),
        )
        return torch._grouped_mm(A_, B_t, offs=offs_)

    import sys
    import types

    mod = types.ModuleType("torchao.prototype.moe_training.fp8_grouped_mm")
    mod._to_fp8_rowwise_then_scaled_grouped_mm = spy
    saved = sys.modules.get(mod.__name__)
    sys.modules[mod.__name__] = mod
    try:
        experts._grouped_mm(A=A.bfloat16(), weight_EOI=weight_EOI, offs=offs)
    finally:
        if saved is not None:
            sys.modules[mod.__name__] = saved
        else:
            del sys.modules[mod.__name__]

    assert seen["shape"] == (E, I, O), seen["shape"]          # (E, K, N)
    assert seen["strides"] == (O * I, 1, I), seen["strides"]  # (K*N, 1, K)
    assert seen["out_dtype"] is torch.bfloat16
    # torchao 0.18.0 asserts on its own padding path; the dispatcher pads instead
    assert seen["pad_flag"] is False
    assert seen["offs"] == [8, 16, 24, 32]
    del moe_mod


def test_bf16_path_is_untouched():
    experts = _experts_config().build()
    assert experts.grouped_mm == "bf16"
    A = torch.randn(8, 16, dtype=torch.bfloat16)
    w = torch.randn(2, 8, 16, dtype=torch.bfloat16)
    offs = torch.tensor([4, 8], dtype=torch.int32)
    if not torch.cuda.is_available():
        pytest.skip("torch._grouped_mm needs CUDA")
    out = experts._grouped_mm(A=A.cuda(), weight_EOI=w.cuda(), offs=offs.cuda())
    ref = torch._grouped_mm(A.cuda(), w.cuda().transpose(-2, -1), offs=offs.cuda())
    assert torch.equal(out, ref)


def test_guard_refuses_unsupported_hardware(monkeypatch):
    """The failure mode this prevents: `torch._scaled_grouped_mm` raising from
    inside the kernel 40 layers deep, with no hint about which card is wrong."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 9))
    with pytest.raises(RuntimeError, match="sm_89"):
        set_moe_grouped_mm(_FakeModelConfig([_FakeLayer()]), "fp8_rowwise")
