"""fp8 training for Qwen4 via torchao.

Covers which layers `apply_float8` is allowed to touch, that a converted model
still trains, and the hardware guard. The end-to-end throughput/loss numbers
live in the README -- these are correctness tests, not benchmarks.
"""
from __future__ import annotations

import re

import pytest
import torch
import torch.nn as nn

from models.qwen4.config import Qwen4Config, Qwen4TextConfig, Qwen4VisionConfig
from models.qwen4 import model as ours
from models.tests.test_qwen4_parity import TINY, randomize

from train.infra import (
    _is_grouped_mm_experts,
    apply_float8,
    apply_float8_moe,
    module_filter_float8_fn,
)

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="fp8 needs CUDA")


def _sm() -> tuple[int, int]:
    return torch.cuda.get_device_capability()


sm89 = pytest.mark.skipif(
    torch.cuda.is_available() and _sm() < (8, 9), reason="fp8 needs SM89+"
)


@pytest.fixture(scope="module")
def tiny_model():
    text = Qwen4TextConfig.from_dict(TINY)
    vision = Qwen4VisionConfig(
        depth=2, hidden_size=32, intermediate_size=64, num_heads=2,
        in_channels=3, patch_size=14, spatial_merge_size=2, temporal_patch_size=2,
        out_hidden_size=text.hidden_size, num_position_embeddings=64,
        hidden_act="silu",
    )
    cfg = Qwen4Config(
        text=text, vision=vision, image_token_id=2, video_token_id=3,
        vision_start_token_id=4, vision_end_token_id=5, tie_word_embeddings=False,
        torch_dtype="bfloat16",
    )
    # the default init leaves the norms and the router at zero, which sends
    # a bf16 forward straight to nan; parity's `randomize` is what the rest
    # of the suite uses to get a usable random model.
    def build():
        torch.manual_seed(0)
        return randomize(ours.Qwen4ForCausalLM(cfg))
    return build


def _linears(model):
    return {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}


def test_filter_skips_the_layers_that_must_stay_high_precision(tiny_model):
    model = tiny_model()
    kept = {n for n, m in _linears(model).items() if module_filter_float8_fn(m, n)}
    skipped = {n for n in _linears(model) if n not in kept}

    # the output projection, the frozen QSA indexer and the vision tower
    assert "lm_head" in skipped
    assert any(".indexer." in n for n in skipped)
    assert all(not n.startswith("model.visual") for n in kept)
    # dims not divisible by 16 cannot be quantized at all
    assert all(".in_proj_a" not in n and ".shared_expert_gate" not in n for n in kept)
    # and the ones that carry the FLOPs are in
    for suffix in ("self_attn.q_proj", "self_attn.o_proj", "linear_attn.in_proj_qkv",
                   "linear_attn.out_proj", "mlp.shared_expert.gate_proj"):
        assert any(n.endswith(suffix) for n in kept), suffix


def test_filter_rejects_dims_not_divisible_by_16():
    assert not module_filter_float8_fn(nn.Linear(17, 32), "a")
    assert not module_filter_float8_fn(nn.Linear(32, 17), "a")
    assert module_filter_float8_fn(nn.Linear(32, 32), "a")


def test_filter_accounts_for_tensor_parallel_sharding():
    """The filter runs before `apply_tp`, so it has to look at the sharded dims.

    A 4096x48 projection passes the plain divisible-by-16 test and then fails at
    trace time under TP=2, where the kernel actually sees 4096x24:
    "Expected both dimensions of mat2 to be divisible by 16".
    """
    layer = nn.Linear(4096, 48)
    assert module_filter_float8_fn(layer, "a", divisor=16)
    assert not module_filter_float8_fn(layer, "a", divisor=32)
    assert module_filter_float8_fn(nn.Linear(4096, 96), "a", divisor=32)


@cuda_only
@sm89
def test_tp_size_narrows_the_conversion(tiny_model):
    """Same model, wider divisor: TP=2 may only convert a subset of TP=1's."""
    at_1, _ = apply_float8(tiny_model().cuda(), "tensorwise", tp_size=1)
    at_2, _ = apply_float8(tiny_model().cuda(), "tensorwise", tp_size=2)
    assert at_2 <= at_1


@cuda_only
@sm89
@pytest.mark.parametrize("recipe", ["tensorwise", "rowwise"])
def test_conversion_swaps_exactly_the_filtered_layers(tiny_model, recipe):
    from torchao.float8.float8_linear import Float8Linear

    model = tiny_model().cuda()
    expected = {n for n, m in _linears(model).items() if module_filter_float8_fn(m, n)}
    converted, total = apply_float8(model, recipe)

    got = {n for n, m in model.named_modules() if isinstance(m, Float8Linear)}
    assert got == expected
    assert converted == len(expected)
    assert total == len(expected) + len(_linears(model)) - len(expected)


@cuda_only
@sm89
def test_experts_are_left_alone(tiny_model):
    """The 3D `_grouped_mm` expert weights are not linears and must not move."""
    model = tiny_model().cuda()
    before = {n: p.dtype for n, p in model.named_parameters() if p.ndim == 3}
    apply_float8(model, "tensorwise")
    after = {n: p.dtype for n, p in model.named_parameters() if p.ndim == 3}
    assert before == after
    assert any("experts" in n for n in before)


@cuda_only
@sm89
def test_converted_model_trains(tiny_model):
    """Forward and backward run, and every trainable parameter gets a finite grad."""
    torch.manual_seed(0)
    model = tiny_model().cuda().to(torch.bfloat16)
    apply_float8(model, "tensorwise")

    total = 128
    ids = torch.randint(0, TINY["vocab_size"], (1, total), device="cuda")
    cu_seqlens = torch.tensor([0, 64, total], dtype=torch.int32, device="cuda")
    loss = model(input_ids=ids, attention_mask=cu_seqlens, labels=ids).loss
    assert torch.isfinite(loss)
    loss.backward()

    trained = [
        (n, p) for n, p in model.named_parameters()
        if p.requires_grad and p.grad is not None
    ]
    assert trained
    for n, p in trained:
        assert torch.isfinite(p.grad).all(), n


@cuda_only
@sm89
def test_float8_tracks_bfloat16(tiny_model):
    """fp8 GEMMs perturb the logits, but not by more than the quantization step."""
    torch.manual_seed(0)
    reference = tiny_model().cuda().to(torch.bfloat16)
    quantized = tiny_model().cuda().to(torch.bfloat16)
    quantized.load_state_dict(reference.state_dict())
    apply_float8(quantized, "tensorwise")

    total = 128
    ids = torch.randint(0, TINY["vocab_size"], (1, total), device="cuda")
    cu_seqlens = torch.tensor([0, 64, total], dtype=torch.int32, device="cuda")
    with torch.no_grad():
        a = reference(input_ids=ids, attention_mask=cu_seqlens)
        b = quantized(input_ids=ids, attention_mask=cu_seqlens)

    # e4m3 carries 3 mantissa bits, so ~6% relative per operand; the tiny config
    # stacks 4 layers of that. Compare against the signal, not an absolute bound.
    error = (a.float() - b.float()).norm() / a.float().norm()
    assert error < 0.1, error


@cuda_only
def test_rejects_hardware_without_fp8(tiny_model, monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (8, 0))
    with pytest.raises(RuntimeError, match="SM89"):
        apply_float8(tiny_model().cuda(), "tensorwise")


# ---------------------------------------------------------------------------
# MoE experts (`torch._grouped_mm`)
#
# The kernel behind these is SM90/SM100 only, and this box is SM120, so what is
# testable here is the guard, the parameter selection and the swap -- never the
# numerics. The capability check is monkeypatched to reach the swap at all.
# ---------------------------------------------------------------------------

SM90 = (9, 0)


def test_expert_filter_matches_only_the_stacked_experts(tiny_model):
    model = tiny_model()
    matched = [
        fqn for fqn, m in model.named_modules() if _is_grouped_mm_experts(m, fqn)
    ]
    assert matched, "the tiny config has MoE layers"
    assert all(fqn.endswith(".mlp.experts") for fqn in matched), matched

    # a Conv1d also owns a 3D weight; it must not be mistaken for an expert
    assert not _is_grouped_mm_experts(nn.Conv1d(8, 8, 3, groups=8), "conv1d")
    assert not _is_grouped_mm_experts(nn.Linear(16, 16), "proj")


@cuda_only
def test_moe_rejects_hardware_without_scaled_grouped_mm(tiny_model, monkeypatch):
    """SM120 is newer than SM90 and still not in the kernel's set."""
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: (12, 0))
    with pytest.raises(RuntimeError, match="SM90 or SM100"):
        apply_float8_moe(tiny_model().cuda(), "fp8_rowwise")

    # mxfp8 is datacenter-Blackwell only, so SM90 is not enough for it
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    with pytest.raises(RuntimeError, match="SM100"):
        apply_float8_moe(tiny_model().cuda(), "mxfp8")


def test_moe_rejects_an_unknown_recipe(tiny_model):
    with pytest.raises(ValueError, match="unknown MoE scaling type"):
        apply_float8_moe(tiny_model(), "tensorwise")


@cuda_only
def test_moe_swaps_the_expert_parameters(tiny_model, monkeypatch):
    from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor

    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    model = tiny_model().cuda()
    expected = {
        f"{fqn}.{name}"
        for fqn, m in model.named_modules()
        if _is_grouped_mm_experts(m, fqn)
        for name, _ in m.named_parameters(recurse=False)
    }
    swapped, blocks = apply_float8_moe(model, "fp8_rowwise")

    got = {
        n for n, p in model.named_parameters()
        if isinstance(p.data, ScaledGroupedMMTensor)
    }
    assert got == expected
    assert swapped == len(expected)
    # one expert block per MoE layer, two 3D parameters each
    assert blocks == sum(
        _is_grouped_mm_experts(m, fqn) for fqn, m in model.named_modules()
    )
    assert swapped == 2 * blocks
    # shapes and dtypes are untouched -- only the tensor subclass changed
    for name in expected:
        assert model.get_parameter(name).ndim == 3


@cuda_only
def test_moe_and_linear_float8_are_independent(tiny_model, monkeypatch):
    """Neither swap should touch what the other one owns."""
    from torchao.float8.float8_linear import Float8Linear
    from torchao.prototype.moe_training.tensor import ScaledGroupedMMTensor

    if _sm() < (8, 9):
        pytest.skip("fp8 linears need SM89+")
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    model = tiny_model().cuda()
    apply_float8(model, "tensorwise")
    apply_float8_moe(model, "fp8_rowwise")

    experts = {
        n for n, p in model.named_parameters()
        if isinstance(p.data, ScaledGroupedMMTensor)
    }
    linears = {n for n, m in model.named_modules() if isinstance(m, Float8Linear)}
    assert experts and linears
    # no Float8Linear owns a swapped expert parameter, and vice versa
    assert all(not any(e.startswith(l + ".") for l in linears) for e in experts)


@cuda_only
def test_moe_errors_when_there_are_no_experts(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda *a, **k: SM90)
    with pytest.raises(ValueError, match="no stacked expert block"):
        apply_float8_moe(nn.Sequential(nn.Linear(16, 16)).cuda(), "fp8_rowwise")
