"""FlashQLA vs FLA for the Qwen4 GatedDeltaNet kernel.

FlashQLA (https://github.com/QwenLM/FlashQLA) is the Qwen team's TileLang
implementation of the same chunked gated delta rule, reporting 2-3x forward and
2x backward over the FLA Triton kernels. Its entry point takes the same
arguments as FLA's, so `models/qwen4/compile_ops.py` wraps both behind matching
custom ops and `set_gdn_backend` picks between them.

These tests are the evidence that the swap is safe. They skip unless FlashQLA
actually builds on the current device:

* SM90 (H100 / GH200, the HPC targets) - forward and backward
* SM100 / SM103                        - forward and backward
* SM120 / SM121                        - forward only
* anything else                        - unsupported, import raises

TileLang JITs through `nvcc`, so a toolkit older than the target architecture
needs (CUDA 12.8+ for SM120) fails at the first call rather than at import.
"""
from __future__ import annotations

import sys

import pytest
import torch

from models.qwen4.config import Qwen4TextConfig
from models.qwen4 import model as ours
from models.qwen4 import compile_ops as ops
from models.tests.test_qwen4_parity import TINY, randomize, close

# FlashQLA's kkt_solve asserts `K == 128`, so the linear-attention head dims are
# not free the way the rest of the tiny config is. The released Qwen4-Exp config
# uses 128/128, so this matches production rather than working around it.
QLA_TINY = {
    **TINY,
    "hidden_size": 256,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "ple_layer_ids": [],
}


def _probe() -> str | None:
    """Run the smallest possible FlashQLA call; return a skip reason or None."""
    if not torch.cuda.is_available():
        return "no CUDA device"
    if ops.flashqla() is None:
        return f"FlashQLA not importable here: {ops.flashqla_unavailable_reason()}"
    try:
        n, h, d = 64, 2, 128
        kw = dict(device="cuda", dtype=torch.bfloat16)
        q = torch.randn(1, n, h, d, **kw)
        v = torch.randn(1, n, h, d, **kw)
        g = torch.randn(1, n, h, device="cuda", dtype=torch.float32)
        beta = torch.rand(1, n, h, **kw)
        cu = torch.tensor([0, n], dtype=torch.int64, device="cuda")
        with torch.no_grad():
            ops.flashqla()(q, q.clone(), v, g, beta, cu_seqlens=cu)
    except Exception as exc:  # nvcc too old, JIT failure, ...
        return f"FlashQLA kernel build failed: {type(exc).__name__}: {str(exc)[-200:]}"
    return None


_SKIP = _probe()
pytestmark = [
    pytest.mark.cuda_only,
    pytest.mark.skipif(_SKIP is not None, reason=str(_SKIP)),
]

SEQ = 512
# Two independent kernel implementations of the same recurrence in bf16; the
# repo's Qwen3.5 suite uses atol=0.5 / rtol=0.1 for whole-model logits.
ATOL = 2e-2
RTOL = 5e-2


@pytest.fixture
def gdn():
    torch.manual_seed(0)
    cfg = Qwen4TextConfig.from_dict(QLA_TINY)
    module = randomize(ours.GatedDeltaNet(cfg)).cuda().to(torch.bfloat16)
    yield cfg, module
    ours.set_gdn_backend("fla")


def _both_backends(module, *args, **kwargs):
    ours.set_gdn_backend("fla")
    with torch.no_grad():
        fla = module(*args, **kwargs)
    ours.set_gdn_backend("flashqla")
    with torch.no_grad():
        qla = module(*args, **kwargs)
    return fla, qla


def test_backend_selection_is_explicit():
    """`auto` must never pick a backend that cannot train."""
    assert ours.set_gdn_backend("fla") == "fla"
    chosen = ours.set_gdn_backend("auto")
    if ops.flashqla_has_backward():
        assert chosen == "flashqla"
    else:
        assert chosen == "fla", "auto picked a forward-only backend"
    ours.set_gdn_backend("fla")


def test_forward_matches_fla(gdn):
    cfg, module = gdn
    x = torch.randn(1, SEQ, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, SEQ], dtype=torch.int32, device="cuda")
    fla, qla = _both_backends(module, x, cu_seqlens=cu)
    close(fla, qla, "GatedDeltaNet forward", atol=ATOL, rtol=RTOL)


def test_forward_matches_fla_packed(gdn):
    """Same, with several documents packed into one row."""
    cfg, module = gdn
    lens = [128, 192, 192]
    total = sum(lens)
    x = torch.randn(1, total, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    bounds = [0]
    for n in lens:
        bounds.append(bounds[-1] + n)
    cu = torch.tensor(bounds, dtype=torch.int32, device="cuda")

    fla, qla = _both_backends(module, x, cu_seqlens=cu)
    close(fla, qla, "GatedDeltaNet forward (packed)", atol=ATOL, rtol=RTOL)

    # and each document must still be independent under the new kernel
    ours.set_gdn_backend("flashqla")
    lo = 0
    with torch.no_grad():
        for n in lens:
            solo = module(
                x[:, lo : lo + n],
                cu_seqlens=torch.tensor([0, n], dtype=torch.int32, device="cuda"),
            )
            close(qla[:, lo : lo + n], solo, f"FlashQLA packed doc @{lo}",
                  atol=ATOL, rtol=RTOL)
            lo += n


@pytest.mark.skipif(
    not ops.flashqla_has_backward(),
    reason="this architecture ships the FlashQLA forward kernel only",
)
def test_backward_matches_fla(gdn):
    cfg, module = gdn
    x = torch.randn(1, SEQ, cfg.hidden_size, device="cuda", dtype=torch.bfloat16)
    cu = torch.tensor([0, SEQ], dtype=torch.int32, device="cuda")

    grads = {}
    for backend in ("fla", "flashqla"):
        ours.set_gdn_backend(backend)
        module.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_(True)
        module(xi, cu_seqlens=cu).square().mean().backward()
        grads[backend] = {
            "input": xi.grad.clone(),
            **{n: p.grad.clone() for n, p in module.named_parameters() if p.grad is not None},
        }

    assert set(grads["fla"]) == set(grads["flashqla"])
    for name in grads["fla"]:
        close(grads["fla"][name], grads["flashqla"][name],
              f"grad {name}", atol=ATOL, rtol=RTOL)
