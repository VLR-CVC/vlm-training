"""`causal_conv1d` is an optional CUDA extension.

These tests cover the path where it is not installed: the plain-torch stand-in
has to agree with the kernel it replaces, and importing the model must not
depend on the package being importable.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen4 import compile_ops as qwen4_ops  # noqa: E402
from models.qwen3_5 import compile_ops as qwen3_5_ops  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
MODULES = pytest.mark.parametrize(
    "ops", [qwen4_ops, qwen3_5_ops], ids=["qwen4", "qwen3_5"]
)


def _reference(x, weight, bias, seq_idx):
    """Direct transcription of the kernel's contract, one output at a time."""
    B, C, L = x.shape
    K = weight.shape[-1]
    out = torch.zeros_like(x, dtype=torch.float64)
    for b in range(B):
        for t in range(L):
            acc = torch.zeros(C, dtype=torch.float64)
            for j in range(K):
                src = t - (K - 1) + j
                if src < 0 or seq_idx[b, src] != seq_idx[b, t]:
                    continue  # tap falls outside this document
                acc = acc + x[b, :, src].double() * weight[:, j].double()
            if bias is not None:
                acc = acc + bias.double()
            out[b, :, t] = acc
    return F.silu(out)


def _inputs(device="cpu", dtype=torch.float32, B=2, C=8, L=11, K=4, bias=True):
    # C is a multiple of 8: the fused kernel rejects anything else
    torch.manual_seed(0)
    # channel-last, as the model's `qkv.transpose(1, 2)` produces: the kernel
    # rejects anything else once `seq_idx` is passed
    x = torch.randn(B, L, C, device=device, dtype=dtype).transpose(1, 2)
    weight = torch.randn(C, K, device=device, dtype=dtype)
    b = torch.randn(C, device=device, dtype=dtype) if bias else None
    # two packed documents of unequal length, so the segment reset is exercised
    seq_idx = torch.zeros(B, L, device=device, dtype=torch.int32)
    seq_idx[:, 5:] = 1
    return x, weight, b, seq_idx


@MODULES
@pytest.mark.parametrize("bias", [True, False])
def test_torch_fallback_matches_reference(ops, bias):
    x, weight, b, seq_idx = _inputs(bias=bias)
    got = ops.causal_conv1d_torch(x, weight, b, seq_idx)
    assert torch.allclose(got.double(), _reference(x, weight, b, seq_idx), atol=1e-6)


@MODULES
def test_torch_fallback_respects_document_boundary(ops):
    """A token may not see the document before it."""
    x, weight, b, seq_idx = _inputs()
    out = ops.causal_conv1d_torch(x, weight, b, seq_idx)
    poisoned = x.clone()
    poisoned[:, :, :5] = 100.0  # rewrite document 0 only
    out2 = ops.causal_conv1d_torch(poisoned, weight, b, seq_idx)
    assert torch.allclose(out[:, :, 5:], out2[:, :, 5:], atol=1e-6)
    assert not torch.allclose(out[:, :, :5], out2[:, :, :5])


@MODULES
def test_torch_fallback_gradients(ops):
    x, weight, b, seq_idx = _inputs(dtype=torch.float64)
    x, weight, b = (t.requires_grad_(True) for t in (x, weight, b))
    torch.autograd.gradcheck(
        lambda *args: ops.causal_conv1d_torch(*args, seq_idx), (x, weight, b)
    )


@MODULES
def test_torch_fallback_compiles_fullgraph(ops):
    """The layers around it are compiled with `fullgraph=True`."""
    x, weight, b, seq_idx = _inputs()
    compiled = torch.compile(ops.causal_conv1d_torch, fullgraph=True)
    assert torch.allclose(
        compiled(x, weight, b, seq_idx), ops.causal_conv1d_torch(x, weight, b, seq_idx)
    )


@MODULES
@pytest.mark.skipif(not torch.cuda.is_available(), reason="the kernel is CUDA-only")
def test_torch_fallback_matches_kernel(ops):
    if not ops.causal_conv1d_available():
        pytest.skip("causal_conv1d is not installed")
    x, weight, b, seq_idx = _inputs(device="cuda")
    x, weight, b = (t.detach().requires_grad_(True) for t in (x, weight, b))
    xr, wr, br = (t.detach().clone().requires_grad_(True) for t in (x, weight, b))

    fused = ops.causal_conv1d(x, weight, b, seq_idx)
    torch_out = ops.causal_conv1d_torch(xr, wr, br, seq_idx)
    assert torch.allclose(fused, torch_out, atol=1e-5, rtol=1e-4)

    grad = torch.randn_like(fused)
    fused.backward(grad)
    torch_out.backward(grad)
    for a, c in ((x, xr), (weight, wr), (b, br)):
        assert torch.allclose(a.grad, c.grad, atol=1e-5, rtol=1e-4)


_WITHOUT_PACKAGE = textwrap.dedent(
    """
    import sys

    class Blocker:
        def find_module(self, name, path=None):
            return self.find_spec(name, path)

        def find_spec(self, name, path=None, target=None):
            if name == "causal_conv1d" or name.startswith("causal_conv1d."):
                raise ImportError("causal_conv1d is not installed (test blocker)")
            return None

    sys.meta_path.insert(0, Blocker())
    for name in list(sys.modules):
        if name.startswith("causal_conv1d"):
            del sys.modules[name]

    import torch
    from models.{pkg} import compile_ops as ops
    from models.{pkg}.model import GatedDeltaNet

    assert not ops.causal_conv1d_available()
    # fla lives behind its own import and must survive the missing extension
    assert hasattr(torch.ops.{ns}, "gated_delta_rule")

    x = torch.randn(1, 4, 8)
    weight = torch.randn(4, 3)
    seq_idx = torch.zeros(1, 8, dtype=torch.int32)
    out = GatedDeltaNet._run_conv1d(x, weight, None, seq_idx)
    assert out.shape == x.shape
    assert torch.allclose(out, ops.causal_conv1d_torch(x, weight, None, seq_idx))
    print("OK")
    """
)


@pytest.mark.parametrize(
    "pkg,ns", [("qwen4", "qwen4"), ("qwen3_5", "qwen3_5")]
)
def test_imports_without_the_package(pkg, ns):
    """Import and run the conv with `causal_conv1d` made unimportable."""
    proc = subprocess.run(
        [sys.executable, "-c", _WITHOUT_PACKAGE.format(pkg=pkg, ns=ns)],
        capture_output=True, text=True, cwd=ROOT,
        env={"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin", "HOME": str(Path.home()),
             "CUDA_VISIBLE_DEVICES": ""},
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    assert "OK" in proc.stdout
