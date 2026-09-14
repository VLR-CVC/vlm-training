"""Environment checks for the three model families.

The models here lean on compiled CUDA extensions (`causal_conv1d`, `torchvision`
via `transformers`, optionally `flash_attn` and `flash_qla`) and on torch APIs
that only exist in recent releases. Two failure modes have bitten this repo, and
neither shows up as a missing package:

* **Stale extension ABI.** A `.so` built against one libtorch and loaded against
  another fails at import with `undefined symbol`. torch 2.14 added a sixth
  parameter to `c10::cuda::c10_cuda_check_implementation`, which is what
  `C10_CUDA_CHECK` expands to, so *every* extension built before it broke. `pip
  list` still reports the package as installed.

* **Silent fallback.** `models/*/compile_ops.py` wraps its imports in `try`, so a
  broken extension degrades to a plain-torch path (or, on older revisions, to a
  `NameError` at the first forward). Training still starts, several times slower.

So the tests assert on *imported symbols and running kernels*, never on version
strings alone. Required dependencies fail; optional ones report and skip, since
both are legitimate configurations -- not every box has an ARM `flash_attn`
build, and `flash_qla` only ships kernels for some architectures.

Run anywhere::

    pytest models/tests/test_dependencies.py -v

The GPU tests skip cleanly on a CPU-only box. `-rs` shows what was skipped and
why, which is the interesting output when comparing two machines.
"""
from __future__ import annotations

import importlib
import platform
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

CUDA = torch.cuda.is_available()
requires_cuda = pytest.mark.skipif(not CUDA, reason="needs a CUDA device")


# --------------------------------------------------------------------------
# what each family needs
# --------------------------------------------------------------------------
# (module, attributes). An empty attribute tuple means the import alone is the
# check. Attributes matter because `compile_ops` imports private, low-level
# entry points that fla moves between releases -- the package can be importable
# while the specific symbol is gone.

FLA_MODULES = [
    ("fla.ops.gated_delta_rule", ("chunk_gated_delta_rule",)),
    (
        "fla.ops.gated_delta_rule.chunk",
        (
            "l2norm_fwd",
            "l2norm_bwd",
            "prepare_chunk_indices",
            "chunk_gated_delta_rule_fwd",
            "chunk_gated_delta_rule_bwd",
        ),
    ),
    (
        "fla.modules.fused_norm_gate",
        ("rms_norm_gated", "layer_norm_gated_fwd", "layer_norm_gated_bwd"),
    ),
]

CONV1D_MODULES = [
    ("causal_conv1d", ("causal_conv1d_fn",)),
    (
        "causal_conv1d.causal_conv1d_interface",
        ("causal_conv1d_fwd_function", "causal_conv1d_bwd_function"),
    ),
]

# Required: no fallback, or a fallback so slow it is not a way to train.
REQUIRED = {
    "qwen3_vl": [("transformers", ()), ("torch", ())],
    "qwen3_5": [("transformers", ()), ("torch", ())] + FLA_MODULES,
}

# Optional: a documented fallback exists. Reported, never fatal.
OPTIONAL = {
    # `varlen_attn` covers this from torch 2.11; flash_attn has no ARM build.
    "qwen3_vl": [("flash_attn", ("flash_attn_varlen_func",))],
    # `causal_conv1d_torch` stands in, at a real cost. flash_qla is picked up
    # by FLA's backend dispatcher for `chunk_gated_delta_rule` -- no repo code
    # calls it, so its absence only costs speed.
    "qwen3_5": CONV1D_MODULES + [("flash_qla", ("chunk_gated_delta_rule",))],
}

FAMILIES = sorted(REQUIRED)


def _probe(name: str, attrs: tuple[str, ...]) -> tuple[bool, str]:
    """Import `name` and check `attrs`. Returns (ok, detail)."""
    try:
        mod = importlib.import_module(name)
    except Exception as exc:  # ImportError, but a bad .so can raise anything
        return False, f"{name}: {type(exc).__name__}: {exc}"
    missing = [a for a in attrs if not hasattr(mod, a)]
    if missing:
        version = getattr(mod, "__version__", "unknown")
        return False, f"{name} ({version}) is missing {missing}"
    return True, f"{name} {getattr(mod, '__version__', '')}".strip()


def _flat(table) -> list:
    seen, out = set(), []
    for family, entries in table.items():
        for mod, attrs in entries:
            if (family, mod) in seen:
                continue
            seen.add((family, mod))
            out.append(pytest.param(family, mod, attrs, id=f"{family}-{mod}"))
    return out


# --------------------------------------------------------------------------
# torch itself
# --------------------------------------------------------------------------

def test_torch_version_floor():
    """`torch.nn.attention.varlen` landed in 2.11; every family imports it."""
    major, minor = (int(p) for p in torch.__version__.split(".")[:2])
    assert (major, minor) >= (2, 11), (
        f"torch {torch.__version__} is too old; varlen attention needs >= 2.11"
    )


@pytest.mark.parametrize(
    "path",
    [
        "torch.nn.attention.varlen:varlen_attn",
        "torch.nn.attention.flex_attention:flex_attention",
        "torch.nn.attention.flex_attention:create_block_mask",
        "torch:_grouped_mm",
    ],
)
def test_torch_api_present(path):
    """The specific torch entry points the models call.

    `varlen_attn` is the attention for all three families, `flex_attention` is
    how Qwen4 expresses the QSA mask, and `_grouped_mm` is the MoE expert
    matmul. All three are private-ish and have moved before.
    """
    module, _, attr = path.partition(":")
    ok, detail = _probe(module, (attr,))
    assert ok, detail


def test_cuda_available():
    """Not fatal by itself, but every kernel test below depends on it."""
    if not CUDA:
        pytest.skip("no CUDA device visible")
    assert torch.cuda.get_device_capability(0) >= (8, 0), (
        "bf16 kernels need SM80 or newer"
    )


# --------------------------------------------------------------------------
# per-family dependencies
# --------------------------------------------------------------------------

@pytest.mark.parametrize("family,module,attrs", _flat(REQUIRED))
def test_required_dependency(family, module, attrs):
    ok, detail = _probe(module, attrs)
    assert ok, f"{family} requires {detail}"


@pytest.mark.parametrize("family,module,attrs", _flat(OPTIONAL))
def test_optional_dependency(family, module, attrs):
    """Skips rather than fails -- but an `undefined symbol` is still an error.

    A package that is absent is a choice. A package that is installed and
    refuses to load is a broken environment, and that distinction is the whole
    point of this test: the ABI breaks we hit reported as the latter.
    """
    ok, detail = _probe(module, attrs)
    if ok:
        return
    if "undefined symbol" in detail or "No module named" not in detail:
        assert ok, f"{family}: {module} is installed but does not load -- {detail}"
    pytest.skip(f"{family}: optional {detail}")


@pytest.mark.parametrize("family", FAMILIES)
def test_model_package_imports(family):
    """The repo's own module tree, which is what pulls the above together."""
    pytest.importorskip(
        f"models.{family}", reason=f"models/{family} not present in this checkout"
    )
    ok, detail = _probe(f"models.{family}.model", ())
    assert ok, detail


# qwen3_vl has no `compile_ops`; it uses `varlen_attn`/`flash_attn` directly.
@pytest.mark.parametrize("family", ["qwen3_5"])
def test_compile_ops_bound_every_symbol(family):
    """Catch the silent-fallback case.

    `compile_ops` swallows import errors by design. This walks the private names
    it binds and reports the ones that are missing, so a degraded environment is
    visible here instead of showing up as a mysteriously slow training run.
    """
    ops = pytest.importorskip(
        f"models.{family}.compile_ops", reason=f"models/{family} not in this checkout"
    )
    fla_names = [
        "_fla_chunk_gated_delta_rule",
        "_fla_rms_norm_gated",
        "_l2norm_fwd",
        "_l2norm_bwd",
        "_prepare_chunk_indices",
        "_gdr_fwd",
        "_gdr_bwd_kernel",
        "_lng_fwd",
        "_lng_bwd",
    ]
    missing = [n for n in fla_names if not hasattr(ops, n)]
    assert not missing, (
        f"models/{family}/compile_ops.py failed to bind {missing}. One failed "
        "import in a shared `try:` block drops every name after it."
    )


def test_conv1d_fallback_reports_honestly():
    """`causal_conv1d_available()` must agree with reality, either way."""
    ops = pytest.importorskip("models.qwen3_5.compile_ops")
    if not hasattr(ops, "causal_conv1d_available"):
        pytest.skip("this revision predates the `causal_conv1d_available` guard")
    claimed = ops.causal_conv1d_available()
    actual, detail = _probe("causal_conv1d", ("causal_conv1d_fn",))
    assert claimed == actual, (
        f"causal_conv1d_available() says {claimed}, but importing says "
        f"{actual} ({detail})"
    )


# --------------------------------------------------------------------------
# the kernels have to actually run
# --------------------------------------------------------------------------
# Importing proves the ABI matches. It does not prove the kernel was compiled
# for this GPU's architecture -- a wrong `-gencode` set fails at launch, not at
# import.

@requires_cuda
def test_varlen_attn_runs():
    from torch.nn.attention.varlen import varlen_attn

    total, heads, dim = 128, 4, 64
    q, k, v = (
        torch.randn(total, heads, dim, device="cuda", dtype=torch.bfloat16)
        for _ in range(3)
    )
    cu = torch.tensor([0, 64, total], device="cuda", dtype=torch.int32)
    # same call shape as `models/qwen3_5/model.py`: causality is a left-only
    # window, there is no `is_causal` flag on this API
    out = varlen_attn(
        q, k, v, cu_seq_q=cu, cu_seq_k=cu, max_q=64, max_k=64, window_size=(-1, 0)
    )
    torch.cuda.synchronize()
    assert out.shape == (total, heads, dim)
    assert out.isfinite().all()


@requires_cuda
def test_grouped_mm_runs():
    """Qwen4's MoE expert matmul. bf16/fp16 only."""
    experts, tokens, din, dout = 4, 32, 16, 24
    x = torch.randn(tokens, din, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(experts, din, dout, device="cuda", dtype=torch.bfloat16)
    offs = torch.tensor([8, 16, 24, tokens], device="cuda", dtype=torch.int32)
    out = torch._grouped_mm(x, w, offs=offs)
    torch.cuda.synchronize()
    assert out.shape == (tokens, dout)
    assert out.isfinite().all()


@requires_cuda
def test_fla_gated_delta_rule_runs():
    """Qwen3.5 and Qwen4 linear-attention layers. Compiles Triton on first call."""
    chunk = pytest.importorskip("fla.ops.gated_delta_rule").chunk_gated_delta_rule
    b, t, h, d = 1, 256, 4, 64
    kw = dict(device="cuda", dtype=torch.bfloat16)
    out, _ = chunk(
        q=torch.randn(b, t, h, d, **kw),
        k=torch.randn(b, t, h, d, **kw),
        v=torch.randn(b, t, h, d, **kw),
        g=torch.rand(b, t, h, device="cuda", dtype=torch.float32).log(),
        beta=torch.rand(b, t, h, **kw),
        use_qk_l2norm_in_kernel=True,
    )
    torch.cuda.synchronize()
    assert out.shape == (b, t, h, d)
    assert out.isfinite().all()


@requires_cuda
def test_fla_rms_norm_gated_runs():
    mod = pytest.importorskip("fla.modules.fused_norm_gate")
    kw = dict(device="cuda", dtype=torch.bfloat16)
    out = mod.rms_norm_gated(
        torch.randn(64, 128, **kw),
        torch.randn(64, 128, **kw),
        torch.randn(128, **kw),
        None,
        activation="swish",
        eps=1e-6,
    )
    torch.cuda.synchronize()
    assert out.isfinite().all()


@requires_cuda
def test_causal_conv1d_runs_forward_and_backward():
    """Gradients too: a forward-only kernel is useless for training."""
    mod = pytest.importorskip("causal_conv1d")
    batch, channels, length, width = 1, 64, 128, 4
    # the kernel needs channel-last (stride(1) == 1) when seq_idx is used
    x = torch.randn(batch, length, channels, device="cuda", dtype=torch.bfloat16)
    x = x.transpose(1, 2).requires_grad_(True)
    weight = torch.randn(channels, width, device="cuda", dtype=torch.bfloat16)
    weight.requires_grad_(True)
    bias = torch.randn(channels, device="cuda", dtype=torch.bfloat16)
    seq_idx = torch.zeros(batch, length, device="cuda", dtype=torch.int32)
    seq_idx[:, length // 2:] = 1  # two packed documents

    out = mod.causal_conv1d_fn(
        x=x, weight=weight, bias=bias, seq_idx=seq_idx, activation="silu"
    )
    out.sum().backward()
    torch.cuda.synchronize()
    assert out.shape == (batch, channels, length)
    assert out.isfinite().all()
    assert x.grad.isfinite().all() and weight.grad.isfinite().all()


@requires_cuda
def test_transformers_loads_modeling_utils():
    """The import that a broken torchvision takes down.

    `transformers.modeling_utils` reaches `image_utils`, which imports
    torchvision unconditionally when it is installed. A torchvision built
    against another libtorch therefore breaks `AutoProcessor`, and with it
    `train/train_qwen.py`, with an error naming neither package.
    """
    ok, detail = _probe("transformers.modeling_utils", ("PreTrainedModel",))
    assert ok, detail
    ok, detail = _probe("transformers", ("AutoProcessor",))
    assert ok, detail


@pytest.mark.parametrize(
    "module,family",
    [
        ("transformers.models.qwen3_vl", "qwen3_vl"),
        ("transformers.models.qwen3_5", "qwen3_5"),
    ],
)
def test_transformers_reference_architecture(module, family):
    """Only the parity tests need these; training uses our own implementation.

    A transformers too old to ship the reference architecture skips instead
    of failing; training does not need it.
    """
    ok, detail = _probe(module, ())
    if not ok:
        pytest.skip(
            f"transformers {getattr(importlib.import_module('transformers'), '__version__', '?')} "
            f"has no reference {family} -- parity tests cannot run: {detail}"
        )


# --------------------------------------------------------------------------
# summary, for eyeballing two machines side by side
# --------------------------------------------------------------------------

def test_report_environment(capsys):
    """Always passes. Run with `-s` to print the table."""
    lines = [
        "",
        f"python   {platform.python_version()} ({platform.machine()})",
        f"torch    {torch.__version__}  cuda {torch.version.cuda}",
    ]
    if CUDA:
        cap = torch.cuda.get_device_capability(0)
        lines.append(
            f"gpu      {torch.cuda.get_device_name(0)}  sm_{cap[0]}{cap[1]}"
        )
    else:
        lines.append("gpu      none")

    for label, table in (("required", REQUIRED), ("optional", OPTIONAL)):
        seen = set()
        for entries in table.values():
            for mod, attrs in entries:
                if mod in seen:
                    continue
                seen.add(mod)
                ok, detail = _probe(mod, attrs)
                lines.append(f"{label:8} {'ok  ' if ok else 'FAIL'} {detail}")

    with capsys.disabled():
        print("\n".join(lines))
