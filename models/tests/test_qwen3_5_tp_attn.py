"""Qwen3.5 decoder layers under real tensor parallelism.

Two bugs lived here, both invisible on a single GPU and both fatal on the first
training step after the allocation was granted:

* **Global head count after unwrap.** `_dtensor_unwrap` returns this rank's
  shard, holding `num_attention_heads // tp_size` heads, but `SelfAttention`
  reshaped with the global count:
  `RuntimeError: shape '[10240, 16, 256]' is invalid for input of size 10485760`.
  With `tp_size == 1` local and global agree, so nothing caught it.

* **fp32 leaking into the attention kernel.** `F.rms_norm` is not on autocast's
  cast list, so with `master_dtype = "float32"` the fp32 `OffsetRMSNorm.weight`
  promoted a bf16 input back to fp32, all the way into the kernel:
  `RuntimeError: FlashAttention only support fp16 and bf16 data type`.

The worker therefore runs the way the trainer does -- fp32 params plus
`torch.autocast` -- and reports which attention path served the call, so a
future regression cannot pass by quietly falling back to the SDPA path.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch

WORKER = Path(__file__).with_name("_tp_worker_qwen3_5_attn.py")


@pytest.fixture(scope="module")
def worker_results():
    tp = min(4, torch.cuda.device_count())
    out = subprocess.run(
        [sys.executable, "-m", "torch.distributed.run",
         f"--nproc_per_node={tp}", "--master_port=29901", str(WORKER)],
        capture_output=True, text=True, timeout=900,
    )
    lines = [l[len("RESULT: "):] for l in out.stdout.splitlines() if l.startswith("RESULT: ")]
    if not lines:
        pytest.fail(
            f"worker printed no RESULT line\n"
            f"stdout:\n{out.stdout[-2000:]}\nstderr:\n{out.stderr[-2000:]}"
        )
    return {l.split()[0]: l for l in lines}


pytestmark = pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs >=2 GPUs to shard anything"
)


@pytest.mark.parametrize("layer_type", ["full_attention", "linear_attention"])
def test_layer_runs_under_tp(worker_results, layer_type):
    line = worker_results.get(layer_type, "<missing>")
    assert " ok " in line, line
    assert "finite=True" in line and "grad=True" in line, line


def test_attention_kernel_gets_bf16(worker_results):
    """Not the SDPA fallback: the real kernel, with a dtype it accepts."""
    line = worker_results["full_attention"]
    assert "kernel_in=torch.bfloat16" in line, line
    assert "path=varlen_attn" in line, line
