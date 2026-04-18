"""Smoke test: Qwen3.5 forward pass under 2-way tensor parallelism.

Goal: catch DTensor/kernel compatibility regressions. This test only
asserts that the forward pass runs to completion without exceptions —
it does NOT check numerical parity against a non-TP run (bf16 reductions
don't match bit-for-bit across reductions anyway; see the exhaustive
suite for full parity checks on a single GPU).

Requires at least 2 CUDA devices. Launches `_tp_worker_qwen3_5.py` via
torchrun in a subprocess and checks the exit code.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch


HERE = Path(__file__).resolve().parent
WORKER = HERE / "_tp_worker_qwen3_5.py"


def _pick_two_free_gpus(min_free_gib: int = 20) -> list[int] | None:
    """Pick two CUDA devices with enough free memory for the 2B model."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    free: list[tuple[int, int]] = []
    for line in out.strip().splitlines():
        idx_s, mib_s = [p.strip() for p in line.split(",")]
        free.append((int(idx_s), int(mib_s)))

    free.sort(key=lambda p: p[1], reverse=True)
    picked = [idx for idx, mib in free[:2] if mib >= min_free_gib * 1024]
    return picked if len(picked) == 2 else None


@pytest.mark.skipif(
    torch.cuda.device_count() < 2,
    reason="TP smoke test requires at least 2 CUDA devices",
)
def test_qwen3_5_tp2_forward() -> None:
    gpus = _pick_two_free_gpus()
    if gpus is None:
        pytest.skip("no two CUDA devices with >= 20 GiB free")

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus)
    # Keep FLA's autotune off for reproducibility / speed.
    env.setdefault("FLA_USE_FAST_OPS", "1")

    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--nproc_per_node=2",
        "--standalone",
        str(WORKER),
    ]

    # 5 minute ceiling — model load dominates.
    result = subprocess.run(
        cmd, env=env, capture_output=True, text=True, timeout=600
    )

    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)
        pytest.fail(
            f"TP worker exited with code {result.returncode}. "
            "DTensor regression? See captured output above."
        )
