"""Qwen3.5 9B TP=2 benchmark on GPUs 6,7.

Launches ``_tp_bench_worker_qwen3_5_9b.py`` under torchrun with
``CUDA_VISIBLE_DEVICES=6,7``, runs warmup + timed fwd+bwd passes on a
packed seq_len=4096 input, and asserts that the worker exits cleanly.
Memory and FLOPs/s numbers are printed by rank 0 and surfaced here.

Run directly:
    python -m models.tests.test_qwen3_5_9b_tp2_bench

Or as a plain script:
    python models/tests/test_qwen3_5_9b_tp2_bench.py
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[2]
WORKER = Path(__file__).resolve().parent / "_tp_bench_worker_qwen3_5_9b.py"


def _run(cuda_devices: str, seq_len: int = 2048, warmup: int = 2, steps: int = 5) -> str:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_devices
    env["BENCH_SEQ_LEN"] = str(seq_len)
    env["BENCH_WARMUP"] = str(warmup)
    env["BENCH_STEPS"] = str(steps)
    env.setdefault("PYTHONPATH", str(REPO_ROOT))

    # Pick a free rendezvous port tied to the test pid to avoid collisions
    # when multiple tests share the same machine.
    env["MASTER_ADDR"] = "127.0.0.1"
    env["MASTER_PORT"] = str(29500 + (os.getpid() % 1000))

    cmd = [
        sys.executable, "-m", "torch.distributed.run",
        "--nproc_per_node=2",
        "--rdzv_backend=c10d",
        f"--master_addr={env['MASTER_ADDR']}",
        f"--master_port={env['MASTER_PORT']}",
        str(WORKER),
    ]
    result = subprocess.run(
        cmd, env=env, cwd=str(REPO_ROOT), capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Worker failed (rc={result.returncode}).\n"
            f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
    return result.stdout


@pytest.mark.skipif(
    not Path("/data/151-1/users/tockier/qwen_finetune/cache/qwen35_9b").exists(),
    reason="Qwen3.5 9B snapshot not available",
)
def test_qwen3_5_9b_tp2_bench_gpus_6_7() -> None:
    stdout = _run("6,7", seq_len=2048, warmup=2, steps=5)
    # Sanity: the bench banner must appear in rank-0 stdout.
    assert "Qwen3.5 9B | TP=2 | seq_len=2048" in stdout, stdout


if __name__ == "__main__":
    gpus = os.environ.get("BENCH_GPUS", "6,7")
    seq_len = int(os.environ.get("BENCH_SEQ_LEN", "2048"))
    warmup = int(os.environ.get("BENCH_WARMUP", "2"))
    steps = int(os.environ.get("BENCH_STEPS", "5"))
    print(_run(gpus, seq_len=seq_len, warmup=warmup, steps=steps))
