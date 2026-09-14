"""Qwen4-Exp under 2-way tensor parallelism: gradients must match TP=1.

A forward-only smoke test is not enough here. Every gradient bug found so far
(the double-counted `q_norm`/`k_norm` all-reduce, and the missing reduction of
the routed-expert input gradient) left the forward exact -- the loss agreed to
six digits -- and only showed up as gradients that were off by a factor of two
in one direction or the other.

Requires two CUDA devices. `_tp_worker_qwen4.py` is launched twice via torchrun,
once with one rank and once with two, and the saved gradients are compared.
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
WORKER = HERE / "_tp_worker_qwen4.py"
MODEL_DIR = Path(
    os.environ.get("QWEN4_615M_DIR", "/data/151-2/users/tockier/models/qwen4_615m")
)


def _pick_two_free_gpus(min_free_gib: int = 20) -> list[int] | None:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    free = []
    for line in out.strip().splitlines():
        idx_s, mib_s = [p.strip() for p in line.split(",")]
        free.append((int(idx_s), int(mib_s)))
    free.sort(key=lambda p: p[1], reverse=True)
    picked = [idx for idx, mib in free[:2] if mib >= min_free_gib * 1024]
    return picked if len(picked) == 2 else None


def _run(worker_out: Path, gpus: list[int], nproc: int, port: int) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in gpus[:nproc])
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc_per_node={nproc}",
        "--master_addr=127.0.0.1",
        f"--master_port={port}",
        str(WORKER),
        str(worker_out),
    ]
    result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=900)
    if result.returncode != 0:
        print("STDOUT:\n" + result.stdout)
        print("STDERR:\n" + result.stderr)
        pytest.fail(f"TP worker (nproc={nproc}) exited with {result.returncode}")


@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs at least 2 CUDA devices"
)
def test_qwen4_tp2_gradients_match_tp1(tmp_path) -> None:
    if not (MODEL_DIR / "config.json").exists():
        pytest.skip(f"no small Qwen4 config at {MODEL_DIR}")
    gpus = _pick_two_free_gpus()
    if gpus is None:
        pytest.skip("no two CUDA devices with >= 20 GiB free")

    one, two = tmp_path / "tp1.pt", tmp_path / "tp2.pt"
    _run(one, gpus, nproc=1, port=29551)
    _run(two, gpus, nproc=2, port=29553)

    a, b = torch.load(one), torch.load(two)
    assert abs(a["loss"] - b["loss"]) < 1e-3, (a["loss"], b["loss"])

    ga, gb = a["grads"], b["grads"]
    assert set(ga) == set(gb)

    # bf16 kernels reduce in a different order under TP, so this is a closeness
    # bound, not equality. The bugs it is here to catch were 50-100% errors.
    num = den = 0.0
    worst = (0.0, "")
    for name, x in ga.items():
        y = gb[name]
        assert x.shape == y.shape, name
        num += ((x - y) ** 2).sum().item()
        den += (x**2).sum().item()
        rel = (x - y).norm().item() / (x.norm().item() + 1e-20)
        worst = max(worst, (rel, name))

    global_rel = (num**0.5) / (den**0.5)
    assert global_rel < 0.01, f"global relative gradient error {global_rel:.4f}"
    assert worst[0] < 0.05, f"worst parameter {worst[1]}: relative error {worst[0]:.4f}"
