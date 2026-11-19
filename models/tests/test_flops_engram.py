"""Engram FLOPs and the per-device MFU peak (`train/flops_estimation.py`)."""

import json
from pathlib import Path

from models.qwen3_5.engram import EngramArgs
from train.flops_estimation import (
    build_flops_model,
    engram_flops_per_token,
    engram_lookup_bytes,
    peak_bf16_tflops,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs/models/qwen3_5_0_8b_text.json"


def test_engram_adds_projection_flops():
    hf = json.loads(CONFIG.read_text())
    hidden = hf.get("text_config", hf)["hidden_size"]
    base = build_flops_model(hf)
    args = EngramArgs(layers=(2, 10))
    with_engram = build_flops_model(hf, engram=args)
    extra = with_engram.dense_per_token - base.dense_per_token
    # two modules, each W_K + W_V (hidden -> hidden) and the gate dot, x3 (fwd+bwd) x2 (FMA)
    assert extra == engram_flops_per_token(args, hidden) == 2 * 6 * (2 * hidden * hidden + hidden)
    assert with_engram.text_attn_pair == base.text_attn_pair
    assert build_flops_model(hf, engram=None) == base


def test_engram_lookup_bytes():
    args = EngramArgs(dim=512)
    assert engram_lookup_bytes(args, 1024) == 512 * (2 + 8)


def test_peak_lookup(monkeypatch):
    monkeypatch.delenv("QWEN_PEAK_TFLOPS", raising=False)
    assert peak_bf16_tflops("NVIDIA A40") == 149.7
    assert peak_bf16_tflops("NVIDIA L40S") == 362.05
    assert peak_bf16_tflops("NVIDIA A100-SXM-64GB") == 312.0
    assert peak_bf16_tflops("NVIDIA H100 80GB HBM3") == 989.4
    assert peak_bf16_tflops("some future card") == 989.4
    monkeypatch.setenv("QWEN_PEAK_TFLOPS", "100")
    assert peak_bf16_tflops("NVIDIA A40") == 100.0
