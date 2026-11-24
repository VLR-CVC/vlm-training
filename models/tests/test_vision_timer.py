"""VisionTimer: the per-rank vision compute it reports matches an independent synchronized
measurement, under eager and per-block activation checkpointing, heavier images read
heavier, it costs nothing measurable, and topk_metrics turns the gathered rows into the
imbalance summary. Needs a GPU.

    python models/tests/test_vision_timer.py
"""

import sys
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.configs import qwen35_config_from_hf  # noqa: E402
from train.utils import PERF_ROW, topk_metrics  # noqa: E402
from train.vision_timer import VISION_TIMER  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
LIGHT = [[1, 32, 32]] * 2                        # 2 x 1024 patches
HEAVY = [[1, 64, 64], [1, 96, 64], [1, 16, 16]]  # 4096 + 6144 + 256 patches


def build():
    torch.manual_seed(0)
    cfg = qwen35_config_from_hf(str(ROOT / "configs/models/qwen3_5_2b_so400m.json"), seq_len=1024)
    enc = cfg.vision_encoder.build().cuda()
    for p in enc.parameters():
        torch.nn.init.normal_(p, std=0.02)
    return enc.to(torch.bfloat16)


def checkpoint_blocks(enc):
    for layer in enc.layers.values():
        fwd = layer.forward
        layer.forward = lambda *a, _f=fwd, **k: checkpoint(_f, *a, use_reentrant=False, **k)


def inputs(enc, grids):
    grid = torch.tensor(grids, device="cuda")
    n = int(grid.prod(-1).sum())
    return torch.randn(n, enc.patch_embed.weight.shape[1], device="cuda", dtype=torch.bfloat16), grid


def step(enc, pixels, grid):
    """One timed step; returns the synchronized reference (fwd ms, bwd ms)."""
    VISION_TIMER.new_step()
    torch.cuda.synchronize(); t = time.perf_counter()
    out = enc(pixels, grid_thw=grid)
    torch.cuda.synchronize(); fwd = (time.perf_counter() - t) * 1e3
    loss = (out.float() ** 2).mean()
    torch.cuda.synchronize(); t = time.perf_counter()
    loss.backward()
    torch.cuda.synchronize(); bwd = (time.perf_counter() - t) * 1e3
    enc.zero_grad(set_to_none=True)
    return fwd, bwd


def measure(enc, grids, reps=5):
    pixels, grid = inputs(enc, grids)
    for _ in range(2):
        step(enc, pixels, grid)  # warm-up
    rows = []
    for _ in range(reps):
        ref = step(enc, pixels, grid)
        VISION_TIMER.new_step()   # the next step's start makes this one readable
        got = VISION_TIMER.collect()
        rows.append((got, ref))
    return rows


def check_against_reference(label, rows, patches):
    for (fwd, bwd, n, attn), (rf, rb) in rows:
        assert n == patches, (n, patches)
        # the timer leaves out what lies outside its markers (input cast, merger backward,
        # patch-embed weight grad) and the host launch gaps the synchronized wall time has
        assert 0.7 * rf <= fwd <= 1.02 * rf, (label, "fwd", fwd, rf)
        assert 0.7 * rb <= bwd <= 1.02 * rb, (label, "bwd", bwd, rb)
    (fwd, bwd, *_), (rf, rb) = rows[-1]
    print(f"{label:28s} timer fwd {fwd:6.1f} bwd {bwd:6.1f} ms | synchronized fwd {rf:6.1f} bwd {rb:6.1f} ms")
    return fwd + bwd


def test_timer():
    if not torch.cuda.is_available():
        print("no GPU: skipped")
        return
    VISION_TIMER.enabled = True
    enc = build()
    n_light = sum(t * h * w for t, h, w in LIGHT)
    n_heavy = sum(t * h * w for t, h, w in HEAVY)
    light = check_against_reference("eager, light", measure(enc, LIGHT), n_light)
    heavy = check_against_reference("eager, heavy", measure(enc, HEAVY), n_heavy)
    assert heavy > 2 * light, (heavy, light)
    checkpoint_blocks(enc)
    ck = check_against_reference("block checkpoint, heavy", measure(enc, HEAVY), n_heavy)
    assert ck > heavy, "recompute should land in the backward window"

    # disabled: nothing recorded
    VISION_TIMER.enabled = False
    VISION_TIMER.new_step(); VISION_TIMER.new_step()
    pixels, grid = inputs(enc, HEAVY)
    step(enc, pixels, grid)
    assert VISION_TIMER.collect()[:2] == (0.0, 0.0)

    # overhead: whole steps, enabled vs disabled
    def wall(enabled, reps=10):
        VISION_TIMER.enabled = enabled
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(reps):
            VISION_TIMER.new_step()
            out = enc(pixels, grid_thw=grid)
            (out.float() ** 2).mean().backward()
            enc.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return (time.perf_counter() - t) / reps * 1e3
    wall(True, 2)
    off, on = wall(False), wall(True)
    print(f"step wall: timer off {off:.1f} ms, on {on:.1f} ms ({100 * (on - off) / off:+.2f}%)")
    assert on <= off * 1.02 + 0.5, (on, off)


def test_topk_summary():
    rows = torch.zeros(4, len(PERF_ROW))
    i = {n: k for k, n in enumerate(PERF_ROW)}
    rows[:, i["vit_ms"]] = torch.tensor([100.0, 100.0, 200.0, 100.0])
    rows[:, i["vit_kpatches"]] = torch.tensor([4.0, 4.0, 10.0, 2.0])
    rows[:, i["vit_attn_m"]] = torch.tensor([1.0, 1.0, 5.0, 1.0])
    rows[:, i["step_time"]] = 1.0
    m = topk_metrics(rows, 2)
    assert m["perf/vit_ms_max"] == 200 and m["perf/vit_ms_mean"] == 125
    assert m["perf/vit_imbalance_ms"] == 75 and abs(m["perf/vit_imbalance_frac"] - 0.075) < 1e-6
    assert abs(m["perf/vit_max_over_mean"] - 1.6) < 1e-6
    assert m["perf_topk/vit_ms_slow_0_rank"] == 2 and m["perf_topk/vit_kpatches_slow_0_rank"] == 2
    assert abs(m["perf/vit_attn_max_over_mean"] - 2.5) < 1e-6
    # no vision timings (timer off): no summary
    rows[:, i["vit_ms"]] = 0
    assert "perf/vit_ms_max" not in topk_metrics(rows, 2)
    print("topk summary ok")


if __name__ == "__main__":
    test_topk_summary()
    test_timer()
    print("ok")
