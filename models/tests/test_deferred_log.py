"""The logging path reads its reduced counters out of a pinned buffer by index.
Eight magic indices written in one method and read in another is exactly the
kind of thing that silently mislabels a metric, so pin the mapping.

CPU-only: `log` is the half of the deferred path that touches no device."""
import types

import pytest

import train.train_qwen as tq
from train.logger import Color

# slot order, per `_stage_log`
(
    LOSS_SUM, TOKENS, ASSISTANT, SAMPLES, NTOK_LOG, FLOPS,
    LOSS_MAX, PEAK_ALLOC, PEAK_RESV,
    NTOK_BATCH, GNORM,
) = range(11)

# The staged vector and this map have to agree; `_LOG_SLOTS` is the contract.
from train.train_qwen import _LOG_SLOTS

assert _LOG_SLOTS == GNORM + 1, (
    f"_LOG_SLOTS is {_LOG_SLOTS} but this test maps {GNORM + 1} slots -- "
    "a slot was added to the staged log vector without updating the names here"
)

DP_SIZE = 4
TP_SIZE = 2
WORLD_SIZE = DP_SIZE * TP_SIZE
SEQ_LEN = 1000
PEAK_TFLOPS = 100.0


def _trainer():
    t = object.__new__(tq.Trainer)
    t.dp_size = DP_SIZE
    t.tp_size = TP_SIZE
    t.world_size = WORLD_SIZE
    t.data_args = types.SimpleNamespace(seq_len=SEQ_LEN, microbatch_tokens=SEQ_LEN)
    t.wandb_args = types.SimpleNamespace(top_k=4)
    t.peak_tflops_per_gpu = PEAK_TFLOPS
    t.color = Color()
    return t


def _rec(**over):
    rec = {
        "step": 7,
        "lr": 1e-4,
        "time_delta": 2.0,
        "train_step_delta": 2.0,
        "fwd_bwd_time": 1.5,
        "data_time_delta": 0.2,
        "peak_alloc": 60.5,
        "peak_resv": 70.5,
        "log_wait": 0.0004,
        "log_emit": 0.0021,
        "sections": {},
        "gathered": None,
    }
    rec.update(over)
    return rec


def _h(**over):
    h = [0.0] * _LOG_SLOTS
    h[LOSS_SUM] = 8.0       # 4 ranks x loss 2.0
    h[TOKENS] = 123456.0
    h[ASSISTANT] = 6543.0
    h[SAMPLES] = 40.0
    h[LOSS_MAX] = 2.5
    h[NTOK_LOG] = 4000.0
    h[NTOK_BATCH] = 900.0
    h[GNORM] = 0.75
    # dp SUM of a per-rank value already divided by tp; `log` multiplies by tp
    h[FLOPS] = 8e12
    h[PEAK_ALLOC] = 60.5
    h[PEAK_RESV] = 70.5
    for k, v in over.items():
        h[globals()[k]] = v
    return h


def _capture(t, rec, h):
    """Run `log` with wandb and the logger stubbed out, return the metrics dict."""
    seen = {}
    real_wandb, real_logger = tq.wandb, tq.logger
    tq.wandb = types.SimpleNamespace(
        log=lambda m, step=None: seen.update(metrics=m, step=step)
    )
    tq.logger = types.SimpleNamespace(info=lambda *a, **k: None)
    try:
        t.log(rec, h)
    finally:
        tq.wandb, tq.logger = real_wandb, real_logger
    return seen


def test_slot_mapping():
    seen = _capture(_trainer(), _rec(), _h())
    m = seen["metrics"]

    # the loss is SUM-reduced and divided here, not AVG-reduced on device
    assert m["train/loss"] == 8.0 / DP_SIZE == 2.0
    assert m["train/max_loss"] == 2.5
    assert m["train/tokens_seen"] == 123456
    assert m["train/assistant_tokens_seen"] == 6543
    assert m["train/num_samples"] == 40
    assert m["train/grad_norm"] == 0.75

    # tps is the dp-reduced token count: a global rate, plus its per-GPU share
    assert m["perf/tokens_per_second"] == 4000.0 / 2.0
    assert m["perf/tokens_per_second_per_gpu"] == 4000.0 / 2.0 / WORLD_SIZE

    # batch_efficiency comes from the last batch's token count
    assert m["train/batch_efficiency"] == (900.0 / SEQ_LEN) * 100

    # counts come back as ints, not floats carried through float64 staging
    for k in ("train/tokens_seen", "train/assistant_tokens_seen", "train/num_samples"):
        assert isinstance(m[k], int), k


def test_record_fields_not_live_state():
    """Everything time- and memory-related comes from the staged record, so a
    record emitted one step late still describes the step it was staged on."""
    seen = _capture(_trainer(), _rec(step=99, train_step_delta=3.0, lr=5e-5), _h())
    assert seen["step"] == 99
    assert seen["metrics"]["perf/step_time"] == 3.0
    assert seen["metrics"]["perf/fwd_bwd_time"] == 1.5
    assert seen["metrics"]["train/lr"] == 5e-5
    assert seen["metrics"]["perf/mem_gib"] == 60.5
    assert seen["metrics"]["perf/mem_reserved_gib"] == 70.5
    # h[FLOPS] x tp / world / 2 s = 8e12 x 2 / 8 / 2 s = 1 TFLOP/s per GPU
    assert seen["metrics"]["perf/tflops_per_second"] == 1.0
    assert seen["metrics"]["perf/mfu"] == 1.0
    assert seen["metrics"]["perf/data_time_pct"] == 10.0
    assert seen["metrics"]["perf/log_wait_ms"] == 0.4
    assert round(seen["metrics"]["perf/log_emit_ms"], 6) == 2.1


def test_section_timings_are_logged_when_present():
    """QWEN_SECTION_TIMING=1 fills rec["sections"]; empty otherwise and then no
    perf_fwd/* keys should appear."""
    plain = _capture(_trainer(), _rec(), _h())["metrics"]
    assert not any(k.startswith("perf_fwd/") for k in plain)

    timed = _capture(
        _trainer(), _rec(sections={"visual": 12.5, "layers": 98.0}), _h()
    )["metrics"]
    assert timed["perf_fwd/visual_ms"] == 12.5
    assert timed["perf_fwd/layers_ms"] == 98.0


def test_perf_topk_decomposes_the_headline():
    """The fairness invariant: `perf_topk/*` is a decomposition of the headline,
    not a parallel set of numbers in different units.

    The per-rank rows come from the real `Trainer.perf_rank_share`, the same one
    `_gather_perf` uses, so this fails if that formula drifts from what `log`
    reports. It is exactly what broke: the row held a per-DP-replica token rate
    while the headline was per GPU, and the two disagreed by a factor of `tp`.
    """
    import torch

    from train.utils import PERF_METRIC_NAMES

    TPS, TFLOPS, MEM = (PERF_METRIC_NAMES.index(n) for n in ("tps", "tflops", "mem_gib"))
    time_delta = 2.0

    # Per-rank counters as they stand at log time: ranks of a TP group hold the
    # same micro-batch, DP groups hold different ones (uneven, as real packing is).
    per_replica_tokens = [700.0, 900.0, 1100.0, 1300.0]     # sums to h[NTOK_LOG]
    rows = torch.zeros(WORLD_SIZE, len(PERF_METRIC_NAMES))
    for r in range(WORLD_SIZE):
        ntokens = per_replica_tokens[r // TP_SIZE]
        # h[FLOPS] is the dp SUM, and both ranks of a TP group hold the same value
        flops = 8e12 / DP_SIZE
        tps, tflops, mfu = tq.Trainer.perf_rank_share(
            ntokens, flops, tp=TP_SIZE, time_delta=time_delta,
            peak_tflops=PEAK_TFLOPS,
        )
        rows[r, TPS], rows[r, TFLOPS] = tps, tflops
        rows[r, MEM] = 40.0 + r
    h = _h()
    h[NTOK_LOG] = sum(per_replica_tokens)
    h[PEAK_ALLOC] = float(rows[:, MEM].max())     # the world MAX, per `_stage_log`

    m = _capture(_trainer(), _rec(gathered=rows), h)["metrics"]

    # additive: the rows sum to the job total and average to the per-GPU headline
    assert m["perf/tokens_per_second"] == pytest.approx(float(rows[:, TPS].sum()))
    assert m["perf/tokens_per_second_per_gpu"] == pytest.approx(float(rows[:, TPS].mean()))
    assert m["perf/tflops_per_second"] == pytest.approx(float(rows[:, TFLOPS].mean()))
    # saturating: the headline is the max, not rank 0 and not the mean
    assert m["perf/mem_gib"] == pytest.approx(float(rows[:, MEM].max()))
    assert m["perf_topk/mem_gib_slow_0"] == pytest.approx(float(rows[:, MEM].max()))


def test_flush_without_pending_is_a_noop():
    t = object.__new__(tq.Trainer)
    t._log_pending = None
    t._log_event = None  # would raise if touched
    t._flush_log()


if __name__ == "__main__":
    test_slot_mapping()
    test_record_fields_not_live_state()
    test_section_timings_are_logged_when_present()
    test_perf_topk_decomposes_the_headline()
    test_flush_without_pending_is_a_noop()
    print("deferred log: 5 checks passed")
