"""`topk_metrics` moves the gathered rows to the host once instead of calling
`.item()` 96 times on a CUDA tensor. That is meant to be behaviour-preserving;
this pins the ordering and the returned types so it stays that way."""
import torch

from train.utils import PERF_METRIC_NAMES, topk_metrics


def _rows():
    # 4 ranks x 6 metrics, columns in PERF_METRIC_NAMES order:
    # tps, step_time, fwd_bwd_time, tflops, mfu, mem_gib
    return torch.tensor(
        [
            [100.0, 1.0, 0.8, 50.0, 5.0, 60.0],  # rank 0: fastest
            [200.0, 2.0, 1.6, 40.0, 4.0, 70.0],  # rank 1
            [300.0, 3.0, 2.4, 30.0, 3.0, 80.0],  # rank 2
            [400.0, 4.0, 3.2, 20.0, 2.0, 90.0],  # rank 3: slowest
        ]
    )


def test_ordering():
    m = topk_metrics(_rows(), top_k=2)

    # tps: higher is better -> rank 3 fastest, rank 0 slowest
    assert m["perf_topk/tps_fast_0_rank"] == 3
    assert m["perf_topk/tps_fast_0"] == 400.0
    assert m["perf_topk/tps_slow_0_rank"] == 0
    assert m["perf_topk/tps_slow_0"] == 100.0

    # step_time: lower is better -> the ranking flips
    assert m["perf_topk/step_time_fast_0_rank"] == 0
    assert m["perf_topk/step_time_slow_0_rank"] == 3

    # second-place entries are present and correctly ordered
    assert m["perf_topk/tps_fast_1_rank"] == 2
    assert m["perf_topk/tps_slow_1_rank"] == 1


def test_plain_python_types():
    """No tensors leak into the dict -- wandb.log would take them, but every
    one would be an unnecessary sync at the call site."""
    m = topk_metrics(_rows(), top_k=2)
    assert len(m) == len(PERF_METRIC_NAMES) * 2 * 2 * 2
    for k, v in m.items():
        assert isinstance(v, (float, int)), f"{k} is {type(v)}"
        assert not isinstance(v, torch.Tensor)


def test_k_clamped_to_rank_count():
    m = topk_metrics(_rows(), top_k=10)
    assert "perf_topk/tps_fast_3" in m
    assert "perf_topk/tps_fast_4" not in m


if __name__ == "__main__":
    test_ordering()
    test_plain_python_types()
    test_k_clamped_to_rank_count()
    print("topk_metrics: 3 checks passed")
