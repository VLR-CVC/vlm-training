import contextlib
import io
import json
import os
import pstats

import torch
from torch.profiler import profile, ProfilerActivity, schedule

from train.logger import logger


def write_batch_stats(batch: dict, out_dir: str, global_step: int, micro_step: int) -> None:
    """Write batch diagnostics to disk before the forward pass.

    Each file is fsynced so the data survives a SIGKILL from the OOM killer. The
    filename encodes both the optimizer step and the accumulation index, making
    it easy to identify the exact micro-batch that triggered an OOM.
    """
    cu = batch['attention_mask'].cpu()
    seq_lens = (cu[1:] - cu[:-1]).tolist()

    stats: dict = {
        "global_step": global_step,
        "micro_step": micro_step,
        "total_tokens": int(batch['input_ids'].shape[1]),
        "num_samples": len(seq_lens),
        "seq_lens": seq_lens,
        "max_seqlen": max(seq_lens) if seq_lens else 0,
        "cuda_mem_allocated_gib": round(torch.cuda.memory_allocated() / 2**30, 3),
        "cuda_mem_reserved_gib": round(torch.cuda.memory_reserved() / 2**30, 3),
    }

    for key, pv_key, grid_key in (
        ("images", "pixel_values", "image_grid_thw"),
        ("videos", "pixel_values_videos", "video_grid_thw"),
    ):
        if batch.get(pv_key) is None:
            continue
        pv = batch[pv_key]
        grids = batch[grid_key].cpu().tolist()
        stats[key] = {
            "num": len(grids),
            "grids_thw": grids,
            "total_patches": int(sum(t * h * w for t, h, w in grids)),
            "pixel_values_shape": list(pv.shape),
            "pixel_values_bytes": pv.numel() * pv.element_size(),
        }

    fname = f"step_{global_step:07d}_accum_{micro_step:02d}.json"
    path = os.path.join(out_dir, fname)
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
        f.flush()
        os.fsync(f.fileno())


def dump_cprofile(prof, output_dir: str, rank: int) -> None:
    """Dump sorted cProfile stats for ``rank`` under ``output_dir``."""
    out_path = os.path.join(output_dir, f"cprofile_rank_{rank}.txt")
    stream = io.StringIO()
    pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(50)
    with open(out_path, "w") as f:
        f.write(stream.getvalue())
    logger.info(f"cProfile stats written to {out_path}")


def make_trace_handler(output_dir: str, rank: int, is_log_rank: bool):
    """Build an ``on_trace_ready`` handler that writes a chrome trace + a text
    summary per profiler window."""
    def trace_handler(prof):
        trace_path = os.path.join(output_dir, f"trace_rank_{rank}_step_{prof.step_num}.json")
        prof.export_chrome_trace(trace_path)
        summary_path = trace_path.replace(".json", "_summary.txt")
        with open(summary_path, "w") as f:
            f.write(prof.key_averages(group_by_stack_n=5).table(
                sort_by="self_cpu_time_total", row_limit=40
            ))
        if is_log_rank:
            logger.info(f"Torch profiler trace → {trace_path}")
            logger.info(f"Torch profiler summary → {summary_path}")
    return trace_handler


def build_debug_profiler(debug_mode: bool, output_dir: str, rank: int, is_log_rank: bool):
    """Return ``(profiler_ctx, cprofile, cprof_start, cprof_stop)``.

    When ``debug_mode`` is False this is a no-op context and ``None`` handle, so
    the training loop stays branch-light.
    """
    if not debug_mode:
        return contextlib.nullcontext(), None, 0, 0

    # wait=10 skips the torch.compile warmup steps; active=2 records 2 full steps
    prof_schedule = schedule(wait=10, warmup=2, active=2, repeat=1)
    prof_ctx = profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        schedule=prof_schedule,
        on_trace_ready=make_trace_handler(output_dir, rank, is_log_rank),
        record_shapes=True,
        profile_memory=False,
        with_stack=True,
    )
    # cProfile window: steady-state steps well after compilation
    import cProfile
    return prof_ctx, cProfile.Profile(), 50, 65
