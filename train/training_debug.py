import contextlib
import io
import json
import os
import pstats

import torch
from torch.profiler import profile, ProfilerActivity, schedule

from train.logger import logger

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
