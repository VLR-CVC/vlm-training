"""Throughput benchmark: run a config for N steps, print wall-clock rates.

    torchrun --nproc_per_node=N -m train.bench --config <toml> [--training.total-steps 60 ...]

or `scripts/bench.sh`. A `Trainer` whose steps end in `torch.cuda.synchronize()`,
so every step's wall time is exact (the training loop proper only tiles
intervals and is exact on average). Per step it records

* ``wall``: host time from the end of the previous step to the end of this one
  (data loading, H2D, forward, backward, clip, optimizer: everything);
* ``fwd_bwd`` / ``optim``: CUDA-event time of `forward_backward` and of the
  optimizer step (clip + `step` + `zero_grad`);
* ``engram_fwd``: CUDA-event time inside the Engram modules' forward;
* peak allocated / reserved memory.

``BENCH_UNFREEZE=<regex>`` additionally unfreezes the matching parameters (the
frozen-backbone, no-Engram reference).

and, over the steps after ``BENCH_WARMUP`` (default 10), prints one
``BENCH_RESULT {json}`` line: global tok/s, tok/s per GPU, TFLOP/s and MFU per
GPU (`train.flops_estimation`, Engram included), the mean breakdown, and the
peak memory over all ranks. ``BENCH_OUT=<path>`` also writes the JSON there.

Nothing is saved: no checkpoint, no HF export, no dataloader state.
"""

import json
import os
import statistics
import sys
import time

import torch

import train.step as step_module
import train.train_qwen as train_qwen_module
from train.config import Config
from train.config_manager import ConfigManager
from train.logger import init_logger, logger
from train.train_qwen import Trainer


def _set_trainable_parts_plus(model_args, model, _inner=train_qwen_module.set_trainable_parts):
    """`set_trainable_parts`, then unfreeze every parameter whose name matches
    BENCH_UNFREEZE (a regex). A frozen backbone without Engram has nothing to
    train; unfreezing e.g. `layers\\.3\\.attention_norm` gives the no-Engram
    reference the same backward extent as Engram after layer 2."""
    import re

    _inner(model_args, model)
    pattern = os.environ.get("BENCH_UNFREEZE")
    if pattern:
        for name, p in model.named_parameters():
            if re.search(pattern, name):
                p.requires_grad = True


train_qwen_module.set_trainable_parts = _set_trainable_parts_plus


class _Timer:
    def __init__(self):
        self.start = torch.cuda.Event(enable_timing=True)
        self.end = torch.cuda.Event(enable_timing=True)
        self.used = False

    def ms(self) -> float:
        return self.start.elapsed_time(self.end) if self.used else 0.0


class BenchTrainer(Trainer):
    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.bench_warmup = int(os.environ.get("BENCH_WARMUP", "10"))
        self.bench_rows: list[dict] = []
        self._bench_step_end = None
        self._engram_timers: list[_Timer] = []
        self._install_engram_hooks()

    # --- instrumentation ---------------------------------------------------
    def _install_engram_hooks(self):
        engram = getattr(self.model, "engram", None)
        if engram is None:
            return

        def pre(module, args):
            timer = _Timer()
            timer.start.record()
            module._bench_timer = timer

        def post(module, args, output):
            timer = module._bench_timer
            timer.end.record()
            timer.used = True
            self._engram_timers.append(timer)

        for module in engram.values():
            module.register_forward_pre_hook(pre)
            module.register_forward_hook(post)

    def train_step(self, batches, optimizer):
        if self._bench_step_end is None:
            torch.cuda.synchronize()
            self._bench_step_end = time.perf_counter()
        fwd_bwd, optim = _Timer(), _Timer()
        self._engram_timers = []
        local_tokens = sum(int(b["positions"].numel()) for b in batches)

        inner_fb = step_module.forward_backward

        def timed_forward_backward(*args, **kwargs):
            fwd_bwd.start.record()
            out = inner_fb(*args, **kwargs)
            fwd_bwd.end.record()
            fwd_bwd.used = True
            optim.start.record()  # clip + stats + step + zero_grad follow
            return out

        inner_zero = optimizer.zero_grad

        def timed_zero(*args, **kwargs):
            out = inner_zero(*args, **kwargs)
            optim.end.record()
            optim.used = True
            return out

        step_module.forward_backward = timed_forward_backward
        optimizer.zero_grad = timed_zero
        try:
            super().train_step(batches, optimizer)
        finally:
            step_module.forward_backward = inner_fb
            del optimizer.zero_grad  # back to the class method

        torch.cuda.synchronize()
        now = time.perf_counter()
        wall = now - self._bench_step_end
        self._bench_step_end = now
        self.bench_rows.append({
            "step": self.global_step,
            "wall": wall,
            "tokens": local_tokens,
            "flops": self._bench_flops,
            "fwd_bwd": fwd_bwd.ms() / 1e3,
            "optim": optim.ms() / 1e3,
            "engram_fwd": sum(t.ms() for t in self._engram_timers) / 1e3,
            "mem_alloc": self._bench_mem[0],
            "mem_resv": self._bench_mem[1],
        })
        if self.if_log_rank():
            r = self.bench_rows[-1]
            logger.info(
                f"[bench] step {r['step']} wall {r['wall']*1e3:.0f}ms "
                f"fwd_bwd {r['fwd_bwd']*1e3:.0f} optim {r['optim']*1e3:.0f} "
                f"engram_fwd {r['engram_fwd']*1e3:.1f} "
                f"tok/s/gpu {local_tokens / wall:.0f} "
                f"mem {r['mem_alloc']:.1f}/{r['mem_resv']:.1f}G"
            )

    def _stage_log(self, loss, lr, time_delta, gathered, loss_max=None):
        # read before `train_step` zeroes the counters and resets the peak
        gib = 1024 ** 3
        self._bench_flops = float(self.flops_since_last_log)
        self._bench_mem = (
            torch.cuda.max_memory_allocated(self.device) / gib,
            torch.cuda.max_memory_reserved(self.device) / gib,
        )
        return super()._stage_log(loss, lr, time_delta, gathered, loss_max)

    def save_checkpoint(self):
        pass

    # --- summary -----------------------------------------------------------
    def end_run(self):
        self._flush_log()
        host_stats = None
        if self.engram_host is not None:
            self.engram_host.close()
            st = self.engram_host.stats
            host_stats = torch.tensor(
                [st["fetched_rows"] - st["miss_rows"], st["miss_rows"], st["miss_fetches"]],
                dtype=torch.float64, device=self.device,
            )
            torch.distributed.all_reduce(host_stats)
        rows = [r for r in self.bench_rows if r["step"] > self.bench_warmup]
        if not rows:
            rows = self.bench_rows
        n = len(rows)
        keys = ("wall", "tokens", "flops", "fwd_bwd", "optim", "engram_fwd")
        local = torch.tensor(
            [sum(r[k] for r in rows) for k in keys]
            + [max(r["mem_alloc"] for r in rows), max(r["mem_resv"] for r in rows)],
            dtype=torch.float64, device=self.device,
        )
        world = self.world_size
        sums = local[:len(keys)].clone()
        maxes = local[len(keys):].clone()
        walls = local[:1].clone()
        torch.distributed.all_reduce(sums)
        torch.distributed.all_reduce(maxes, op=torch.distributed.ReduceOp.MAX)
        torch.distributed.all_reduce(walls, op=torch.distributed.ReduceOp.MAX)
        if self.if_log_rank():
            wall = walls.item()  # the slowest rank's wall time sets the job's rate
            tokens = sums[1].item() / self.tp_size  # a TP group shares one batch
            tflops = sums[2].item() * self.tp_size / wall / world / 1e12
            step_walls = [r["wall"] for r in rows]
            result = {
                "steps": n,
                "warmup": self.bench_warmup,
                "world": world,
                "device": torch.cuda.get_device_name(self.device),
                "tok_s": tokens / wall,
                "tok_s_gpu": tokens / wall / world,
                "tflops_gpu": tflops,
                "mfu": 100 * tflops / self.peak_tflops_per_gpu,
                "peak_tflops": self.peak_tflops_per_gpu,
                "step_s": wall / n,
                "step_s_median": statistics.median(step_walls),
                "fwd_bwd_s": sums[3].item() / world / n,
                "optim_s": sums[4].item() / world / n,
                "engram_fwd_s": sums[5].item() / world / n,
                "tokens_per_step": tokens / n,
                "peak_alloc_gib": maxes[0].item(),
                "peak_resv_gib": maxes[1].item(),
            }
            if host_stats is not None:
                # summed over ranks, whole run (warmup included)
                result.update(host_prefetched_rows=int(host_stats[0]),
                              host_miss_rows=int(host_stats[1]),
                              host_miss_fetches=int(host_stats[2]))
            logger.info("BENCH_RESULT " + json.dumps(result))
            out = os.environ.get("BENCH_OUT")
            if out:
                os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
                with open(out, "w") as f:
                    json.dump(result, f, indent=1)
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def main(argv: list[str]) -> None:
    config = ConfigManager(Config).parse_args(argv)
    init_logger()
    torch.manual_seed(42)
    BenchTrainer(config).train()


if __name__ == "__main__":
    main(sys.argv[1:])
