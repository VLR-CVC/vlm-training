import json
import time

import torch
import wandb

from train.logger import logger
from train.utils import MOE_STAT_NAMES, PERF_ROW, dist_all_gather, dist_max, dist_sum, topk_metrics
from train.vision_timer import VISION_TIMER

WALL_CHECK_AFTER = 10  # steps after the start step: compile warmup excluded

class TrainLogging:
    def _log_batch_shapes(self, batch: dict, ntokens: int) -> None:
        positions = batch["positions"]
        starts = (positions == 0).sum()
        grid = batch.get("grid_thw")
        self._batch_log.write(json.dumps({
            "wall": time.time(),
            "step": self.global_step,
            "tokens": int(ntokens),
            "docs": int(starts),
            "max_doc": int(torch.diff(
                torch.cat([(positions == 0).nonzero().flatten(),
                           torch.tensor([positions.numel()])])
            ).max()) if starts else 0,
            "patches": int(grid.prod(-1).sum()) if grid is not None and grid.numel() else 0,
            "images": int(grid.shape[0]) if grid is not None and grid.numel() else 0,
        }) + "\n")
        self._batch_log.flush()

    @staticmethod
    def perf_rank_share(ntokens, flops, *, tp, time_delta, peak_tflops):
        """This rank's SHARE of the job's rates, so that the gathered rows are a
        decomposition of the log line rather than a second opinion about it."""
        tps = ntokens / tp / time_delta
        tflops = flops / time_delta / 1e12
        return tps, tflops, (tflops / peak_tflops) * 100

    def _gather_perf(self, time_delta):
        """Per-rank perf row, gathered across the world."""
        tps, tflops_per_sec, mfu = self.perf_rank_share(
            self.ntokens_since_last_log, self.flops_since_last_log,
            tp=self.tp_size, time_delta=time_delta,
            peak_tflops=self.peak_tflops_per_gpu,
        )
        peak_mem_gib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        # vision-encoder compute of the PREVIOUS step (its events have completed by now);
        # see train/vision_timer.py for why only the vision encoder is timed
        vit_fwd, vit_bwd, patches, attn = VISION_TIMER.collect()

        local = torch.empty(len(PERF_ROW), dtype=torch.float32, device=self.device)
        local[0] = tps
        local[1] = self.microbatch_time
        local[2] = tflops_per_sec
        local[3] = mfu
        local[4] = peak_mem_gib
        local[5] = vit_fwd + vit_bwd
        local[6] = patches / 1e3
        local[7] = vit_fwd
        local[8] = attn / 1e6
        local[9] = time_delta
        return dist_all_gather(local, torch.distributed.group.WORLD)

    def _stage_log(self, loss, lr, time_delta, gathered, loss_max=None):
        """We store the current step log and show it in the next step.
        Reduces step time.
        """
        loss64 = loss.detach().to(torch.float64).reshape(1)
        sums = torch.cat([
            loss64,
            self.tokens_seen.to(torch.float64).reshape(1),
            self.tokens_seen_assistant.to(torch.float64).reshape(1),
            torch.full(
                (1,), float(self.samples_since_last_log),
                dtype=torch.float64, device=self.device,
            ),
            self.ntokens_since_last_log.to(torch.float64).reshape(1),
            self.flops_since_last_log.reshape(1),
        ])
        gib = 1024 ** 3
        mem = torch.tensor(
            [torch.cuda.max_memory_allocated(self.device) / gib,
             torch.cuda.max_memory_reserved(self.device) / gib],
            dtype=torch.float64, device=self.device,
        )
        mx_in = torch.cat([
            (loss64 if loss_max is None else loss_max.detach().to(torch.float64).reshape(1)),
            mem,
        ])
        if self.dp_mesh is not None:
            sums = dist_sum(sums, self.dp_mesh)
        # MAX over WORLD
        mx = dist_max(mx_in, torch.distributed.group.WORLD) if self.world_size > 1 else mx_in

        grad_norm = self.grad_norm
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        ckpt_time, self._ckpt_time = self._ckpt_time, 0.0
        # every rank ran the collectives above; only the log rank reads values
        if not self.if_log_rank():
            return

        grad_norm = torch.as_tensor(
            grad_norm, dtype=torch.float64, device=self.device
        ).reshape(1)

        vec = torch.cat([
            sums, mx,
            grad_norm,
        ])
        self._log_host.copy_(vec, non_blocking=True)
        self._log_event.record()

        self._log_pending = {
            "step": self.global_step,
            "lr": lr,
            "time_delta": time_delta,
            "n_micro": self.n_micro,
            "fwd_bwd_time": self.fwd_bwd_time,
            "opt_time": self.opt_time,
            "data_time_delta": self.data_time_delta,
            # cost of the flush that ran immediately before this stage
            "log_wait": self._log_wait,
            # checkpoint saved since the previous step, kept out of time_delta
            "ckpt_time": ckpt_time,
            "gathered": gathered,
            "moe": self.moe_stats.get("vec"),
            "loss_paths": dict(self.loss_metrics) if self.loss_metrics else None,
        }

    def _flush_log(self):
        """Emit the previous step's record. The copy it reads was queued a full
        step ago, so the event wait below is already satisfied and returns
        without stalling."""
        rec = self._log_pending
        if rec is None:
            return
        self._log_pending = None

        t0 = time.perf_counter()
        self._log_event.synchronize()
        self._log_wait = time.perf_counter() - t0
        self.log(rec, self._log_host.tolist())

    def log(self, rec, h):
        """`rec` is the host-side snapshot taken when the step was staged; `h`
        is the reduced counter vector read back out of pinned memory. Slot order
        is fixed by `_stage_log`."""
        time_delta = rec["time_delta"]

        # SUM/n rather than a second collective on ReduceOp.AVG
        dp_size = self.dp_size
        avg_loss = h[0] / dp_size
        global_tokens = int(h[1])
        global_assistant_tokens = int(h[2])
        global_samples = int(h[3])
        max_loss = h[6]
        # global: summed over dp, and a TP group shares one micro-batch
        tps = h[4] / time_delta
        tps_per_gpu = tps / self.world_size
        # real tokens over the step's capacity: dp x micro-batches x microbatch_tokens
        batch_efficiency = h[4] / (dp_size * rec["n_micro"] * self.data_args.microbatch_tokens) * 100
        grad_norm = h[9]

        job_flops_per_sec = h[5] * self.tp_size / time_delta
        tflops_per_sec = job_flops_per_sec / self.world_size / 1e12
        mfu = (tflops_per_sec / self.peak_tflops_per_gpu) * 100

        color = self.color

        data_time_pct = (rec["data_time_delta"] / time_delta) * 100

        peak_alloc = h[7]
        peak_resv = h[8]

        logger.info(
            f"{color.red}{rec['step']}{color.reset} - "
                f"{color.green}loss {avg_loss:.4f} "
                f"{color.blue}tps {tps:.0f} ({tps_per_gpu:.0f}/gpu) "
                f"{color.magenta}mfu {mfu:.1f}% "
                f"{color.cyan}tflops {tflops_per_sec:.1f} "
                f"{color.reset}"
                f"gnorm {grad_norm:.3f} "
                f"time {time_delta:.3f}s "
                + (f"fwd {rec['fwd_bwd_time']:.3f}s opt {rec['opt_time']:.3f}s "
                   if rec["opt_time"] else "") +
                f"mem {peak_alloc:.1f}/{peak_resv:.1f}G "
                f"data_pct {data_time_pct:.2f}% "
                f"nsamples {global_samples} "
                f"batch_util {batch_efficiency:.1f}% "
        )

        log_metrics = {
            "train/loss": avg_loss,
            "train/max_loss": max_loss,
            "train/tokens_seen": global_tokens,
            "train/assistant_tokens_seen": global_assistant_tokens,
            "train/num_samples": global_samples,
            "train/lr": rec["lr"],
            "train/grad_norm": grad_norm,
            "train/batch_efficiency": batch_efficiency,

            # performance related
            "perf/tokens_per_second": tps,
            "perf/tokens_per_second_per_gpu": tps_per_gpu,
            "perf/data_time_pct": data_time_pct,
            "perf/microbatch_time": time_delta / rec["n_micro"],
            "perf/opt_step_time": time_delta,  # whole optimizer step
            "perf/tflops_per_second": tflops_per_sec,
            "perf/mfu": mfu,
            "perf/log_wait_ms": rec["log_wait"] * 1e3,
            "perf/mem_gib": peak_alloc,
            "perf/mem_reserved_gib": peak_resv,
        }

        if rec["opt_time"]:  # QWEN_SYNC_TIMING=1: GPU time
            log_metrics["perf/fwd_bwd_time"] = rec["fwd_bwd_time"]
            log_metrics["perf/opt_time"] = rec["opt_time"]

        if rec.get("loss_paths"):
            paths = {k: v.item() for k, v in rec["loss_paths"].items()}
            logger.info(
                "  mtp: " + "  ".join(f"{k.replace('loss_', '')} {v:.4f}"
                                      for k, v in paths.items())
            )
            log_metrics.update({f"train/{k}": v for k, v in paths.items()})

        if rec.get("moe") is not None:
            moe = dict(zip(MOE_STAT_NAMES, rec["moe"].tolist(), strict=True))
            logger.info(
                f"  moe: max/mean {moe['max_over_mean']:.2f} "
                f"(worst {moe['max_over_mean_worst']:.2f} @ layer "
                f"{int(moe['worst_layer'])})  cv {moe['cv']:.3f}  "
                f"dead {100 * moe['dead_expert_frac']:.1f}%  "
                f"|bias| {moe['expert_bias_absmax']:.4f}"
            )
            log_metrics.update({f"moe/{k}": v for k, v in moe.items()})

        if rec["ckpt_time"]:
            logger.info(f"  checkpoint save {rec['ckpt_time']:.1f}s (not in the step time)")
            log_metrics["perf/checkpoint_time_s"] = rec["ckpt_time"]

        if rec["gathered"] is not None:
            topk = topk_metrics(rec["gathered"], self.wandb_args.top_k)
            log_metrics.update(topk)
            if "perf/vit_ms_max" in topk:
                logger.info(
                    f"  vit compute (prev step) mean {topk['perf/vit_ms_mean']:.0f} ms, "
                    f"max {topk['perf/vit_ms_max']:.0f} ms (rank {topk['perf_topk/vit_ms_slow_0_rank']}), "
                    f"max/mean {topk['perf/vit_max_over_mean']:.2f}, "
                    f"imbalance {topk['perf/vit_imbalance_ms']:.0f} ms "
                    f"({100 * topk['perf/vit_imbalance_frac']:.1f}% of the step)"
                )

        wandb.log(log_metrics, step=rec["step"])

    def _mark_wall_ref(self):
        """Reference point for `_log_wall_clock_rate`, WALL_CHECK_AFTER steps in."""
        if self.global_step == self._start_step + WALL_CHECK_AFTER:
            torch.cuda.synchronize()
            self._wall_ref = (self.global_step, time.perf_counter(),
                              self.tokens_seen.clone(), self._ckpt_time_total)

    def _log_wall_clock_rate(self):
        """tok/s from a synced clock and the token counter since `_wall_ref`: an
        outside check on the per-step tps, which should agree within a few percent."""
        if self._wall_ref is None:
            return
        step, t0, tokens0, ckpt0 = self._wall_ref
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        tokens = (self.tokens_seen - tokens0).to(torch.float64).reshape(1)
        if self.dp_mesh is not None:
            tokens = dist_sum(tokens, self.dp_mesh)
        # per-step tps leaves saves out: compare it with the second figure
        ckpt = self._ckpt_time_total - ckpt0
        if self.if_log_rank() and elapsed > ckpt:
            logger.info(
                f"wall-clock tok/s since step {step}: "
                f"{float(tokens) / elapsed:.0f} over {elapsed:.0f}s; "
                f"{float(tokens) / (elapsed - ckpt):.0f} without the "
                f"{ckpt:.0f}s of checkpoint saves (compare with the per-step tps above)"
            )

class VisionTimer:
    def __init__(self) -> None:
        self.enabled = False
        self._cur: list[dict] = []
        self._prev: list[dict] = []
        self._cur_counts = [0, 0]  # patches, sum over images of patches^2 (attention cost)
        self._prev_counts = [0, 0]

    def new_step(self) -> None:
        if self.enabled:
            self._prev, self._cur = self._cur, []
            self._prev_counts, self._cur_counts = self._cur_counts, [0, 0]
        else:  # nothing stale survives a switch-off
            self._prev, self._cur = [], []
            self._prev_counts, self._cur_counts = [0, 0], [0, 0]

    @staticmethod
    def _event() -> torch.cuda.Event:
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def forward_start(self, grids) -> dict | None:
        """``grids``: list of [t, h, w] in patches. Returns the record to pass on."""
        if not self.enabled or torch.compiler.is_compiling():
            return None
        rec = {"f0": self._event()}
        self._cur.append(rec)
        for t, h, w in grids:
            self._cur_counts[0] += t * h * w
            self._cur_counts[1] += t * (h * w) ** 2
        return rec

    def forward_end(self, rec: dict | None) -> None:
        if rec is not None:
            rec["f1"] = self._event()

    def backward_marks(self, rec: dict | None, out_side: torch.Tensor, in_side: torch.Tensor) -> None:
        """Record when backward reaches ``out_side`` (start) and ``in_side`` (end)."""
        if rec is None or not (out_side.requires_grad and in_side.requires_grad):
            return

        def mark(name):
            def hook(grad):
                rec[name] = self._event()  # autograd runs the hook on the op's stream
            return hook

        out_side.register_hook(mark("b0"))
        in_side.register_hook(mark("b1"))

    def collect(self) -> tuple[float, float, int, int]:
        """Previous step's (forward ms, backward ms, patches, sum patches^2) on this rank."""
        fwd = bwd = 0.0
        for rec in self._prev:
            if "f1" in rec:
                rec["f1"].synchronize()
                fwd += rec["f0"].elapsed_time(rec["f1"])
            if "b0" in rec and "b1" in rec:
                rec["b1"].synchronize()
                bwd += rec["b0"].elapsed_time(rec["b1"])
        return fwd, bwd, self._prev_counts[0], self._prev_counts[1]

VISION_TIMER = VisionTimer()
