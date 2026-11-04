"""Step logging for `train.train_qwen.Trainer`: stage the reduced counters, read
them back a step later once the event has fired, and write the console line and
the wandb metrics. A mixin: the methods read the trainer's own state."""

import json
import time

import torch
import wandb

from train.logger import logger
from train.utils import MOE_STAT_NAMES, dist_all_gather, dist_max, dist_sum, topk_metrics


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

        local = torch.empty(6, dtype=torch.float32, device=self.device)
        local[0] = tps
        local[1] = self.train_step_delta
        local[2] = self.fwd_bwd_time
        local[3] = tflops_per_sec
        local[4] = mfu
        local[5] = peak_mem_gib
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

        # Popped on every rank so the dict does not grow on the ones that
        # never read it. Empty unless QWEN_SECTION_TIMING=1.
        sections = {}

        grad_norm = self.grad_norm
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()

        # every rank ran the collectives above; only the log rank reads values
        if not self.if_log_rank():
            return

        grad_norm = torch.as_tensor(
            grad_norm, dtype=torch.float64, device=self.device
        ).reshape(1)

        vec = torch.cat([
            sums, mx,
            self.ntokens_last_batch.to(torch.float64).reshape(1),
            grad_norm,
        ])
        self._log_host.copy_(vec, non_blocking=True)
        self._log_event.record()

        self._log_pending = {
            "step": self.global_step,
            "sections": sections,
            "lr": lr,
            "time_delta": time_delta,
            "train_step_delta": self.train_step_delta,
            "fwd_bwd_time": self.fwd_bwd_time,
            "data_time_delta": self.data_time_delta,
            # cost of the flush that ran immediately before this stage
            "log_wait": self._log_wait,
            "log_emit": self._log_emit,
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
        t1 = time.perf_counter()
        self.log(rec, self._log_host.tolist())
        self._log_wait = t1 - t0
        self._log_emit = time.perf_counter() - t1

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
        batch_efficiency = (h[9] / self.data_args.microbatch_tokens) * 100
        grad_norm = h[10]

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
                f"time {rec['train_step_delta']:.3f}s "
                f"fwd {rec['fwd_bwd_time']:.3f}s "
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
            "perf/step_time": rec["train_step_delta"],
            "perf/fwd_bwd_time": rec["fwd_bwd_time"],
            "perf/tflops_per_second": tflops_per_sec,
            "perf/mfu": mfu,
            "perf/log_wait_ms": rec["log_wait"] * 1e3,
            "perf/log_emit_ms": rec["log_emit"] * 1e3,
            "perf/mem_gib": peak_alloc,
            "perf/mem_reserved_gib": peak_resv,
        }

        if rec["sections"]:
            # forward only, and only under QWEN_SECTION_TIMING=1
            total = sum(rec["sections"].values())
            parts = "  ".join(
                f"{k} {v:.1f}ms {100*v/total:.0f}%"
                for k, v in sorted(rec["sections"].items(), key=lambda kv: -kv[1])
            )
            logger.info(f"  fwd sections ({total:.1f}ms): {parts}")
            log_metrics.update(
                {f"perf_fwd/{k}_ms": v for k, v in rec["sections"].items()}
            )

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

        if rec["gathered"] is not None:
            log_metrics.update(topk_metrics(rec["gathered"], self.wandb_args.top_k))

        wandb.log(log_metrics, step=rec["step"])
