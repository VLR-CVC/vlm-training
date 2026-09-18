import datetime
import json
import faulthandler
import os
import signal
import sys
import torch
import wandb
import transformers
from itertools import cycle

import time
from pathlib import Path

from transformers import AutoProcessor

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.checkpoint.state_dict import _init_optim_state

from torch.profiler import record_function

# data imports
from megatron.energon import get_train_dataset, get_loader, get_savable_loader, WorkerConfig
from data.task_encoder_factory import build_task_encoder

# training imports
from train.config_manager import ConfigManager
from train.config import Config
from train.parallel.parallel_dims import ParallelDims
from train.logger import init_logger, logger, Color

_LOG_SLOTS = 11
from train.utils import (
    build_adamw,
    dist_max,
    MASTER_DTYPES,
    clip_grad_norm_mixed,
    cast_master_weights,
    set_determinism,
    generate_accumulation_pattern,
    get_scheduler,
    dist_sum,
    dist_all_gather,
    set_trainable_parts,
    build_optimizer_param_groups,
    topk_metrics,
)
from train.checkpoint import (
    save_distributed_checkpoint,
    load_distributed_checkpoint,
    find_latest_checkpoint_step,
    save_dataloader_state,
    load_dataloader_state,
)
from train.training_debug import (
    dump_cprofile,
    build_debug_profiler,
)
from train.flops_estimation import build_flops_model

torch._inductor.config.fx_graph_cache = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

def _apply_chat_template(processor, path: str) -> None:
    """Override the processor's chat template, and warn about the Qwen default."""
    if path and path != "NULL":
        with open(path) as f:
            processor.chat_template = f.read()
        logger.info(f"chat template overridden from {path}")
        return

    # The comparison, not the bare name: `ns.last_query_index` is still computed
    # in the fixed template (it is used for tool-call handling), so matching the
    # variable alone warns on a template that is already correct.
    template = getattr(processor, "chat_template", None) or ""
    if "> ns.last_query_index" in template:
        c = Color()
        logger.warning(
            f"{c.red}chat template in {processor.__class__.__name__} drops "
            f"<think> spans from all but the last turn. Fine for inference, "
            f"wrong for SFT on reasoning data. Set "
            f"training.chat_template=assets/chat_template_sft.jinja if this "
            f"dataset has reasoning.{c.reset}"
        )

class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, cfg: Config):
        self.model_args = cfg.model
        self.training_args = cfg.training
        self.data_args = cfg.data
        self.wandb_args = cfg.wandb
        self.debug_mode = bool(os.environ.get("DEBUG", False))

        timeout = datetime.timedelta(seconds=int(os.environ.get("QWEN_NCCL_TIMEOUT_S", 600)))
        torch.distributed.init_process_group(backend='nccl', timeout=timeout)
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)

        # spmd_types searches the TP group on a thread-local mesh
        torch.autograd.set_multithreading_enabled(False)

        tp = self.training_args.tp_size
        replicate = self.training_args.data_parallel == "ddp"
        self.parallel_dims = ParallelDims(
            dp_replicate=self.world_size // tp if replicate else 1,
            dp_shard=1 if replicate else -1,
            cp=1, tp=tp, pp=1, ep=1, world_size=self.world_size,
        )
        self.parallel_dims.build_mesh()
        # data-parallel ("batch") and TP axes; None when the axis has size 1
        self.dp_mesh = self.parallel_dims.get_optional_mesh("batch")
        self.tp_mesh = self.parallel_dims.get_optional_mesh("tp")
        self.dp_size = self.dp_mesh.size() if self.dp_mesh is not None else 1

        self.device = torch.device(f"cuda:{self.local_rank}")
        if self.if_log_rank():
            wandb.init(
                name=self.wandb_args.run_name,
                project=self.wandb_args.project_name,
                entity=self.wandb_args.entity_name,
                config={
                    **vars(self.model_args),
                    **vars(self.training_args),
                    **vars(self.data_args),
                    "world_size": self.world_size,
                    "dp_size": self.dp_size,
                },
            )

            logger.info('using directory:')
            logger.info(os.getcwd())
            logger.info(f"world_size: {self.world_size}")
            logger.info("starting finetune job")
            logger.info(f"dp={self.dp_size} tp={self.training_args.tp_size}")

            logger.info(self.model_args)
            logger.info(self.training_args)
            logger.info(self.data_args)

        set_determinism(seed=42 + self.local_rank, deterministic=self.training_args.deterministic, world_mesh=self.parallel_dims.world_mesh, debug_mode=self.debug_mode)

        if self.rank() == 0:
            if not os.path.exists(self.training_args.output_dir):
                os.makedirs(self.training_args.output_dir)

        self._setup_model()

        # this rank's data shard: its index on the data-parallel axis
        self.data_rank = self.dp_mesh.get_local_rank() if self.dp_mesh is not None else 0
        self.is_data_leader = self.tp_mesh is None or self.tp_mesh.get_local_rank() == 0

        logger.info('sharding/parallelism applied')

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.training_args.model_dir,
            model_max_length=int(self.data_args.seq_len),
            padding_side="right",
            use_fast=False,
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        self.processor = AutoProcessor.from_pretrained(
            self.training_args.model_dir,
            max_pixels=1048576,
        )
        _apply_chat_template(self.processor, self.training_args.chat_template)

        worker_config = WorkerConfig(
            rank=self.data_rank,
            world_size=self.dp_size,
            data_parallel_group=self.dp_mesh.get_group() if self.dp_mesh is not None else None,
            num_workers=2,
        )

        task_encoder, extra_ds_kwargs = build_task_encoder(
            self.data_args,
            self.processor,
            image_token_id=self.image_token_id,
            video_token_id=self.video_token_id,
            spatial_merge_size=self.spatial_merge_size,
        )
        ds = get_train_dataset(
            self.data_args.data_path,
            batch_size=1,
            repeat=self.data_args.repeat,
            shuffle_buffer_size=self.data_args.shuffle_buffer_size,
            max_samples_per_sequence=self.data_args.max_samples_per_sequence,
            task_encoder=task_encoder,
            worker_config=worker_config,
            **extra_ds_kwargs,
        )

        # creation of dataloader
        if self.data_args.save_dataloader_state:
            self.data_loader = get_savable_loader(ds)
        else:
            self.data_loader = get_loader(ds)

        self.setup_accumulation(self.training_args.tpi_multiplier)

        self.global_step = 0
        self.micro_step = 0

        self.tokens_seen = torch.zeros((), dtype=torch.int64, device=self.device)
        self.tokens_seen_assistant = torch.zeros((), dtype=torch.int64, device=self.device)
        self.ntokens_since_last_log = torch.zeros((), dtype=torch.int64, device=self.device)
        self.ntokens_last_batch = torch.zeros((), dtype=torch.int64, device=self.device)
        self.flops_since_last_log = torch.zeros((), dtype=torch.float64, device=self.device)
        self.grad_norm = torch.zeros((), dtype=torch.float32, device=self.device)

        # host-side already: it comes from a tensor *shape*, not device memory
        self.samples_since_last_log = 0

        # Deferred logging -- see `_stage_log` / `_flush_log`.
        self._log_host = torch.zeros(_LOG_SLOTS, dtype=torch.float64).pin_memory()
        self._log_event = torch.cuda.Event()
        self._log_pending = None
        self._log_wait = 0.0   # host wait for the staged copy; should stay ~0
        self._log_emit = 0.0   # formatting + wandb.log
        self._flag_stream = None

        self.time_last_log = time.perf_counter()
        self.color = Color()

    def _setup_model(self):
        """Qwen3.5 via `models/qwen3_5_tt`, Qwen3-VL via `models/qwen3_vl_tt`
        (TITAN_MIGRATION_v2.md).

        torchtitan's order: build on meta -> freeze -> cast master dtype ->
        per-block compile -> FSDP/replicate -> `to_empty` -> load HF (DCP straight
        into the shards) or init. No rank ever holds a materialised full model.
        """
        from models.qwen3_5_tt.checkpoint import build_meta, load_hf, materialize
        from models.qwen3_5_tt.configs import resolve_model_config
        from train.parallel.parallelize import parallelize_qwen3_5

        tp = self.training_args.tp_size

        model_dir = self.training_args.model_dir
        config_path = resolve_model_config(
            self.model_args.model_config, model_dir, self.model_args.use_model_dir_config
        )
        logger.info(f"architecture from {config_path}")
        # the HF-format config: `model_type` picks the model, and the dataloader and
        # the loss need its special-token ids and the vision merge size
        self.hf_config = json.loads(Path(config_path).read_text())
        self.image_token_id = self.hf_config["image_token_id"]
        self.video_token_id = self.hf_config["video_token_id"]
        self.spatial_merge_size = self.hf_config["vision_config"]["spatial_merge_size"]
        seq_len = int(self.data_args.seq_len)
        enable_sp = tp > 1 and self.training_args.sequence_parallel
        if enable_sp and self.data_args.microbatch_tokens % tp:
            raise ValueError(
                f"sequence parallel needs tokens per micro-batch ({self.data_args.microbatch_tokens}) "
                f"divisible by tp ({tp})"
            )
        self.model = build_meta(
            config_path, seq_len=seq_len, tp=tp, enable_sp=enable_sp,
            attn_backend=self.model_args.attn_backend, decoder_mask=self.model_args.decoder_mask,
        )

        # FLOPs come from the batch, not from seq_len: attention is causal per
        # document and the ViT runs per image patch (train/flops_estimation.py).
        num_params = sum(p.numel() for p in self.model.parameters())
        self.flops_model = build_flops_model(self.hf_config)
        # per GPU: a TP group shares one micro-batch, so each rank does 1/tp of it
        self.tp_size = tp
        self.peak_tflops_per_gpu = 989.4
        logger.info(f"Number params: {num_params}")

        set_trainable_parts(self.model_args, self.model)

        master_dtype = cast_master_weights(self.model, self.training_args.master_dtype)
        logger.info(f"master weights in {master_dtype}")

        if self.training_args.dynamo_recompile_limit > 0:
            import torch._dynamo.config as dynamo_config

            dynamo_config.recompile_limit = self.training_args.dynamo_recompile_limit

        # TP (Module.parallelize) -> per-block fullgraph compile -> FSDP on the
        # storage mesh. Compute is bf16 (attn_gym's fused GDN kernel takes fp16/bf16 only).
        parallelize_qwen3_5(
            self.model,
            self.parallel_dims,
            mode=self.training_args.data_parallel,
            compile=self.training_args.compile,
            param_dtype=torch.bfloat16,
            reduce_dtype=MASTER_DTYPES[self.training_args.grad_reduce_dtype],
            reshard_after_forward=self.training_args.reshard_after_forward == "always",
        )
        logger.info(f"tp={tp} sp={enable_sp} compile={self.training_args.compile} "
                    f"data_parallel={self.training_args.data_parallel}")

        materialize(self.model, self.device)
        if self.training_args.random_init:
            with torch.no_grad():
                self.model.init_states(buffer_device=self.device)
            logger.info("random init (init_states)")
        else:
            load_hf(self.model, model_dir)
            logger.info(f"loaded HF weights from {model_dir}")
        self.model.train()
        self.optimizer = None

    def rank(self):
        return torch.distributed.get_rank()

    def if_log_rank(self):
        return self.rank() == 0

    def _all_ranks_have_batch(self, local_has_batch: bool) -> bool:
        """Collective agreement on whether EVERY rank still has data.

        The answer is needed before the step runs, so unlike the logging
        counters this one cannot be deferred. It can be kept off the training
        stream, though: the flag is built from a host bool and has no data
        dependency on anything the model computed. Running it on its own stream
        means the `.item()` waits for a one-int all-reduce instead of for every
        kernel the previous step queued.
        """
        if self._flag_stream is None:
            self._flag_stream = torch.cuda.Stream(device=self.device)
        with torch.cuda.stream(self._flag_stream):
            flag = torch.tensor(
                [1 if local_has_batch else 0], dtype=torch.int32, device=self.device
            )
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
            return bool(flag.item())

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        weight_decay = self.training_args.weight_decay
        lr_by_group = {
            "mlp": self.training_args.lr_mlp,
            "vit": self.training_args.lr_vit,
            "llm": self.training_args.lr_llm,
        }

        optimizer_grouped_parameters = build_optimizer_param_groups(
            self.model.named_parameters(),
            lr_by_group,
            weight_decay,
            log=self.if_log_rank(),
        )

        # different types of AdamW supported
        self.optimizer = build_adamw(
            optimizer_grouped_parameters,
            lr=self.training_args.lr_llm,
            weight_decay=weight_decay,
            impl=self.training_args.adamw_impl,
            stochastic_round=self.training_args.adamw_stochastic_round,
            betas=self.training_args.adam_betas,
            eps=self.training_args.eps,
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            self.training_args
        )
        return self.optimizer, self.scheduler

    def save_checkpoint(self):
        # Hand reserved-but-unallocated blocks back to the driver first.
        # Checkpointing creates new NCCL communicators, and those allocate
        # *outside* the caching allocator. Job 1782009 finished all 60 steps at
        # 10240 with 91.6 GiB reserved and then died in `ncclCuMemAlloc` with
        # "Cuda failure 2 'out of memory'" -- the training loop fit, the
        # communicator did not. This costs a sync and some re-allocation on the
        # next step, once per save.
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        state_dict = {
            "model": self.model,
            "step": self.global_step,
            "tokens_seen": int(self.tokens_seen),
            "tokens_seen_assistant": int(self.tokens_seen_assistant),
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }
        save_distributed_checkpoint(
            self.training_args.output_dir,
            self.global_step,
            state_dict,
            self.rank(),
            self.if_log_rank(),
        )

        if self.data_args.save_dataloader_state:
            save_dataloader_state(
                self.training_args.output_dir,
                self.global_step,
                self.data_loader,
                self.data_rank,
                self.is_data_leader,
                self.if_log_rank(),
            )

    def load_checkpoint(self, step_num, ckpt_dir=None):
        # where to read the resume checkpoint from; defaults to output_dir
        ckpt_dir = ckpt_dir or self.training_args.output_dir

        # init AdamW state by calling step() with zero grads
        _init_optim_state(self.optimizer)

        state_dict = {
            "model": self.model,
            "step": step_num,
            "tokens_seen": None,
            "tokens_seen_assistant": None,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        loaded = load_distributed_checkpoint(
            ckpt_dir, step_num, state_dict, self.rank()
        )
        if loaded is None:
            return

        self.tokens_seen.fill_(loaded['tokens_seen'])
        self.tokens_seen_assistant.fill_(loaded['tokens_seen_assistant'])
        self.global_step = loaded['step']
        self.optimizer = loaded['optimizer']
        self.scheduler = loaded['scheduler']

        if self.data_args.save_dataloader_state:
            if self.data_args.restore_dataloader_state:
                load_dataloader_state(
                    ckpt_dir,
                    step_num,
                    self.data_loader,
                    self.data_rank,
                )
            elif self.if_log_rank():
                logger.info("restore_dataloader_state=false; data stream starts from scratch")

        if self.if_log_rank():
            logger.info(f"{self.color.red}load checkpoint at step {self.global_step}{self.color.reset}")
        return self.optimizer, self.scheduler

    def may_save(self):
        if self.global_step % self.training_args.save_steps == 0:
            return True
        return False

    def batch_generator(self):
        """Packed micro-batches (`data/energon_dataloader.py:PackedBatchEncoder`) to the
        device, plus the logging counters"""
        data_iter = iter(self.data_loader)

        while True:
            data_start_time = time.perf_counter()
            try:
                batch = next(data_iter)
            except StopIteration:
                return

            ntokens = batch.pop("num_tokens")
            self.samples_since_last_log += batch.pop("num_samples")
            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device=self.device, non_blocking=True)
            if "pixel_values" in batch:
                batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)

            self.ntokens_last_batch.fill_(ntokens)
            self.tokens_seen_assistant.add_(batch["num_valid_tokens"])
            self.tokens_seen.add_(ntokens)
            self.ntokens_since_last_log.add_(ntokens)
            self.flops_since_last_log.add_(
                self.flops_model.batch_flops(batch["positions"], batch.get("grid_thw"))
                / self.tp_size
            )
            self.data_time_delta = time.perf_counter() - data_start_time

            yield batch

    def train_step(self, batches, optimizer):
        """One optimizer step over `batches` (the accumulation window), torchtitan's
        way (`trainer.py:872`): count valid tokens across every micro-batch and every
        DP rank first, then give each token the weight 1/global_tokens."""
        from train.step import forward_backward

        s_model = time.perf_counter()
        accumulated, local_tokens, denom = forward_backward(
            self.model,
            batches,
            dp_group=self.dp_mesh,
            special_tokens={"image_id": self.image_token_id},
            ddp=self.training_args.data_parallel == "ddp",
            loss_chunks=self.training_args.loss_chunks,
            compile_loss=self.training_args.compile,
            parallel_dims=self.parallel_dims,
        )
        self.fwd_bwd_time = time.perf_counter() - s_model

        with record_function("optimizer_step"):
            params = [p for p in self.model.parameters() if p.grad is not None]
            if self.training_args.max_grad_norm > 0:
                self.grad_norm = clip_grad_norm_mixed(params, self.training_args.max_grad_norm)
            else:
                self.grad_norm = torch.nn.utils.get_total_norm([p.grad for p in params])
            gn = self.grad_norm.full_tensor() if hasattr(self.grad_norm, "full_tensor") else self.grad_norm
            # torchtitan's check: stop before the optimizer can write a non-finite
            # update, without a host sync (TITAN_MIGRATION_v2.md D5)
            torch._assert_async(
                torch.isfinite(accumulated) & torch.isfinite(gn),
                "loss or gradient norm is not finite; stopping before the optimizer step",
            )
            optimizer.step()
            optimizer.zero_grad()

        n = self.training_args.clear_cache_vram
        if n > 0 and self.global_step % n == 0:
            torch.cuda.synchronize()
            torch.cuda.empty_cache()

        lr = optimizer.param_groups[0]['lr']
        self.global_step += 1
        time_delta = time.perf_counter() - self.time_last_log
        self.train_step_delta = time_delta / len(batches)

        gathered = None
        topk_interval = max(1, self.wandb_args.topk_interval)
        if self.wandb_args.log_topk and self.global_step % topk_interval == 0:
            gathered = self._gather_perf(time_delta)

        # `accumulated` is local_sum / global_tokens, so the SUM over DP is the exact
        # global mean. `_stage_log` divides its summed slot by dp_size, hence * dp.
        # The max slot gets this rank's own per-token mean.
        local_mean = accumulated * denom / max(local_tokens, 1)
        self._flush_log()
        self._stage_log(accumulated * self.dp_size, lr, time_delta, gathered, loss_max=local_mean)

        self.ntokens_since_last_log.zero_()
        self.flops_since_last_log.zero_()
        self.samples_since_last_log = 0
        torch.cuda.reset_peak_memory_stats(self.device)
        self.time_last_log = time.perf_counter()
        self.current_accum_count = 0
        self.current_accum_target = next(self.accum_schedule)

    @staticmethod
    def perf_rank_share(ntokens, flops, *, tp, time_delta, peak_tflops):
        """This rank's SHARE of the job's rates, so that the gathered rows are a
        decomposition of the headline rather than a second opinion about it.

        A TP group works on one micro-batch, so each of its ranks is credited
        1/tp of its tokens; `flops` is already per-rank (divided by tp when it was
        accumulated). Summing either column over WORLD gives the job total, so the
        mean over WORLD is the per-GPU headline `log` reports. Shared by both
        paths on purpose -- they diverged once, and nothing caught it.
        """
        tps = ntokens / tp / time_delta
        tflops = flops / time_delta / 1e12
        return tps, tflops, (tflops / peak_tflops) * 100

    def _gather_perf(self, time_delta):
        """Per-rank perf row, gathered across the world.

        Assembled on device. `ntokens_since_last_log` is a device tensor now, so
        building this row on the host would reintroduce the sync that
        `_stage_log` exists to avoid. The returned tensor is read on the host by
        `topk_metrics`, one step later, on the log rank only.
        """
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
        """Reduce this step's counters and start a non-blocking copy of the
        result into pinned memory. Touches the host for nothing that came off
        the device; `_flush_log` reads the values on the next step, by which
        time the copy has long landed.

        Both collectives here are unavoidable -- the loss and the token counts
        are genuinely per-rank. What was avoidable was reading them.
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
        }

    def _flush_log(self):
        """Emit the previous step's record. The copy it reads was queued a full
        step ago, so the event wait below is already satisfied and returns
        without stalling."""
        rec = self._log_pending
        if rec is None:
            return
        self._log_pending = None

        # Two numbers, both reported on the *next* record. `log_wait` is the
        # host waiting on a copy that was queued a full step ago -- if it is
        # not ~0, the deferral is not buying anything and something upstream
        # is still draining the stream. `log_emit` is the part people assume
        # is expensive: string formatting and `wandb.log`. Measure before
        # trimming what gets printed.
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

        if rec["gathered"] is not None:
            log_metrics.update(topk_metrics(rec["gathered"], self.wandb_args.top_k))

        wandb.log(log_metrics, step=rec["step"])

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def train(self):
        data_iterator = self.batch_generator()

        optimizer, scheduler = self.create_optimizer()
        if self.training_args.resume_checkpoint:
            load_dir = self.training_args.load_dir
            if load_dir in ("NULL", "", None):
                load_dir = self.training_args.output_dir
            resume_step = self.training_args.start_step
            if resume_step <= 0:
                resume_step = find_latest_checkpoint_step(load_dir)
            if resume_step is None:
                logger.info('could not resume')
                raise Exception("Could not found initial checkpoint, killing run")
            optimizer, scheduler = self.load_checkpoint(resume_step, load_dir)

        prof_ctx, _cprof, _CPROF_START, _CPROF_STOP = build_debug_profiler(
            self.debug_mode, self.training_args.output_dir, self.rank(), self.if_log_rank()
        )
        profile_dir = os.environ.get("PROFILE_DIR")
        if profile_dir and not self.debug_mode:
            # Kernel-level trace without DEBUG=1, which also changes determinism
            # settings and so the run being measured. Steps are optimizer steps
            # (`prof.step()` below runs once per loop iteration; the loop does
            # a whole accumulation window per iteration).
            from torch.profiler import ProfilerActivity, profile, schedule
            from train.training_debug import make_trace_handler

            os.makedirs(profile_dir, exist_ok=True)
            wait = int(os.environ.get("PROFILE_WAIT", "40"))
            prof_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=schedule(wait=wait, warmup=3, active=2, repeat=1),
                on_trace_ready=make_trace_handler(profile_dir, self.rank(), self.if_log_rank()),
            )
        _cprof_active = False

        with prof_ctx as prof:
            
            # training loop
            while self.global_step < self.training_args.total_steps:
                self.micro_step += 1

                if self.debug_mode and not _cprof_active and self.global_step >= _CPROF_START:
                    _cprof.enable()
                    _cprof_active = True

                try:
                    # retrieve the batch
                    batch = next(data_iterator)
                    local_has_batch = True
                except StopIteration:
                    batch = None
                    local_has_batch = False

                # check for data exhaustion
                if not self.data_args.repeat:
                    if not self._all_ranks_have_batch(local_has_batch):
                        if self.if_log_rank():
                            logger.info(f"data exhausted on at least one rank at step {self.global_step}; stopping")
                        break
                elif not local_has_batch:
                    break

                # TRAINING STEP
                batches = [batch]
                have_window = True
                for _ in range(self.current_accum_target - 1):
                    try:
                        batches.append(next(data_iterator))
                    except StopIteration:
                        have_window = False
                        break

                if not self.data_args.repeat:
                    if not self._all_ranks_have_batch(have_window):
                        if self.if_log_rank():
                            logger.info(f"data exhausted mid-window on at least one rank at step {self.global_step}; stopping")
                        break
                elif not have_window:
                    break
                self.train_step(batches, optimizer)
                optimizer_updated = True

                if prof is not None:
                    prof.step()

                # this is a flag because of grad_accm
                if optimizer_updated:
                    scheduler.step()

                    if _cprof_active and self.global_step >= _CPROF_STOP:
                        _cprof.disable()
                        _cprof_active = False
                        dump_cprofile(_cprof, self.training_args.output_dir, self.rank())

                    if self.may_save() and self.global_step < self.training_args.total_steps:
                        self.save_checkpoint()

        self.end_run()

    def end_run(self):
        self._flush_log()  # the last step staged a record nobody has emitted yet
        if self.if_log_rank():
            logger.info(f"finalizing run at step {self.global_step}")
            logger.info(f"tokens seen: {int(self.tokens_seen)}")
            logger.info(f"assistant tokens seen: {int(self.tokens_seen_assistant)}")
            logger.info("saving final checkpoint...")

        self.save_checkpoint()

        # make sure every rank has finished writing before tearing down NCCL
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # `kill -USR1 <pid>` prints every thread's Python stack to stderr: how to see where
    # a hung rank is without py-spy
    faulthandler.register(signal.SIGUSR1, all_threads=True)
    # patch how error are reported
    #real_stdout = redirect_rank_io()

    config_manager = ConfigManager(Config)
    args = sys.argv[1:]
    config = config_manager.parse_args(args)

    #init_logger(stream=real_stdout)
    init_logger()

    torch.manual_seed(42)

    trainer = Trainer(config)
    trainer.train()
