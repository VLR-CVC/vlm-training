import re
import os
import sys
import torch
import wandb
import transformers
from itertools import cycle
from pathlib import Path

import time

from transformers import AutoProcessor

import torch.distributed as dist
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable.replicate import replicate

from torch.profiler import profile, record_function, ProfilerActivity, schedule

# data imports
from megatron.energon import get_train_dataset, get_loader, WorkerConfig
from data.task_encoder_factory import build_task_encoder

from models.qwen3_5.utils import causal_lm_loss, load_stage_weights

# training imports
from train.config_manager import ConfigManager
from train.config import Config, ModelType
from train.logger import init_logger, logger, Color
from train.infra import (
    get_mesh,
    get_mesh_group,

    apply_fsdp,
    apply_tp,
    apply_ep,
    apply_ac,
    apply_pp,

    ACConfig,
    compile_model,
)
from models.qwen3_5.model import initialize_missing_weights

from train.utils import (
    set_determinism,
    generate_accumulation_pattern,
    get_scheduler,

    init_qwen35,
    init_qwen3vl,

    dist_mean,
    dist_max,
    dist_sum,

    get_dense_model_nparams_and_flops,

    select_text_model,
    select_model_class,
    set_model,
    load_text_model,
)

torch._logging.set_logs(graph_code=True)
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, cfg: Config):
        self.model_args = cfg.model
        self.training_args = cfg.training
        self.data_args = cfg.data
        self.p_args = cfg.parallel
        self.wandb_args = cfg.wandb
        self.debug_mode = bool(os.environ.get("DEBUG", False))

        torch.distributed.init_process_group(backend='nccl')
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)

        self.mesh = get_mesh(self.p_args, self.world_size)

        self.tp_group = get_mesh_group(self.mesh, 'tp')
        self.pp_group = get_mesh_group(self.mesh, 'pp')
        self.ep_group = get_mesh_group(self.mesh, 'ep')
        self.shard_group = get_mesh_group(self.mesh, 'dp_shard')
        self.replicate_group = get_mesh_group(self.mesh, 'dp_replicate')
        # this mesh group unifies `shard` and `replicate`
        self.dp_group = get_mesh_group(self.mesh, "dp")

        self.pp_size  = getattr(self.p_args, "pp_size", 1)
        self.ep_size  = getattr(self.p_args, "ep_size", 1)

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
                    **vars(self.p_args),
                    "mesh": self.mesh,
                    "world_size": self.world_size,
                    # TODO: add all parallel args here
                },
            )

            logger.info('using directory:')
            logger.info(os.getcwd())
            logger.info(f"world_size: {self.world_size}")
            logger.info("starting finetune job")
            logger.info(f"mesh: {self.mesh}")

            logger.info(self.model_args)
            logger.info(self.training_args)
            logger.info(self.data_args)

        set_determinism(seed=42 + self.local_rank, deterministic=True, world_mesh=self.mesh, debug_mode=self.debug_mode)

        self.setup_accumulation(self.training_args.tpi_multiplier)

        if self.rank() == 0:
            if not os.path.exists(self.training_args.output_dir):
                os.makedirs(self.training_args.output_dir)

        if "Qwen3.5" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_5
        elif "Qwen3-VL" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_vl
        elif "Qwen3" in self.model_args.model_name:
            self.model_type = ModelType.Qwen3_text
        else:
            raise NotImplementedError(f"model not supported: {self.model_args.model_name}")

        # Load the model on CPU with weights; for PP the split stage is moved to GPU below.
        self.model = select_model_class(
            self.model_type, self.model_args, self.training_args
        )

        # we calculate the flops per token used to get the MFU number
        # (works on meta tensors: shapes are valid even without data)
        num_params, self.flops_per_token = get_dense_model_nparams_and_flops(
            self.model_args.model_name,
            self.model,
            seq_len=int(self.data_args.seq_len),
        )

        logger.info(f"Number params: {num_params}")

        self.optimizer = None  # defined later on

        # -- PIPELINE PARALLEL
        if self.p_args.pp_size > 1:

            def pp_loss_fn(outputs, labels):
                logits, aux_loss = outputs
                ce_loss = causal_lm_loss(logits, labels)
                aux_loss = aux_loss.squeeze()

                self._recent_ce_loss = ce_loss.detach()
                self._recent_aux_loss = aux_loss.detach()

                return (ce_loss + 0.01 * aux_loss) / self.current_accum_target

            self.pp_microbatches, self.pp_schedule = apply_pp(
                self.model, self.mesh, self.p_args, self.training_args, self.device, pp_loss_fn
            )
            self.pp_has_first_stage = self.model.model.visual is not None
            self.pp_has_last_stage = self.model.lm_head is not None

        logger.info("model loaded")

        # -- WEIGHT INIT
        if self.training_args.random_init:
            if self.model_type == ModelType.Qwen3_5:
                logger.info('initilizing decoder and projecter of Qwen3.5')
                #init_qwen35(self.model)
            elif self.model_type == ModelType.Qwen3_vl:
                logger.info('initilizing projector of Qwen3-VL')
                init_qwen3vl(self.model)
            else:
                logger.info('model not initlized, incompatible')
            initialize_missing_weights(self.model)

        # -- MIXED PRECISION
        self.model.train()
        if self.training_args.bfloat16:
            self.model = self.model.to(torch.bfloat16)

        # -- TENSOR PARALLEL
        if self.p_args.tp_size > 1:
            apply_tp(self.model, self.model_type, self.tp_group, self.p_args.async_tp)

        # -- ACTIVATION CHECKPOINTING
        ac_mode = getattr(self.training_args, "ac_mode", "off")
        if ac_mode != "off":
            ac_cfg = ACConfig(enabled=True, full=(ac_mode == "full"))
            apply_ac(
                self.model.model.language_model,
                ac_cfg,
                model_compile_enabled=self.training_args.compile,
            )
            logger.info(f"activation checkpointing applied ({ac_mode})")

        # -- EXPERT PARALLEL
        if self.ep_size > 1:
            if self.model_type != ModelType.Qwen3_5:
                raise NotImplementedError("EP is only supported for Qwen3.5 MoE models")
            tp_mesh = self.tp_group if self.p_args.tp_size > 1 else None
            apply_ep(self.model, self.ep_group, tp_mesh=tp_mesh)
            logger.info(f"expert parallelism applied (ep_size={self.ep_size})")

        # -- DATA PARALLEL
        dp_shard_mesh = get_mesh_group(self.mesh, "dp_shard")
        dp_replicate_mesh = get_mesh_group(self.mesh, "dp_replicate")

        if dp_shard_mesh is not None and dp_shard_mesh.size() > 1:
            apply_fsdp(self.model_type, self.model, mesh=dp_shard_mesh)
            logger.info(f"FSDP applied (dp_shard={dp_shard_mesh.size()})")
        elif dp_replicate_mesh is not None and dp_replicate_mesh.size() > 1:
            self.model = replicate(self.model, device_mesh=dp_replicate_mesh)
            logger.info(f"DDP applied (dp_replicate={dp_replicate_mesh.size()})")
        else:
            logger.info(f"no DP applied (dp=1)")

        # loading into GPU
        self.model = self.model.to(device=self.device)
        if self.training_args.bfloat16:
            self.model = self.model.to(torch.bfloat16)

        logger.info('sharding/parallelism applied')

        if self.training_args.compile and self.pp_size == 1:
            compile_model(self.model)
            logger.info("model (will be) compiled")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.training_args.model_dir,
            model_max_length=int(self.data_args.seq_len),
            padding_side="right",
            use_fast=False,
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        self.processor = AutoProcessor.from_pretrained(
            self.training_args.model_dir,
        )

        # set_model freezes/unfreezes param groups; skip for PP (stage module
        # doesn't have the full VLM wrapper structure)
        if self.pp_size == 1:
            self.model = set_model(self.model_type, self.model_args, self.model)

        # get rank of local GPU that belongs to the DP group
        data_rank = self.dp_group.get_local_rank()
        data_world_size = self.dp_group.size()

        worker_config = WorkerConfig(
            rank=data_rank,
            world_size=data_world_size,
            data_parallel_group=self.dp_group,
            num_workers=1,
        )

        task_encoder, extra_ds_kwargs = build_task_encoder(
            self.data_args,
            tokenizer=self.tokenizer,
            processor=self.processor,
        )
        ds = get_train_dataset(
            self.data_args.data_path,
            batch_size=1,
            shuffle_buffer_size=self.data_args.shuffle_buffer_size,
            max_samples_per_sequence=self.data_args.max_samples_per_sequence,
            task_encoder=task_encoder,
            worker_config=worker_config,
            **extra_ds_kwargs,
        )

        self.data_loader = get_loader(ds)

        self.global_step = 0
        self.micro_step = 0

        self.tokens_seen = 0
        self.tokens_seen_assistant = 0

        self.ntokens_since_last_log = 0
        self.total_ntokens_since_last_log = 0
        self.samples_since_last_log = 0

        self.time_last_log = time.perf_counter()
        self.color = Color()

    def rank(self):
        return torch.distributed.get_rank()

    def if_log_rank(self):
        # Log only from global rank 0 (always pp_rank=0 and dp_rank=0)
        return self.rank() == 0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lr_mlp = self.training_args.lr_mlp
        lr_vit = self.training_args.lr_vit
        lr_llm = self.training_args.lr_llm

        mlp_params = []
        vision_params = []
        llm_params = []

        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "visual.merger" in n:
                mlp_params.append(p)
            elif "visual.deepstack_merger_list":
                mlp_params.append(p)
            elif "visual.patch_embed" in n:
                vision_params.append(p)
            elif "visual.blocks" in n:
                vision_params.append(p)
            else:
                llm_params.append(p)

        optimizer_grouped_parameters = [
            {
                "params": mlp_params,
                "lr": lr_mlp,
            },
            {
                "params": vision_params,
                "lr": lr_vit,
            },
            {
                "params": llm_params,
                "lr": lr_llm,
            },
        ]

        # TODO: add weight decay exclusion for bias and LayerNorm
        #no_decay = ["bias", "LayerNorm.weight"]

        # the "global learning rate" is the LLM learning rate
        self.optimizer = torch.optim.AdamW(
            optimizer_grouped_parameters,
            lr=lr_llm,
            foreach=False,
            weight_decay=self.training_args.weight_decay,
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            self.training_args
        )
        return self.optimizer, self.scheduler

    def save_checkpoint(self):
        step = self.global_step

        checkpoint_dir = os.path.join(
            self.training_args.output_dir,
            f"checkpoint-step-{step}",
        )

        state_dict = {
            "model": self.model,
            "step": step,
            "tokens_seen": self.tokens_seen,
            "tokens_seen_assistant": self.tokens_seen_assistant,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        try:
            logger.info(f"checkpointing at {checkpoint_dir}")
            torch.distributed.checkpoint.save(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir,
            )
        except Exception as e:
            logger.info(f"rank: {self.rank()}")
            logger.info(f"exception during checkpointing: {e}")
        else:
            if self.if_log_rank():
                logger.info(f"checkpoint at step {step} saved.")

    def load_checkpoint(self, step_num):
        checkpoint_dir = os.path.join(
            self.training_args.output_dir,
            f"checkpoint-step-{step_num}",
        )

        state_dict = {
            "model": self.model,
            "step": step_num,
            "tokens_seen": None,
            "tokens_seen_assistant": None,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        # we syncronize all of the processes
        torch.distributed.barrier()

        try:
            logger.info(f"checkpointing at {checkpoint_dir}")
            torch.distributed.checkpoint.load(
                state_dict=state_dict,
                checkpoint_id=checkpoint_dir,
            )
        except Exception as e:
            logger.info(f"rank: {self.rank()}")
            logger.info(f"exception during checkpointing: {e}")
        else:
            self.tokens_seen = state_dict['tokens_seen']
            self.tokens_seen_assistant = state_dict['tokens_seen_assistant']
            self.global_step = state_dict['step']
            self.optimizer = state_dict['optimizer']
            self.scheduler = state_dict['scheduler']

            if self.if_log_rank():
                logger.info(f"{self.color.red}load checkpoint at step {self.global_step}{self.color.reset}")
            return self.optimizer, self.scheduler

    def may_save(self):
        if self.global_step % self.training_args.save_steps == 0:
            return True
        return False

    def batch_generator(self):
        data_iter = iter(self.data_loader)

        while True:
            data_start_time = time.perf_counter()
            batch = next(data_iter)

            if batch['cu_seqlens'].ndim > 1:
                batch['cu_seqlens'].squeeze_()

            if batch['image_grid_thw'].ndim > 1:
                # do not use squeeze because we need to have two dims
                batch['image_grid_thw'] = batch['image_grid_thw'][0]

            batch['attention_mask'], batch['original_mask'] = batch['cu_seqlens'], batch['attention_mask']

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            # the first and last numbers in cu_seqlens do not count towards the sample count
            # (pun intented)
            batch_samples = batch['attention_mask'].shape[0] - 2
            
            ntokens_batch = (batch['input_ids'] != self.pad_token_id).sum().item()
            ntokens_batch_assistant = (batch['labels'] != -100).sum().item()

            self.batch_efficiency = (ntokens_batch / self.data_args.seq_len ) * 100
            self.tokens_seen_assistant += ntokens_batch_assistant
            self.tokens_seen += ntokens_batch
            self.ntokens_since_last_log += ntokens_batch
            self.total_ntokens_since_last_log += self.data_args.seq_len
            self.samples_since_last_log += batch_samples

            self.data_time_delta = time.perf_counter() - data_start_time

            yield batch

    def log(self, avg_loss, aux_loss, max_loss, global_tokens, global_assistant_tokens, global_samples, lr):

        time_delta = time.perf_counter() - self.time_last_log

        tps = self.ntokens_since_last_log / time_delta

        step_flops = self.flops_per_token * self.total_ntokens_since_last_log
        flops_per_sec = step_flops / time_delta
        tflops_per_sec = flops_per_sec / 1e12

        # GB200 (JUP) and SXM H100 (MN5)
        peak_tflops_per_gpu = 989.4

        # BLACKWELL 6000
        peak_tflops_per_gpu = 504

        # L40S
        #peak_tflops_per_gpu = 362

        mfu = (flops_per_sec / (peak_tflops_per_gpu * 1e12)) * 100

        color = self.color

        data_time_pct = (self.data_time_delta / time_delta) * 100

        logger.info(
            f"{color.red}step {self.global_step} "
                f"{color.green}loss {avg_loss:.4f} "
                f"{color.green}aux {aux_loss:.4f} "
                f"{color.blue}tps {tps:.2f} "
                f"{color.magenta}mfu {mfu:.1f}% "
                f"{color.reset}"
                f"time {self.train_step_delta:.3f}s "
                f"fwd {self.fwd_bwd_time:.3f}s "
                f"data_pct {data_time_pct:.2f}% "
                f"nsamples {global_samples} "
                f"batch_util {self.batch_efficiency:.1f}% "
        )

        log_metrics = {
            "train/loss": avg_loss,
            "train/max_loss": max_loss,
            "train/tokens_seen": global_tokens,
            "train/assistant_tokens_seen": global_assistant_tokens,
            "train/num_samples": global_samples,
            "train/lr": lr,
            "train/batch_efficiency": self.batch_efficiency,

            # performance related
            "perf/tokens_per_second": tps,
            "perf/data_time_pct": data_time_pct,
            "perf/step_time": self.train_step_delta,
            "perf/fwd_bwd_time": self.fwd_bwd_time,
            "perf/tflops_per_second": tflops_per_sec,
            "perf/mfu": mfu,
        }

        wandb.log(log_metrics, step=self.global_step)

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def _train_step_pp(self, data_iterator, optimizer):
        batch = next(data_iterator)

        labels = batch.pop('labels', None)
        input_ids = batch.pop('input_ids')
        batch['input_ids'] = input_ids

        # Schedule chunks the positional input_ids and target along dim 0;
        # tile the (1, total) packed sample to (N, total) so n_microbatches > 1
        # produces N actual chunks. Each microbatch is identical content — fine
        # for benchmarking the schedule, not for real training.
        n = self.pp_microbatches
        tiled_input_ids = input_ids.repeat(n, 1) if n > 1 else input_ids
        tiled_labels = labels.repeat(n, 1) if (n > 1 and labels is not None) else labels

        losses = [] if self.pp_has_last_stage else None
        target = tiled_labels if self.pp_has_last_stage else None

        s_model = time.perf_counter()
        with record_function("pp_forward_backward"):
            with torch.autocast('cuda', torch.bfloat16):
                if self.pp_has_first_stage:
                    self.pp_schedule.step(tiled_input_ids, **batch, target=target, losses=losses)
                else:
                    self.pp_schedule.step(**batch, target=target, losses=losses)

        if self.ep_size > 1 and self.dp_group.size() > 1:
            is_last_accum = (self.current_accum_count + 1 >= self.current_accum_target)
            if is_last_accum:
                # we use a custom bucking system instead of the replicate hooks
                self._sync_gradients()

        self.fwd_bwd_time = time.perf_counter() - s_model

        scaled_loss = torch.stack(losses).sum() if losses else torch.tensor(0.0, device=self.device)
        loss_for_logging = scaled_loss * self.current_accum_target
        torch.distributed.all_reduce(loss_for_logging, group=self.pp_group.get_group())

        # TODO: FIX THIS
        ce_loss = getattr(self, '_recent_ce_loss', torch.tensor(0.0, device=self.device))
        aux_loss = getattr(self, '_recent_aux_loss', torch.tensor(0.0, device=self.device))

        torch.distributed.all_reduce(ce_loss, group=self.pp_group.get_group())
        torch.distributed.all_reduce(aux_loss, group=self.pp_group.get_group())
        
        return self._maybe_optimizer_step(loss_for_logging, ce_loss, aux_loss, optimizer)

    def _sync_gradients(self):
        """Bucketed grad all_reduce across dp_group. One collective per ~25 MB
        bucket (per dtype) instead of one per parameter, so DP scales by NCCL
        bandwidth rather than per-launch latency. Used instead of DDP hooks when
        EP is active.
        """
        dp_size = self.dp_group.size()
        if dp_size <= 1:
            return
        grp = self.dp_group.get_group()

        from torch.distributed.tensor import DTensor
        from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors

        # NCCL all_reduce requires uniform dtype within a call.
        by_dtype: dict[torch.dtype, list[torch.Tensor]] = {}
        for p in self.model.parameters():
            if p.grad is None:
                continue
            g = p.grad
            # TP-sharded params (e.g. shared_expert.*) have DTensor grads; reduce the local shard.
            if isinstance(g, DTensor):
                g = g.to_local()
            by_dtype.setdefault(g.dtype, []).append(g)

        bucket_max_elems = 25 * 1024 * 1024  # ~50 MB at bf16, ~100 MB at fp32
        inv_dp = 1.0 / dp_size

        def _flush(bucket: list[torch.Tensor]) -> None:
            flat = _flatten_dense_tensors(bucket)
            dist.all_reduce(flat, group=grp)
            flat.mul_(inv_dp)
            for g, synced in zip(bucket, _unflatten_dense_tensors(flat, bucket)):
                g.copy_(synced)

        for grads in by_dtype.values():
            bucket: list[torch.Tensor] = []
            bucket_elems = 0
            for g in grads:
                n = g.numel()
                if bucket and bucket_elems + n > bucket_max_elems:
                    _flush(bucket)
                    bucket, bucket_elems = [], 0
                bucket.append(g)
                bucket_elems += n
            if bucket:
                _flush(bucket)

    def _train_step(self, data_iterator, optimizer):
        batch = next(data_iterator)
        input_ids = batch.pop('input_ids')
        labels = batch.pop('labels', None)

        s_model = time.perf_counter()
        with record_function("forward_pass"):
            with torch.autocast('cuda', torch.bfloat16):
                logits, aux_loss = self.model(input_ids, **batch)
                ce_loss = causal_lm_loss(logits, labels)
                loss = ce_loss + (.01 * aux_loss)

        with record_function("backward_pass"):
            scaled_loss = loss / self.current_accum_target
            scaled_loss.backward()

        if self.ep_size > 1 and self.dp_group.size() > 1:
            is_last_accum = (self.current_accum_count + 1 >= self.current_accum_target)
            if is_last_accum:
                self._sync_gradients()

        self.fwd_bwd_time = time.perf_counter() - s_model
        return self._maybe_optimizer_step(loss, ce_loss, aux_loss, optimizer)

    def train_step(self, data_iterator, optimizer):
        if self.pp_size == 1:
            return self._train_step(data_iterator, optimizer)
        else:
            return self._train_step_pp(data_iterator, optimizer)

    def _maybe_optimizer_step(self, loss, ce_loss, aux_loss, optimizer):
        """Shared optimizer-step logic after fwd+bwd (regular and PP paths)."""
        self.current_accum_count += 1

        if self.current_accum_count >= self.current_accum_target:
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

            lr = optimizer.param_groups[0]['lr']
            self.global_step += 1

            avg_loss, aux_loss, max_loss, global_tokens, global_assistant, global_samples = (
                dist_mean(ce_loss, self.dp_group),
                dist_mean(aux_loss, self.dp_group),
                dist_max(ce_loss, self.dp_group),
                dist_sum(
                    torch.tensor(self.tokens_seen, dtype=torch.int64, device=self.device),
                    self.dp_group,
                ),
                dist_sum(
                    torch.tensor(self.tokens_seen_assistant, dtype=torch.int64, device=self.device),
                    self.dp_group,
                ),
                dist_sum(
                    torch.tensor(self.samples_since_last_log, dtype=torch.int32, device=self.device),
                    self.dp_group,
                ),
            )

            self.train_step_delta = (
                (time.perf_counter() - self.time_last_log) / self.current_accum_target
            )

            if self.if_log_rank():
                self.log(avg_loss, aux_loss, max_loss, global_tokens, global_assistant, global_samples, lr)

            self.total_ntokens_since_last_log = 0
            self.ntokens_since_last_log = 0
            self.samples_since_last_log = 0
            self.time_last_log = time.perf_counter()
            self.current_accum_count = 0
            self.current_accum_target = next(self.accum_schedule)

            return True

        return False

    def train(self):
        data_iterator = self.batch_generator()

        optimizer, scheduler = self.create_optimizer()
        if self.training_args.resume_checkpoint:
            paths = os.listdir(self.training_args.output_dir)
            possible_steps = []
            for path in paths:
                match = re.search(r"(\d+\.?\d*)$", path)
                try:
                    if match:
                        step = match.group(1)
                    possible_steps.append(int(step))
                except Exception as e:
                    pass

            if possible_steps:
                largest_step = max(possible_steps)
                optimizer, scheduler = self.load_checkpoint(largest_step)

            else:
                logger.info('could not resume')
                raise Exception("Could not found initial checkpoint, killing run")
        
        # Custom handler for Chrome Trace export instead of TensorBoard
        def trace_handler(prof):
            trace_path = os.path.join(self.training_args.output_dir, f"trace_rank_{self.rank()}_step_{prof.step_num}.json")
            prof.export_chrome_trace(trace_path)
            if self.if_log_rank():
                logger.info(f"Profiler trace saved to: {trace_path}")

        prof_schedule = schedule(wait=5, warmup=2, active=3, repeat=1)

        #with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], schedule=prof_schedule, on_trace_ready=trace_handler, record_shapes=True, profile_memory=True, with_stack=True) as prof:
        try:
            while self.global_step < self.training_args.total_steps:
                self.micro_step += 1

                # training step executed here
                optimizer_updated = self.train_step(data_iterator, optimizer)

                if optimizer_updated:
                    scheduler.step()
                    # Save checkpoint only if we haven't reached the target steps
                    if self.may_save() and self.global_step < self.training_args.total_steps:
                        self.save_checkpoint()
                #prof.step()

        except StopIteration as e:
            if self.if_log_rank():
                logger.info(f"data iterator exhausted at step {self.global_step}: {e}")

        if self.if_log_rank():
            logger.info(f"tokens seen: {self.tokens_seen}")
            logger.info(f"assistant tokens seen: {self.tokens_seen_assistant}")
            logger.info(f"Training completed at step {self.global_step}. Saving final checkpoint...")

        self.save_checkpoint()

        torch.distributed.destroy_process_group()
        exit()

if __name__ == "__main__":
    config_manager = ConfigManager(Config)
    args = sys.argv[1:]
    config = config_manager.parse_args(args)

    init_logger()

    torch.manual_seed(42)

    trainer = Trainer(config)
    trainer.train()
