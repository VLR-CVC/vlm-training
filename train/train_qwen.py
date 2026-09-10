import os
import sys
import torch
import wandb
import transformers
from itertools import cycle

import time

from transformers import AutoProcessor

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable.replicate import replicate

from torch.profiler import record_function

# data imports
from megatron.energon import get_train_dataset, get_loader, get_savable_loader, WorkerConfig
from data.task_encoder_factory import build_task_encoder

# training imports
from train.config_manager import ConfigManager
from train.config import Config, ModelType
from train.logger import init_logger, redirect_rank_io, logger, Color
from train.infra import (
    get_mesh,
    get_tp_group,
    get_dp_group,
    apply_fsdp,
    apply_tp,
    compile_model,
)
from train.utils import (
    build_adamw,
    clip_grad_norm_mixed,
    cast_master_weights,
    set_determinism,
    generate_accumulation_pattern,
    get_scheduler,

    init_qwen35,
    init_qwen3vl,

    dist_mean,
    dist_max,
    dist_sum,
    dist_all_gather,

    select_text_model,
    select_vision_model,
    select_model_class,
    set_model,
    load_text_model,
    load_vision_model,

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
    write_batch_stats,
    dump_cprofile,
    build_debug_profiler,
)
from train.flops_estimation import get_dense_model_nparams_and_flops

torch._logging.set_logs(graph_code=True)
torch._inductor.config.fx_graph_cache = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, cfg: Config):
        self.model_args = cfg.model
        self.training_args = cfg.training
        self.data_args = cfg.data
        self.wandb_args = cfg.wandb
        self.debug_mode = bool(os.environ.get("DEBUG", False))

        torch.distributed.init_process_group(backend='nccl')
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)

        self.mesh = get_mesh(self.training_args, self.world_size)
        self.tp_group = get_tp_group(self.mesh)
        self.dp_group = get_dp_group(self.mesh)

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
                    "mesh": self.mesh,
                    "world_size": self.world_size,
                    "dp_group": self.dp_group,
                    "tp_group": self.tp_group,
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

        self.model, self.cfg_model = select_model_class(self.model_type, self.model_args, self.training_args)

        # we calculate the flops per token used to get the MFU number
        num_params, self.flops_per_token = get_dense_model_nparams_and_flops(
            self.model_type,
            self.cfg_model,
            self.model,
            seq_len=int(self.data_args.seq_len),
        )

        self.flops_per_token = self.flops_per_token / self.training_args.tp_size

        # peak bf16 TFLOPs per GPU, used for the MFU number
        # SXM H100/GH200 (MN5): 989.4 ; L40S: 362
        self.peak_tflops_per_gpu = 989.4

        logger.info(f"Number params: {num_params}")

        if self.training_args.load_text_model:
            self.text_model = select_text_model(self.training_args)
            self.model = load_text_model(self.model, self.text_model)
            del self.text_model

        if self.training_args.load_vision_model:
            self.vision_model = select_vision_model(self.training_args)
            self.model = load_vision_model(self.model, self.vision_model)
            del self.vision_model

        # MOVE TO cuda:{self.local_rank}
        self.model.to(self.device)
        
        if self.training_args.random_init:
            if self.model_type == ModelType.Qwen3_5:
                logger.info('initilizing decoder and projecter of Qwen3.5')
                init_qwen35(self.model)
            elif self.model_type == ModelType.Qwen3_vl:
                logger.info('initilizing projector of Qwen3-VL')
                init_qwen3vl(self.model)
            else:
                logger.info('model not initlized, incompatible')

        # replace flash_attn
        self.model.train()
        self.optimizer = None # its defined later on

        # we only cast the weights
        master_dtype = cast_master_weights(self.model, self.training_args.master_dtype)
        if master_dtype is torch.bfloat16 and not self.training_args.bf16_compute:
            logger.warning(
                "master_dtype='bfloat16' with bf16_compute=false: nothing is in "
                "fp32 any more. Autocast is what keeps softmax, the norms and "
                "the loss in fp32, and it also gates the FSDP "
                "MixedPrecisionPolicy below, so gradients get reduced in bf16 too."
            )

        if self.model_type == ModelType.Qwen3_vl:
            from models.qwen3_vl.model import set_loss_chunk_mb

            set_loss_chunk_mb(self.training_args.loss_chunk_mb)

        logger.info("model loaded")

        if self.training_args.tp_size > 1:
            apply_tp(self.model, self.model_type, self.tp_group, self.training_args.async_tp)

        ac_memory_budget = getattr(self.training_args, "ac_memory_budget", None)
        if ac_memory_budget is not None:
            import torch._functorch.config as functorch_config
            functorch_config.activation_memory_budget = ac_memory_budget
            logger.info(f"activation memory budget set to {ac_memory_budget}")

        # Compile before sharding, the way torchtitan orders it.
        if self.training_args.compile:
            compile_model(self.model)
            logger.info("model (will be) compiled")

        if self.training_args.data_parallel == 'fsdp':
            # bf16 compute + comms, fp32 master shards + fp32 gradient reduce
            mp_policy = None
            if self.training_args.bf16_compute:
                from torch.distributed.fsdp import MixedPrecisionPolicy
                mp_policy = MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                )
            apply_fsdp(self.model_type, self.model, mesh=self.dp_group, mp_policy=mp_policy)
        elif self.training_args.data_parallel == 'ddp':
            # params stay at master_dtype; torch.autocast in train_step handles bf16 compute
            self.model = replicate(self.model, device_mesh=self.dp_group)
        else:
            raise Exception('invalid sharding strategy for Data Parallel')

        # get rank of local GPU that belongs to the DP group
        data_rank = self.dp_group.get_local_rank()
        data_world_size = self.dp_group.size()

        # ranks sharing a data_rank (a TP group) read identical data, so only the
        # TP-group leader persists the (shared) dataloader state on checkpoint.
        self.data_rank = data_rank
        self.is_data_leader = self.tp_group is None or self.tp_group.get_local_rank() == 0

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

        self.model = set_model(self.model_type, self.model_args, self.model)

        worker_config = WorkerConfig(
            rank=data_rank,
            world_size=data_world_size,
            data_parallel_group=self.dp_group,
            num_workers=2,
        )

        task_encoder, extra_ds_kwargs = build_task_encoder(
            self.data_args,
            tokenizer=self.tokenizer,
            processor=self.processor,
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

        if self.training_args.debug_batch_stats:
            self._batch_stats_dir = os.path.join(
                self.training_args.output_dir, "batch_debug", f"rank_{self.rank()}"
            )
            os.makedirs(self._batch_stats_dir, exist_ok=True)

        # creation of dataloader
        if self.data_args.save_dataloader_state:
            self.data_loader = get_savable_loader(ds)
        else:
            self.data_loader = get_loader(ds)

        self.setup_accumulation(self.training_args.tpi_multiplier)

        self.global_step = 0
        self.micro_step = 0

        self.tokens_seen = 0
        self.tokens_seen_assistant = 0

        self.ntokens_since_last_log = 0
        self.total_ntokens_since_last_log = 0
        self.samples_since_last_log = 0
        self.grad_norm = 0.0

        self.time_last_log = time.perf_counter()
        self.color = Color()

    def rank(self):
        return torch.distributed.get_rank()

    def if_log_rank(self):
        return self.rank() == 0

    def _all_ranks_have_batch(self, local_has_batch: bool) -> bool:
        """Collective agreement on whether EVERY rank still has data."""
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
        )
        self.scheduler = get_scheduler(
            self.optimizer,
            self.training_args
        )
        return self.optimizer, self.scheduler

    def save_checkpoint(self):
        state_dict = {
            "model": self.model,
            "step": self.global_step,
            "tokens_seen": self.tokens_seen,
            "tokens_seen_assistant": self.tokens_seen_assistant,
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

    def load_checkpoint(self, step_num):
        state_dict = {
            "model": self.model,
            "step": step_num,
            "tokens_seen": None,
            "tokens_seen_assistant": None,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
        }

        loaded = load_distributed_checkpoint(
            self.training_args.output_dir, step_num, state_dict, self.rank()
        )
        if loaded is None:
            return

        self.tokens_seen = loaded['tokens_seen']
        self.tokens_seen_assistant = loaded['tokens_seen_assistant']
        self.global_step = loaded['step']
        self.optimizer = loaded['optimizer']
        self.scheduler = loaded['scheduler']

        if self.data_args.save_dataloader_state:
            load_dataloader_state(
                self.training_args.output_dir,
                step_num,
                self.data_loader,
                self.data_rank,
            )

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
            try:
                batch = next(data_iter)
            except StopIteration:
                return

            if batch['cu_seqlens'].ndim > 1:
                batch['cu_seqlens'].squeeze_()

            if 'image_grid_thw' in batch and batch['image_grid_thw'].ndim > 1:
                # do not use squeeze because we need to have two dims
                batch['image_grid_thw'] = batch['image_grid_thw'][0]

            batch['attention_mask'], batch['original_mask'] = batch['cu_seqlens'], batch['attention_mask']

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(device=torch.cuda.current_device(), non_blocking=True)

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

    def _gather_perf(self, time_delta):
        tps = self.ntokens_since_last_log / time_delta
        flops_per_sec = (self.flops_per_token * self.total_ntokens_since_last_log) / time_delta
        tflops_per_sec = flops_per_sec / 1e12
        mfu = (flops_per_sec / (self.peak_tflops_per_gpu * 1e12)) * 100
        peak_mem_gib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 3)

        local = torch.tensor(
            [tps, self.train_step_delta, self.fwd_bwd_time, tflops_per_sec, mfu, peak_mem_gib],
            dtype=torch.float32,
            device=self.device,
        )
        return dist_all_gather(local, torch.distributed.group.WORLD)

    def log(self, avg_loss, max_loss, global_tokens, global_assistant_tokens, global_samples, lr, time_delta, gathered=None):
        tps = self.ntokens_since_last_log / time_delta

        grad_norm = self.grad_norm
        if hasattr(grad_norm, "full_tensor"):
            grad_norm = grad_norm.full_tensor()
        grad_norm = float(grad_norm)

        step_flops = self.flops_per_token * self.total_ntokens_since_last_log
        flops_per_sec = step_flops / time_delta
        tflops_per_sec = flops_per_sec / 1e12

        mfu = (flops_per_sec / (self.peak_tflops_per_gpu * 1e12)) * 100

        color = self.color

        data_time_pct = (self.data_time_delta / time_delta) * 100

        logger.info(
            f"{color.red}{self.global_step}{color.reset} - "
                f"{color.green}loss {avg_loss:.4f} "
                f"{color.blue}tps {tps:.2f} "
                f"{color.magenta}mfu {mfu:.1f}% "
                f"{color.cyan}tflops {tflops_per_sec:.1f} "
                f"{color.reset}"
                f"gnorm {grad_norm:.3f} "
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
            "train/grad_norm": grad_norm,
            "train/batch_efficiency": self.batch_efficiency,

            # performance related
            "perf/tokens_per_second": tps,
            "perf/data_time_pct": data_time_pct,
            "perf/step_time": self.train_step_delta,
            "perf/fwd_bwd_time": self.fwd_bwd_time,
            "perf/tflops_per_second": tflops_per_sec,
            "perf/mfu": mfu,
        }

        if gathered is not None:
            log_metrics.update(topk_metrics(gathered, self.wandb_args.top_k))

        wandb.log(log_metrics, step=self.global_step)

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def train_step(self, batch, optimizer):
        if self.training_args.debug_batch_stats:
            write_batch_stats(
                batch, self._batch_stats_dir, self.global_step, self.current_accum_count
            )

        s_model = time.perf_counter()
        with record_function("forward_pass"):
            with torch.autocast('cuda', torch.bfloat16, enabled=self.training_args.bf16_compute):
                outputs = self.model(
                    **batch
                )
                loss = outputs.loss

        with record_function("backward_pass"):
            scaled_loss = loss / self.current_accum_target
            with torch.autocast('cuda', torch.bfloat16, enabled=self.training_args.bf16_compute):
                scaled_loss.backward()

        self.fwd_bwd_time = time.perf_counter() - s_model

        self.current_accum_count += 1

        if self.current_accum_count >= self.current_accum_target:
            with record_function("optimizer_step"):
                if self.training_args.max_grad_norm > 0:
                    self.grad_norm = clip_grad_norm_mixed(
                        self.model.parameters(), self.training_args.max_grad_norm
                    )
                optimizer.step()
                optimizer.zero_grad()

            n = self.training_args.clear_cache_vram
            if n > 0 and self.global_step % n == 0:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()

            lr = optimizer.param_groups[0]['lr']

            self.global_step += 1

            avg_loss, max_loss, global_tokens, global_assistant, global_samples = (
                dist_mean(loss, self.dp_group),
                dist_max(loss, self.dp_group),
                dist_sum(
                    torch.tensor(
                        self.tokens_seen, dtype=torch.int64, device=self.device
                    ),
                    self.dp_group,
                ),
                dist_sum(
                    torch.tensor(
                        self.tokens_seen_assistant, dtype=torch.int64, device=self.device
                    ),
                    self.dp_group,
                ),
                dist_sum(
                    torch.tensor(self.samples_since_last_log, dtype=torch.int32, device=self.device),
                    self.dp_group,
                )
            )

            time_delta = time.perf_counter() - self.time_last_log
            self.train_step_delta = time_delta / self.current_accum_target

            gathered = None
            if self.wandb_args.log_topk:
                gathered = self._gather_perf(time_delta)

            if self.if_log_rank():
                self.log(avg_loss, max_loss, global_tokens, global_assistant, global_samples, lr, time_delta, gathered)

            self.total_ntokens_since_last_log = 0
            self.ntokens_since_last_log = 0
            self.samples_since_last_log = 0
            torch.cuda.reset_peak_memory_stats(self.device)
            self.time_last_log = time.perf_counter()

            self.current_accum_count = 0
            self.current_accum_target = next(self.accum_schedule)

            return True

        return False

    def train(self):
        data_iterator = self.batch_generator()

        optimizer, scheduler = self.create_optimizer()
        if self.training_args.resume_checkpoint:
            largest_step = find_latest_checkpoint_step(self.training_args.output_dir)
            if largest_step is None:
                logger.info('could not resume')
                raise Exception("Could not found initial checkpoint, killing run")
            optimizer, scheduler = self.load_checkpoint(largest_step)

        prof_ctx, _cprof, _CPROF_START, _CPROF_STOP = build_debug_profiler(
            self.debug_mode, self.training_args.output_dir, self.rank(), self.if_log_rank()
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
                optimizer_updated = self.train_step(batch, optimizer)

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
        if self.if_log_rank():
            logger.info(f"finalizing run at step {self.global_step}")
            logger.info(f"tokens seen: {self.tokens_seen}")
            logger.info(f"assistant tokens seen: {self.tokens_seen_assistant}")
            logger.info("saving final checkpoint...")

        self.save_checkpoint()

        # make sure every rank has finished writing before tearing down NCCL
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    # patch how error are reported
    real_stdout = redirect_rank_io()

    config_manager = ConfigManager(Config)
    args = sys.argv[1:]
    config = config_manager.parse_args(args)

    init_logger(stream=real_stdout)

    torch.manual_seed(42)

    trainer = Trainer(config)
    trainer.train()
