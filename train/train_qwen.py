import re
import os
import sys
import torch
import wandb
import transformers
from itertools import cycle

import time
import contextlib
import cProfile
import pstats
import io

from transformers import AutoProcessor

from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable.replicate import replicate

from torch.profiler import profile, record_function, ProfilerActivity, schedule

# data imports
from megatron.energon import get_train_dataset, get_loader, WorkerConfig
from data.task_encoder_factory import build_task_encoder

# training imports
from train.config_manager import ConfigManager
from train.config import Config, ModelType
from train.logger import init_logger, logger, Color
from train.infra import (
    get_mesh,
    get_tp_group,
    get_dp_group,
    apply_fsdp,
    apply_tp,
    apply_ac,
    ACConfig,
    compile_model,
)
from train.utils import (
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
        if self.model_args.model_impl == "hf":
            self.model.enable_input_require_grads()
        self.optimizer = None # its defined later on

        if self.training_args.bfloat16:
            self.model = self.model.to(torch.bfloat16)

        logger.info("model loaded")

        if self.training_args.tp_size > 1:
            apply_tp(self.model, self.model_type, self.tp_group, self.training_args.async_tp)

        ac_mode = getattr(self.training_args, "ac_mode", "off")
        if ac_mode != "off":
            ac_cfg = ACConfig(enabled=True, full=(ac_mode == "full"))
            apply_ac(
                self.model.model.language_model,
                ac_cfg,
                model_compile_enabled=self.training_args.compile,
            )
            logger.info(f"activation checkpointing applied ({ac_mode})")

        if self.training_args.data_parallel == 'fsdp':
            apply_fsdp(self.model_type, self.model, mesh=self.dp_group)
        elif self.training_args.data_parallel == 'ddp':
            self.model = replicate(self.model, device_mesh=self.dp_group)
        else:
            raise Exception('invalid sharding strategy for Data Parallel')

        # get rank of local GPU that belongs to the DP group
        data_rank = self.dp_group.get_local_rank()
        data_world_size = self.dp_group.size()

        logger.info('sharding/parallelism applied')

        if self.training_args.compile:
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
            repeat=False,
            shuffle_buffer_size=self.data_args.shuffle_buffer_size,
            max_samples_per_sequence=self.data_args.max_samples_per_sequence,
            task_encoder=task_encoder,
            worker_config=worker_config,
            **extra_ds_kwargs,
        )

        self.data_loader = get_loader(ds)

        self.setup_accumulation(self.training_args.tpi_multiplier)

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
            foreach=True,
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
            try: 
                batch = next(data_iter)
            except StopIteration:
                self.end_run()

            if batch['cu_seqlens'].ndim > 1:
                batch['cu_seqlens'].squeeze_()

            if batch['image_grid_thw'].ndim > 1:
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

    # column order produced by `_gather_perf`; True == higher is better
    _PERF_METRIC_NAMES = ("tps", "step_time", "fwd_bwd_time", "tflops", "mfu", "mem_gib")
    _PERF_HIGHER_IS_BETTER = (True, False, False, True, True, False)

    def _topk_metrics(self, gathered):
        """Build the perf_topk/* dict: K slowest and K fastest ranks per metric."""
        metrics = {}
        k = min(self.wandb_args.top_k, gathered.shape[0])
        for j, name in enumerate(self._PERF_METRIC_NAMES):
            col = gathered[:, j]
            higher_better = self._PERF_HIGHER_IS_BETTER[j]
            worst_v, worst_i = torch.topk(col, k, largest=not higher_better)
            best_v, best_i = torch.topk(col, k, largest=higher_better)
            for r in range(k):
                metrics[f"perf_topk/{name}_slow_{r}"] = worst_v[r].item()
                metrics[f"perf_topk/{name}_slow_{r}_rank"] = int(worst_i[r].item())
                metrics[f"perf_topk/{name}_fast_{r}"] = best_v[r].item()
                metrics[f"perf_topk/{name}_fast_{r}_rank"] = int(best_i[r].item())
        return metrics

    def log(self, avg_loss, max_loss, global_tokens, global_assistant_tokens, global_samples, lr, time_delta, gathered=None):
        tps = self.ntokens_since_last_log / time_delta

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

        if gathered is not None:
            log_metrics.update(self._topk_metrics(gathered))

        wandb.log(log_metrics, step=self.global_step)

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def train_step(self, data_iterator, optimizer):
        batch = next(data_iterator)

        s_model = time.perf_counter()
        with record_function("forward_pass"):
            with torch.autocast('cuda', torch.bfloat16):
                outputs = self.model(
                    **batch
                )
                loss = outputs.loss

        with record_function("backward_pass"):
            scaled_loss = loss / self.current_accum_target
            with torch.autocast('cuda', torch.bfloat16):
                scaled_loss.backward()

        self.fwd_bwd_time = time.perf_counter() - s_model

        self.current_accum_count += 1

        if self.current_accum_count >= self.current_accum_target:
            with record_function("optimizer_step"):
                optimizer.step()
                optimizer.zero_grad()

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

    def _dump_cprofile(self, prof: cProfile.Profile) -> None:
        out_path = os.path.join(
            self.training_args.output_dir, f"cprofile_rank_{self.rank()}.txt"
        )
        stream = io.StringIO()
        pstats.Stats(prof, stream=stream).sort_stats("cumulative").print_stats(50)
        with open(out_path, "w") as f:
            f.write(stream.getvalue())
        logger.info(f"cProfile stats written to {out_path}")

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
        
        def trace_handler(prof):
            trace_path = os.path.join(
                self.training_args.output_dir,
                f"trace_rank_{self.rank()}_step_{prof.step_num}.json",
            )
            prof.export_chrome_trace(trace_path)
            summary_path = trace_path.replace(".json", "_summary.txt")
            with open(summary_path, "w") as f:
                f.write(prof.key_averages(group_by_stack_n=5).table(
                    sort_by="self_cpu_time_total", row_limit=40
                ))
            if self.if_log_rank():
                logger.info(f"Torch profiler trace → {trace_path}")
                logger.info(f"Torch profiler summary → {summary_path}")

        if self.debug_mode:
            # wait=30 skips the torch.compile warmup steps; active=5 records 5 full steps
            prof_schedule = schedule(wait=10, warmup=2, active=2, repeat=1)
            prof_ctx = profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=prof_schedule,
                on_trace_ready=trace_handler,
                record_shapes=True,
                profile_memory=False,
                with_stack=True,
            )
            # cProfile window: steady-state steps well after compilation
            _cprof = cProfile.Profile()
            _CPROF_START = 50
            _CPROF_STOP = 65
        else:
            prof_ctx = contextlib.nullcontext()
            _cprof = None
        _cprof_active = False

        with prof_ctx as prof:
            while self.global_step < self.training_args.total_steps:
                self.micro_step += 1

                if self.debug_mode and not _cprof_active and self.global_step >= _CPROF_START:
                    _cprof.enable()
                    _cprof_active = True

                optimizer_updated = self.train_step(data_iterator, optimizer)
                if prof is not None:
                    prof.step()

                if optimizer_updated:
                    scheduler.step()

                    if _cprof_active and self.global_step >= _CPROF_STOP:
                        _cprof.disable()
                        _cprof_active = False
                        self._dump_cprofile(_cprof)

                    if self.may_save() and self.global_step < self.training_args.total_steps:
                        self.save_checkpoint()

    def end_run(self):
        if self.if_log_rank():
            logger.info(f"data iterator exhausted at step {self.global_step}")
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
