import re
import os
import math
import torch
import wandb
import transformers
import random
import sys
from itertools import cycle

import time

from transformers import AutoProcessor

from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable import replicate

# data imports
from data.advanced_datasets import QwenPackedDataset, ShardedParquetSource
from data.data_processor import DataCollatorForSupervisedDataset

# training imports
from train.config_manager import ConfigManager
from train.config import Config
from train.model_attention import replace_attention_qwenvl
from train.logger import init_logger, logger, Color
from train.infra import (
    get_mesh,
    get_tp_group,
    get_dp_group,
    apply_fsdp,
    apply_tp_complex,
    compile_model,
)
from train.utils import (
    set_determinism,
    generate_accumulation_pattern,
    get_scheduler,

    dist_mean,
    dist_max,
    dist_sum,

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
        attn_implementation = None

        self.model_args = cfg.model
        self.training_args = cfg.training
        self.data_args = cfg.data
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
                name=self.model_args.run_name,
                project=self.model_args.project_name,
                entity=self.model_args.entity_name,
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

        self.model, _ = select_model_class(self.model_args, self.data_args, self.training_args)

        if self.training_args.load_text_model:
            self.text_model = select_text_model(self.model_args, self.training_args)

            self.model = load_text_model(self.model, self.text_model)

        # MOVE TO cuda:{self.local_rank}
        self.model.to(self.device)
        
        if self.training_args.random_init_mlp:
            if self.if_log_rank():
                logger.info("Randomly initializing MLP projector weights")

            def init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

            torch.manual_seed(42)
            self.model.visual.merger.apply(init_weights)
            self.model.visual.deepstack_merger_list.apply(init_weights)

            for param in self.model.visual.merger.parameters():
                torch.distributed.broadcast(param.data, src=0)

        # replace flash_attn
        replace_attention_qwenvl(self.training_args)
        self.model.enable_input_require_grads()
        self.optimizer = None # its defined later on

        if self.training_args.bfloat16:
            self.model = self.model.to(torch.bfloat16)

        #apply_float8(self.model)
        logger.info("model loaded")

        if self.tp_group is not None:
            apply_tp_complex(self.model, self.tp_group)
            
        if self.dp_group is not None:
            if self.training_args.data_parallel == 'fsdp':
                apply_fsdp(self.model, mesh=self.dp_group)
            elif self.training_args.data_parallel == 'ddp':
                self.model = replicate(self.model, device_mesh=self.dp_group)
            else:
                raise Exception('invalid sharding strategy for Data Parallel')

            # get rank of local GPU that belongs to the DP group
            data_rank = self.dp_group.get_local_rank()
            data_world_size = self.dp_group.size()
        else:
            data_rank = 0
            data_world_size = 1

            logger.info('WARNING: only one data rank is being used')

        logger.info('sharding/parallelism applied')

        if self.training_args.compile:
            compile_model(self.model)
            logger.info("model compiled")

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.training_args.cache_dir,
            model_max_length=int(self.data_args.seq_len),
            padding_side="right",
            use_fast=False,
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        self.processor = AutoProcessor.from_pretrained(
            self.training_args.cache_dir,
        )

        self.model = set_model(self.model_args, self.model)

        # TODO: patch
        self.datasource = ShardedParquetSource(
            self.data_args.data_path,
            self.data_args.start_idx,
            self.data_args.end_idx,
            data_world_size=data_world_size,
            data_rank=data_rank,
        )
        dataset = QwenPackedDataset(
            dataset=self.datasource,
            processor=self.processor,
            data_args=self.data_args
        )

        collator = DataCollatorForSupervisedDataset(
            tokenizer=self.tokenizer,
            seq_len=int(self.data_args.seq_len)
        )

        self.data_loader = DataLoader(
            dataset,
            batch_size=1,
            collate_fn=collator,
            num_workers=1,
            pin_memory=True,
            worker_init_fn=random.seed,
        )

        self.setup_accumulation(self.training_args.tpi_multiplier)

        self.global_step = 0
        self.micro_step = 0

        self.tokens_seen = 0
        self.tokens_seen_assistant = 0

        self.ntokens_since_last_log = 0
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

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            labels = batch.pop("labels")
            batch_samples = batch['attention_mask'].shape[0] - 1

            ntokens_batch = (batch['input_ids'] != self.pad_token_id).sum().item()
            ntokens_batch_assistant = (labels != -100).sum().item()

            self.batch_efficiency = (ntokens_batch / self.data_args.seq_len ) * 100
            self.tokens_seen_assistant += ntokens_batch_assistant
            self.tokens_seen += ntokens_batch
            self.ntokens_since_last_log += ntokens_batch
            self.samples_since_last_log += batch_samples

            self.data_time_delta = time.perf_counter() - data_start_time

            yield batch, labels

    def log(self, avg_loss, max_loss, global_tokens, global_assistant_tokens, global_samples, lr):

        time_delta = time.perf_counter() - self.time_last_log

        tps = self.ntokens_since_last_log / time_delta

        color = self.color

        data_time_pct = (self.data_time_delta / time_delta) * 100

        logger.info(
            f"{color.red}step {self.global_step} "
                f"{color.green}loss {avg_loss:.4f} "
                f"{color.blue}tps {tps:.2f} "
                f"{color.reset}"
                f"time {self.train_step_delta:.2f}s "
                f"data_pct {data_time_pct:.2f}% "
                f"nsamples {global_samples} "
                f"batch_util {self.batch_efficiency:.1f}% "
        )

        log_metrics = {
            "train/loss": avg_loss,
            "train/max_loss": max_loss,
            "train/tokens_per_second": tps,
            "train/step_time": self.train_step_delta,
            "train/data_time_pct": data_time_pct,
            "train/tokens_seen": global_tokens,
            "train/assistant_tokens_seen": global_assistant_tokens,
            "train/num_samples": global_samples,
            "train/lr": lr,
        }

        wandb.log(log_metrics, step=self.global_step)

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def train_step(self, data_iterator, optimizer):
        batch, labels = next(data_iterator)

        device_type = "cpu" if self.debug_mode else "cuda"

        with torch.autocast(device_type, torch.bfloat16):
            outputs = self.model(
                labels=labels,
                output_hidden_states=False,
                **batch
            )
            loss = outputs.loss

            scaled_loss = loss / self.current_accum_target
            scaled_loss.backward()

        self.current_accum_count += 1

        if self.current_accum_count >= self.current_accum_target:
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

            self.train_step_delta = (time.perf_counter() - self.time_last_log) / self.current_accum_target

            if self.if_log_rank():
                self.log(avg_loss, max_loss, global_tokens, global_assistant, global_samples, lr)

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
        if self.training_args.resume:
            paths = os.listdir(self.training_args.output_dir)
            possible_steps = []
            for path in paths:
                match = re.search(r"(\d+\.?\d*)$", path)
                try:
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

        try:
            while self.global_step < self.training_args.scheduler_steps:
                self.micro_step += 1
                optimizer_updated = self.train_step(data_iterator, optimizer)

                if optimizer_updated:
                    scheduler.step()
                    # Save checkpoint only if we haven't reached the target steps
                    if self.may_save() and self.global_step < self.training_args.scheduler_steps:
                        self.save_checkpoint()

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
    try:
        from local_logging import logger as local_logger
        local_logger.patch_wandb()
    except ImportError:
        pass

    config_manager = ConfigManager(Config)
    args = sys.argv[1:]
    config = config_manager.parse_args(args)

    init_logger()

    torch.manual_seed(42)

    trainer = Trainer(config)
    trainer.train()
