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

from train.config_manager import ConfigManager
from train.config import Config
from train.model_attention import replace_attention_qwenvl
import train.utils as utils

from data.advanced_datasets import QwenPackedDataset, ShardedParquetSource
from data.data_processor import DataCollatorForSupervisedDataset

from transformers import AutoProcessor

from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed.fsdp import fully_shard
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict
from torch.nn.parallel import DistributedDataParallel as DDP

from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import Color

torch._logging.set_logs(graph_code=True)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

def generate_accumulation_pattern(target_multiplier: float, pattern_length: int = 100) -> list[int]:
    if target_multiplier < 1.0:
        raise ValueError("Multiplier must be >= 1.0")

    pattern = []
    current_cumulative = 0.0
    for i in range(pattern_length):
        next_cumulative = (i + 1) * target_multiplier
        steps_this_cycle = math.floor(next_cumulative) - math.floor(current_cumulative)
        
        pattern.append(int(steps_this_cycle))
        current_cumulative = next_cumulative
        
        if math.isclose(current_cumulative, round(current_cumulative)):
            break
            
    return pattern

def set_determinism(
        world_mesh,
        seed: int | None = None,
        deterministic: bool = True,
) -> None:
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    random.seed(seed or 42)
    torch.manual_seed(seed or 42)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.distributed.tensor._random.manual_seed(seed, world_mesh)

def compile_model(model: torch.nn.Module, train_args):
    if train_args.bfloat16:
        model = model.to(torch.bfloat16)

    model.visual = torch.compile(
        model.visual, fullgraph=True, mode='max-autotune',
    )
    model.language_model = torch.compile(
        model.language_model, fullgraph=True, mode='max-autotune',
    )
    model.visual.merger = torch.compile(
        model.visual.merger, fullgraph=True, mode='max-autotune',
    )

    model = torch.compile(
        model, fullgraph=True, #mode='max-autotune',
    ) 

def simple_compile(model, train_args):
    if train_args.bfloat16:
        model = model.to(torch.bfloat16)

    model = torch.compile(model)


def apply_ddp(
        model,
        dp_mesh,
):
    model = DDP(model, device_mesh=dp_mesh)

    logger.info("applied DDP to the model")


def set_model(model_args, model):
    if model_args.train_vit:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.train_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.train_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    return model


class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, cfg: Config):
        attn_implementation = None

        self.model_args = cfg.model
        self.training_args = cfg.training
        self.data_args = cfg.data

        # nccl sees how to init the processes for each GPU
        torch.distributed.init_process_group(backend='nccl')
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])
        torch.cuda.set_device(self.local_rank)

        # this is fine by now, we just shard on every GPU
        self.mesh = torch.distributed.device_mesh.init_device_mesh(
            "cuda",
            (self.world_size,),
            mesh_dim_names=("shard",),
        )

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
                },
            )

            logger.info('using directory:')
            logger.info(os.getcwd())
            logger.info(self.world_size)
            logger.info("starting finetune job")
            logger.info(f"mesh: {self.mesh}")

            logger.info(self.model_args)
            logger.info(self.training_args)
            logger.info(self.data_args)

        set_determinism(seed=42 + self.local_rank, deterministic=True, world_mesh=self.mesh)

        self.model, data_args = utils.select_model_class(self.model_args, self.data_args, self.training_args, attn_implementation)
        self.model.enable_input_require_grads()
        self.optimizer = None # its defined later on

        if self.training_args.compile:
            #compile_model(self.model.to(self.device), self.training_args)
            simple_compile(self.model.to(self.device), self.training_args)
            if self.if_log_rank():
                logger.info("model compiled with torch.compile")

        if self.training_args.shard:
            if self.if_log_rank():
                logger.info(f"applied FSDP with {self.mesh}")
            self.model = fully_shard(
                self.model.to(self.device).to(torch.bfloat16),
                mesh=self.mesh,
            )

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_args.model_name,
            cache_dir=self.training_args.cache_dir,
            model_max_length=self.data_args.seq_len,
            padding_side="right",
            use_fast=False,
        )
        self.pad_token_id = self.tokenizer.pad_token_id

        self.processor = AutoProcessor.from_pretrained(
            self.model_args.model_name,
            cache_dir=self.training_args.cache_dir,
        )

        replace_attention_qwenvl()
        self.model = set_model(self.model_args, self.model)

        self.model.to(self.device)

        self.datasource = ShardedParquetSource(
                self.data_args.data_path,
                self.data_args.start_idx,
                self.data_args.end_idx,
        )
        dataset = QwenPackedDataset(
            dataset=self.datasource,
            processor=self.processor,
            data_args=self.data_args
        )

        collator = DataCollatorForSupervisedDataset(
            tokenizer=self.tokenizer,
            seq_len=self.data_args.seq_len
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

        self.gc_handler = utils.GarbageCollection(
            gc_freq=self.training_args.garbage_steps, debug=False
        )

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
            weight_decay=self.training_args.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.training_args.scheduler_steps
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

        self.ntokens_since_last_log = 0
        self.samples_since_last_log = 0
        self.time_last_log = time.perf_counter()

    def setup_accumulation(self, tpi_multiplier=1.5):
        pattern = generate_accumulation_pattern(tpi_multiplier)
        self.accum_schedule = cycle(pattern)
        self.current_accum_target = next(self.accum_schedule)
        self.current_accum_count = 0

    def train_step(self, data_iterator, optimizer):
        batch, labels = next(data_iterator)

        with torch.autocast("cuda", torch.bfloat16):
            outputs = self.model(
                labels=labels,
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
                    utils.dist_mean(loss, self.mesh['shard']),
                    utils.dist_max(loss, self.mesh['shard']),
                    utils.dist_sum(
                        torch.tensor(
                            self.tokens_seen, dtype=torch.int64, device=self.device
                        ),
                        self.mesh['shard'],
                    ),
                    utils.dist_sum(
                        torch.tensor(
                            self.tokens_seen_assistant, dtype=torch.int64, device=self.device
                        ),
                        self.mesh['shard'],
                    ),
                    utils.dist_sum(
                        torch.tensor(self.samples_since_last_log, dtype=torch.int32, device=self.device),
                        self.mesh['shard'],
                    )
                )

            self.train_step_delta = (time.perf_counter() - self.time_last_log) / self.current_accum_target

            if self.if_log_rank():
                self.log(avg_loss, max_loss, global_tokens, global_assistant, global_samples, lr)

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
                print('could not resume')
                raise Exception("Could not found initial checkpoint, killing run")

        try:
            while True:
                self.micro_step += 1
                self.gc_handler.run(self.micro_step)

                optimizer_updated = self.train_step(data_iterator, optimizer)

                if self.may_save() and optimizer_updated:
                    self.save_checkpoint()

                scheduler.step()

        except StopIteration as e:
            logger.info(f"data iterator exhausted...: {e}")
            logger.info("saving final model...")
            self.save_checkpoint()
            logger.info(f"final model saved, step: {self.step}")

        if self.if_log_rank():
            logger.info(f"tokens seen: {self.tokens_seen}")
            logger.info(f"assistant tokens seen: {self.tokens_seen_assistant}")
            logger.info("Training completed")

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
