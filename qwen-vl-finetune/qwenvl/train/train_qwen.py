import os
import pathlib
import torch
import wandb
import transformers
import random
import sys
from pathlib import Path

import time

project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from qwenvl.data.data_processor import make_supervised_data_module
from qwenvl.train.utils import select_model_class, GarbageCollection
from qwenvl.train.argument import (
    ModelArguments,
    DataArguments,
    TrainingArguments,
)
from transformers import AutoProcessor

import time

from torch.utils.data import DataLoader
from torch.distributed.elastic.multiprocessing.errors import record
from torch.distributed._composable import replicate
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard as FSDP
from torch.distributed.checkpoint import HuggingFaceStorageWriter
from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict


from torchtitan.tools.logging import init_logger, logger
from torchtitan.tools.utils import Color

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

def compile_model(model: torch.nn.Module):
    model.visual = torch.compile(
        model.visual, fullgraph=True
    )
    model.language_model = torch.compile(
        model.language_model, fullgraph=True
    )
    model.visual.merger = torch.compile(
        model.visual.merger, fullgraph=True
    )

    model = torch.compile(
        model, fullgraph=True
    ) 


def apply_ddp(
        model,
        dp_mesh,
        enable_compile=True,
        enable_compile_autograd=True,
):
    if enable_compile:
        if enable_compile_autograd:
            torch._dynamo.config.optimize_ddp = (
                "python_reducer_without_compiled_forward"
            )
        else:
            torch._dynamo.config.optimize_ddp = "ddp_optimizer"

    replicate(model, dp_mesh, bucket_cap_mb=100)

    logger.info("applied DDP to the model")


def set_model(model_args, model):
    if model_args.tune_mm_vision:
        for n, p in model.visual.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_mlp:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False

    if model_args.tune_mm_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False


def loss_fn(pred, labels):
    return torch.nn.functional.cross_entropy(
        pred.flatten(0, 1).float(), labels.flatten(0, 1)
    )


class Trainer(torch.distributed.checkpoint.stateful.Stateful):

    @record
    def __init__(self, *args, **kwargs):
        attn_implementation = None

        parser = transformers.HfArgumentParser(
            (ModelArguments, DataArguments, TrainingArguments)
        )
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

        os.makedirs(training_args.output_dir, exist_ok=True)
        local_rank = int(os.environ["LOCAL_RANK"])

        self.mesh = init_device_mesh(
            "cuda",
            (3,),
            mesh_dim_names=("shard",),
        )

        if self.if_log_rank():
            wandb.init(
                project="qwen-vl-finetune",
                config={
                    **vars(model_args),
                    **vars(data_args),
                    **vars(training_args),
                },
            )

        if self.if_log_rank():
            logger.info("starting finetune job")
            logger.info(f"mesh: {self.mesh}")

        set_determinism(seed=42 + local_rank, deterministic=True, world_mesh=self.mesh)

        self.model, data_args = select_model_class(model_args, data_args, training_args, attn_implementation)
        self.optimizer = None # its defined later on

        self.device = torch.device(f"cuda:{local_rank}")

        if True:
            compile_model(self.model.to(self.device).to(torch.bfloat16))
            logger.info("model compiled with torch.compile")

        if False:
            self.model = FSDP(
                self.model.to(self.device).to(torch.bfloat16),
                mesh=self.mesh,
            )

        self.model_args = model_args
        self.data_args = data_args
        self.training_args = training_args

        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

        self.processor = AutoProcessor.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
        )

        set_model(model_args, self.model)

        self.model.to(self.device)

        self.data_module = make_supervised_data_module(self.processor, data_args=data_args)

        dataset = self.data_module['train_dataset']
        collator = self.data_module['data_collator']

        self.sampler = torch.utils.data.distributed.DistributedSampler(
            dataset,
            num_replicas=torch.distributed.get_world_size(),
            rank=torch.distributed.get_rank(),
            shuffle=False,
            seed=42,
        )

        self.data_loader = DataLoader(
            dataset,
            batch_size=self.training_args.per_device_train_batch_size,
            collate_fn=collator,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            #sampler=self.sampler,
            worker_init_fn=random.seed,
        )

        self.gc_handler = GarbageCollection(
            gc_freq=100000, debug=False
        )

        self.step = 0
        self.tokens_seen = 0
        self.tokens_seen_assistant = 0
        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()
        self.color = Color()

    def rank(self):
        return torch.distributed.get_rank()

    def if_log_rank(self):
        return self.rank() == 0

    def create_optimizer(self):
        if self.optimizer is not None:
            return self.optimizer

        lr_mlp = self.training_args.mm_projector_lr
        lr_vision = self.training_args.vision_tower_lr
        lr_llm = self.training_args.learning_rate

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
                "lr": lr_vision,
            },
            {
                "params": llm_params,
                "lr": lr_llm,
            },
        ]

        # TODO: add weight decay exclusion for bias and LayerNorm
        #no_decay = ["bias", "LayerNorm.weight"]

        self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=lr_llm, weight_decay=self.training_args.weight_decay)

        return self.optimizer

    def save_checkpoint(self):
        step = self.step

        checkpoint_dir = os.path.join(
            self.training_args.output_dir,
            f"checkpoint-step-{step}",
        )
        state_dict = {"model": self.model}

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
            
    def state_dict(self):
        model_state, optimizer_state, = get_state_dict(self.model, self.optimizer)
        return {
            "model": model_state,
            "optimizer": optimizer_state,
        }

    def load_state_dict(self, state_dict):
        set_state_dict(
            self.model,
            self.optimizer,
            model_state_dict=state_dict["model"],
            optimizer_state_dict=state_dict["optimizer"],
        )

    def may_save(self):
        if self.step % self.training_args.save_steps == 0:
            return True
        return False

    def batch_generator(self, data_module):
        data_iter = iter(self.data_loader)

        while True:
            data_start_time = time.perf_counter()
            try:
                batch = next(data_iter)

            except StopIteration:
                raise StopIteration("DataLoader ran out of data.")

            for k, v in batch.items():
                if isinstance(v, torch.Tensor):
                    batch[k] = v.to(self.device, non_blocking=True)

            labels = batch.pop("labels")
            _ = batch.pop("attention_mask", None)

            ntokens_batch = labels.numel()
            ntokens_batch_assistant = (labels != -100).sum().item()

            self.tokens_seen_assistant += ntokens_batch_assistant
            self.tokens_seen += ntokens_batch
            self.ntokens_since_last_log += ntokens_batch

            self.data_time_delta = time.perf_counter() - data_start_time

            yield batch, labels

    def log(self, global_loss):

        time_delta = time.perf_counter() - self.time_last_log

        tps = self.ntokens_since_last_log / time_delta

        color = self.color

        data_time_pct = (self.data_time_delta / time_delta) * 100

        logger.info(
            f"{color.red}step {self.step} "
            f"{color.green}loss {global_loss:.4f} "
            f"{color.blue}tps {tps:.2f} "
            f"{color.reset}"
            f"time_delta {self.train_step_delta:.2f}s "
            f"data_time_pct {data_time_pct:.2f}%"
        )

        log_metrics = {
            "train/loss": global_loss,
            "train/tokens_per_second": tps,
            "train/step_time": self.train_step_delta,
            "train/data_time_pct": data_time_pct,
            "train/tokens_seen": self.tokens_seen,
            "train/assistant_tokens_seen": self.tokens_seen_assistant,
        }

        wandb.log(log_metrics, step=self.step)

        self.ntokens_since_last_log = 0
        self.time_last_log = time.perf_counter()

    def train_step(self, data_iterator):
        self.optimizer.zero_grad()
        batch, labels = next(data_iterator)
        # we use the labels directly

        accumulated_losses = []
        for _microbatch in range(self.training_args.gradient_accumulation_steps):
            with torch.autocast("cuda", torch.bfloat16):
                outputs = self.model(
                    labels=labels,
                    **batch
                )
                loss = outputs.loss
                loss.backward()
            accumulated_losses.append(loss.detach())

        loss = torch.sum(torch.stack(accumulated_losses))

        if self.training_args.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.training_args.max_grad_norm
            )
        self.optimizer.step()
        self.train_step_delta = time.perf_counter() - self.time_last_log

        if self.if_log_rank():
            self.log(loss.item())

    def train(self):
        data_iterator = self.batch_generator(self.data_module)
        self.optimizer = self.create_optimizer()

        try:
            while True:
                self.step += 1
                self.gc_handler.run(self.step)

                self.train_step(data_iterator)

                if self.may_save():
                    self.save_checkpoint()

        except StopIteration as e:
            logger.info(f"data iterator exhausted...: {e}")
        except Exception as e:
            logger.info(f"exception during training: {e}")
        except KeyboardInterrupt:
            logger.info("keyboard interrupt received...")

        if self.if_log_rank():
            logger.info(f"tokens seen: {self.tokens_seen}")
            logger.info(f"assistant tokens seen: {self.tokens_seen_assistant}")
            logger.info("Training completed")

        torch.distributed.destroy_process_group()

if __name__ == "__main__":
    init_logger()

    torch.manual_seed(42)

    trainer = Trainer()
    trainer.train()
