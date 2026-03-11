import torch
from torch import nn
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    AutoModelForCausalLM,
    Qwen3VLMoeForConditionalGeneration
)

import os
import gc
import time
import random
from pathlib import Path
import contextlib

from train.logger import logger

import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor import DTensor

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
    debug_mode: bool = False,
) -> None:
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if not debug_mode:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    if seed is None: seed = 42

    random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed % 2**32)
    if not debug_mode:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    torch.distributed.tensor._random.manual_seed(seed, world_mesh)

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
        for n, p in model.visual.deepstack_merger_list.named_parameters():
            p.required_grad = True
    else:
        for n, p in model.visual.merger.named_parameters():
            p.requires_grad = False
        for n, p in model.visual.deepstack_merger_list.named_parameters():
            p.required_grad = False

    if model_args.train_llm:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = True
        model.lm_head.requires_grad = True
    else:
        for n, p in model.language_model.named_parameters():
            p.requires_grad = False
        model.lm_head.requires_grad = False

    return model

@contextlib.contextmanager
def maybe_enable_profiling(enable_profiling):
    if enable_profiling:
        trace_dir = "/gpfs/scratch/ehpc391/trace/"

        rank = torch.distributed.get_rank()

        def trace_handler(prof):
            curr_trace_dir_name = "iteration_" + str(prof.step_run)
            curr_trace_dir = os.path.join(trace_dir, curr_trace_dir_name)

            if not os.path.exists(curr_trace_dir):
                os.makedirs(curr_trace_dir, exist_ok=True)
            
            logger.info(f"profiling at {curr_trace_dir}")
            output_file = os.path.join(curr_trace_dir, f"rank{rank}_trace.json")
            prof.export_chrome_trace(output_file)
            logger.info("trace saved")
    else:
        return

class GarbageCollection:
    def __init__(self, gc_freq: int = 1000, debug: bool = False):
        assert gc_freq > 0, "gc_freq must be a positive integer"
        self.gc_freq = gc_freq
        self.debug = debug
        gc.disable()
        self.collect("Initial GC collection")
        if debug:
            from torch.utils.viz._cycles import warn_tensor_cycles

            if torch.distributed.get_rank() == 0:
                warn_tensor_cycles()

    def run(self, step_count: int):
        if self.debug:
            self.collect(
                "Force GC to perform collection to obtain debug information",
                generation=2,
            )
            gc.collect()
        elif step_count > 1 and step_count % self.gc_freq == 0:
            self.collect("Performing periodic GC collection")

    @staticmethod
    def collect(reason: str, generation: int = 1):
        begin = time.monotonic()
        gc.collect(generation)
        logger.info("[GC] %s took %.2f seconds", reason, time.monotonic() - begin)
        

def select_model_class(model_args, data_args, training_args):
    logger.info(f'using model: {model_args.model_name}')

    model_name = model_args.model_name.lower()

    if "qwen3" in model_name:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            training_args.cache_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen3vl"

    elif "qwen2.5" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name,
            cache_dir=training_args.cache_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen2.5vl"

    elif "qwen2" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name,
            cache_dir=training_args.cache_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen2vl"

    elif "qwen3" in model_name and "a" in Path(model_name.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name,
            cache_dir=training_args.cache_dir,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen3vl"
        raise NotImplementedError("Qwen3vl-moe finetune is not supported yet.")

    else:
        raise ValueError(f"Unsupported model: {model_args.model_name}")

    return model, data_args

def select_text_model(model_args, training_args):
    model = AutoModelForCausalLM.from_pretrained(
        training_args.text_cache_dir,
        local_files_only=True,
        dtype=(torch.bfloat16 if training_args.bfloat16 else None),
    )
    logger.info(f"Loaded text-only model from {training_args.text_cache_dir}")

    return model


@torch.no_grad()
def load_text_model(vlm_model, text_model):
    logger.info("Starting surgical weight transfer with Prefix Remapping...")
    
    vlm_state = vlm_model.state_dict()
    text_state = text_model.state_dict()
    
    loaded_keys = []
    skipped_keys = []
    shape_mismatch_keys = []

    prefix_map = {
        "model.": "model.language_model.",  # The main backbone shift
        "lm_head.": "lm_head."              # Usually matches exactly, but good to be explicit
    }

    for text_key, text_param in text_state.items():
        vlm_key = None
        for text_prefix, vlm_prefix in prefix_map.items():
            if text_key.startswith(text_prefix):
                suffix = text_key[len(text_prefix):] 
                candidate_key = vlm_prefix + suffix
                
                if candidate_key in vlm_state:
                    vlm_key = candidate_key
                    break
        
        if vlm_key is None and text_key in vlm_state:
            vlm_key = text_key

        if vlm_key is None:
            if len(skipped_keys) < 5: 
                logger.warning(f"Skipping text key '{text_key}': No matching VLM key found.")
            skipped_keys.append(text_key)
            continue

        vlm_param = vlm_state[vlm_key]
        
        if text_param.shape != vlm_param.shape:
            if "embed_tokens" in text_key or "lm_head" in text_key:
                logger.warning(f"Resizing {text_key} -> {vlm_key}: {text_param.shape} -> {vlm_param.shape}")
                
                min_vocab = min(text_param.shape[0], vlm_param.shape[0])
                
                target_param = vlm_model.get_parameter(vlm_key)
                target_param.data[:min_vocab] = text_param.data[:min_vocab]
                loaded_keys.append(vlm_key)
            else:
                shape_mismatch_keys.append(f"{text_key} -> {vlm_key} ({text_param.shape} vs {vlm_param.shape})")
        else:
            target_param = vlm_model.get_parameter(vlm_key)
            target_param.data.copy_(text_param.data)
            loaded_keys.append(vlm_key)

    logger.info(f"Transfer Complete. Loaded: {len(loaded_keys)} keys.")
    logger.info(f"Skipped: {len(skipped_keys)} keys (Vision encoder weights usually).")
    
    if shape_mismatch_keys:
        logger.error(f"CRITICAL: Unresolved shape mismatches:\n{shape_mismatch_keys}")
        raise ValueError("Shape mismatches detected in critical layers!")
    
    return vlm_model

def load_LLM_model(model_args, training_args, model):
    model_name = model_args.model_name.lower()

    if training_args.text_only_path is None:
        raise ValueError('text only cache dir was not provided')

    logger.info(f'loading {model_name} text only weights')
    text_only = Qwen3ForConditionalGeneration.from_pretrained(
            training_args.text_only_path,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
    )

    text_state_dict = text_only.state_dict()
    vl_state_dict = model.state_dict()

    raise NotImplementedError

def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh,
) -> float:
    assert x.numel() == 1  # required by `.item()`
    return funcol.all_reduce(x, reduceOp=reduceOp, group=mesh).item()

def dist_mean(
    x: torch.Tensor,
    mesh,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.AVG.name, mesh=mesh,
    )

def dist_max(
    x: torch.Tensor,
    mesh,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.MAX.name, mesh=mesh,
    )

def dist_sum(
    x: torch.Tensor,
    mesh,
) -> float:
    return _dist_reduce(
        x, reduceOp=c10d.ReduceOp.SUM.name, mesh=mesh,
    )

def create_WSD_scheduler(optimizer, training_args):
    total_steps = training_args.scheduler_steps
    warmup_steps = training_args.warmup_steps
    
    decay_steps = int(training_args.wsd_decay_ratio * total_steps)
    stable_steps = total_steps - warmup_steps - decay_steps
    
    def lr_lambda(current_step):
        # warmup
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        # stable
        if current_step < warmup_steps + stable_steps:
            return 1.0
        
        # decay
        decay_current = current_step - (warmup_steps + stable_steps)
        progress = float(decay_current) / float(max(1, decay_steps))
        
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress))))

    return LambdaLR(optimizer, lr_lambda)

def create_cosine_scheduler(optimizer, training_args):
    total_steps = training_args.scheduler_steps
    warmup_steps = training_args.warmup_steps
    min_lr_ratio = training_args.min_lr_ratio
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

def get_scheduler(optimizer, training_args):
    if training_args.scheduler_type.lower() == "wsd":
        return create_WSD_scheduler(optimizer, training_args)
    elif training_args.scheduler_type.lower() == "cosine":
        return create_cosine_scheduler(optimizer, training_args)
    else:
        raise ValueError(f"Unknown scheduler type: {training_args.scheduler_type}")
