import torch
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3_5ForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration,
    Qwen3ForCausalLM,
    AutoModelForCausalLM,
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

from train.config import Training as TrainArgs
from train.config import Model as ModelArgs
from train.config import ModelType


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

def set_model(model_type: ModelType, model_args: ModelArgs, model):
    if model_type == ModelType.Qwen3_5:
        return set_model_qwen3_5(model_args, model)
    elif model_type == ModelType.Qwen3_vl:
        return set_model_qwen3vl(model_args, model)
    elif model_type == ModelType.Qwen3_text:
        return set_model_qwen3(model_args, model)
    raise NotImplementedError()

def set_model_qwen3_5(model_args: ModelArgs, model):
    # MLP / Projector
    for n, p in model.model.visual.merger.named_parameters():
        p.requires_grad = model_args.train_mlp

    # ViT
    for n, p in model.model.visual.blocks.named_parameters():
        p.requires_grad = model_args.train_vit
    for n, p in model.model.visual.patch_embed.named_parameters():
        p.requires_grad = model_args.train_vit

    # LLM
    for n, p in model.model.language_model.named_parameters():
        p.requires_grad = model_args.train_llm
    model.lm_head.requires_grad = model_args.train_llm

    # MTP Heads (Tie to LLM if computing MTP loss, otherwise force False)
    for n, p in model.named_parameters():
        if "mtp" in n.lower():
            # TODO: implement MTP and unfreeze the Module
            p.requires_grad = False

    return model

def set_model_qwen3vl(model_args: ModelArgs, model):
    # ViT
    for n, p in model.model.visual.named_parameters():
        p.requires_grad = model_args.train_vit

    # MLP / Projector
    for n, p in model.model.visual.merger.named_parameters():
        p.requires_grad = model_args.train_mlp
    for n, p in model.model.visual.deepstack_merger_list.named_parameters():
        p.requires_grad = model_args.train_mlp

    # LLM
    for n, p in model.model.language_model.named_parameters():
        p.requires_grad = model_args.train_llm
    model.lm_head.requires_grad = model_args.train_llm

    return model

def set_model_qwen3(model_args: ModelArgs, model):
    # LLM
    for n, p in model.model.named_parameters():
        p.requires_grad = model_args.train_llm
    model.lm_head.requires_grad = model_args.train_llm

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
        

def select_model_class(model_type: ModelType, model_args: ModelArgs, training_args: TrainArgs):
    """
    TODO: use ModelType instead of model name
    """
    logger.info(f'using model: {model_args.model_name} (impl={model_args.model_impl})')

    if not os.path.exists(training_args.model_dir):
        raise ValueError(f"path with model does not exists, got: {training_args.model_dir}")

    model_name = model_args.model_name.lower()

    if model_args.model_impl == "native":
        return _select_native_model_class(training_args, model_name)
    elif model_args.model_impl != "hf":
        raise ValueError(
            f"Unknown model_impl '{model_args.model_impl}'. Expected 'hf' or 'native'."
        )

    if "qwen3-vl" in model_name:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            training_args.model_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )

    elif "qwen3-vl" in model_name and "a" in Path(model_name.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            training_args.model_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        raise NotImplementedError("Qwen3vl-moe finetune is not supported yet.")
    
    elif "qwen3.5" in model_name:
        model = Qwen3_5ForConditionalGeneration.from_pretrained(
            training_args.model_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None) ,
        )
    
    elif "qwen3" in model_name:
        model = Qwen3ForCausalLM.from_pretrained(
            training_args.model_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None) ,
        )

    elif "qwen3" in model_name:
        model = Qwen3ForCausalLM.from_pretrained(
            training_args.model_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None) ,
        )

    elif "qwen2.5-vl" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            training_args.model_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )

    elif "qwen2" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            training_args.model_dir,
            local_files_only=True,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )

    else:
        raise ValueError(f"Unsupported model: {model_args.model_name}")

    return model


def _select_native_model_class(training_args: TrainArgs, model_name: str):
    """Dispatch to our torch-native model implementations under `models/`."""
    dtype = torch.bfloat16 if training_args.bfloat16 else torch.float32

    if "qwen3-vl" in model_name:
        from models.qwen3_vl.model_qwen3_vl import Qwen3VLForCausalLM as NativeQwen3
    elif "qwen3.5" in model_name:
        raise NotImplementedError()
    elif "qwen3" in model_name:
        from models.qwen3.model_qwen3 import Qwen3ForCausalLM as NativeQwen3
    else:
        raise ValueError(
            f"Unsupported model for native impl: {model_name}"
        )

    model = NativeQwen3.from_pretrained(
        training_args.model_dir,
        dtype=dtype,
        device="cpu",
    )
    logger.info(f"Loaded native {model_name} from {training_args.model_dir}")
    return model

def select_text_model(training_args):
    model = AutoModelForCausalLM.from_pretrained(
        training_args.text_model_dir,
        local_files_only=True,
        dtype=(torch.bfloat16 if training_args.bfloat16 else None),
    )
    logger.info(f"Loaded text-only model from {training_args.text_model_dir}")

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

def create_WSD_scheduler(optimizer, training_args: TrainArgs):
    total_steps = training_args.total_steps
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

def create_cosine_scheduler(optimizer, training_args: TrainArgs):
    total_steps = training_args.total_steps
    warmup_steps = training_args.warmup_steps
    min_lr_ratio = training_args.min_lr_ratio
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda)

def get_scheduler(optimizer, training_args: TrainArgs):
    if training_args.scheduler_type.lower() == "wsd":
        return create_WSD_scheduler(optimizer, training_args)
    elif training_args.scheduler_type.lower() == "cosine":
        return create_cosine_scheduler(optimizer, training_args)
    else:
        raise ValueError(f"Unknown scheduler type: {training_args.scheduler_type}")

def get_dense_model_nparams_and_flops(
    model_name: str,
    model: torch.nn.Module,
    seq_len: int,
) -> tuple[int, int]:
    """
    Args:
        model_name: str (either Qwen/Qwen3-VL-8B or 2B)
        model: nn.Module representing the model.
        seq_len: The sequence length in training configs.

    Returns:
        Tuple of (nparams, num_flops_per_token):
            nparams: Total number of model parameters.
            num_flops_per_token: Estimated number of floating point operations per token.
    """
    nparams = sum(p.numel() for p in model.parameters())
    nparams_embedding = sum(
        sum(p.numel() for p in m.parameters())
        for m in model.children()
        if isinstance(m, torch.nn.Embedding)
    )

    if "8B" in model_name:
        tied = False
    elif "9B" in model_name:
        tied = False
    elif "2B" in model_name:
        tied = True
    elif "4B" in model_name:
        tied = True
    elif "1.7B" in model_name:
        tied = True
    else:
        # ValueError
        return 0, 0
    
    # we take into account the embedding params
    num_flops_per_token = 6 * nparams

    if tied:
        nparams = nparams - nparams_embedding

    return int(nparams), int(num_flops_per_token)