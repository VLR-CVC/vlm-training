import re
import torch
import torch.nn.functional as F
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

import math
import torch

def init_qwen35(model):
    # hf compatibility stuff
    model = model.model
    decoder = model.language_model
    num_layers = len(decoder.layers)
    
    std = 0.02
    scaled_std = std / math.sqrt(2 * num_layers)

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, torch.nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.padding_idx is not None:
                torch.nn.init.zeros_(m.weight[m.padding_idx])
        # many norm variants, this catches them
        elif "Norm" in m.__class__.__name__:
            if hasattr(m, 'weight') and m.weight is not None:
                torch.nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    torch.manual_seed(42)
    decoder.apply(init_weights)
    model.visual.merger.apply(init_weights)

    with torch.no_grad():
        for name, param in decoder.named_parameters():
            if "o_proj.weight" in name or "down_proj.weight" in name:
                torch.nn.init.normal_(param, mean=0.0, std=scaled_std)

    for param in decoder.parameters():
        torch.distributed.broadcast(param.data, src=0)
    for param in model.visual.merger.parameters():
        torch.distributed.broadcast(param.data, src=0)

def init_qwen3vl(model):
    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    torch.manual_seed(42)
    model.visual.merger.apply(init_weights)
    model.visual.deepstack_merger_list.apply(init_weights)

    for param in model.visual.merger.parameters():
        torch.distributed.broadcast(param.data, src=0)

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

    load_vision = not getattr(training_args, "load_vision_model", False)
    return _select_native_model_class(training_args, model_type, load_vision=load_vision)
    # return: model, config

def _select_native_model_class(training_args: TrainArgs, model_type: ModelType, load_vision: bool = True):
    """Dispatch to our torch-native model implementations under `models/`."""
    dtype = torch.bfloat16 if training_args.bfloat16 else torch.float32

    if model_type is ModelType.Qwen3_vl:
        from models.qwen3_vl.model import Qwen3VLForCausalLM as NativeQwen3
    elif model_type is ModelType.Qwen3_5:
        from models.qwen3_5.model import Qwen3_5ForCausalLM as NativeQwen3
    elif model_type is ModelType.Qwen3_text:
        from models.qwen3.model import Qwen3ForCausalLM as NativeQwen3
    else:
        raise ValueError(
            f"Unsupported model for native impl: {model_type}"
        )

    model, config = NativeQwen3.from_pretrained(
        training_args.model_dir,
        dtype=dtype,
        device="cpu",
        load_vision=load_vision,
    )
    logger.info(f"Loaded native {model_type} from {training_args.model_dir} (load_vision={load_vision})")
    return model, config

def select_text_model(training_args):
    model = AutoModelForCausalLM.from_pretrained(
        training_args.text_model_dir,
        local_files_only=True,
        dtype=(torch.bfloat16 if training_args.bfloat16 else None),
    )
    logger.info(f"Loaded text-only model from {training_args.text_model_dir}")

    return model


def select_vision_model(training_args):
    from transformers import SiglipVisionModel
    model = SiglipVisionModel.from_pretrained(
        training_args.vision_model_dir,
        local_files_only=True,
    )
    logger.info(f"Loaded SigLIP2 vision model from {training_args.vision_model_dir}")
    return model


@torch.no_grad()
def load_vision_model(vlm_model, siglip_model):
    """Surgical weight transfer from SigLIP2 vision encoder into Qwen3-VL vision encoder.

    Key transformations performed:
      1. patch_embed: Conv2d kernel inflated to Conv3d by repeating along the
         temporal axis and dividing by temporal_patch_size, preserving the
         response magnitude for static images.
      2. pos_embed: bilinear interpolation from SigLIP2's 32x32 grid to the
         Qwen3-VL target grid size (e.g. 48x48).
      3. attn.qkv: separate q/k/v projections fused into a single [q;k;v] matrix.
      4. Layer name mapping (layer_norm1→norm1, fc1→linear_fc1, out_proj→proj, …).

    Skipped SigLIP2 weights (no equivalent in Qwen3-VL):
      - vision_model.post_layernorm  (final LayerNorm used for contrastive pooling)
      - vision_model.head.*          (attention-pool head for contrastive training)

    Qwen3-VL-specific weights left untouched (random init, trained from scratch):
      - model.visual.merger.*
      - model.visual.deepstack_merger_list.*
    """
    logger.info("Starting SigLIP2 → Qwen3-VL vision encoder weight surgery...")

    siglip_state = dict(siglip_model.state_dict())
    vlm_state = dict(vlm_model.state_dict())
    loaded_keys: list[str] = []

    def copy_to(vlm_key: str, tensor: torch.Tensor) -> None:
        if vlm_key not in vlm_state:
            logger.warning(f"VLM key not found, skipping: {vlm_key}")
            return
        param = vlm_model.get_parameter(vlm_key)
        if param.shape != tensor.shape:
            raise ValueError(
                f"Shape mismatch for {vlm_key}: model expects {tuple(param.shape)}, "
                f"source has {tuple(tensor.shape)}"
            )
        param.data.copy_(tensor.to(dtype=param.data.dtype))
        loaded_keys.append(vlm_key)

    pe_w = siglip_state["vision_model.embeddings.patch_embedding.weight"].float()
    tgt_pe_key = "model.visual.patch_embed.proj.weight"
    t = vlm_state[tgt_pe_key].shape[2]  # temporal_patch_size
    inflated_pe_w = pe_w.unsqueeze(2).repeat(1, 1, t, 1, 1) / t
    copy_to(tgt_pe_key, inflated_pe_w)
    copy_to(
        "model.visual.patch_embed.proj.bias",
        siglip_state["vision_model.embeddings.patch_embedding.bias"],
    )

    pos_w = siglip_state["vision_model.embeddings.position_embedding.weight"].float()
    src_n, embed_dim = pos_w.shape
    src_g = int(round(src_n ** 0.5))
    tgt_pe_key = "model.visual.pos_embed.weight"
    tgt_n = vlm_state[tgt_pe_key].shape[0]
    tgt_g = int(round(tgt_n ** 0.5))
    if src_g != tgt_g:
        logger.info(f"Interpolating pos_embed: {src_g}x{src_g} → {tgt_g}x{tgt_g}")
        pos_2d = pos_w.reshape(1, src_g, src_g, embed_dim).permute(0, 3, 1, 2)
        pos_2d = F.interpolate(pos_2d, size=(tgt_g, tgt_g), mode="bilinear", align_corners=False)
        pos_w = pos_2d.permute(0, 2, 3, 1).reshape(tgt_n, embed_dim)
    copy_to(tgt_pe_key, pos_w)

    layer_indices = sorted({
        int(m.group(1))
        for k in siglip_state
        if (m := re.match(r"vision_model\.encoder\.layers\.(\d+)\.", k))
    })

    for idx in layer_indices:
        sp = f"vision_model.encoder.layers.{idx}"
        qp = f"model.visual.blocks.{idx}"

        for src_norm, tgt_norm in (("layer_norm1", "norm1"), ("layer_norm2", "norm2")):
            for suffix in ("weight", "bias"):
                copy_to(f"{qp}.{tgt_norm}.{suffix}", siglip_state[f"{sp}.{src_norm}.{suffix}"])

        # fuse the qvk into a single weight
        q_w = siglip_state[f"{sp}.self_attn.q_proj.weight"]
        k_w = siglip_state[f"{sp}.self_attn.k_proj.weight"]
        v_w = siglip_state[f"{sp}.self_attn.v_proj.weight"]
        copy_to(f"{qp}.attn.qkv.weight", torch.cat([q_w, k_w, v_w], dim=0))

        q_b = siglip_state[f"{sp}.self_attn.q_proj.bias"]
        k_b = siglip_state[f"{sp}.self_attn.k_proj.bias"]
        v_b = siglip_state[f"{sp}.self_attn.v_proj.bias"]
        copy_to(f"{qp}.attn.qkv.bias", torch.cat([q_b, k_b, v_b], dim=0))

        copy_to(f"{qp}.attn.proj.weight", siglip_state[f"{sp}.self_attn.out_proj.weight"])
        copy_to(f"{qp}.attn.proj.bias", siglip_state[f"{sp}.self_attn.out_proj.bias"])

        copy_to(f"{qp}.mlp.linear_fc1.weight", siglip_state[f"{sp}.mlp.fc1.weight"])
        copy_to(f"{qp}.mlp.linear_fc1.bias", siglip_state[f"{sp}.mlp.fc1.bias"])
        copy_to(f"{qp}.mlp.linear_fc2.weight", siglip_state[f"{sp}.mlp.fc2.weight"])
        copy_to(f"{qp}.mlp.linear_fc2.bias", siglip_state[f"{sp}.mlp.fc2.bias"])

    if hasattr(vlm_model, "cfg"):
        vis = vlm_model.model.visual
        vc  = vlm_model.cfg.vision
        device = vis.merger.linear_fc1.weight.device

        head_dim_v = vc.hidden_size // vc.num_heads
        rdim       = head_dim_v // 2
        inv_freq   = 1.0 / (
            10000.0 ** (torch.arange(0, rdim, 2, dtype=torch.float32, device=device) / rdim)
        )
        vis.rotary_pos_emb.inv_freq = inv_freq
        logger.info("Recomputed vision rotary_pos_emb.inv_freq.")

        # init the MERGER and DEEPSTACK
        def _init(m: torch.nn.Module) -> None:
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.ones_(m.weight)
                torch.nn.init.zeros_(m.bias)

        vis.merger.apply(_init)
        vis.deepstack_merger_list.apply(_init)
        logger.info("Initialised merger and deepstack_merger_list (Xavier / ones-zeros).")

    logger.info(
        f"Vision surgery complete. Loaded {len(loaded_keys)} tensors "
        f"across {len(layer_indices)} transformer blocks."
    )
    return vlm_model

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

def dist_all_gather(x: torch.Tensor, group) -> torch.Tensor:
    """Gather a 1-D per-rank tensor across `group`.

    Returns a [world_size, x.numel()] tensor available on every rank. This is a
    collective, so it MUST be called on all ranks of `group`.
    """
    x = x.contiguous()
    return funcol.all_gather_tensor(x, gather_dim=0, group=group).reshape(-1, x.numel())

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