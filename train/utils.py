import re
import torch
import torch.nn.functional as F
import math
from torch.optim.lr_scheduler import LambdaLR
from transformers import (
    AutoModelForCausalLM, # use for text-only model loading
)

import os
import gc
import time
import random
import contextlib

from train.logger import logger

import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor import DTensor

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
    model = model.model

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    torch.manual_seed(42)
    model.model.visual.merger.apply(init_weights)
    model.model.visual.deepstack_merger_list.apply(init_weights)

    for param in model.model.visual.merger.parameters():
        torch.distributed.broadcast(param.data, src=0)
    for param in model.visual.deepstack_merger_list.parameters():
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

def set_model_titan(model_args: ModelArgs, model):
    """Freezing for `models/qwen3_5_tt` and `models/qwen3_vl_tt`, same policy as
    `set_model_qwen3_5` / `set_model_qwen3vl` under the torchtitan module names. Call
    on the meta model, before sharding."""
    for n, p in model.named_parameters():
        if n.startswith(("vision_encoder.merger.", "vision_encoder.deepstack_mergers.")):
            p.requires_grad = model_args.train_mlp
        elif n.startswith("vision_encoder."):
            p.requires_grad = model_args.train_vit
        else:
            p.requires_grad = model_args.train_llm
    return model

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

    # QSA indexer: non-differentiable by construction, see the module docstring.
    n_idx = 0
    for n, p in model.named_parameters():
        if ".indexer." in n:
            p.requires_grad = False
            n_idx += 1
    if n_idx:
        logger.info(f"QSA indexer frozen: {n_idx} tensors take no gradient")

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
    logger.info(f'using model: {model_args.model_name} (native impl)')

    if not os.path.exists(training_args.model_dir):
        raise ValueError(f"path with model does not exists, got: {training_args.model_dir}")

    load_vision = not getattr(training_args, "load_vision_model", False)
    return _select_native_model_class(training_args, model_type, load_vision=load_vision)
    # return: model, config

def _select_native_model_class(training_args: TrainArgs, model_type: ModelType, load_vision: bool = True):
    """Dispatch to our torch-native model implementations under `models/`."""
    dtype = MASTER_DTYPES[getattr(training_args, "master_dtype", "float32")]

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

    kwargs = {}
    if getattr(training_args, "random_init", False) and model_type is ModelType.Qwen3_5:
        # config.json is enough; see `from_pretrained(load_weights=...)`
        kwargs["load_weights"] = False
        logger.info(
            f"random_init: building {model_type} from config.json only, no "
            "safetensors read (architecture ablations have no matching checkpoint)"
        )

    model, config = NativeQwen3.from_pretrained(
        training_args.model_dir,
        dtype=dtype,
        device="cpu",
        load_vision=load_vision,
        **kwargs,
    )
    logger.info(f"Loaded native {model_type} from {training_args.model_dir} (load_vision={load_vision})")
    return model, config

def select_text_model(training_args):
    model = AutoModelForCausalLM.from_pretrained(
        training_args.text_model_dir,
        local_files_only=True,
        # storage, so it follows the master weights (see `cast_master_weights`)
        dtype=MASTER_DTYPES[getattr(training_args, "master_dtype", "float32")],
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

def dist_sum_max(sums: torch.Tensor, mx: torch.Tensor, mesh):
    """One SUM all-reduce and one MAX all-reduce over `mesh`, no host sync.

    This replaces five separate `dist_mean`/`dist_max`/`dist_sum` calls, each of
    which was typed `-> float` and therefore ended in `.item()`. The collectives
    themselves were never the expensive part -- a handful of small all-reduces
    is microseconds. The `.item()` was: it forces the CPU to wait for the GPU to
    reach that point in the stream, which on a launch-bound model means the CPU
    stops running ahead and the GPU starts going idle between kernels.

    Returns device tensors. The caller is expected to stage them to pinned
    memory and read them on a later step. A mean is a sum divided by the group
    size on the host side; there is no reason to spend a separate collective on
    `ReduceOp.AVG`.
    """
    return (
        funcol.all_reduce(sums, reduceOp=c10d.ReduceOp.SUM.name, group=mesh),
        funcol.all_reduce(mx, reduceOp=c10d.ReduceOp.MAX.name, group=mesh),
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
    
    if training_args.wsd_decay_steps > 0:
        decay_steps = training_args.wsd_decay_steps
    else:
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


def build_optimizer_param_groups(named_parameters, lr_by_group: dict, weight_decay: float, log: bool = False):
    """Bucket trainable params by part (mlp / vit / llm) and by weight-decay
    eligibility, returning the ``optimizer_grouped_parameters`` list for AdamW.

    Only >=2D tensors (weight matrices, embeddings) get weight decay; biases and
    norm scales are 1D and must never be pulled toward zero.
    """
    groups = ("mlp", "vit", "llm")
    decay_params = {g: [] for g in groups}
    no_decay_params = {g: [] for g in groups}

    for n, p in named_parameters:
        if not p.requires_grad:
            continue
        if "visual.merger" in n or "visual.deepstack_merger_list" in n or "vision_encoder.merger" in n:
            group = "mlp"
        elif "visual.patch_embed" in n or "visual.blocks" in n or "vision_encoder." in n:
            group = "vit"
        else:
            group = "llm"

        (decay_params if p.dim() >= 2 else no_decay_params)[group].append(p)

    if log:
        for group in groups:
            logger.info(
                f"optimizer group {group} (lr={lr_by_group[group]}) -> "
                f"decay:{len(decay_params[group])} (wd={weight_decay}) "
                f"no_decay:{len(no_decay_params[group])} (wd=0.0)"
            )

    param_groups = []
    for group in groups:
        if decay_params[group]:
            param_groups.append({
                "params": decay_params[group],
                "lr": lr_by_group[group],
                "weight_decay": weight_decay,
            })
        if no_decay_params[group]:
            param_groups.append({
                "params": no_decay_params[group],
                "lr": lr_by_group[group],
                "weight_decay": 0.0,
            })
    return param_groups


def clip_grad_norm_mixed(parameters, max_norm: float, norm_type: float = 2.0):
    """`clip_grad_norm_` for a model whose grads live on more than one mesh."""
    params = [p for p in parameters if p.grad is not None]
    if not params:
        return torch.zeros(())

    # grouped by the mesh the *gradient* lives on, but keyed to the parameters:
    # `clip_grads_with_norm_` below takes parameters, not gradients.
    groups: dict[object, list[torch.Tensor]] = {}
    for p in params:
        key = p.grad.device_mesh if isinstance(p.grad, DTensor) else None
        groups.setdefault(key, []).append(p)

    if len(groups) == 1:
        # common case, no DTensors
        return torch.nn.utils.clip_grad_norm_(params, max_norm, norm_type)

    # Multi-mesh case. Everything below stays on device: this used to do
    # `float(n.item())` once per group, which is a host sync in the middle of
    # the optimizer step, plus a `full_tensor()` all-gather per group. The
    # caller now stages the returned tensor and reads it a step later.
    total_sq = None
    for ps in groups.values():
        n = torch.nn.utils.get_total_norm([p.grad for p in ps], norm_type)
        if isinstance(n, DTensor):
            n = n.full_tensor()
        n_sq = n.to(torch.float32).pow(norm_type)
        total_sq = n_sq if total_sq is None else total_sq + n_sq

    total_norm = total_sq.pow(1.0 / norm_type)

    if max_norm > 0:
        # upstream's own clipping half, so the DTensor handling matches what
        # `clip_grad_norm_` does on the single-mesh path above. It clamps the
        # scale to <= 1 on device, so the `total_norm > max_norm` test that
        # used to gate this no longer needs a host value.
        for ps in groups.values():
            torch.nn.utils.clip_grads_with_norm_(ps, max_norm, total_norm)

    return total_norm


def zero_grads_if_nonfinite_(parameters, total_norm) -> torch.Tensor:
    """Zero every gradient when `total_norm` is not finite. Returns 1.0 if so.

    One non-finite gradient anywhere is all-reduced to every rank, multiplied
    through `clip_grads_with_norm_`, and written into the weights *and the
    moments* -- after which every later step is `nan`. Measured: 64 nodes died
    at step 25, 256 nodes at step 13, 16 nodes at step 16 and at step 283 on a
    different dataset, with the loss descending normally right up to the step
    that killed it. Zeroing here leaves the weights and the moment history
    intact, so the next step trains normally.

    `total_norm` is already a global reduction, so every rank sees the same
    value and takes the same branch. No host sync, and no rank-divergent
    control flow -- `train_qwen.py` notes that "a collective one rank skips is
    a hang", which is why this cannot be an `if` on a host-side float.

    `masked_fill_` rather than scaling by `isfinite(total_norm)`: `nan * 0.0`
    is `nan`, so the multiply would leave the poison exactly where it was.

    The optimizer still steps. With zero gradients that means the moments decay
    and the parameters take one momentum-only step, which is what a
    zero-gradient step has always meant to Adam -- not a no-op, but not a
    corruption either. A true skip would need the host to know, i.e. a sync.
    """
    n = total_norm
    if isinstance(n, DTensor):
        n = n.full_tensor()
    bad = ~torch.isfinite(torch.as_tensor(n).reshape(()))

    for p in parameters:
        if p.grad is None:
            continue
        g = p.grad
        local = g.to_local() if isinstance(g, DTensor) else g
        local.masked_fill_(bad, 0.0)

    return bad.to(torch.float32)


def round_max_seqlen(n: int) -> int:
    """Round a varlen `max_seqlen` up to a power of two.

    `max_seqlen` is a Python int, and dynamo specialises on int *values*
    (`specialize_int` being False makes them dynamic only in some positions, not
    here). It is the longest document in a packed row, so it takes a different
    value almost every step: tlparse on job 1781928 reported
    `last reason: max_seqlen == 786` against both decoder-layer frames, 32
    recompiles and then eager fallback.

    The flash/varlen kernels use it to size the block grid and only require an
    upper bound -- blocks past a sequence's real end exit immediately. Rounding
    to a power of two collapses the value space to about seven possibilities,
    all of which stay cached, for under 2x of launched-but-idle blocks.
    """
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


MASTER_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16}


def cast_master_weights(model, master_dtype: str) -> torch.dtype:
    """Cast the optimizer's master copy of the *parameters* to ``master_dtype``.
    It only casts the weights and not other parameters/buffer (e.g. RoPE)"""
    if master_dtype not in MASTER_DTYPES:
        raise ValueError(
            f"master_dtype must be one of {sorted(MASTER_DTYPES)}, got {master_dtype!r}"
        )
    dtype = MASTER_DTYPES[master_dtype]
    for param in model.parameters():
        param.data = param.data.to(dtype)
    return dtype


# torchao's AdamW variants, by the `adamw_impl` name that selects them. These are
# the only implementations that can do `adamw_stochastic_round` -- torch.optim.AdamW
# always rounds to nearest, which silently drops updates smaller than ~2^-9 of the
# weight once `master_dtype = "bfloat16"`.
#
# "torchao" is the *unquantized* one. `_AdamW` passes `block_size=inf`, so the
# `local_p.numel() % block_size == 0` test in `_new_buffer` never holds and the
# moments fall through to a plain `torch.zeros_like` in the parameter dtype. No
# `OptimState8bit` subclass, hence none of the `AttributeError: 'Tensor' object has
# no attribute 'codes'` that killed AdamW8bit on a dp x tp mesh.
TORCHAO_ADAMW = {
    "torchao": "_AdamW",
    "fp8": "AdamWFp8",
    "8bit": "AdamW8bit",
    "4bit": "AdamW4bit",
}
# "foreach_sr" is ours: `train/adamw_sr.py`. Same update as torchao's `_AdamW`
# and the same stochastic rounding, but bucketed through flat fp32 scratch
# instead of a Python loop with one compiled call per parameter -- ~18 kernel
# launches per bucket rather than ~500 per step. It is the only implementation
# that is both fast and safe at `master_dtype = "bfloat16"`.
ADAMW_IMPLS = ("foreach_sr", "foreach", "fused", "forloop", *TORCHAO_ADAMW)

def build_adamw(param_groups, lr: float, weight_decay: float, impl: str,
                stochastic_round: bool = False, betas=(0.9, 0.999), eps: float = 1e-8):
    """AdamW over ``param_groups``, picking the implementation by name."""
    if impl not in ADAMW_IMPLS:
        raise ValueError(
            f"adamw_impl must be one of {ADAMW_IMPLS}, got {impl!r}"
        )

    if impl == "foreach_sr":
        from train.adamw_sr import AdamWSR

        return AdamWSR(
            param_groups,
            lr=lr,
            betas=tuple(betas),
            eps=eps,
            weight_decay=weight_decay,
            stochastic_round=stochastic_round,
        )

    if impl in TORCHAO_ADAMW:
        import torchao.optim as ao_optim

        return getattr(ao_optim, TORCHAO_ADAMW[impl])(
            param_groups,
            lr=lr,
            weight_decay=weight_decay,
            bf16_stochastic_round=stochastic_round,
        )

    if stochastic_round:
        # warn, not raise: `adamw_stochastic_round` defaults to true, so every
        # config that picks a torch.optim impl would otherwise fail to build.
        logger.warning(
            f"adamw_stochastic_round ignored: adamw_impl={impl!r} is "
            "torch.optim.AdamW, which always rounds to nearest. Use "
            f"'foreach_sr' or one of {sorted(TORCHAO_ADAMW)} to get stochastic "
            "rounding. With master_dtype='bfloat16' this is not a preference: "
            "at lr 2e-5 a weight of typical magnitude never moves."
        )

    return torch.optim.AdamW(
        param_groups,
        lr=lr,
        betas=tuple(betas),
        eps=eps,
        foreach=impl == "foreach",
        fused=impl == "fused",
        weight_decay=weight_decay,
    )

# column order produced by the perf gather; True == higher is better
PERF_METRIC_NAMES = ("tps", "step_time", "fwd_bwd_time", "tflops", "mfu", "mem_gib")
PERF_HIGHER_IS_BETTER = (True, False, False, True, True, False)

def topk_metrics(gathered, top_k: int) -> dict:
    """Build the ``perf_topk/*`` dict: K slowest and K fastest ranks per metric,
    from the per-rank rows gathered across the world."""
    metrics = {}
    # One device-to-host copy instead of 96. The loop below ends in `.item()`
    # per value (6 metrics x k x 4), and on a CUDA tensor each of those drains
    # the stream. The gathered rows are a few KiB; topk over them on the host
    # is free.
    gathered = gathered.cpu()
    k = min(top_k, gathered.shape[0])
    for j, name in enumerate(PERF_METRIC_NAMES):
        col = gathered[:, j]
        higher_better = PERF_HIGHER_IS_BETTER[j]
        worst_v, worst_i = torch.topk(col, k, largest=not higher_better)
        best_v, best_i = torch.topk(col, k, largest=higher_better)
        for r in range(k):
            metrics[f"perf_topk/{name}_slow_{r}"] = worst_v[r].item()
            metrics[f"perf_topk/{name}_slow_{r}_rank"] = int(worst_i[r].item())
            metrics[f"perf_topk/{name}_fast_{r}"] = best_v[r].item()
            metrics[f"perf_topk/{name}_fast_{r}_rank"] = int(best_i[r].item())
    return metrics