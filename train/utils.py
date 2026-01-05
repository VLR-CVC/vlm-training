import torch
from transformers import (
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Qwen3VLForConditionalGeneration,
    Qwen3VLMoeForConditionalGeneration
)

import time
import gc
from pathlib import Path
import contextlib

from torchtitan.tools.logging import logger

import torch.distributed._functional_collectives as funcol
import torch.distributed.distributed_c10d as c10d
from torch.distributed.tensor import DTensor

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
        

def get_peak_flops(device_name: str) -> int:
    return


def select_model_class(model_args, data_args, training_args, attn_implementation):
    print(f'using model: {model_args.model_name}')

    model_name = model_args.model_name.lower()

    if attn_implementation is None:
        attn_implementation = ""


    if "qwen3" in model_name:
        model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_args.model_name,
            cache_dir=training_args.cache_dir,
            #attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen3vl"

    elif "qwen2.5" in model_name:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_args.model_name,
            cache_dir=training_args.cache_dir,
            #attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen2.5vl"

    elif "qwen2" in model_name:
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_args.model_name,
            cache_dir=training_args.cache_dir,
            #attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen2vl"

    elif "qwen3" in model_name and "a" in Path(model_name.rstrip("/")).name.lower():
        model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
            model_args.model_name,
            cache_dir=training_args.cache_dir,
            #attn_implementation=attn_implementation,
            dtype=(torch.bfloat16 if training_args.bfloat16 else None),
        )
        data_args.model_type = "qwen3vl"
        raise NotImplementedError("Qwen3vl-moe finetune is not supported yet.")

    else:
        raise ValueError(f"Unsupported model: {model_args.model_name_or_path}")

    return model, data_args


def _dist_reduce(
    x: torch.Tensor,
    reduceOp: str,
    mesh,
) -> float:
    """Perform distributed reduction on a tensor.

    Args:
        x (torch.Tensor): Input tensor.
        reduceOp (str): Reduce operation to perform.
        mesh (DeviceMesh): Device mesh to use for reduction.
        extra_pg (dist.ProcessGroup, optional): Extra process group to use for reduction.
            Defaults to None. If provided, this all_reduce will be called for the extra
            process group, and then the result will be all_reduced for the mesh.
    """
    if isinstance(x, DTensor):
        # functional collectives do not support DTensor inputs
        x = x.full_tensor()

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
