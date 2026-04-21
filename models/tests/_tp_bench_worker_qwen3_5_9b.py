"""Per-rank worker for the Qwen3.5 9B TP=2 benchmark.

Launched by ``test_qwen3_5_9b_tp2_bench.py`` under torchrun. Each rank:
  1. Joins the process group and builds a ('dp', 'tp') = (1, 2) mesh.
  2. Loads Qwen3.5 9B text-only onto its GPU.
  3. Applies the new head-sharded TP plan (``apply_tp``).
  4. Runs a few warmup fwd+bwd passes, then measures peak memory and
     the average fwd+bwd time over N timed steps on a packed seq_len
     input.
  5. Rank 0 prints the metrics.

No gradients are actually optimized — backward is run to exercise the
DTensor / Triton kernel paths that matter for training and to obtain a
realistic memory footprint. FLOPs are estimated as ``6 * n_params * tokens``,
following ``train/utils.py:get_dense_model_nparams_and_flops``.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.model import Qwen3_5ForCausalLM
from train.config import ModelType
from train.infra import apply_tp, get_mesh


class _TrainingArgsStub:
    def __init__(self, tp_size: int) -> None:
        self.tp_size = tp_size


def _count_params_replicated(model: torch.nn.Module) -> int:
    """Total parameter count as if the model were un-sharded.

    Multiplies local shards by ``mesh.size()`` to get the global count.
    """
    total = 0
    for p in model.parameters():
        t = p
        if hasattr(t, "to_local"):
            # DTensor — multiply local by the product of shard-dim sizes.
            local = t.to_local()
            placements = t.placements
            mesh = t.device_mesh
            factor = 1
            for dim_idx, placement in enumerate(placements):
                if placement.is_shard():
                    factor *= mesh.size(dim_idx)
            total += local.numel() * factor
        else:
            total += t.numel()
    return total


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    snapshot = os.environ.get(
        "QWEN3_5_SNAPSHOT",
        "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_9b",
    )
    seq_len = int(os.environ.get("BENCH_SEQ_LEN", "4096"))
    warmup = int(os.environ.get("BENCH_WARMUP", "2"))
    timed = int(os.environ.get("BENCH_STEPS", "5"))

    mesh = get_mesh(_TrainingArgsStub(tp_size=world_size), world_size)
    tp_mesh = mesh["tp"]

    if rank == 0:
        print(f"[rank0] world={world_size} device={device} seq_len={seq_len} "
              f"warmup={warmup} timed={timed}", flush=True)
        print(f"[rank0] snapshot={snapshot}", flush=True)

    dist.barrier()
    model = Qwen3_5ForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device=device,
        load_vision=False,
    )
    # Text-only bench: drop the uninitialized vision tower so its weights
    # don't waste ~800 MiB per rank.
    if hasattr(model.model, "visual"):
        del model.model.visual
    torch.cuda.empty_cache()
    model.train()

    apply_tp(model, ModelType.Qwen3_5, tp_mesh, enable_tp_async=False)

    n_params = _count_params_replicated(model)
    flops_per_token = 6 * n_params  # fwd + bwd, matches train/utils.py
    tokens = seq_len  # packed single-row batch

    # Fake packed input: a single sequence of length ``seq_len``.
    torch.manual_seed(0 + rank)
    input_ids = torch.randint(
        0, model.cfg.text.vocab_size, (1, seq_len), device=device, dtype=torch.long
    )
    cu_seqlens = torch.tensor([0, seq_len], device=device, dtype=torch.int32)
    labels = input_ids.clone()

    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    def step():
        out = model(
            input_ids=input_ids,
            attention_mask=cu_seqlens,
            labels=labels,
        )
        out.loss.backward()
        # Zero grads without building an optimizer; we're only benching fwd+bwd.
        for p in model.parameters():
            p.grad = None

    for _ in range(warmup):
        step()

    dist.barrier()
    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    for _ in range(timed):
        step()
    torch.cuda.synchronize(device)
    t1 = time.perf_counter()

    avg_step = (t1 - t0) / timed
    peak_mem_gib = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
    peak_reserved_gib = torch.cuda.max_memory_reserved(device) / (1024 ** 3)

    step_flops = flops_per_token * tokens
    tflops_per_sec = step_flops / avg_step / 1e12
    peak_tflops_per_gpu_l40s_bf16 = 362.0
    mfu = tflops_per_sec / peak_tflops_per_gpu_l40s_bf16 * 100.0

    # Gather per-rank mem so rank 0 can print all.
    mems = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(mems, torch.tensor([peak_mem_gib], device=device))
    reserved = [torch.zeros(1, device=device) for _ in range(world_size)]
    dist.all_gather(reserved, torch.tensor([peak_reserved_gib], device=device))

    if rank == 0:
        print("")
        print("=" * 64)
        print(f"Qwen3.5 9B | TP={world_size} | seq_len={seq_len}")
        print(f"n_params (replicated count) : {n_params/1e9:.2f} B")
        print(f"avg fwd+bwd time            : {avg_step*1000:.1f} ms")
        print(f"TFLOPs/s (per GPU)          : {tflops_per_sec:.1f}")
        print(f"MFU vs L40S bf16 362 TFLOPs : {mfu:.1f}%")
        for r in range(world_size):
            print(f"peak alloc GPU{r}            : {mems[r].item():.2f} GiB "
                  f"(reserved {reserved[r].item():.2f} GiB)")
        print("=" * 64, flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
