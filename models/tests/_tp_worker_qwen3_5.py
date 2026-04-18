"""Per-rank worker for the Qwen3.5 TP smoke test.

Runs under torchrun. Each rank:
  1. Joins the process group and builds a 2D device mesh (dp=1, tp=world_size).
  2. Loads `Qwen3_5ForCausalLM` without vision weights (text-only smoke).
  3. Applies `apply_tp` from `train.infra`.
  4. Runs a single forward pass on a short text input and exits.

The test only checks that no exception propagates out of the forward pass
— it does NOT check numerical parity. The goal is to catch DTensor/kernel
compatibility regressions (e.g. a Triton kernel that receives a DTensor
by mistake).
"""
from __future__ import annotations

import os
import sys
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


def main() -> None:
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    dist.init_process_group(backend="nccl")

    snapshot = os.environ.get(
        "QWEN3_5_SNAPSHOT",
        "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b",
    )

    # 2D mesh with dp=1 and tp=world_size.
    mesh = get_mesh(_TrainingArgsStub(tp_size=world_size), world_size)
    tp_mesh = mesh["tp"]

    if rank == 0:
        print(f"[rank0] world_size={world_size} device={device}", flush=True)

    model = Qwen3_5ForCausalLM.from_pretrained(
        snapshot,
        dtype=torch.bfloat16,
        device=device,
        load_vision=False,
    ).eval()

    apply_tp(model, ModelType.Qwen3_5, tp_mesh, enable_tp_async=False)

    # Short text prompt — exercises both full_attention and linear_attention
    # decoder layers (alternating pattern inside Qwen3.5).
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(snapshot)
    input_ids = tok("The quick brown fox jumps over the lazy dog", return_tensors="pt").input_ids.to(device)

    with torch.no_grad():
        logits = model(input_ids=input_ids)

    torch.cuda.synchronize(device)

    assert logits.shape[0] == 1, f"unexpected batch dim: {logits.shape}"
    assert logits.shape[1] == input_ids.shape[1], (
        f"unexpected seq dim: logits={logits.shape}, input={input_ids.shape}"
    )

    if rank == 0:
        print(f"[rank0] forward OK, logits shape={tuple(logits.shape)}", flush=True)

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
