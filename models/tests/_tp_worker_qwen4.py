"""Per-rank worker for the Qwen4 TP gradient test.

Runs under torchrun with 1 or 2 ranks. Each rank builds the small Qwen4-Exp
config, random-initializes it with a fixed seed, applies `apply_tp` when the
world has more than one rank, runs one forward/backward on a fixed packed input
and (rank 0) saves every parameter gradient to the path given on the command
line.

Gradients of the weights the TP pass permutes (the fused GDN qkv projection and
the stacked expert `gate_up_proj`) are written back in the single-rank layout,
so the caller can compare them element by element.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen4.config import Qwen4Config
from models.qwen4.model import Qwen4ForCausalLM, set_gdn_backend
from train.config import ModelType
from train.infra import apply_tp
from train.utils import init_qwen4

MODEL_DIR = os.environ.get(
    "QWEN4_615M_DIR", "/data/151-2/users/tockier/models/qwen4_615m"
)


def _unpermute_qkv(t: torch.Tensor, cfg, tp: int) -> torch.Tensor:
    """`[q_r0|k_r0|v_r0|q_r1|...]` (rank-grouped) back to `[q|k|v]`."""
    key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim
    val_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim
    kper, vper = key_dim // tp, val_dim // tp
    q, k, v, off = [], [], [], 0
    for _ in range(tp):
        q.append(t[off : off + kper]); off += kper
        k.append(t[off : off + kper]); off += kper
        v.append(t[off : off + vper]); off += vper
    return torch.cat(q + k + v, dim=0)


def _unpermute_gate_up(t: torch.Tensor, cfg, tp: int) -> torch.Tensor:
    """`(E, [g_r0|u_r0|g_r1|u_r1], H)` back to `(E, [g|u], H)`."""
    per = cfg.moe_intermediate_size // tp
    gate, up, off = [], [], 0
    for _ in range(tp):
        gate.append(t[:, off : off + per]); off += per
        up.append(t[:, off : off + per]); off += per
    return torch.cat(gate + up, dim=1)


def main() -> None:
    out_path = sys.argv[1]
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    cfg = Qwen4Config.from_json(os.path.join(MODEL_DIR, "config.json"))
    # The PLE n-gram table is head-sharded at construction, not by `apply_tp`.
    cfg.text.ple_tp_size = world_size
    cfg.text.ple_tp_rank = rank
    cfg.text.validate()

    torch.manual_seed(0)
    model = Qwen4ForCausalLM(cfg).to(device)
    set_gdn_backend("auto")
    init_qwen4(model)

    # `init_qwen4` deliberately gives each rank's head shard its own draw, so
    # the one- and two-rank runs would otherwise start from different tables.
    # Seed the packed (unsharded) table identically instead; every rank then
    # keeps the rows for the heads it owns. The runtime row layout does not
    # depend on the TP size, so a `full_tensor()` of the sharded gradient is
    # directly comparable with the single-rank one.
    table_gen = torch.Generator(device="cpu").manual_seed(1234)
    for module in model.modules():
        if hasattr(module, "load_packed_table"):
            packed = torch.empty(
                module.packed_rows, module.ngram_embedding.embedding_dim
            ).normal_(0.0, 0.02, generator=table_gen)
            module.load_packed_table(packed.to(device))

    model = model.float()
    model.train()

    if world_size > 1:
        mesh = init_device_mesh(
            "cuda", (1, world_size), mesh_dim_names=("dp", "tp")
        )
        apply_tp(model, ModelType.Qwen4, mesh["tp"], False)

    total = 512
    gen = torch.Generator(device="cpu").manual_seed(7)
    ids = torch.randint(0, 1000, (1, total), generator=gen).to(device)
    cu_seqlens = torch.tensor([0, 256, total], dtype=torch.int32, device=device)

    out = model(input_ids=ids, attention_mask=cu_seqlens, labels=ids.clone())
    out.loss.backward()

    grads = {}
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        grad = param.grad
        if isinstance(grad, DTensor):
            grad = grad.full_tensor()
        grad = grad.detach().float().cpu()
        if world_size > 1:
            if name.endswith("in_proj_qkv.weight") or name.endswith(
                "linear_attn.conv1d.weight"
            ):
                grad = _unpermute_qkv(grad, cfg.text, world_size)
            elif name.endswith("experts.gate_up_proj"):
                grad = _unpermute_gate_up(grad, cfg.text, world_size)
        grads[name] = grad

    if rank == 0:
        torch.save({"loss": out.loss.item(), "grads": grads}, out_path)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
