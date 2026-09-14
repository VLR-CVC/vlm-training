"""Worker for test_qwen3_5_tp_attn.py -- run under torchrun, not directly.

Exercises a DecoderLayer of each type under real tensor parallelism, set up the
way the trainer does: params left at ``master_dtype = "float32"`` with
``torch.autocast`` doing the bf16 compute (train/train_qwen.py:229,558).

Prints one RESULT line per layer type. Besides "did it run", it reports the
dtype the attention kernel actually received and which path served it, so a
regression cannot hide by silently falling back to the slow SDPA path.
"""
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import Replicate
from torch.distributed.tensor.parallel import (
    ColwiseParallel, PrepareModuleInput, RowwiseParallel, SequenceParallel,
    parallelize_module,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import models.qwen3_5.model as m5
from models.qwen3_5.config import Qwen3_5TextConfig
from models.qwen3_5.model import DecoderLayer
from train.infra import _shard_gated_delta_net

SEEN = {}
_orig_varlen, _orig_sdpa = m5.varlen_attn, m5._varlen_sdpa


def _probe_varlen(q, k, v, **kw):
    SEEN.update(kernel_dtype=q.dtype, path="varlen_attn")
    return _orig_varlen(q, k, v, **kw)


def _probe_sdpa(q, k, v, cu, causal):
    SEEN.update(kernel_dtype=q.dtype, path="sdpa-fallback")
    return _orig_sdpa(q, k, v, cu, causal)


m5.varlen_attn, m5._varlen_sdpa = _probe_varlen, _probe_sdpa

dist.init_process_group("nccl")
rank, TP = dist.get_rank(), dist.get_world_size()
torch.cuda.set_device(rank)
mesh = init_device_mesh("cuda", (TP,), mesh_dim_names=("tp",))

H, HD, T = 1024, 128, 1024
cfg = Qwen3_5TextConfig(
    vocab_size=4096, hidden_size=H, intermediate_size=2048, num_hidden_layers=4,
    num_attention_heads=8, num_key_value_heads=2, head_dim=HD,
    max_position_embeddings=8192, rms_norm_eps=1e-6, tie_word_embeddings=False,
    layer_types=["full_attention"], full_attention_interval=4,
    linear_conv_kernel_dim=4, linear_key_head_dim=128, linear_num_key_heads=8,
    linear_num_value_heads=16, linear_value_head_dim=128,
    mtp_num_hidden_layers=0, mtp_use_dedicated_embeddings=False,
    rope_parameters={"rope_theta": 10000.0, "rope_type": "default"},
)

ATTN_PLAN = {
    "self_attn": PrepareModuleInput(
        input_kwarg_layouts={"hidden_states": Replicate()},
        desired_input_kwarg_layouts={"hidden_states": Replicate()},
    ),
    "self_attn.q_proj": ColwiseParallel(use_local_output=False),
    "self_attn.k_proj": ColwiseParallel(use_local_output=False),
    "self_attn.v_proj": ColwiseParallel(use_local_output=False),
    "self_attn.q_norm": SequenceParallel(sequence_dim=2),
    "self_attn.k_norm": SequenceParallel(sequence_dim=2),
    "self_attn.o_proj": RowwiseParallel(output_layouts=Replicate()),
    "mlp.gate_proj": ColwiseParallel(),
    "mlp.up_proj": ColwiseParallel(),
    "mlp.down_proj": RowwiseParallel(output_layouts=Replicate()),
}

results = []
for layer_type in ("full_attention", "linear_attention"):
    SEEN.clear()
    torch.manual_seed(0)
    # fp32 params, as with the default master_dtype
    layer = DecoderLayer(cfg, layer_type).to("cuda").to(torch.float32)
    if layer_type == "full_attention":
        parallelize_module(layer, mesh, ATTN_PLAN)
    else:
        _shard_gated_delta_net(layer, mesh, ColwiseParallel, RowwiseParallel)

    x = torch.randn(1, T, H, device="cuda", dtype=torch.float32, requires_grad=True)
    cos = torch.randn(T, HD, device="cuda")
    sin = torch.randn(T, HD, device="cuda")
    cu = torch.tensor([0, T // 2, T], device="cuda", dtype=torch.int32)
    try:
        with torch.autocast("cuda", torch.bfloat16, enabled=True):
            out = layer(x, cos, sin, cu, T // 2)
        out.float().sum().backward()
        torch.cuda.synchronize()
        results.append(
            f"{layer_type} ok kernel_in={SEEN.get('kernel_dtype')} "
            f"path={SEEN.get('path')} finite={torch.isfinite(out).all().item()} "
            f"grad={x.grad is not None and torch.isfinite(x.grad).all().item()}"
        )
    except Exception as exc:
        results.append(f"{layer_type} FAIL {type(exc).__name__}: {str(exc).splitlines()[0][:100]}")

if rank == 0:
    for line in results:
        print("RESULT:", line)
dist.destroy_process_group()
