"""Step-by-step test to find hang in MoeEPOnly forward/backward."""
import os, sys
os.chdir("/home/tockier/vlm_training_upstream")
sys.path.insert(0, "/home/tockier/vlm_training_upstream")

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

torch.backends.cuda.matmul.allow_tf32 = True

from models.qwen3_5.config import Qwen3_5TextConfig
from models.qwen3_5.model import TopKRouter, MoeMLP, MoeExperts
from models.qwen3_5.dispatcher import TokenDispatcher


def get_cfg():
    return Qwen3_5TextConfig(
        vocab_size=1024, hidden_size=256, moe_intermediate_size=512,
        num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
        head_dim=64, max_position_embeddings=512, rms_norm_eps=1e-6,
        tie_word_embeddings=False, num_experts=8, num_experts_per_tok=2,
        router_aux_loss_coef=0.01, shared_expert_intermediate_size=512,
        layer_types=["linear_attention"], full_attention_interval=2,
        linear_conv_kernel_dim=4, linear_key_head_dim=64, linear_num_key_heads=2,
        linear_num_value_heads=4, linear_value_head_dim=64, mtp_num_hidden_layers=0,
        mtp_use_dedicated_embeddings=False,
        rope_parameters={"rope_theta": 10000.0, "mrope_section": [16, 16, 16]}
    )


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    ep_size = world_size
    dp_size = 1
    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    ep_mesh = mesh["ep"]
    ep_rank = ep_mesh.get_local_rank()

    cfg = get_cfg()
    cfg.num_experts = ep_size * 2  # 8 with ep_size=4

    num_local = cfg.num_experts // ep_size

    if rank == 0: print(f"[step 1] Building modules...", flush=True)
    dist.barrier()

    gate = TopKRouter(cfg).to(device).to(torch.bfloat16)
    nn.init.normal_(gate.weight, std=0.02)

    # Local experts only
    experts_gate_up = nn.Parameter(
        torch.empty(num_local, 2 * cfg.moe_intermediate_size, cfg.hidden_size, device=device, dtype=torch.bfloat16)
    )
    experts_down = nn.Parameter(
        torch.empty(num_local, cfg.hidden_size, cfg.moe_intermediate_size, device=device, dtype=torch.bfloat16)
    )
    nn.init.normal_(experts_gate_up, std=0.02)
    nn.init.normal_(experts_down, std=0.02)

    shared_expert = MoeMLP(cfg, cfg.shared_expert_intermediate_size).to(device).to(torch.bfloat16)
    shared_gate = nn.Linear(cfg.hidden_size, 1, bias=False, device=device, dtype=torch.bfloat16)
    nn.init.normal_(shared_gate.weight, std=0.02)

    dispatcher = TokenDispatcher(num_experts=cfg.num_experts, top_k=cfg.num_experts_per_tok)
    dispatcher.ep_mesh = ep_mesh

    if rank == 0: print(f"[step 2] Modules built. Running forward...", flush=True)
    dist.barrier()

    torch.manual_seed(42 + rank)
    seq_len = 128
    hidden = torch.randn(1, seq_len, cfg.hidden_size, device=device, dtype=torch.bfloat16, requires_grad=True)
    x = hidden.view(-1, cfg.hidden_size)

    if rank == 0: print(f"[step 3] Router forward...", flush=True)
    dist.barrier()
    router_logits, routing_weights, selected_experts = gate(x)
    if rank == 0: print(f"[step 4] Dispatch...", flush=True)
    dist.barrier()

    routed_input, num_tokens_per_expert, metadata = dispatcher.dispatch(x, routing_weights, selected_experts)
    if rank == 0: print(f"[step 5] Dispatch done. Routed shape={routed_input.shape}, counts={num_tokens_per_expert}", flush=True)
    dist.barrier()

    # Expert compute
    if rank == 0: print(f"[step 6] Expert compute...", flush=True)
    import torch.nn.functional as F
    num_local_experts = experts_gate_up.shape[0]
    offsets = torch.zeros(num_local_experts + 1, dtype=torch.long, device=device)
    offsets[1:] = num_tokens_per_expert.long().cumsum(0)
    outputs = []
    for i in range(num_local_experts):
        s, e = int(offsets[i]), int(offsets[i+1])
        if e > s:
            chunk = routed_input[s:e]
            gu = F.linear(chunk, experts_gate_up[i])
            g, u = gu.chunk(2, dim=-1)
            outputs.append(F.linear(F.silu(g) * u, experts_down[i]))
    routed_out = torch.cat(outputs, dim=0) if outputs else routed_input.new_zeros(0, cfg.hidden_size)
    if rank == 0: print(f"[step 7] Expert done. Combine...", flush=True)
    dist.barrier()

    expert_output = dispatcher.combine(routed_out, metadata, x)
    if rank == 0: print(f"[step 8] Combine done. expert_output shape={expert_output.shape}", flush=True)
    dist.barrier()

    shared_out = F.sigmoid(shared_gate(x)) * shared_expert(x)
    final = (expert_output + shared_out).reshape(1, seq_len, cfg.hidden_size)

    if rank == 0: print(f"[step 9] Backward...", flush=True)
    dist.barrier()

    loss = final.sum()
    loss.backward()
    if rank == 0: print(f"[step 10] Backward done!", flush=True)
    dist.barrier()

    if rank == 0: print("[PASSED] All steps completed.", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
