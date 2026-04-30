"""Minimal test to isolate all-to-all hang in dispatcher."""
import os, sys
os.chdir("/home/tockier/vlm_training_upstream")
sys.path.insert(0, "/home/tockier/vlm_training_upstream")

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from models.qwen3_5.dispatcher import TokenDispatcher


def main():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    print(f"Rank {rank}/{world_size} started", flush=True)
    dist.barrier()

    # EP=4 test
    ep_size = min(4, world_size)
    dp_size = world_size // ep_size
    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    ep_mesh = mesh["ep"]
    ep_rank = ep_mesh.get_local_rank()

    print(f"Rank {rank}: ep_rank={ep_rank}", flush=True)
    dist.barrier()

    num_experts = 4  # divisible by ep_size=2
    top_k = 2
    num_tokens = 8
    hidden = 16

    dispatcher = TokenDispatcher(num_experts=num_experts, top_k=top_k)
    dispatcher.ep_mesh = ep_mesh

    torch.manual_seed(42 + rank)
    x = torch.randn(num_tokens, hidden, device=device, dtype=torch.bfloat16)
    # selected_experts: [num_tokens, top_k]
    selected = torch.randint(0, num_experts, (num_tokens, top_k), device=device)
    scores = torch.softmax(torch.randn(num_tokens, top_k, device=device), dim=-1).to(torch.bfloat16)

    print(f"Rank {rank}: calling dispatch...", flush=True)
    dist.barrier()

    try:
        routed, counts, meta = dispatcher.dispatch(x, scores, selected)
        print(f"Rank {rank}: dispatch OK, routed shape={routed.shape}, counts={counts}", flush=True)
        dist.barrier()

        # Fake expert output
        routed_out = routed.clone()
        result = dispatcher.combine(routed_out, meta, x)
        print(f"Rank {rank}: combine OK, result shape={result.shape}", flush=True)
        dist.barrier()
        print(f"Rank {rank}: EP=2 test PASSED", flush=True)
    except Exception as e:
        print(f"Rank {rank}: FAILED with {e}", flush=True)
        import traceback; traceback.print_exc()

    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
