import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh

# Import directly from your existing files
from config import Qwen3VLConfig, Qwen3_5TextConfig, Qwen3_5VisionConfig
from model import Qwen3_5ForCausalLM, initialize_missing_weights

def get_debug_config() -> Qwen3VLConfig:
    text_cfg = Qwen3_5TextConfig(
        vocab_size=128, hidden_size=64, moe_intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        head_dim=32, max_position_embeddings=128, rms_norm_eps=1e-6,
        tie_word_embeddings=False, num_experts=4, num_experts_per_tok=2,
        router_aux_loss_coef=0.01, shared_expert_intermediate_size=128,
        layer_types=["linear_attention", "full_attention"], full_attention_interval=2,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_num_key_heads=2,
        linear_num_value_heads=2, linear_value_head_dim=32, mtp_num_hidden_layers=0,
        mtp_use_dedicated_embeddings=False, rope_parameters={"rope_theta": 10000.0, "mrope_section": [16, 16, 16]}
    )
    
    vision_cfg = Qwen3_5VisionConfig(
        depth=1, hidden_size=64, intermediate_size=128, num_heads=2,
        in_channels=3, patch_size=14, temporal_patch_size=2, spatial_merge_size=2,
        num_position_embeddings=1024, out_hidden_size=64, hidden_act="silu",
        deepstack_visual_indexes=[]
    )
    
    return Qwen3VLConfig(
        text=text_cfg, vision=vision_cfg,
        image_token_id=120, video_token_id=121, 
        vision_start_token_id=122, vision_end_token_id=123,
        tie_word_embeddings=False, torch_dtype="bfloat16"
    )

def main():
    # 1. Initialize Distributed Env
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    
    # Set up a 2D mesh: (dp, ep). For this test, we map all ranks to EP.
    dp_size = 1
    ep_size = world_size
    mesh = init_device_mesh("cuda", (dp_size, ep_size), mesh_dim_names=("dp", "ep"))
    
    if rank == 0:
        print(f"Initialized DeviceMesh: {mesh}")

    # 2. Build Model
    cfg = get_debug_config()
    # Note: As you modify model.py to accept DeviceMesh for EP, pass it here
    model = Qwen3_5ForCausalLM(cfg).cuda().bfloat16()
    initialize_missing_weights(model)

    # 3. Dummy Data (Packed varlen format as expected by your forward pass)
    seq_len = 64
    # (1, total) shape expected by your model
    input_ids = torch.randint(0, cfg.text.vocab_size, (1, seq_len), device="cuda")
    
    # attention_mask is used as cu_seqlens in your varlen implementation
    cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device="cuda")
    
    # 4. Forward Pass
    if rank == 0: print("\nStarting Forward Pass...")
    logits, aux_loss = model(
        input_ids=input_ids,
        attention_mask=cu_seqlens
    )
    
    if rank == 0:
        print(f"Forward Success. Logits shape: {logits.shape}, Aux Loss: {aux_loss.item()}")

    # 5. Backward Pass
    if rank == 0: print("Starting Backward Pass...")
    
    # Ensure aux_loss is a scalar for summation
    if aux_loss.dim() > 0:
        aux_loss = aux_loss.sum()
        
    loss = logits.sum() + aux_loss
    loss.backward()

    # Quick check if gradients flowed through the experts
    # Path depends on your current model.py structure, adjust if needed
    expert_grad_exists = False
    for name, param in model.named_parameters():
        if "experts" in name and param.grad is not None:
            expert_grad_exists = True
            break

    if rank == 0:
        print(f"Backward Success. Expert gradients populated: {expert_grad_exists}\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
