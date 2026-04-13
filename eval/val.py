import os
import torch
import torch.distributed as dist
import math
import argparse
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AutoModelForCausalLM, AutoProcessor

from data.energon_dataloader import QwenTextEncoder
from megatron.energon import get_val_dataset, get_loader, WorkerConfig

import numpy as np
import random

@torch.no_grad()
def evaluate(model_path, data_path, batch_size=14, limit_samples=None):
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    dist.init_process_group(backend="nccl")
    local_rank = int(os.environ["LOCAL_RANK"])
    global_rank = dist.get_rank()
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if global_rank == 0:
        print(f"Loading model from {model_path}...")
        
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map={"": local_rank}  # Overrides the auto spreading
    )
    model = DDP(model, device_ids=[local_rank])
    model.eval()
    
    processor = AutoProcessor.from_pretrained(model_path)

    task_encoder = QwenTextEncoder(
        tokenizer=processor,
        max_len=2048,
    )

    val_ds = get_val_dataset(
        data_path,
        batch_size=batch_size,
        max_samples_per_sequence=1,
        task_encoder=task_encoder,
        worker_config=WorkerConfig.default_worker_config()
    )
    val_loader = get_loader(val_ds)

    total_loss = torch.tensor(0.0, device=device)
    num_batches = torch.tensor(0.0, device=device)

    max_batches = None
    if limit_samples:
        world_size = dist.get_world_size()
        global_batch_size = batch_size * world_size
        max_batches = max(1, limit_samples // global_batch_size)
        if global_rank == 0:
            print(f"Limiting to {limit_samples} global samples ({max_batches} local batches).")

    if global_rank == 0:
        print("Starting evaluation...")
        
    for batch in val_loader:
        if max_batches is not None and num_batches >= max_batches:
            break
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

        total_loss += outputs.loss
        num_batches += 1

    dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    dist.all_reduce(num_batches, op=dist.ReduceOp.SUM)

    if global_rank == 0:
        if num_batches == 0:
            print("Error: Validation dataloader was empty.")
        else:
            avg_loss = (total_loss / num_batches).item()
            perplexity = math.exp(avg_loss) if avg_loss < 20 else float('inf')

            print("\n" + "="*30)
            print(f"Validation Loss: {avg_loss}")
            print(f"Perplexity:      {perplexity}")
            print("="*30)
            
    dist.destroy_process_group()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to HF model step directory")
    parser.add_argument("--data_path", default="/gpfs/scratch/ehpc543/c4_ca", help="Path to Energon dataset root")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--limit_samples", type=int, default=None)
    
    args = parser.parse_args()
    evaluate(args.model_path, args.data_path, args.batch_size, args.limit_samples)