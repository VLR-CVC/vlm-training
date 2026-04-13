import torch
import torch.distributed.checkpoint as dcp
from transformers import AutoModelForCausalLM, AutoProcessor
import argparse
import os
import glob

def convert_nested_dcp_batch(base_model_path, checkpoint_dir):
    models_out_dir = os.path.join(checkpoint_dir, "models")
    os.makedirs(models_out_dir, exist_ok=True)

    search_pattern = os.path.join(checkpoint_dir, "checkpoint-step-*")
    checkpoint_dirs = [d for d in glob.glob(search_pattern) if os.path.isdir(d)]
    
    if not checkpoint_dirs:
        print(f"No checkpoints found in {checkpoint_dir} matching 'checkpoint-step-*'")
        return

    print(f"Loading base model from {base_model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
    )
    processor = AutoProcessor.from_pretrained(base_model_path, trust_remote_code=True)
    
    hf_state_dict = model.state_dict()
    NESTING_PREFIX = "model."

    for ckpt_path in sorted(checkpoint_dirs):
        step_name = os.path.basename(ckpt_path)
        step_num = step_name.split('-')[-1]
        output_path = os.path.join(models_out_dir, f"step-{step_num}")
        
        print(f"Processing {step_name} -> {output_path}")

        reader = dcp.FileSystemReader(ckpt_path)
        metadata = reader.read_metadata()
        checkpoint_keys = set(metadata.state_dict_metadata.keys())
        
        load_plan = {}
        for hf_key, tensor in hf_state_dict.items():
            target_key = NESTING_PREFIX + hf_key
            if target_key in checkpoint_keys:
                load_plan[target_key] = tensor

        dcp.load(
            state_dict=load_plan,
            checkpoint_id=ckpt_path,
        )

        restored_state_dict = {}
        for hf_key in hf_state_dict.keys():
            target_key = NESTING_PREFIX + hf_key
            if target_key in load_plan:
                restored_state_dict[hf_key] = load_plan[target_key]
                
        model.load_state_dict(restored_state_dict, strict=False)
        
        processor.save_pretrained(output_path)
        model.save_pretrained(output_path, safe_serialization=True)
        print(f"Saved HF snapshot to {output_path}\n")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to original HF model")
    parser.add_argument("--checkpoint_dir", required=True, help="Path to the directory containing checkpoint folders")
    
    args = parser.parse_args()
    convert_nested_dcp_batch(args.base_model, args.checkpoint_dir)