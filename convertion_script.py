import torch
import torch.distributed.checkpoint as dcp
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
import argparse
import os

def convert_nested_dcp(base_model_path, checkpoint_path, output_path):
    print(f"1. Initializing base model structure from: {base_model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", 
        cache_dir=base_model_path,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        device_map="cpu"
    )
    
    hf_state_dict = model.state_dict()
    
    print(f"2. Inspecting checkpoint metadata...")
    reader = dcp.FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata.keys())
    
    print(f"   Checkpoint contains {len(checkpoint_keys)} keys.")
    
    load_plan = {}
    mapped_count = 0
    
    NESTING_PREFIX = "model."

    print("3. Mapping keys (adding 'model.' prefix)...")
    
    for hf_key, tensor in hf_state_dict.items():
        target_key = NESTING_PREFIX + hf_key
        if target_key in checkpoint_keys:
            load_plan[target_key] = tensor

    print(f"   Mapped {mapped_count} / {len(hf_state_dict)} tensors.")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct", cache_dir=base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)
    
    print("4. Loading weights from disk...")
    # dcp.load will read 'target_key' from disk and write into 'tensor' (which is linked to 'model')
    dcp.load(
        state_dict=load_plan,
        checkpoint_id=checkpoint_path,
    )
    
    # 5. Save Result
    print(f"5. Saving Hugging Face model to: {output_path}")
    model.save_pretrained(output_path, safe_serialization=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to original HF model")
    parser.add_argument("--checkpoint", required=True, help="Path to sharded checkpoint folder")
    parser.add_argument("--output", required=True, help="Path to save converted model")
    
    args = parser.parse_args()
    convert_nested_dcp(args.base_model, args.checkpoint, args.output)