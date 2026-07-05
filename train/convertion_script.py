import torch
import torch.distributed.checkpoint as dcp
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import argparse

def convert_nested_dcp(base_model_path, checkpoint_path, output_path):
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-2B-Instruct", 
        cache_dir=base_model_path,
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True, 
        device_map="cpu"
    )
    
    hf_state_dict = model.state_dict()

    reader = dcp.FileSystemReader(checkpoint_path)
    metadata = reader.read_metadata()
    checkpoint_keys = set(metadata.state_dict_metadata.keys())

    NESTING_PREFIX = "model."
    ckpt_model_keys = {k for k in checkpoint_keys if k.startswith(NESTING_PREFIX)}

    load_plan = {}
    for hf_key, tensor in hf_state_dict.items():
        target_key = NESTING_PREFIX + hf_key
        if target_key in checkpoint_keys:
            load_plan[target_key] = tensor

    matched = set(load_plan.keys())
    unmapped_ckpt = sorted(ckpt_model_keys - matched)
    missing_hf = sorted({NESTING_PREFIX + k for k in hf_state_dict} - ckpt_model_keys)

    print(f"[convert] checkpoint model tensors : {len(ckpt_model_keys)}")
    print(f"[convert] HF model tensors         : {len(hf_state_dict)}")
    print(f"[convert] matched (will be loaded) : {len(matched)}")

    if not matched:
        raise RuntimeError(
            "No checkpoint tensors matched the HF model keys — the exported model "
            "would be the stock base model. Check the nesting prefix / key naming "
            f"(checkpoint sample: {sorted(ckpt_model_keys)[:3]})."
        )

    if unmapped_ckpt:
        raise RuntimeError(
            f"{len(unmapped_ckpt)} trained checkpoint tensor(s) have no HF "
            "counterpart and would be silently dropped (native<->HF naming drift). "
            "Fix the key mapping before exporting. Examples:\n  "
            + "\n  ".join(unmapped_ckpt[:10])
        )

    if missing_hf:
        print(f"[convert] {len(missing_hf)} HF tensor(s) not in checkpoint (kept as "
              f"base-model init; expected: tied lm_head / rotary buffers):")
        for k in missing_hf[:10]:
            print(f"    {k}")

    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-2B-Instruct", cache_dir=base_model_path, trust_remote_code=True)
    processor.save_pretrained(output_path)

    dcp.load(
        state_dict=load_plan,
        checkpoint_id=checkpoint_path,
    )
    model.save_pretrained(output_path, safe_serialization=True)
    print(f"[convert] saved HF model to {output_path} ({len(matched)} trained tensors loaded)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model", required=True, help="Path to original HF model")
    parser.add_argument("--checkpoint", required=True, help="Path to sharded checkpoint folder")
    parser.add_argument("--output", required=True, help="Path to save converted model")
    
    args = parser.parse_args()
    convert_nested_dcp(args.base_model, args.checkpoint, args.output)
