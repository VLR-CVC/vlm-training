import os

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

import torch
import torch.distributed.checkpoint as DCP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save, torch_save_to_dcp

CHECKPOINT_DIR = "/data/users/tockier/qwen_finetune/checkpoints/checkpoint-step-10"
TORCH_SAVE_CHECKPOINT_DIR = "torch_save_checkpoint.pth"

# convert dcp model to torch.save (assumes checkpoint was generated as above)
dcp_to_torch_save(CHECKPOINT_DIR, TORCH_SAVE_CHECKPOINT_DIR)

state_dict = torch.load(TORCH_SAVE_CHECKPOINT_DIR, map_location="cpu")
model_state_dict = state_dict["model"]

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-VL-3B-Instruct",
)

model.load_state_dict(model_state_dict, strict=False)
model.to("cuda")

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")

messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "image": "test_image.jpg",
            },
            {"type": "text", "text": "whats on the board?"},
        ],
    }
]

# Preparation for inference
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Inference: Generation of the output
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)