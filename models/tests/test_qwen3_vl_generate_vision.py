"""Greedy multimodal generation parity.

Feeds a real image + "describe this image" prompt through the HF processor,
greedy-decodes N tokens with both our model and HF, and checks that the
produced token sequences match.

Run:
    python models/tests/test_qwen3_vl_generate_vision.py
    # or point at a different image:
    IMAGE_PATH=/path/to/img.jpg python models/tests/test_qwen3_vl_generate_vision.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.model import Qwen3_5ForCausalLM

SNAPSHOT = os.environ.get(
    "QWEN3VL_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b",
)
IMAGE_PATH = os.environ.get(
    "IMAGE_PATH",
    "/home/tockier/vlm-training/benchmarks_float8/qwen_benchmark_plot.png",
)
MAX_NEW_TOKENS = int(os.environ.get("MAX_NEW_TOKENS", "16"))


def greedy_generate_ours(model, input_ids, pixel_values, image_grid_thw, max_new_tokens):
    generated = input_ids.clone()
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits = model(
                input_ids=generated,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
            )
        next_tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_tok], dim=1)
    return generated


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    image = Image.open(IMAGE_PATH).convert("RGB")
    max_side = 384
    if max(image.size) > max_side:
        w, h = image.size
        scale = max_side / max(w, h)
        image = image.resize((int(w * scale), int(h * scale)))
    print(f"Loaded image {IMAGE_PATH}  size={image.size}")

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(SNAPSHOT)
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Describe this image in one sentence."},
            ],
        }
    ]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)

    input_ids = inputs["input_ids"]
    pixel_values = inputs["pixel_values"].to(torch.bfloat16)
    image_grid_thw = inputs["image_grid_thw"]
    mm_token_type_ids = inputs.get("mm_token_type_ids")

    print(f"input_ids shape = {tuple(input_ids.shape)}")
    print(f"image_grid_thw  = {image_grid_thw.tolist()}")
    print(f"pixel_values    = {tuple(pixel_values.shape)}")

    print(f"Loading our Qwen3-VL from {SNAPSHOT} ...")
    ours = Qwen3_5ForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, device=device, load_vision=True
    ).eval()

    print("Loading HF Qwen3-VL ...")
    hf = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            SNAPSHOT, torch_dtype=torch.bfloat16, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )

    print(f"Greedy decoding {MAX_NEW_TOKENS} tokens ...")
    with torch.no_grad():
        hf_out = hf.generate(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            mm_token_type_ids=mm_token_type_ids,
            do_sample=False,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=1,
            use_cache=True,
        )
    ours_out = greedy_generate_ours(
        ours, input_ids, pixel_values, image_grid_thw, MAX_NEW_TOKENS
    )

    hf_new = hf_out[0, input_ids.shape[1]:].tolist()
    ours_new = ours_out[0, input_ids.shape[1]:].tolist()

    print("HF   :", hf_new)
    print("ours :", ours_new)

    tok = processor.tokenizer
    print("HF   text:", tok.decode(hf_new, skip_special_tokens=False))
    print("ours text:", tok.decode(ours_new, skip_special_tokens=False))

    assert hf_new == ours_new, f"token mismatch\nHF  : {hf_new}\nours: {ours_new}"
    print("[OK] vision generation parity")


if __name__ == "__main__":
    main()
