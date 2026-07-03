#!/usr/bin/env python3
"""Gradio chat interface for Qwen3-VL checkpoints.

Usage:
    python -m utils.chat --model_dir /path/to/checkpoint [--port 7860] [--max_new_tokens 512]

Greedy decode with no KV cache: each step re-runs the full model (including the
full visual injection — merged embeddings AND deepstack features), matching the
computation used in training and by eval/test_suite.py.
"""

import argparse
import re
import torch
import gradio as gr
from transformers import AutoProcessor

parser = argparse.ArgumentParser()
parser.add_argument("--model_dir", required=True, help="Path to converted safetensors checkpoint")
parser.add_argument("--max_new_tokens", type=int, default=512)
parser.add_argument("--port", type=int, default=7860)
args = parser.parse_args()

print(f"Loading model from {args.model_dir} …")
from models.qwen3_vl.model import Qwen3VLForCausalLM

model, cfg = Qwen3VLForCausalLM.from_pretrained(
    args.model_dir, dtype=torch.bfloat16, device="cpu", load_vision=True
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(torch.bfloat16).to(device).eval()

processor = AutoProcessor.from_pretrained(args.model_dir, trust_remote_code=True)
eos_id = processor.tokenizer.eos_token_id
print(f"Ready on {device}.")


def _parse(text: str) -> tuple[str, str]:
    """Split decoded text into (thinking, answer).

    Returns:
        thinking: content inside <think>…</think>, empty string if none.
        answer:   everything outside the think block.
                  During a still-open <think> block the answer is empty.
    """
    # Complete think block present
    m = re.match(r"<think>(.*?)</think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    # Mid-block: <think> opened but not yet closed
    m = re.match(r"<think>(.*)", text, re.DOTALL)
    if m:
        return m.group(1).strip(), ""
    # No thinking at all
    return "", text.strip()


def generate(image, question):
    has_image = image is not None
    has_text = bool(question.strip())
    if not has_image and not has_text:
        yield "", "(no input provided)"
        return

    content = []
    if has_image:
        content.append({"type": "image"})
    if has_text:
        content.append({"type": "text", "text": question.strip()})
    messages = [{"role": "user", "content": content}]

    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    images = [image] if image is not None else None
    inputs = processor(text=[rendered], images=images, padding=False, return_tensors="pt")

    input_ids = inputs["input_ids"].to(device)
    pixel_values = inputs.get("pixel_values")
    image_grid_thw = inputs.get("image_grid_thw")

    if pixel_values is not None:
        pixel_values = pixel_values.to(torch.bfloat16).to(device)
        image_grid_thw = image_grid_thw.to(device)

    # 3D MRoPE position_ids for the prompt.
    cu_seqlens = torch.tensor([0, input_ids.shape[1]], device=device, dtype=torch.int32)
    if image_grid_thw is not None:
        position_ids = model.get_rope_index(input_ids, cu_seqlens, image_grid_thw)
    else:
        S = input_ids.shape[1]
        position_ids = torch.arange(S, device=device).view(1, 1, S).expand(3, 1, S).clone()

    last_pos = int(position_ids.max().item())
    generated_ids = []

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            logits = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                attention_mask=None,
            )

            next_id = int(logits[0, -1].argmax().item())
            generated_ids.append(next_id)

            raw = processor.tokenizer.decode(generated_ids, skip_special_tokens=True)
            thinking, answer = _parse(raw)
            yield thinking, answer

            if next_id == eos_id:
                break

            last_pos += 1
            input_ids = torch.cat(
                [input_ids, torch.tensor([[next_id]], device=device)], dim=1
            )
            new_pos = torch.full((3, 1, 1), last_pos, dtype=torch.int64, device=device)
            position_ids = torch.cat([position_ids, new_pos], dim=2)


with gr.Blocks(title="Qwen3-VL") as demo:
    gr.Markdown(f"## Qwen3-VL Chat\n`{args.model_dir}`")

    with gr.Row():
        with gr.Column(scale=1):
            image_in = gr.Image(
                type="pil",
                label="Image — paste (Ctrl+V) or upload",
                height=380,
            )
        with gr.Column(scale=1):
            text_in = gr.Textbox(
                label="Question",
                placeholder="Describe this image. / What is shown here? / …",
                lines=4,
            )
            with gr.Row():
                submit = gr.Button("Ask", variant="primary")
                stop = gr.Button("Stop", variant="stop")
                clear = gr.ClearButton(variant="secondary", value="Clear")
            output = gr.Textbox(label="Response", lines=10, interactive=False)
            with gr.Accordion("Thinking tokens", open=False) as think_box:
                thinking_out = gr.Textbox(
                    label="",
                    lines=6,
                    interactive=False,
                    placeholder="(none — thinking was suppressed successfully)",
                )

    clear.add([image_in, text_in, output, thinking_out])

    gen1 = submit.click(fn=generate, inputs=[image_in, text_in], outputs=[thinking_out, output])
    gen2 = text_in.submit(fn=generate, inputs=[image_in, text_in], outputs=[thinking_out, output])
    stop.click(fn=None, cancels=[gen1, gen2])

demo.launch(server_port=args.port, server_name="0.0.0.0", share=False, show_error=True)
