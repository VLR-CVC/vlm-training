import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
import argparse
import requests
from PIL import Image

def chat(model_path):
    print(f"Loading model from {model_path}...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("\nChat started! Type 'quit' to exit.")
    print("Tip: Start your message with an http(s) link to include an image.")
    
    chat_history = []
    session_images = [] # Required to persist images across multi-turn processor calls

    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break

        user_content = []
        parts = user_input.split(" ", 1)
        
        # Check if the first token is an image URL
        if parts[0].startswith("http"):
            try:
                img = Image.open(requests.get(parts[0], stream=True).raw).convert("RGB")
                session_images.append(img)
                user_content.append({"type": "image"})
                # Extract remaining text, or leave empty if it's an image-only prompt
                user_input = parts[1] if len(parts) > 1 else ""
            except Exception as e:
                print(f"Failed to load image: {e}")
                continue

        # Append text content (or empty text if model requires text placeholder)
        if user_input or not user_content:
            user_content.append({"type": "text", "text": user_input})

        chat_history.append({"role": "user", "content": user_content})

        text = processor.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        
        # Unpack conditionally based on whether the session has images
        processor_kwargs = {"text": [text], "return_tensors": "pt"}
        if session_images:
            processor_kwargs["images"] = session_images
            
        inputs = processor(**processor_kwargs).to(model.device)

        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        
        response = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        
        print(f"Model: {response}")
        chat_history.append({"role": "assistant", "content": [{"type": "text", "text": response}]})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="Path to the saved step folder")
    args = parser.parse_args()
    
    with torch.no_grad():
        chat(args.model_path)
