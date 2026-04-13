import torch
from transformers import AutoModelForCausalLM, AutoProcessor
import argparse

def chat(model_path):
    print(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    
    print("\nChat started! Type 'quit' to exit.")
    chat_history = []

    while True:
        user_input = input("\nYou: ")
        if user_input.lower() in ['quit', 'exit']:
            break

        chat_history.append({"role": "user", "content": user_input})

        text = processor.apply_chat_template(chat_history, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors="pt").to(model.device)

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