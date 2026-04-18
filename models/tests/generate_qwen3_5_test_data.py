"""Generate synthetic test data for the Qwen3.5 test suite using the HF model.

Produces `qwen3_5_test_data.pt` next to this script. The file contains gold
logits and token sequences from the HF reference model, which the exhaustive
pytest suite (`test_qwen3_5_exhaustive.py`) loads to avoid re-running the HF
model during testing.

Usage:
    pip install transformers accelerate flash-attn safetensors
    python models/tests/generate_qwen3_5_test_data.py

Environment:
    QWEN3_5_SNAPSHOT  path to the model snapshot (default: same default used
                      by the rest of the test suite)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

SNAPSHOT = os.environ.get(
    "QWEN3_5_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b",
)

OUT_PATH = Path(__file__).with_name("qwen3_5_test_data.pt")

PROMPTS = [
    # short (4–10 tokens)
    "Hello",
    "The sky is blue",
    "One two three",
    # medium (15–30 tokens)
    "The capital of France is Paris, and the capital of Germany is Berlin.",
    "In computer science, a linked list is a linear data structure.",
    "The quick brown fox jumps over the lazy dog near the river bank.",
    # longer (~50 tokens)
    (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. "
        "Ut enim ad minim veniam."
    ),
    (
        "Artificial intelligence has transformed how we interact with "
        "technology. Large language models can generate human-like text, "
        "translate languages, summarize documents, and answer questions."
    ),
    # code-flavoured
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    # numbers / structured
    "The year is 2024. The answer is 42. Pi is approximately 3.14159.",
]

MAX_NEW_TOKENS = 15


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(42)

    print(f"Loading HF Qwen3.5 from {SNAPSHOT} ...")
    from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    hf = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            SNAPSHOT,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        )
        .to(device)
        .eval()
    )

    records: list[dict] = []
    print(f"Generating data for {len(PROMPTS)} prompts ...")

    with torch.no_grad():
        for i, prompt in enumerate(PROMPTS):
            enc = tok(prompt, return_tensors="pt")
            input_ids = enc.input_ids.to(device)
            seq_len = input_ids.shape[1]

            # --- gold logits (forward pass only) ---
            logits = hf(input_ids=input_ids).logits.cpu().float()

            # --- greedy continuation ---
            gen_ids = hf.generate(
                input_ids,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                num_beams=1,
                temperature=1.0,
                top_p=1.0,
                use_cache=False,
            )
            new_tokens = gen_ids[0, seq_len:].cpu()

            records.append(
                {
                    "prompt": prompt,
                    "input_ids": input_ids.cpu(),
                    "hf_logits": logits,
                    "hf_new_tokens": new_tokens,
                }
            )
            print(f"  [{i:2d}] len={seq_len:3d}  "
                  f"generated={tok.decode(new_tokens.tolist(), skip_special_tokens=True)!r}")

    data = {
        "records": records,
        "snapshot": SNAPSHOT,
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    torch.save(data, OUT_PATH)
    print(f"\nSaved {len(records)} records → {OUT_PATH}")


if __name__ == "__main__":
    main()
