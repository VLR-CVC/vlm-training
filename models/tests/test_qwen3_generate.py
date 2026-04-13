"""End-to-end generation parity test: our Qwen3 vs HF Qwen3 on a real prompt.

Greedy-decodes a short completion from both models and asserts the
produced token sequences are identical, and that the decoded strings match.

Run:
    python models/test_qwen3_generate.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from models.qwen3.model_qwen3 import Qwen3ForCausalLM

SNAPSHOT = os.environ.get(
    "QWEN3_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_1_7b",
)

PROMPT = "The capital of France is"
MAX_NEW_TOKENS = 20


@torch.no_grad()
def greedy_generate(model: Qwen3ForCausalLM, input_ids: torch.Tensor, max_new: int) -> torch.Tensor:
    """Simple no-cache greedy decode: re-runs the full prefix each step."""
    ids = input_ids
    for _ in range(max_new):
        logits = model(ids)
        next_tok = logits[:, -1, :].argmax(-1, keepdim=True)
        ids = torch.cat([ids, next_tok], dim=1)
    return ids


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    input_ids = tok(PROMPT, return_tensors="pt").input_ids.to(device)
    prompt_len = input_ids.shape[1]
    print(f"Prompt: {PROMPT!r}  ({prompt_len} tokens)")

    print("Loading our Qwen3 ...")
    ours = Qwen3ForCausalLM.from_pretrained(SNAPSHOT, dtype=torch.float32, device=device).eval()

    print("Loading HF Qwen3 ...")
    hf = AutoModelForCausalLM.from_pretrained(
        SNAPSHOT, torch_dtype=torch.float32, attn_implementation="eager"
    ).to(device).eval()

    our_out = greedy_generate(ours, input_ids, MAX_NEW_TOKENS)
    hf_out = hf.generate(
        input_ids,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        num_beams=1,
        temperature=1.0,
        top_p=1.0,
        use_cache=False,
    )

    our_new = our_out[0, prompt_len:].tolist()
    hf_new = hf_out[0, prompt_len:].tolist()

    our_text = tok.decode(our_new, skip_special_tokens=True)
    hf_text = tok.decode(hf_new, skip_special_tokens=True)

    print(f"\n--- ours ---\n{our_text}\n--- hf ---\n{hf_text}\n")
    print(f"ours tokens: {our_new}")
    print(f"hf   tokens: {hf_new}")

    assert our_new == hf_new, "Generated token ids diverge"
    assert our_text == hf_text, "Decoded text differs"
    print("\n[OK] outputs match")


if __name__ == "__main__":
    main()
