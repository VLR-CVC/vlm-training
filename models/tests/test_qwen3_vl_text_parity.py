"""Stage A parity: our Qwen3-VL text backbone vs HF Qwen3VLForConditionalGeneration
on a text-only input.

Run:
    python models/tests/test_qwen3_vl_text_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_vl.model import Qwen3VLForCausalLM

SNAPSHOT = os.environ.get(
    "QWEN3VL_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_2b",
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ids = tok(
        "The capital of France is Paris, and the capital of Germany is",
        return_tensors="pt",
    ).input_ids.to(device)

    print(f"Loading our Qwen3-VL (text only) from {SNAPSHOT} ...")
    ours = Qwen3VLForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, device=device, load_vision=False
    ).eval()

    print("Loading HF Qwen3-VL ...")
    from transformers import Qwen3VLForConditionalGeneration

    hf = Qwen3VLForConditionalGeneration.from_pretrained(
        SNAPSHOT, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to(device).eval()

    with torch.no_grad():
        a = ours(input_ids=ids).float()
        b = hf(input_ids=ids).logits.float()

    print(f"ours shape={tuple(a.shape)}  hf shape={tuple(b.shape)}")
    max_abs = (a - b).abs().max().item()
    mean_abs = (a - b).abs().mean().item()
    print(f"max abs diff  = {max_abs:.3e}")
    print(f"mean abs diff = {mean_abs:.3e}")
    print(f"argmax equal  = {torch.equal(a.argmax(-1), b.argmax(-1))}")
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    torch.testing.assert_close(a, b, atol=0.5, rtol=0.1)
    print("[OK] text-only parity")


if __name__ == "__main__":
    main()
