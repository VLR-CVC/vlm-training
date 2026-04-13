"""Varlen packing sanity: two text samples concatenated into one row with
`attention_mask = cu_seqlens` must yield the same per-sample logits as running
each sample separately.

Run:
    python models/tests/test_qwen3_vl_varlen_pack.py
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
    device = torch.device("cuda")

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)
    ids_a = tok("The capital of France is", return_tensors="pt").input_ids.to(device)
    ids_b = tok("The quick brown fox jumps over the", return_tensors="pt").input_ids.to(device)
    Sa, Sb = ids_a.shape[1], ids_b.shape[1]

    packed = torch.cat([ids_a, ids_b], dim=1)
    cu_seqlens = torch.tensor([0, Sa, Sa + Sb], device=device, dtype=torch.int32)

    print(f"Loading Qwen3-VL from {SNAPSHOT} ...")
    m = Qwen3VLForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, device=device, load_vision=False
    ).eval()

    with torch.no_grad():
        logits_a_solo = m(input_ids=ids_a).float()
        logits_b_solo = m(input_ids=ids_b).float()
        logits_pack = m(input_ids=packed, attention_mask=cu_seqlens).float()

    pack_a = logits_pack[:, :Sa, :]
    pack_b = logits_pack[:, Sa:, :]

    def stats(a, b, name):
        d = (a - b).abs()
        print(f"{name}: max={d.max().item():.3e}  mean={d.mean().item():.3e}  "
              f"argmax_eq={torch.equal(a.argmax(-1), b.argmax(-1))}")

    stats(pack_a, logits_a_solo, "sample A")
    stats(pack_b, logits_b_solo, "sample B")
    assert torch.equal(pack_a.argmax(-1), logits_a_solo.argmax(-1))
    assert torch.equal(pack_b.argmax(-1), logits_b_solo.argmax(-1))
    print("[OK] varlen packing")


if __name__ == "__main__":
    main()
