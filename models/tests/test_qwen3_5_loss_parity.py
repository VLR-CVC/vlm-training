"""Training loss parity for Qwen3.5.

Two checks that must hold:

  1. Single-sample loss parity (ours vs HF).
     On a batch of one sample there is no packing, so HF's `GatedDeltaNet`
     behaves correctly (the whole row is one sequence). Losses must agree
     within bf16 noise.

  2. Packed+padded self-consistency (ours vs ours).
     When two samples are packed and padded into one row with
     `attention_mask = cu_seqlens`, our per-sample loss must match the
     loss computed on each sample alone. This is the test that catches
     state leakage across packed samples in the linear-attention layers.

Why we do NOT check ours-vs-HF under packing:

  HF's `Qwen3_5Model.forward` derives `linear_attn_mask` from `attention_mask`
  via `_update_linear_attn_mask` and feeds *that* to `GatedDeltaNet`. The
  varlen `cu_seqlens` we pass for flash-attention is ignored by HF's linear
  path, so sample A's recurrent state leaks into sample B. Our native impl
  threads `cu_seqlens` into FLA's `chunk_gated_delta_rule` and runs the causal
  Conv1d per-segment, so it is correct — but exact parity with HF is
  impossible under packing.

Run:
    python models/tests/test_qwen3_5_loss_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_5.model import Qwen3_5ForCausalLM

SNAPSHOT = os.environ.get(
    "QWEN3_5_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b",
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(SNAPSHOT)

    prompts = [
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. "
        "Sed do eiusmod tempor incididunt ut labore et dolore magna aliqua.",
        "The quick brown fox jumps over the lazy dog near the river bank.",
        "In computer science, a list is an abstract data type that represents "
        "a countable number of ordered values.",
    ]

    print(f"Loading our Qwen3.5 from {SNAPSHOT} ...")
    ours = Qwen3_5ForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, device=device, load_vision=False
    ).eval()

    # ----------- Check 1: single-sample loss parity (ours vs HF).
    print("\n-- Check 1: single-sample loss (ours vs HF) --")
    from transformers import Qwen3_5ForConditionalGeneration
    hf = (
        Qwen3_5ForConditionalGeneration.from_pretrained(
            SNAPSHOT, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to(device)
        .eval()
    )

    max_vs_hf = 0.0
    for i, text in enumerate(prompts):
        ids = tok(text, return_tensors="pt").input_ids.to(device)
        labels = ids.clone()
        with torch.no_grad():
            l_ours = ours(input_ids=ids, labels=labels).loss.float().item()
            l_hf = hf(input_ids=ids, labels=labels).loss.float().item()
        d = abs(l_ours - l_hf)
        max_vs_hf = max(max_vs_hf, d)
        print(f"[{i}] len={ids.shape[1]:3d}  ours={l_ours:.6f}  hf={l_hf:.6f}  |diff|={d:.3e}")
    tol_vs_hf = 1e-2
    print(f"max |diff| vs HF = {max_vs_hf:.3e}  (tol={tol_vs_hf:.0e})")
    assert max_vs_hf < tol_vs_hf, f"ours-vs-HF single-sample loss diverges: {max_vs_hf:.3e}"

    del hf
    torch.cuda.empty_cache()

    # ----------- Check 2: packed+padded self-consistency (ours vs ours).
    print("\n-- Check 2: packed+padded self-consistency (ours vs ours) --")
    ids_list = [tok(t, return_tensors="pt").input_ids.to(device) for t in prompts[:2]]
    Sa, Sb = ids_list[0].shape[1], ids_list[1].shape[1]
    pad_to = 128
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    pad = torch.full((1, pad_to - Sa - Sb), pad_id, dtype=ids_list[0].dtype, device=device)
    packed = torch.cat([ids_list[0], ids_list[1], pad], dim=1)
    cu = torch.tensor([0, Sa, Sa + Sb, pad_to], dtype=torch.int32, device=device)

    # Build labels: real tokens contribute, pad is -100.
    labels = packed.clone()
    labels[0, Sa + Sb :] = -100

    with torch.no_grad():
        # Packed run.
        out_packed = ours(input_ids=packed, attention_mask=cu, labels=labels)
        logits_packed = out_packed.logits if hasattr(out_packed, "logits") else None
        # When labels is passed our forward returns CausalLMOutput.
        logits_packed = out_packed.logits
        # Solo runs for reference.
        logits_a_solo = ours(input_ids=ids_list[0]).float()
        logits_b_solo = ours(input_ids=ids_list[1]).float()

    pack_a = logits_packed[:, :Sa].float()
    pack_b = logits_packed[:, Sa : Sa + Sb].float()
    a_eq = torch.equal(pack_a.argmax(-1), logits_a_solo.argmax(-1))
    b_eq = torch.equal(pack_b.argmax(-1), logits_b_solo.argmax(-1))
    a_max = (pack_a - logits_a_solo).abs().max().item()
    b_max = (pack_b - logits_b_solo).abs().max().item()
    print(f"sample A: argmax_eq={a_eq}  max_abs={a_max:.3e}")
    print(f"sample B: argmax_eq={b_eq}  max_abs={b_max:.3e}")
    assert a_eq, "packed sample A argmax diverges from solo"
    assert b_eq, "packed sample B argmax diverges from solo (state leakage)"

    print("\n[OK] loss parity (single-sample vs HF, packed self-consistency)")


if __name__ == "__main__":
    main()
