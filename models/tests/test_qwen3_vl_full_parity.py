"""Stage C parity: full Qwen3-VL multimodal forward.

Compares our text+vision+DeepStack+MRoPE forward against HF
`Qwen3VLForConditionalGeneration` on a synthetic image+text prompt.

Run:
    python models/tests/test_qwen3_vl_full_parity.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.qwen3_vl.model import Qwen3VLForCausalLM

# bf16 is what training runs, but it is a bad medium for a parity assertion:
# reduction order across 28 decoder layers moves individual logits by ~1 unit
# out of a std of 4, and an argmax can flip on a pair that is exactly tied. The
# implementations are compared in fp32 instead, where the answer is either right
# or it is not, and bf16 is checked only for the looser property that it stays
# within its own noise. `QWEN3VL_DTYPE=bf16` runs the bf16 pass alone.
_DTYPES = {
    # dtype, HF attn impl, per-element atol, mean-abs-diff bound
    "fp32": (torch.float32, "sdpa", 1e-3, 1e-4),
    "bf16": (torch.bfloat16, "flash_attention_2", 2.0, 0.25),
}


def _dtype_cases():
    only = os.environ.get("QWEN3VL_DTYPE")
    return [(k, *v) for k, v in _DTYPES.items() if only in (None, k)]


def _compare(name, a, b, atol, mean_tol):
    a, b = a.float(), b.float()
    diff = (a - b).abs()
    max_abs, mean_abs = diff.max().item(), diff.mean().item()
    print(f"  {name}: max abs = {max_abs:.3e}   mean abs = {mean_abs:.3e}")

    # Where the argmax disagrees, the only question is whether the top two were
    # separated by more than the noise -- a flipped tie is not a parity failure.
    am_a, am_b = a.argmax(-1), b.argmax(-1)
    bad = (am_a != am_b).nonzero()
    for pos in bad.tolist():
        row = b[tuple(pos)]
        top2 = row.topk(2).values
        gap = (top2[0] - top2[1]).item()
        print(f"    argmax differs at {tuple(pos)}, hf top1-top2 gap = {gap:.3f}")
        assert gap <= 4 * mean_abs, (
            f"argmax flipped on a clearly-separated pair (gap {gap:.3f})"
        )

    assert mean_abs < mean_tol, f"{name}: mean abs diff {mean_abs:.3e} > {mean_tol:.0e}"
    assert max_abs < atol, f"{name}: max abs diff {max_abs:.3e} > {atol:.0e}"


SNAPSHOT = os.environ.get(
    "QWEN3VL_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_2b",
)



SNAPSHOT = os.environ.get(
    "QWEN3VL_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_2b",
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    from transformers import Qwen3VLForConditionalGeneration

    for name, dtype, attn, atol, mean_tol in _dtype_cases():
        print(f"[{name}] loading ...")
        ours, _cfg = Qwen3VLForCausalLM.from_pretrained(
            SNAPSHOT, dtype=dtype, device=device, load_vision=True
        )
        ours = ours.eval()
        cfg = ours.cfg

        # One synthetic "image" at grid 4x4 -> merged to 2x2 = 4 visual tokens.
        grid = torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)
        merge = cfg.vision.spatial_merge_size
        num_visual_tokens = int(
            (grid[:, 0] * (grid[:, 1] // merge) * (grid[:, 2] // merge)).sum().item()
        )
        patch_dim = (
            cfg.vision.in_channels
            * cfg.vision.temporal_patch_size
            * cfg.vision.patch_size ** 2
        )
        total_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
        torch.manual_seed(0)
        pixel_values = torch.randn(total_patches, patch_dim, device=device, dtype=dtype)

        # [text text <vision_start> <img>*N <vision_end> text text text]
        input_ids = torch.cat([
            torch.tensor([10, 11], device=device, dtype=torch.long),
            torch.tensor([cfg.vision_start_token_id], device=device, dtype=torch.long),
            torch.full((num_visual_tokens,), cfg.image_token_id,
                       device=device, dtype=torch.long),
            torch.tensor([cfg.vision_end_token_id], device=device, dtype=torch.long),
            torch.tensor([20, 21, 22], device=device, dtype=torch.long),
        ]).unsqueeze(0)

        hf = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                SNAPSHOT, torch_dtype=dtype, attn_implementation=attn
            )
            .to(device)
            .eval()
        )
        mm_type = torch.zeros_like(input_ids, dtype=torch.int32)
        mm_type[input_ids == cfg.image_token_id] = 1

        with torch.no_grad():
            a = ours(input_ids=input_ids, pixel_values=pixel_values,
                     image_grid_thw=grid)
            b = hf(input_ids=input_ids, pixel_values=pixel_values,
                   image_grid_thw=grid, mm_token_type_ids=mm_type).logits
        _compare(name, a, b, atol, mean_tol)
        del ours, hf
        torch.cuda.empty_cache()

    print("[OK] multimodal parity")


def test_full_parity():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("both attention backends need a GPU")
    main()


if __name__ == "__main__":
    main()
