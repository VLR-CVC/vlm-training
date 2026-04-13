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

SNAPSHOT = os.environ.get(
    "QWEN3VL_SNAPSHOT",
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_2b",
)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)

    print(f"Loading our Qwen3-VL from {SNAPSHOT} ...")
    ours = Qwen3VLForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.bfloat16, device=device, load_vision=True
    ).eval()
    cfg = ours.cfg

    # One synthetic "image" at grid 4x4 → merged to 2x2 = 4 visual tokens.
    grid = torch.tensor([[1, 4, 4]], dtype=torch.long, device=device)
    num_visual_tokens = int(
        (grid[:, 0] * (grid[:, 1] // cfg.vision.spatial_merge_size)
         * (grid[:, 2] // cfg.vision.spatial_merge_size)).sum().item()
    )
    patch_dim = (
        cfg.vision.in_channels
        * cfg.vision.temporal_patch_size
        * cfg.vision.patch_size ** 2
    )
    total_patches = int((grid[:, 0] * grid[:, 1] * grid[:, 2]).sum().item())
    pixel_values = torch.randn(total_patches, patch_dim, device=device, dtype=torch.bfloat16)

    # Sequence: [text_0 text_1 <vision_start> <img>*N <vision_end> text_2 text_3]
    img_id = cfg.image_token_id
    vs_id = cfg.vision_start_token_id
    ve_id = cfg.vision_end_token_id
    prefix = torch.tensor([10, 11], device=device, dtype=torch.long)
    img = torch.full((num_visual_tokens,), img_id, device=device, dtype=torch.long)
    suffix = torch.tensor([20, 21, 22], device=device, dtype=torch.long)
    seq = torch.cat([prefix, torch.tensor([vs_id], device=device), img,
                     torch.tensor([ve_id], device=device), suffix])
    input_ids = seq.unsqueeze(0)

    print("Loading HF Qwen3-VL ...")
    from transformers import Qwen3VLForConditionalGeneration

    hf = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            SNAPSHOT, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
        )
        .to(device)
        .eval()
    )

    mm_type = torch.zeros_like(input_ids, dtype=torch.int32)
    mm_type[input_ids == img_id] = 1

    with torch.no_grad():
        a = ours(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
        )
        b = hf(
            input_ids=input_ids,
            pixel_values=pixel_values,
            image_grid_thw=grid,
            mm_token_type_ids=mm_type,
        ).logits

    a = a.float()
    b = b.float()
    print(f"ours={tuple(a.shape)}  hf={tuple(b.shape)}")
    diff = (a - b).abs()
    print(f"max abs diff  = {diff.max().item():.3e}")
    print(f"mean abs diff = {diff.mean().item():.3e}")
    print(f"argmax equal  = {torch.equal(a.argmax(-1), b.argmax(-1))}")
    assert torch.equal(a.argmax(-1), b.argmax(-1))
    print("[OK] multimodal parity")


if __name__ == "__main__":
    main()
