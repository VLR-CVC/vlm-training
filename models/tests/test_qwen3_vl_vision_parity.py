"""Stage B parity: our Qwen3VLVisionModel vs HF `model.visual`.

Synthetic inputs: random pixel_values + a small grid_thw. Compares the merged
hidden states and every deepstack feature tensor.

Run:
    python models/tests/test_qwen3_vl_vision_parity.py
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

    # Two small "images": grids of size 4x4 and 2x6 spatial patches (already
    # in patch-units; multiples of spatial_merge_size=2).
    grid_thw = torch.tensor([[1, 4, 4], [1, 2, 6]], dtype=torch.long, device=device)
    total_patches = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum().item())

    print(f"Loading our Qwen3-VL (with vision) from {SNAPSHOT} ...")
    ours = Qwen3VLForCausalLM.from_pretrained(
        SNAPSHOT, dtype=torch.float32, device=device, load_vision=True
    ).eval()
    vc = ours.cfg.vision
    patch_dim = vc.in_channels * vc.temporal_patch_size * vc.patch_size * vc.patch_size

    pixel_values = torch.randn(total_patches, patch_dim, device=device, dtype=torch.float32)

    print("Loading HF Qwen3-VL ...")
    from transformers import Qwen3VLForConditionalGeneration

    hf = (
        Qwen3VLForConditionalGeneration.from_pretrained(
            SNAPSHOT, torch_dtype=torch.float32, attn_implementation="sdpa"
        )
        .to(device)
        .eval()
    )

    with torch.no_grad():
        merged_ours, deepstack_ours = ours.model.visual(pixel_values, grid_thw)
        out_hf = hf.model.visual(pixel_values, grid_thw)

    merged_hf = out_hf.last_hidden_state if hasattr(out_hf, "last_hidden_state") else out_hf[0]
    # In HF's BaseModelOutputWithDeepstackFeatures, the merged tensor is
    # `pooler_output` and the un-merged final hidden states are `last_hidden_state`.
    # We expose only the merged version, so prefer pooler_output when present.
    if hasattr(out_hf, "pooler_output") and out_hf.pooler_output is not None:
        merged_hf = out_hf.pooler_output
    deepstack_hf = out_hf.deepstack_features

    print(f"merged ours={tuple(merged_ours.shape)}  hf={tuple(merged_hf.shape)}")
    diff = (merged_ours - merged_hf).abs()
    print(f"merged max={diff.max().item():.3e}  mean={diff.mean().item():.3e}")
    torch.testing.assert_close(merged_ours, merged_hf, atol=1e-3, rtol=1e-3)

    assert len(deepstack_ours) == len(deepstack_hf), (
        f"deepstack count mismatch: {len(deepstack_ours)} vs {len(deepstack_hf)}"
    )
    for i, (a, b) in enumerate(zip(deepstack_ours, deepstack_hf)):
        d = (a - b).abs()
        print(f"deepstack[{i}] shape={tuple(a.shape)}  max={d.max().item():.3e}  mean={d.mean().item():.3e}")
        torch.testing.assert_close(a, b, atol=1e-3, rtol=1e-3)

    print("[OK] vision parity")


if __name__ == "__main__":
    main()
