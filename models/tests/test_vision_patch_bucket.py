"""Padding the ViT patch count to a bucket does not change the output.

`vision_encoder.VISION_PATCH_BUCKET` rounds the packed patch count up so the shapes
reaching `torch.compile` come from a small fixed set (see the constant's comment).
The padding rides through every ViT block as its own mask segment, so it must not
reach a real patch -- this asserts exactly that, by running the same input with the
bucket on and off and comparing outputs.

    QWEN3_VL_CONFIG=configs/models/qwen3_vl_2b.json python models/tests/test_vision_patch_bucket.py

Runs on CPU with a randomly initialised tower (no snapshot needed); it checks the
masking, not the weights. It is deliberately tiny: the ViT is quadratic in the patch
count and CPU attention over 4k patches is slow.
"""

import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common import vision_encoder as ve_common
from models.qwen3_5 import vision_encoder as ve
from models.qwen3_5.checkpoint import build_meta, materialize

# Inductor's CPU flex-attention codegen fails to compile here ("'cur_qSplitSize8' was
# not declared in this scope"), both for the mask build and for the attention itself.
# Irrelevant to what this checks -- the masking is the same either way -- so run both
# eagerly; on GPU the compiled path is the one that runs.
from torch.nn.attention.flex_attention import flex_attention  # noqa: E402

from models.common.attention import FlexInnerAttention  # noqa: E402

ve_common.compiled_create_block_mask = ve_common.create_block_mask
FlexInnerAttention._compiled_flex_attn = staticmethod(flex_attention)

CONFIG = os.environ.get("QWEN3_VL_CONFIG", "configs/models/qwen3_vl_2b.json")
BUCKET = 512  # small, so the test pads without a 4096-patch forward


def run(model, pixel_values, grid_thw, bucket: int) -> torch.Tensor:
    ve.VISION_PATCH_BUCKET = bucket
    with torch.no_grad():
        return model.vision_encoder(pixel_values, grid_thw=grid_thw)


def main() -> None:
    torch.manual_seed(0)
    model = build_meta(CONFIG, seq_len=1024)
    # two small images, patch counts chosen so the total is NOT a multiple of BUCKET
    grid_thw = torch.tensor([[1, 8, 8], [1, 12, 12]])
    total = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    assert total % BUCKET, f"pick a grid whose total ({total}) is not a multiple of {BUCKET}"

    # only the tower is needed; build it on CPU and initialise it
    materialize(model, "cpu")
    with torch.no_grad():
        model.init_states(buffer_device=torch.device("cpu"))
    model.eval()

    dim = model.vision_encoder.config.patch_dim if hasattr(
        model.vision_encoder.config, "patch_dim") else model.vision_encoder.patch_embed.weight.shape[1]
    pixel_values = torch.randn(total, dim, dtype=model.vision_encoder.patch_embed.weight.dtype)

    original = ve.VISION_PATCH_BUCKET
    try:
        off = run(model, pixel_values, grid_thw, 0)
        on = run(model, pixel_values, grid_thw, BUCKET)
    finally:
        ve.VISION_PATCH_BUCKET = original

    assert off.shape == on.shape, (off.shape, on.shape)
    torch.testing.assert_close(off, on, rtol=1e-4, atol=1e-4)
    padded = -(-total // BUCKET) * BUCKET
    print(f"vision patch bucket: {total} patches padded to {padded}, "
          f"output unchanged {tuple(off.shape)}")


if __name__ == "__main__":
    main()
