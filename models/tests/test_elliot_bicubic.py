"""Parity of `bicubic_pos_embeds` (bicubic resize as two GEMMs) against the F.interpolate
implementation it replaced: forward and gradients, on SigLIP2 so400m's 32x32 position grid,
for upsampled, downsampled, non-square and identity grids.

    python models/tests/test_elliot_bicubic.py          # CPU, or the current GPU if any

float64 shows the two are the same linear map (error at rounding level); fp32 shows the
error the training dtype sees. In bf16 the old kernel is itself the less accurate one (its
backward accumulates the atomics in bf16), so there both are scored against the float64
result and the new one must be at least as close.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.elliot.vision_encoder import bicubic_pos_embeds  # noqa: E402

SIDE, DIM, MERGE = 32, 1152, 2
# (t, h, w) in patches: identity, up, down, non-square, tall 4 MP-like, a video-like t=2
GRIDS = [[1, 32, 32], [1, 64, 48], [1, 16, 20], [1, 22, 94], [2, 128, 128], [1, 30, 8]]
# max |new - old| relative to max |old|
TOL = {torch.float64: 1e-12, torch.float32: 1e-5}
BF16_TOL = 1.6e-2  # vs the float64 result: ~2 bf16 ulps

def reference(pos_embed, grids, merge):
    """The previous implementation, verbatim."""
    n, dim = pos_embed.shape
    side = int(n**0.5)
    base = pos_embed.view(side, side, dim).permute(2, 0, 1).unsqueeze(0)
    out = []
    for t, h, w in grids:
        grid = base if (h, w) == (side, side) else F.interpolate(
            base, size=(h, w), mode="bicubic", align_corners=False)
        grid = grid[0].permute(1, 2, 0)
        block = grid.reshape(h // merge, merge, w // merge, merge, dim).permute(0, 2, 1, 3, 4)
        out.append(block.reshape(h * w, dim).repeat(t, 1))
    return torch.cat(out)

def rel(a, b):
    return ((a.double() - b.double()).abs().max() / b.double().abs().max()).item()

@pytest.mark.parametrize("dtype", [torch.float64, torch.float32, torch.bfloat16])
def test_parity(dtype, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    g = torch.Generator().manual_seed(0)
    p0 = torch.randn(SIDE * SIDE, DIM, generator=g, dtype=torch.float64).to(device, dtype)
    p_new, p_ref = p0.clone().requires_grad_(), p0.clone().requires_grad_()
    out_new = bicubic_pos_embeds(p_new, GRIDS, MERGE)
    out_ref = reference(p_ref, GRIDS, MERGE)
    assert out_new.shape == out_ref.shape and out_new.dtype == out_ref.dtype
    upstream = torch.randn(out_ref.shape, generator=g, dtype=torch.float64).to(device, dtype)
    out_new.backward(upstream)
    out_ref.backward(upstream)
    fwd, bwd = rel(out_new, out_ref), rel(p_new.grad, p_ref.grad)
    print(f"{str(dtype):15s} {device}: new vs old: forward rel err {fwd:.2e}  grad rel err {bwd:.2e}")
    if dtype == torch.bfloat16:
        p64 = p0.double().requires_grad_()
        out64 = reference(p64, GRIDS, MERGE)
        out64.backward(upstream.double())
        errs = {n: (rel(o, out64), rel(gr, p64.grad)) for n, o, gr in
                (("new", out_new, p_new.grad), ("old", out_ref, p_ref.grad))}
        print(f"{'':15s} vs float64: new forward {errs['new'][0]:.2e} grad {errs['new'][1]:.2e} | "
              f"old forward {errs['old'][0]:.2e} grad {errs['old'][1]:.2e}  (tol {BF16_TOL:.0e})")
        for i in (0, 1):
            assert errs["new"][i] <= max(errs["old"][i], BF16_TOL), errs
    else:
        assert fwd <= TOL[dtype], fwd
        assert bwd <= TOL[dtype], bwd
    # the identity grid is passed through untouched
    assert torch.equal(out_new[: SIDE * SIDE], out_ref[: SIDE * SIDE])

if __name__ == "__main__":
    for dt in (torch.float64, torch.float32, torch.bfloat16):
        test_parity(dt)
    print("ok")
