"""The ViT's bidirectional varlen attention matches the flex block-diagonal mask it
replaced: outputs and gradients, several packed images of different sizes, so400m
geometry (head dim 72, the case flex was slow on). Needs a GPU.

The kernels only differ in bf16 rounding, so both bf16 paths are scored against an
fp32 flex reference, and varlen must be no worse than flex itself.

    python models/tests/test_vision_varlen.py
"""

import copy
import sys
from pathlib import Path

import torch
from torch.nn.attention.flex_attention import create_block_mask

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.attention import FlexInnerAttention  # noqa: E402
from models.qwen3_5 import vision_encoder as qve  # noqa: E402
from models.qwen3_5.configs import qwen35_config_from_hf  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
VARLEN_MASK = qve.create_block_diagonal_mask


def flex_block_diagonal_mask(segment_lengths, total_tokens, device):
    """The mask the ViT used before (flex, bidirectional within a segment)."""
    ids = torch.repeat_interleave(
        torch.arange(segment_lengths.shape[0], device=device),
        segment_lengths.to(device), output_size=total_tokens)
    return create_block_mask(lambda b, h, q, kv: ids[q] == ids[kv],
                             1, None, total_tokens, total_tokens, device=device)


def run(enc, pixels, grid, flex):
    if flex:
        attn = FlexInnerAttention(FlexInnerAttention.Config())
        for layer in enc.layers.values():
            layer.attn.inner_attention = attn
    qve.create_block_diagonal_mask = flex_block_diagonal_mask if flex else VARLEN_MASK
    enc.zero_grad(set_to_none=True)
    out = enc(pixels.to(next(enc.parameters()).dtype), grid_thw=grid)
    (out.float() ** 2).mean().backward()
    return out.float(), {n: p.grad.float() for n, p in enc.named_parameters()}


def main():
    torch.manual_seed(0)
    cfg = qwen35_config_from_hf(str(ROOT / "configs/models/qwen3_5_2b_so400m.json"), seq_len=1024)
    enc32 = cfg.vision_encoder.build().cuda()
    for p in enc32.parameters():
        torch.nn.init.normal_(p, std=0.02)
    enc16 = copy.deepcopy(enc32).to(torch.bfloat16)
    grid = torch.tensor([[1, 32, 48], [1, 64, 64], [1, 16, 20], [1, 2, 2]], device="cuda")
    n = int((grid[:, 1] * grid[:, 2]).sum())
    pixels = torch.randn(n, enc32.patch_embed.weight.shape[1], device="cuda")

    out_v, g_v = run(enc16, pixels, grid, flex=False)
    out_f, g_f = run(copy.deepcopy(enc16), pixels, grid, flex=True)
    out_r, g_r = run(enc32, pixels, grid, flex=True)  # fp32 reference

    rel = lambda a, b: ((a - b).norm() / b.norm()).item()
    # attn.wk.bias has an exactly-zero gradient in exact math (a bias shared by all
    # keys cancels in the softmax); its ~1e-11 residue is noise, so it is skipped
    keys = [k for k in g_r if not k.endswith("attn.wk.bias")]
    worst = lambda g: max(rel(g[k], g_r[k]) for k in keys)
    ov, of, gv, gf = rel(out_v, out_r), rel(out_f, out_r), worst(g_v), worst(g_f)
    print(f"tokens {n}  vs fp32: output varlen {ov:.2e} flex {of:.2e}  "
          f"worst grad varlen {gv:.2e} flex {gf:.2e}")
    assert ov <= 1.5 * of + 1e-3 and gv <= 1.5 * gf + 1e-3, (ov, of, gv, gf)
    print("ok")


if __name__ == "__main__":
    main()
