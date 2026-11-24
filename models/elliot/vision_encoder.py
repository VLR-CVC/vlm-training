from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F

from models.qwen3_5.vision_encoder import _compute_2d_rope_cache, Qwen35VisionEncoder

_INTERP_CACHE: dict = {}

def _bicubic_matrix(side: int, size: int, device) -> torch.Tensor:
    """`(size, side)` matrix of 1D bicubic resizing, align_corners=False: F.interpolate of the
    identity, so the weights and border clamping are PyTorch's own. Bicubic upsample_2d is
    separable, so a 2D resize is A_h @ P @ A_w^T."""
    key = (side, size, str(device))
    mat = _INTERP_CACHE.get(key)
    if mat is None:
        eye = torch.eye(side, dtype=torch.float64, device=device).view(1, side, side, 1)
        mat = F.interpolate(eye, size=(size, 1), mode="bicubic", align_corners=False)[0, :, :, 0].T
        mat = _INTERP_CACHE[key] = mat.contiguous()
    return mat

def bicubic_resize(base: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """`(side, side, dim)` -> `(h, w, dim)`, equal to F.interpolate(mode="bicubic",
    align_corners=False) up to fp32 rounding. As two GEMMs its backward is two GEMMs too;
    upsample_bicubic2d_backward scatters every output gradient into the small source grid
    with atomics (389 ms of a 3.8 s step at MBS 4 x 8192, 4 MP images, on GH200)."""
    side, _, dim = base.shape
    ct = torch.promote_types(base.dtype, torch.float32)  # bf16 params resize in fp32
    a_h = _bicubic_matrix(side, h, base.device).to(ct)
    a_w = _bicubic_matrix(side, w, base.device).to(ct)
    x = (a_h @ base.to(ct).reshape(side, side * dim)).view(h, side, dim)  # rows
    x = torch.einsum("wj,hjd->hwd", a_w, x)  # columns
    return x.to(base.dtype)

def bicubic_pos_embeds(
    pos_embed: torch.Tensor, grids: list[list[int]], merge: int
) -> torch.Tensor:
    """Packed `(sum t*h*w, dim)` position embeddings in 2x2 block order."""
    n, dim = pos_embed.shape
    side = int(n**0.5)
    base = pos_embed.view(side, side, dim)
    out = []
    for t, h, w in grids:
        grid = base if (h, w) == (side, side) else bicubic_resize(base, h, w)  # (h, w, dim)
        block = grid.reshape(h // merge, merge, w // merge, merge, dim).permute(0, 2, 1, 3, 4)
        out.append(block.reshape(h * w, dim).repeat(t, 1))
    packed = torch.cat(out)
    if spmd.is_type_checking():
        packed = spmd.mutate_type(packed, src=spmd.R, dst={"dp": spmd.V, "tp": spmd.I})
    return packed

class ElliotVisionEncoder(Qwen35VisionEncoder):
    @dataclass(kw_only=True, slots=True)
    class Config(Qwen35VisionEncoder.Config):
        pass

    def compute_position_embeddings(self, grids):
        head_dim = self.config.dim // self.config.num_heads
        max_hw = max(max(h, w) for _, h, w in grids)
        if self._cached_freq_table is None or self._cached_freq_table.shape[0] < max_hw:
            self._cached_freq_table = self.rotary_pos_emb(max_hw)
        learned = bicubic_pos_embeds(self.pos_embed, grids, self.spatial_merge_size)
        rope = _compute_2d_rope_cache(self._cached_freq_table, grids, self.spatial_merge_size, head_dim)
        return learned, rope
