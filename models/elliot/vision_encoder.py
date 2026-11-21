from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F

from models.qwen3_5.vision_encoder import _compute_2d_rope_cache, Qwen35VisionEncoder

def bicubic_pos_embeds(
    pos_embed: torch.Tensor, grids: list[list[int]], merge: int
) -> torch.Tensor:
    """Packed `(sum t*h*w, dim)` position embeddings in 2x2 block order."""
    n, dim = pos_embed.shape
    side = int(n**0.5)
    base = pos_embed.view(side, side, dim).permute(2, 0, 1).unsqueeze(0)
    out = []
    for t, h, w in grids:
        grid = base if (h, w) == (side, side) else F.interpolate(
            base, size=(h, w), mode="bicubic", align_corners=False)
        grid = grid[0].permute(1, 2, 0)  # (h, w, dim), row-major
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
