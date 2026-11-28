"""A video folded into per-frame images gets HF Qwen3.5's MRoPE positions and keeps
every pixel row, in sequence order, with images and videos interleaved.

    QWEN35_PROCESSOR=<hf dir or hub id> python data/tests/test_video_fold.py
"""

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.model_batch import fold_video_frames, mrope_positions  # noqa: E402


def frames(n, w, h, seed):
    g = torch.Generator().manual_seed(seed)
    return [Image.fromarray(torch.randint(0, 256, (h, w, 3), generator=g, dtype=torch.uint8).numpy()) for _ in range(n)]


def hf_positions(input_ids, mm_types, image_grid, video_grid, merge):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Model

    stub = SimpleNamespace(config=SimpleNamespace(vision_config=SimpleNamespace(spatial_merge_size=merge)))
    stub.get_vision_position_ids = Qwen3_5Model.get_vision_position_ids.__get__(stub)
    pos, _ = Qwen3_5Model.get_rope_index(stub, input_ids[None], mm_types[None], image_grid, video_grid.clone())
    return pos[:, 0].T  # (T, 3)


def main():
    from transformers import AutoProcessor
    from transformers.video_utils import VideoMetadata

    proc = AutoProcessor.from_pretrained(os.environ.get("QWEN35_PROCESSOR", "Qwen/Qwen3.5-2B"))
    tok = proc.tokenizer
    image_id, video_id = tok.convert_tokens_to_ids("<|image_pad|>"), tok.convert_tokens_to_ids("<|video_pad|>")
    merge = proc.image_processor.merge_size

    # video, image, video: odd frame count (temporal padding), two frame sizes
    videos = [frames(5, 96, 64, 0), frames(4, 64, 128, 1)]
    image = frames(1, 80, 80, 2)[0]
    messages = [
        {"role": "user", "content": [{"type": "video"}, {"type": "image"}, {"type": "text", "text": "and"}, {"type": "video"}]},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
    ]
    text = proc.apply_chat_template(messages, tokenize=False)
    inputs = proc(
        text=[text], images=[image], videos=videos, padding=False, return_tensors="pt",
        video_metadata=[VideoMetadata(total_num_frames=len(v), fps=2.0, frames_indices=list(range(len(v)))) for v in videos],
        do_sample_frames=False,
    )
    ids = inputs["input_ids"][0]
    # one timestamp per temporal patch, the mean of its 2 frames: 0.25 s, 1.25 s, 2.0 s (last frame repeated)
    assert all(f"<{t} seconds>" in tok.decode(ids) for t in ("0.2", "1.2", "2.0")), tok.decode(ids)

    folded, pv, grid = fold_video_frames(
        ids, inputs["pixel_values"], inputs["image_grid_thw"],
        inputs["pixel_values_videos"], inputs["video_grid_thw"],
        image_token_id=image_id, video_token_id=video_id,
    )
    vg = inputs["video_grid_thw"]
    assert (folded != video_id).all() and int((folded == image_id).sum()) == int((ids == image_id).sum() + (ids == video_id).sum())
    assert grid.shape[0] == int(vg[:, 0].sum()) + 1 and (grid[:, 0] == 1).all()

    # pixel rows: video 0's frames, then the image, then video 1's frames
    n0 = int(vg[0].prod())
    expect = torch.cat([inputs["pixel_values_videos"][:n0], inputs["pixel_values"], inputs["pixel_values_videos"][n0:]])
    assert torch.equal(pv, expect)
    assert torch.equal(grid[int(vg[0, 0])], inputs["image_grid_thw"][0])

    ours = mrope_positions(folded, torch.tensor([0, len(ids)]), grid,
                           image_token_id=image_id, video_token_id=video_id, spatial_merge_size=merge)
    theirs = hf_positions(ids, inputs["mm_token_type_ids"][0], inputs["image_grid_thw"], vg, merge)
    assert torch.equal(ours, theirs.to(ours.dtype)), (ours - theirs).abs().max()
    print(f"ok: {len(ids)} tokens, {grid.shape[0]} vision items, video grids {vg.tolist()}")


if __name__ == "__main__":
    main()
