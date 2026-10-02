"""A sample whose image placeholders and images disagree must be skipped.

Both shapes kill the run rather than the sample, which is why they are caught in
`encode_sample` and not left to the caller:

  placeholders > images   the processor itself raises a bare `StopIteration`
  placeholders, no image  the placeholder is left unexpanded, so `mrope_positions`
                          runs `next(grids)` past the end of an empty grid list

Energon wraps either one as `FatalSampleError` and 64 nodes go down with it --
job 46700737, ranks 252-255. The second shape is what a nemotron member whose
json declares an image but whose tar entry has none produces.
"""

import pytest
import torch
from PIL import Image

pytest.importorskip("transformers")

# whichever box this runs on -- the processor config is all that is read
MODEL_DIRS = (
    "/data/151-1/users/tockier/qwen_finetune/cache/qwen3_vl_4b",
    "/gpfs/scratch/ehpc543/tockier/qwen_models/qwen3vl_2b",
)


@pytest.fixture(scope="module")
def processor():
    from transformers import AutoProcessor

    for path in MODEL_DIRS:
        try:
            return AutoProcessor.from_pretrained(path)
        except Exception:
            continue
    pytest.skip(f"no processor config in any of {MODEL_DIRS}")


def chat(processor, n_placeholders):
    content = [{"type": "image"}] * n_placeholders + [{"type": "text", "text": "hi"}]
    return processor.apply_chat_template(
        conversation=[
            {"role": "user", "content": content},
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
        ],
        tokenize=False,
        add_generation_prompt=False,
    )


def image_runs(input_ids, image_token_id):
    """The count `mrope_positions` will ask the grid list for."""
    m = input_ids == image_token_id
    return int(m[:1].sum() + (m[1:] & ~m[:-1]).sum())


def test_matched_sample_is_consistent(processor):
    img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    out = processor(
        text=[chat(processor, 2)],
        images=[Image.new("RGB", (64, 64))] * 2,
        padding=False,
        return_tensors="pt",
    )
    assert image_runs(out["input_ids"][0], img_id) == out["image_grid_thw"].shape[0] == 2


def test_more_placeholders_than_images_raises_stopiteration(processor):
    """Not a hypothesis about the processor -- the behaviour `encode_sample`
    catches. If a transformers upgrade stops doing this, the catch is dead."""
    with pytest.raises(StopIteration):
        processor(
            text=[chat(processor, 2)],
            images=[Image.new("RGB", (64, 64))],
            padding=False,
            return_tensors="pt",
        )


def test_placeholder_without_image_is_detectable(processor):
    """The shape that actually killed job 46700737: no error from the processor,
    an unexpanded placeholder, and no grid for `mrope_positions` to consume."""
    img_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")
    out = processor(text=[chat(processor, 1)], images=None, padding=False, return_tensors="pt")
    assert out.get("image_grid_thw") is None
    assert image_runs(out["input_ids"][0], img_id) == 1


def test_mrope_positions_would_die_on_it():
    """Why the check has to be the run count and not the cooker's placeholders:
    this is the failure it prevents, at `data/model_batch.py:41`."""
    from data.model_batch import mrope_positions

    ids = torch.tensor([5, 5, 99, 5])  # one image run, no grid
    with pytest.raises(StopIteration):
        mrope_positions(
            ids,
            torch.tensor([0, 4], dtype=torch.int32),
            None,
            image_token_id=99,
            video_token_id=98,
            spatial_merge_size=2,
        )
