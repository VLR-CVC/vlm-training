"""The invariant `_embed` relies on but never checks.

`models/qwen3_vl/model.py:214` computes the image-token count as
`grid_thw.prod(-1) // spatial_merge_unit` and scatters the ViT output onto exactly
that many `image_token_id` positions. A synthetic batch whose token count disagrees
with its grid does not raise -- it scatters the wrong number of embeddings. So the
precompile specs have to be checked, or a bad default silently trains on garbage
during warmup.
"""

import torch

from train.precompile import DEFAULT_SPECS, synthetic_batch

MERGE_UNIT = 4          # spatial_merge_size=2
PATCH_DIM = 3 * 2 * 16 * 16
IMAGE_TOKEN_ID = 151655
SEQ_LEN = 32768


def build(images, patches, docs, seq_len=SEQ_LEN):
    return synthetic_batch(
        images, patches, docs,
        seq_len=seq_len,
        image_token_id=IMAGE_TOKEN_ID,
        vocab_size=151936,
        patch_dim=PATCH_DIM,
        spatial_merge_unit=MERGE_UNIT,
        device=torch.device("cpu"),
        generator=torch.Generator().manual_seed(0),
    )


def test_image_tokens_match_grid():
    batch = build(2, 2560, 3)
    total_patches = int(batch["grid_thw"].prod(-1).sum())
    assert total_patches == 2 * 2560
    assert batch["pixel_values"].shape == (total_patches, PATCH_DIM)
    assert (batch["input"] == IMAGE_TOKEN_ID).sum() == total_patches // MERGE_UNIT


def test_text_only_has_no_vision_keys():
    batch = build(0, 0, 4)
    assert "pixel_values" not in batch
    assert "grid_thw" not in batch
    assert (batch["input"] == IMAGE_TOKEN_ID).sum() == 0


def test_positions_restart_once_per_document():
    for docs in (1, 3, 22):
        batch = build(1, 2560, docs)
        assert (batch["positions"] == 0).sum() == docs
        assert batch["positions"].shape == (SEQ_LEN,)
        assert batch["mrope_positions"].shape == (SEQ_LEN, 3)


def test_every_default_spec_is_valid():
    """The specs ship as defaults; a bad one would only show up at 32 nodes."""
    for images, patches, docs in DEFAULT_SPECS:
        batch = build(images, patches, docs)
        n_img = int((batch["input"] == IMAGE_TOKEN_ID).sum())
        if images:
            total = int(batch["grid_thw"].prod(-1).sum())
            assert total == images * patches
            assert n_img == total // MERGE_UNIT
            assert n_img < SEQ_LEN, "image tokens must leave room for text"
        else:
            assert n_img == 0
        assert batch["labels"][-1] == -100


def test_rejects_unmergeable_patch_count():
    try:
        build(1, 2561, 1)
    except ValueError:
        return
    raise AssertionError("patch count not divisible by the merge unit must raise")


def test_rejects_batch_that_cannot_fit():
    try:
        build(64, 25088, 1)
    except ValueError:
        return
    raise AssertionError("image tokens exceeding seq_len must raise")


def test_grids_are_near_square():
    """Degenerate grids compile the wrong RoPE table size (vision_encoder.py:334)."""
    for images, patches, docs in DEFAULT_SPECS:
        if not images:
            continue
        _, h, w = build(images, patches, docs)["grid_thw"][0].tolist()
        assert h * w == patches
        assert h % 2 == 0 and w % 2 == 0
        assert max(h, w) / min(h, w) <= 4, f"{h}x{w} is not a plausible image shape"


def _runs(t, val):
    """Number of contiguous runs of `val` -- what get_vision_positions counts."""
    m = (t == val).to(torch.int8)
    return int(((m == 1) & (torch.cat([torch.zeros(1, dtype=torch.int8), m[:-1]]) == 0)).sum())


def test_one_placeholder_run_per_image():
    """`models/common/multimodal.py:47` rejects any other layout.

    Regression: job 1897947 died here, with all images in one block and stray
    image_token_id values drawn at random from the full vocab.
    """
    for images, patches, docs in DEFAULT_SPECS:
        batch = build(images, patches, docs)
        assert _runs(batch["input"], IMAGE_TOKEN_ID) == images, f"{images} images"
