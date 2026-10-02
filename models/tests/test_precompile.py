"""The invariant `_embed` relies on but never checks.

`models/qwen3_vl/model.py:214` computes the image-token count as
`grid_thw.prod(-1) // spatial_merge_unit` and scatters the ViT output onto exactly
that many `image_token_id` positions. A synthetic batch whose token count disagrees
with its grid does not raise -- it scatters the wrong number of embeddings. So the
precompile specs have to be checked, or a bad default silently trains on garbage
during warmup.
"""

import torch

from train.precompile import DEFAULT_SPECS, spec_fits, synthetic_batch

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


def test_short_seq_len_drops_oversized_specs():
    """seq_len=8192 does not fit every default spec; precompile must not die on it.

    Regression: a run configured with seq_len=8192 aborted inside
    `synthetic_batch` because (4, 8192, 6) and (2, 24576, 3) need more image
    tokens than there are positions.
    """
    fits = [
        s for s in DEFAULT_SPECS
        if spec_fits(s[0], s[1], seq_len=8192, spatial_merge_unit=MERGE_UNIT)
    ]
    assert [s for s in DEFAULT_SPECS if s not in fits] == [(4, 8192, 6), (2, 24576, 3)]
    from train.precompile import vlm_specs

    specs, _ = vlm_specs(8192, spatial_merge_unit=MERGE_UNIT)
    assert any(im == 0 for im, *_ in specs), "text-only must survive any seq_len"

    # what survives the filter must actually build
    for images, patches, docs in fits:
        build(images, patches, docs, seq_len=8192)

    # and what it drops must be exactly what would have raised
    for images, patches, docs in DEFAULT_SPECS:
        if (images, patches, docs) in fits:
            continue
        try:
            build(images, patches, docs, seq_len=8192)
        except ValueError:
            continue
        raise AssertionError(f"({images}, {patches}, {docs}) should not fit 8192")


# --- text-only specs -------------------------------------------------------
#
# The text path has the same failure mode as the vision one above: a spec that
# does not match what the packer produces trains a graph nothing will use, and
# every real batch then recompiles. These pin the two graph-shaping quantities.

import pytest

from models.common.attention import create_varlen_metadata_for_document
from train.precompile import _synthetic_positions, synthetic_batch_text, text_specs


def _key(docs, seq_len, longest):
    md = create_varlen_metadata_for_document(_synthetic_positions(docs, seq_len, longest))
    return int(md.cu_seq_q.numel()), int(md.max_k)


@pytest.mark.parametrize("seq_len", [8192, 16384])
def test_text_specs_are_distinct_graphs(seq_len):
    """A duplicate spec is pure startup cost: same graph compiled twice."""
    keys = [_key(d, seq_len, l) for d, l in text_specs(seq_len)]
    assert len(keys) == len(set(keys)), sorted(keys)


@pytest.mark.parametrize("seq_len", [8192, 16384])
def test_text_specs_cover_every_max_k_bucket(seq_len):
    """`max_k` is the axis that actually moves on real data -- measured at
    seq_len 16384 it spans all five buckets while the document count stays at
    64. Equal-length splits (the old specs) could only ever produce 1024."""
    covered = {k[1] for k in (_key(d, seq_len, l) for d, l in text_specs(seq_len))}
    expected, m = set(), 1024
    while m <= seq_len:
        expected.add(m)
        m *= 2
    assert covered == expected, sorted(covered)


@pytest.mark.parametrize("seq_len", [8192, 16384])
def test_text_specs_cover_the_document_buckets(seq_len):
    """MN5 packs 90-469 documents per row -> buckets 128/256/512; the local set
    packs 1-89 -> 64. Both datasets have to be warmed by the same spec list."""
    covered = {k[0] for k in (_key(d, seq_len, l) for d, l in text_specs(seq_len))}
    assert {65, 129, 257, 513} <= covered, sorted(covered)


def test_positions_are_fully_initialised():
    """Regression: `_synthetic_positions` allocated with `torch.empty` and only
    wrote the remainder when `documents > 1`. With one document the tail kept
    whatever the allocator held, so stray zeros invented extra documents and the
    spec was non-deterministic."""
    for _ in range(20):
        pos = _synthetic_positions(1, 4096, 1024)
        assert int((pos == 0).sum()) == 1, int((pos == 0).sum())
        assert torch.equal(pos, torch.arange(4096))


def test_synthetic_batch_text_matches_the_spec_positions():
    """The batch the precompiler runs must have the layout the specs were
    deduplicated on, or it warms a graph the real data never uses."""
    gen = torch.Generator().manual_seed(0)
    for docs, longest in text_specs(8192):
        batch = synthetic_batch_text(
            documents=docs, seq_len=8192, longest=longest,
            device=torch.device("cpu"), generator=gen,
        )
        assert torch.equal(batch["positions"], _synthetic_positions(docs, 8192, longest))


@pytest.mark.parametrize("seq_len", [8192, 16384, 24576])
def test_vlm_specs_warm_every_decoder_graph(seq_len):
    """The regression this file exists for: DEFAULT_SPECS alone warms five of
    the twenty-one graphs the packer reaches at seq_len 24576, and never the
    `max_k=1024` bucket that packed SFT conversations land in almost every step.
    The vision specs plus the image-free top-up must cover the text set exactly."""
    from train.precompile import vlm_specs

    specs, _ = vlm_specs(seq_len, spatial_merge_unit=4)
    covered = {_key(docs, seq_len, longest) for _, _, docs, longest in specs}
    assert {_key(d, seq_len, l) for d, l in text_specs(seq_len)} <= covered
    # the image specs may share a decoder graph -- they differ on the vision
    # axis -- but a top-up spec that warms nothing new is pure startup cost
    image = {_key(docs, seq_len, l) for im, _, docs, l in specs if im}
    topup = [_key(docs, seq_len, l) for im, _, docs, l in specs if not im]
    assert len(topup) == len(set(topup)), sorted(topup)
    assert not (image & set(topup)), sorted(image & set(topup))
