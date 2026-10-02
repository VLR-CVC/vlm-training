"""`cooker_nemotron` has to find the image whichever way the shard spells it.

Multi-image samples are repacked as `<key>.0.png`, so energon hands the cooker
`0.png`. A single-image shard keeps `<key>.png` and the cooker gets a bare `png`.
Only the numbered spelling was matched, so every single-image sample came out
with `image=None` while `nemotron_messages` still emitted an `{"type": "image"}`
placeholder -- local plotqa_cot lost 100% of its samples to that.
"""

import pytest

from data.cookers import cooker_nemotron

TURNS = [
    {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "what?"}]},
    {"role": "assistant", "content": [{"type": "text", "text": "a chart"}]},
]


def crude(**payload):
    return {
        "__key__": "00000000",
        "__restore_key__": ("Webdataset", 0, 0),
        "__subflavors__": {"type_dataset": "plotqa_cot"},
        "json": {"messages": TURNS},
        **payload,
    }


def n_placeholders(sample):
    return sum(
        part.get("type") == "image"
        for message in sample.messages
        for part in message["content"]
    )


@pytest.mark.parametrize("key", ["png", "jpg", "jpeg", "webp", "0.png"])
def test_one_image_however_it_is_spelled(key):
    sample = cooker_nemotron(crude(**{key: "PIXELS"}))
    assert sample.image == "PIXELS", key
    assert sample.images is None
    # one placeholder, one image -- the balance encode_sample checks
    assert n_placeholders(sample) == 1


def test_numbered_keys_stay_in_order():
    """The fallback must not disturb multi-image samples: `10.png` sorts after
    `9.png` numerically, not lexically."""
    keys = {f"{i}.png": f"PIX{i}" for i in (0, 1, 9, 10)}
    sample = cooker_nemotron(crude(**keys))
    assert sample.image is None
    assert sample.images == ["PIX0", "PIX1", "PIX9", "PIX10"]


def test_numbered_keys_win_over_a_bare_one():
    sample = cooker_nemotron(crude(**{"0.png": "A", "1.png": "B", "png": "STALE"}))
    assert sample.images == ["A", "B"]


def test_a_textless_sample_still_has_no_image():
    """No image key at all is a different bug; the fallback must not invent one."""
    sample = cooker_nemotron(crude())
    assert sample.image is None and sample.images is None
