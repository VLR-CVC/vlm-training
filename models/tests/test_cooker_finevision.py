"""`cooker_finevision` has to read the schema FineVision is actually staged in.

It was written against a `{"conversations": [{"from": ..., "value": ...}]}` layout
with the image under a bare `jpg`. The staged shards carry neither: every json is
`{"source": ..., "texts": [{"user": ..., "assistant": ...}]}` and images sit under
numbered keys (`0.jpg`, `1.jpg`, ...). Probing 20 samples from each of the 185
members in `finevision_debug.yaml` raised `KeyError: 'conversations'` 185/185 times,
and the loader died on the first batch -- energon turns a cooker exception into
`FatalSampleError` after 100 in a row, so the whole run goes down, no skips.

One `texts` entry is one user/assistant EXCHANGE, not one turn, so it has to emit
two messages; getting that wrong would leave the assistant text unsupervised.
"""

import pytest

from data.cookers import cooker_finevision


def crude(texts, **payload):
    return {
        "__key__": "vqav2_000_000000",
        "__restore_key__": ("Webdataset", 0, 0),
        "__subflavors__": {"type_dataset": "finevision"},
        "json": {"source": "vqav2", "texts": texts},
        **payload,
    }


def n_placeholders(sample):
    return sum(
        part.get("type") == "image"
        for message in sample.messages
        for part in message["content"]
    )


def roles(sample):
    return [m["role"] for m in sample.messages]


def test_one_exchange_becomes_two_messages():
    sample = cooker_finevision(crude([{"user": "what?", "assistant": "a net"}], **{"0.jpg": "PIX"}))
    assert roles(sample) == ["user", "assistant"]
    assert sample.messages[1]["content"] == [{"type": "text", "text": "a net"}]


def test_every_exchange_is_supervised():
    """The real failure this guards: assistant text folded into a user turn trains
    on nothing. encode_sample skips a sample with no supervised tokens, so a wrong
    split here costs the whole member silently."""
    texts = [{"user": f"q{i}", "assistant": f"a{i}"} for i in range(4)]
    sample = cooker_finevision(crude(texts, **{"0.jpg": "PIX"}))
    assert roles(sample) == ["user", "assistant"] * 4
    answers = [m["content"][0]["text"] for m in sample.messages if m["role"] == "assistant"]
    assert answers == ["a0", "a1", "a2", "a3"]


def test_single_image_lands_on_image_not_images():
    sample = cooker_finevision(crude([{"user": "q", "assistant": "a"}], **{"0.jpg": "PIX"}))
    assert sample.image == "PIX" and sample.images is None
    # one placeholder, one image -- the balance encode_sample checks
    assert n_placeholders(sample) == 1


def test_numbered_keys_stay_in_numeric_order():
    """drivelm ships 21 views; `10.jpg` sorts after `9.jpg` numerically, not lexically."""
    keys = {f"{i}.jpg": f"PIX{i}" for i in (0, 1, 9, 10)}
    sample = cooker_finevision(crude([{"user": "q", "assistant": "a"}], **keys))
    assert sample.image is None
    assert sample.images == ["PIX0", "PIX1", "PIX9", "PIX10"]
    assert n_placeholders(sample) == 4


def test_placeholders_only_on_the_first_exchange():
    """Every later exchange repeating the placeholders would outnumber the images and
    walk the processor off the end of the grid list (StopIteration -> skip)."""
    texts = [{"user": "q1", "assistant": "a1"}, {"user": "q2", "assistant": "a2"}]
    sample = cooker_finevision(crude(texts, **{"0.jpg": "A", "1.jpg": "B"}))
    assert n_placeholders(sample) == 2
    assert sample.messages[0]["content"][:2] == [{"type": "image"}, {"type": "image"}]
    assert all(p.get("type") == "text" for p in sample.messages[2]["content"])


def test_a_text_only_member_gets_no_image_and_no_placeholder():
    """The 20-odd `text_*` members carry no image member at all. A placeholder with no
    image is skipped by encode_sample (n_runs != n_grids)."""
    sample = cooker_finevision(crude([{"user": "q", "assistant": "a"}]))
    assert sample.image is None and sample.images is None
    assert n_placeholders(sample) == 0


@pytest.mark.parametrize("field", ["user", "assistant"])
def test_a_literal_image_marker_is_stripped(field):
    """11 members carry "<image>" in their text. Left in, the chat template renders it
    as text next to a real placeholder."""
    turn = {"user": "q", "assistant": "a"}
    turn[field] = "<image>\n" + turn[field]
    sample = cooker_finevision(crude([turn], **{"0.jpg": "PIX"}))
    assert n_placeholders(sample) == 1
    texts = [p.get("text", "") for m in sample.messages for p in m["content"]]
    assert not any("<image>" in t for t in texts)


def test_a_null_field_does_not_crash():
    sample = cooker_finevision(crude([{"user": None, "assistant": "a"}], **{"0.jpg": "PIX"}))
    assert sample.messages[0]["content"][-1] == {"type": "text", "text": ""}
    assert sample.messages[1]["content"] == [{"type": "text", "text": "a"}]
