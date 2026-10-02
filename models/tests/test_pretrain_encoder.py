"""`PretrainBatchEncoder` is the from-scratch decoder-only path: raw text, no
chat template, loss on every token, long documents chunked instead of dropped.

Needs the tokenizer from the local Qwen3.5 snapshot; no GPU, no dataset."""
import types

import pytest
import torch

from data.cookers import cooker_text
from data.energon_dataloader import PretrainBatchEncoder

MODEL_DIR = "/data/151-1/users/tockier/qwen_finetune/cache/qwen35_2b"
SEQ = 64
KEYS = {"__key__": "k", "__restore_key__": (), "__subflavors__": None,
        "__sources__": ()}


@pytest.fixture(scope="module")
def enc():
    tok = pytest.importorskip("transformers").AutoTokenizer.from_pretrained(MODEL_DIR)
    return PretrainBatchEncoder(
        types.SimpleNamespace(tokenizer=tok),
        SEQ, SEQ,
        image_token_id=-1, video_token_id=-1, spatial_merge_size=1,
    )


def _chunks(enc, words):
    sample = cooker_text(dict(KEYS, json={"text": " ".join(words)}))
    return list(enc.encode_sample(sample))


def test_no_chat_template(enc):
    """The SFT path wraps every document in `<|im_start|>assistant`. Pretraining
    must not: the model would learn the markup as part of the language."""
    ids = _chunks(enc, ["hello"] * 5)[0].input_ids.tolist()
    text = enc.tokenizer.decode(ids)
    assert "<|im_start|>" not in text, text
    assert text.endswith(enc.tokenizer.decode([enc.EOS_token]))


def test_every_token_is_supervised(enc):
    """`assistant_labels` masks all but assistant spans; here only the final
    token is unsupervised, because nothing follows it to predict."""
    c = _chunks(enc, ["hello"] * 5)[0]
    assert (c.labels[:-1] == c.input_ids[1:]).all()
    assert c.labels[-1] == -100
    assert (c.labels == -100).sum() == 1


def test_long_document_is_chunked_not_dropped(enc):
    """The SFT encoder raises SkipSample above `seq_len`. FineWeb keeps a large
    share of its tokens in documents that long."""
    chunks = _chunks(enc, ["hello"] * 400)
    assert len(chunks) > 1, "document did not exceed one row; widen the test"
    assert all(c.length <= SEQ for c in chunks)
    # no token lost and none duplicated: the chunks concatenate back to the whole
    joined = torch.cat([c.input_ids for c in chunks])
    assert joined[-1] == enc.EOS_token
    assert len(joined) == sum(c.length for c in chunks)
    assert (joined[:-1] != enc.EOS_token).all(), "EOS inside the document"


def test_text_only_payload(enc):
    c = _chunks(enc, ["hello"] * 5)[0]
    assert c.pixel_values is None and c.image_grid_thw is None
    assert c.mrope_positions.shape == (c.length, 3)
    assert (c.positions == torch.arange(c.length)).all()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
