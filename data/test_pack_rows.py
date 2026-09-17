"""PackedBatchEncoder: label masking, and micro-batches of `rows` rows that keep every
document whole, in the model's input layout.  python -m data.test_pack_rows"""

import random
from types import SimpleNamespace

import torch

from data.energon_dataloader import PackedBatchEncoder

IM_START, ASSISTANT, NL, IM_END, PAD = 1, 2, 3, 4, 0


def fake_sample(uid: int, length: int, image: bool):
    ids = torch.full((length,), 100 + uid)
    return SimpleNamespace(
        input_ids=ids, labels=ids.clone(), positions=torch.arange(length),
        mrope_positions=torch.arange(length).unsqueeze(-1).expand(-1, 3), length=length,
        pixel_values=torch.full((length // 4 + 1, 3), float(uid)) if image else None,
        image_grid_thw=torch.tensor([[1, 2, 2]]) if image else None,
    )


def main():
    enc = PackedBatchEncoder.__new__(PackedBatchEncoder)
    enc.tokenizer = SimpleNamespace(pad_token_id=PAD)
    enc.assistant_prefix, enc.EOS_token = [IM_START, ASSISTANT, NL], IM_END

    # --- labels: only real assistant turns, through <|im_end|> and the newline -----
    user = [IM_START, 50, NL, ASSISTANT, 51, 52, IM_END, NL]  # "assistant" inside a user turn
    answer = [IM_START, ASSISTANT, NL, 60, 61, IM_END, NL]
    ids = user + answer + [IM_START, ASSISTANT, NL, 70]      # last answer truncated
    labels = enc.assistant_labels(ids).tolist()
    expect = [-100] * len(user) + [-100, -100, -100, 60, 61, IM_END, NL] + [-100] * 4
    assert labels == expect, labels

    # --- packing: 2 rows of 32 -----------------------------------------------------
    random.seed(0)
    enc.row_length, enc.rows = 32, 2
    samples = [fake_sample(i, random.randint(1, 32), image=i % 3 == 0) for i in range(1, 200)]
    seen = set()
    for group in enc.select_samples_to_pack(list(samples)):
        b = enc.pack_selected_samples(group)
        T = enc.row_length * enc.rows
        assert b["input"].shape == b["labels"].shape == b["positions"].shape == (T,)
        assert b["mrope_positions"].shape == (T, 3)
        starts = (b["positions"] == 0).nonzero().flatten().tolist()
        assert 32 in starts, "row boundary must be a document boundary"
        bounds = starts + [T]
        for s, e in zip(bounds, bounds[1:]):
            doc = b["input"][s:e]
            uid = int(doc[0])
            assert (doc == uid).all() and b["positions"][s:e].tolist() == list(range(e - s))
            if uid != PAD:
                assert uid not in seen
                seen.add(uid)
        assert b["num_samples"] == len(group)
        assert b["num_tokens"] == sum(s.length for s in group)
        assert b["num_valid_tokens"] == int((b["labels"] != -100).sum())
        imgs = [s for s in group if s.pixel_values is not None]
        assert ("pixel_values" in b) == bool(imgs)
        if imgs:
            assert b["pixel_values"].shape[0] == sum(s.pixel_values.shape[0] for s in imgs)
            assert b["grid_thw"].shape == (len(imgs), 3)
    assert seen == {100 + i for i in range(1, 200)}
    print("pack_rows ok")


if __name__ == "__main__":
    main()
