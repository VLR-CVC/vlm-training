"""PackedBatchEncoder with pack_rows > 1: rows stay within their length and every
sample lands exactly once, in whole documents.  python -m data.test_pack_rows"""

import random
from types import SimpleNamespace

import torch

from data.energon_dataloader import PackedBatchEncoder


def fake_sample(uid: int, length: int):
    ids = torch.full((length,), uid)
    return SimpleNamespace(
        input_ids=ids, labels=ids.clone(), attention_mask=torch.ones(length),
        mm_token_type_ids=torch.zeros(length, dtype=torch.long), length=length,
        pixel_values=torch.full((length // 4, 3), float(uid)), image_grid_thw=torch.tensor([[1, 2, 2]]),
    )


def main():
    random.seed(0)
    encoder = PackedBatchEncoder.__new__(PackedBatchEncoder)
    encoder.tokenizer = SimpleNamespace(pad_token_id=-1)
    encoder.max_length, encoder.rows, encoder.row_length = 64, 2, 32

    samples = [fake_sample(i, random.randint(1, 32)) for i in range(1, 200)]
    lengths = {id(s): s.length for s in samples}
    seen = set()
    for group in encoder.select_samples_to_pack(list(samples)):
        batch = encoder._pack_rows(group)
        cu = batch["cu_seqlens"].tolist()
        assert cu[0] == 0 and cu[-1] == 64 == batch["input_ids"].numel()
        assert 32 in cu, "row boundary must be a document boundary"
        for start, end in zip(cu, cu[1:]):
            doc = batch["input_ids"][start:end]
            uid = int(doc[0])
            assert (doc == uid).all()
            if uid != -1:
                assert uid not in seen
                seen.add(uid)
        assert batch["pixel_values"].shape[0] == sum(s.length // 4 for s in group)
    assert seen == set(range(1, 200))
    print("pack_rows ok")


if __name__ == "__main__":
    main()
