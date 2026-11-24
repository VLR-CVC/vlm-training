"""Rolling packing picks `rows` first-fit rows in arrival order, never over a row's length,
and pack_selected_samples' greedy split recovers exactly those rows.

    python data/tests/test_rolling_packing.py
"""

import random
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from data.energon_dataloader import RollingPackedBatchEncoder, RollingPackingDataset  # noqa: E402


def greedy_rows(samples, row_length):
    """pack_selected_samples' row split, without the tensors."""
    rows, row, used = [], [], 0
    for s in samples:
        if used + s.length > row_length:
            rows.append(row)
            row, used = [], 0
        row.append(s)
        used += s.length
    return rows + [row]


def main():
    enc = object.__new__(RollingPackedBatchEncoder)
    rng = random.Random(0)
    for rows in (1, 2, 4):
        enc.rows, enc.row_length = rows, 1000
        for _ in range(200):
            buf = [SimpleNamespace(length=rng.randint(1, 1000)) for _ in range(64)]
            (picked,) = enc.select_samples_to_pack(list(buf))
            assert picked[0] is buf[0], "the oldest sample must lead"
            split = greedy_rows(picked, enc.row_length)
            assert len(split) <= rows
            assert all(sum(s.length for s in r) <= enc.row_length for r in split)
            # every passed-over sample was too long for the room left in each row
            left = [s for s in buf if all(s is not p for p in picked)]
            for r in split:
                room = enc.row_length - sum(s.length for s in r)
                assert all(s.length > room for s in left)

    # the dataset keeps one pack and returns the rest to the buffer, in arrival order
    ds = object.__new__(RollingPackingDataset)
    ds._select, ds._reading_buffer = enc.select_samples_to_pack, []
    buf = [SimpleNamespace(length=rng.randint(1, 1000)) for _ in range(64)]
    (kept,) = ds._first_pack_only(list(buf))
    assert [id(s) for s in ds._reading_buffer] == [id(s) for s in buf if all(s is not k for k in kept)]
    print("ok")


if __name__ == "__main__":
    main()
