"""`_permute` must size `repeat_interleave` from the counts, not from the tensor.

`routed_input_RD.shape[0]` is symbolic under compile -- job 2268203 traced it as
`256*Max(1, (2048//s56))` -- and a floor division on a symbolic dim does not agree
with the integer sum of the per-expert counts. `repeat_interleave` asserts
`output_size == repeats.sum()` on device, so the disagreement killed the backward
(AC recomputes the dispatch there) with 27,360 copies of

    Repeat.cu:20 Assertion `result_size == cumsum_ptr[size - 1]` failed

off by one in either direction. Here the row count is deliberately wrong so a
regression reproduces that instead of passing on an even split.
"""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from models.common.token_dispatcher import AllToAllTokenDispatcher

EP_SIZE, E, D = 2, 2, 2
# (rank, expert) counts: r0e0=3 r0e1=2 r1e0=4 r1e1=1 -> 10 real rows
COUNTS = [3, 2, 4, 1]
TOTAL = sum(COUNTS)
# expert-major order: e0 takes r0's rows 0-2 then r1's 5-8; e1 takes r0's 3-4, r1's 9
EXPECTED = [0, 1, 2, 5, 6, 7, 8, 3, 4, 9]


class _Mesh:
    def size(self):
        return EP_SIZE


class _Stub:
    """`_permute` only touches `self.ep_mesh`."""
    ep_mesh = _Mesh()


def _inputs():
    counts = torch.tensor(COUNTS, dtype=torch.long)
    # one row MORE than the counts describe: stands in for the symbolic shape
    # disagreeing with the real sum
    rows = torch.arange(TOTAL + 1, dtype=torch.float32).unsqueeze(1).repeat(1, D)
    return rows, counts


def test_permute_uses_total_rows_not_the_tensor_shape():
    rows, counts = _inputs()
    shape, permuted, indices, per_expert = AllToAllTokenDispatcher._permute(
        _Stub(), rows, counts, total_rows=TOTAL
    )

    assert indices.tolist() == EXPECTED, indices.tolist()
    assert permuted.shape[0] == TOTAL
    torch.testing.assert_close(permuted[:, 0], rows[EXPECTED, 0])
    # counts collapsed over ranks: e0 = 3+4, e1 = 2+1
    assert per_expert.tolist() == [COUNTS[0] + COUNTS[2], COUNTS[1] + COUNTS[3]]
    assert tuple(shape) == tuple(rows.shape)


def test_falling_back_to_the_tensor_shape_is_what_used_to_fail():
    """Guards the guard: omitting total_rows reproduces the production failure.

    The wording is device-specific -- CUDA trips the kernel assert and names
    `repeat_interleave`, CPU raises `allocated size does not match required size`
    from the same mismatch -- so accept either.
    """
    rows, counts = _inputs()
    with pytest.raises(
        RuntimeError, match="repeat_interleave|allocated size does not match"
    ):
        AllToAllTokenDispatcher._permute(_Stub(), rows, counts)


def test_total_rows_is_accepted_by_the_torchao_override():
    """`dispatch` calls whichever override is installed with the same keyword."""
    import inspect

    from models.common.token_dispatcher import TorchAOTokenDispatcher

    for cls in (AllToAllTokenDispatcher, TorchAOTokenDispatcher):
        params = inspect.signature(cls._permute).parameters
        assert "total_rows" in params, f"{cls.__name__} would raise TypeError"
