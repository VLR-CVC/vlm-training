"""AllToAllTokenDispatcher's rank-major -> expert-major reshuffle.

    pytest models/tests/test_ep_dispatcher.py -q

`_permute` is the half of the EP dispatcher that needs no collectives, and it is
the half that corrupts routing silently: after the dispatch all-to-all the rows
arrive rank-major ((e0,r0), (e1,r0), (e0,r1), ...) while `_grouped_mm` needs one
contiguous run per local expert. Get the index arithmetic wrong and every token
is still computed by SOME expert, so the loss keeps going down -- just on the
wrong weights. See `models/common/token_dispatcher.py`.

The all-to-all itself needs a process group, so it is not covered here.
"""

from __future__ import annotations

import pytest
import torch

from models.common.token_dispatcher import AllToAllTokenDispatcher


class _FakeMesh:
    """`_permute` only ever asks the mesh for its size."""

    def __init__(self, size: int):
        self._size = size

    def size(self) -> int:
        return self._size


def _dispatcher(ep_size: int, *, num_experts: int = 4, top_k: int = 2):
    d = AllToAllTokenDispatcher.Config(num_experts=num_experts, top_k=top_k).build()
    d.ep_mesh = _FakeMesh(ep_size)
    return d


def test_permute_groups_rows_by_local_expert():
    """Hand-built case: 2 ranks, 2 local experts, counts (r0: 1, 2), (r1: 3, 1).

    Rank-major rows:   e0r0 | e1r0 e1r0 | e0r1 e0r1 e0r1 | e1r1
    Expert-major rows: e0r0 e0r1 e0r1 e0r1 | e1r0 e1r0 e1r1
    """
    d = _dispatcher(ep_size=2)
    counts = torch.tensor([1, 2, 3, 1])  # (r0e0, r0e1, r1e0, r1e1)
    rows = torch.tensor([[0.0], [10.0], [11.0], [1.0], [2.0], [3.0], [12.0]])

    shape, permuted, indices, per_local_expert = d._permute(rows, counts)

    assert shape == rows.shape
    assert per_local_expert.tolist() == [4, 3]  # e0: 1+3, e1: 2+1
    assert permuted.flatten().tolist() == [0, 1, 2, 3, 10, 11, 12]
    assert indices.tolist() == [0, 3, 4, 5, 1, 2, 6]


def test_unpermute_is_the_exact_inverse():
    """`combine` relies on this: the expert output has to land back on the row
    the dispatch all-to-all brought in, or it is returned to the wrong rank."""
    d = _dispatcher(ep_size=2)
    counts = torch.tensor([2, 1, 0, 3])
    rows = torch.arange(6, dtype=torch.float32).reshape(6, 1)

    shape, permuted, indices, _ = d._permute(rows, counts)
    assert torch.equal(d._unpermute(permuted, shape, indices), rows)


def test_an_empty_expert_segment_is_handled():
    """Routing leaves experts with zero tokens all the time; a zero-length
    segment must not shift the rows after it."""
    d = _dispatcher(ep_size=2)
    counts = torch.tensor([0, 2, 2, 0])  # e0 only from r1, e1 only from r0
    rows = torch.tensor([[10.0], [11.0], [0.0], [1.0]])

    _, permuted, _, per_local_expert = d._permute(rows, counts)
    assert per_local_expert.tolist() == [2, 2]
    assert permuted.flatten().tolist() == [0, 1, 10, 11]


def test_ep1_mesh_takes_the_local_path():
    """`ep_mesh is None` must fall back to the EP=1 argsort, so the same class
    works at any degree and EP=1 pays for no collectives."""
    d = AllToAllTokenDispatcher.Config(num_experts=4, top_k=2).build()
    d.wire_meshes(ep_mesh=None)

    x_TD = torch.randn(3, 2)
    topk_scores_TK = torch.rand(3, 2)
    topk_expert_ids_TK = torch.tensor([[0, 3], [1, 2], [0, 1]])
    counts = torch.tensor([2, 2, 1, 1])

    routed, out_counts, metadata = d.dispatch(
        x_TD, topk_scores_TK, topk_expert_ids_TK, counts
    )
    assert routed.shape == (6, 2)  # T*K, nothing moved off-rank
    assert torch.equal(out_counts, counts)
    assert not hasattr(metadata, "permuted_indices")

    out = d.combine(torch.ones(6, 2), metadata, x_TD)
    # every token is routed to top_k experts, so each row gets both its scores
    assert torch.allclose(out, topk_scores_TK.sum(dim=1, keepdim=True).expand(3, 2))
