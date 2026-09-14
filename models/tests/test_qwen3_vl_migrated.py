"""The qwen3.5 forward-path optimisations ported to qwen3-vl.

Covers the two pieces with real logic (`packed_positions`, `_round_max_seqlen`)
and the key-name contract between the trainer and the model, which is the way
the `max_seqlen` plumbing fails silently: a rename leaves `kwargs.get` returning
None, the model falls back to the `.item()` path, and everything still trains --
just with the host sync back and one recompile per distinct length.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import torch

from models.qwen3_vl.model import _round_max_seqlen, packed_positions

ROOT = Path(__file__).resolve().parents[2]


def _reference_positions(cu_seqlens: torch.Tensor, total: int) -> torch.Tensor:
    """The per-document Python loop this replaced."""
    pos = torch.zeros(total, dtype=torch.int64)
    for start, end in zip(cu_seqlens[:-1].tolist(), cu_seqlens[1:].tolist()):
        pos[start:end] = torch.arange(end - start)
    return pos


@pytest.mark.parametrize(
    "bounds",
    [
        [0, 4],                 # single document
        [0, 3, 10],             # two, uneven
        [0, 1, 2, 3, 4, 5],     # many length-1 documents
        [0, 7, 7, 12],          # an empty document in the middle
        [0, 16],
    ],
)
def test_packed_positions_matches_the_loop(bounds):
    cu = torch.tensor(bounds, dtype=torch.int32)
    total = bounds[-1]
    torch.testing.assert_close(
        packed_positions(cu, total), _reference_positions(cu, total)
    )


def test_packed_positions_restarts_at_every_boundary():
    cu = torch.tensor([0, 3, 8], dtype=torch.int32)
    assert packed_positions(cu, 8).tolist() == [0, 1, 2, 0, 1, 2, 3, 4]


@pytest.mark.parametrize(
    "n,expected", [(1, 1), (2, 2), (3, 4), (5, 8), (8, 8), (9, 16), (10240, 16384)]
)
def test_round_max_seqlen_is_a_power_of_two_upper_bound(n, expected):
    got = _round_max_seqlen(n)
    assert got == expected
    assert got >= n                      # never truncates the varlen block grid
    assert got & (got - 1) == 0          # power of two


def test_round_max_seqlen_handles_degenerate_input():
    assert _round_max_seqlen(0) == 1
    assert _round_max_seqlen(-5) == 1


@pytest.mark.parametrize("key", ["max_seqlen", "vision_max_seqlen"])
def test_trainer_and_model_agree_on_the_kwarg_names(key):
    """`train_qwen.batch_generator` writes these; the model reads them."""
    trainer = (ROOT / "train" / "train_qwen.py").read_text()
    model = (ROOT / "models" / "qwen3_vl" / "model.py").read_text()

    assert re.search(rf"batch\[{re.escape(repr(key))}\]\s*=", trainer) or re.search(
        rf'batch\["{key}"\]\s*=', trainer
    ), f"trainer no longer sets {key}"
    assert f'kwargs.get("{key}")' in model, f"qwen3-vl no longer reads {key}"
