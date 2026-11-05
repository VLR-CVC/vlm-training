"""`roll_mtp_sequence`: the shift kernel Multi-Token Prediction is built on.

    pytest models/tests/test_mtp_roll.py -q

MTP depth k predicts the token k ahead, so its input and label rows are the
original row shifted left by k. The only hard part is that our rows are PACKED:
several documents share one row, and a shift must not let document A's tail read
document B's head. Upstream (torchtitan b21f7d43e, `models/deepseek_v3/MTP.md`
section 2.2) decides validity from reset-style position IDs:

    position i is valid for shift k  iff  positions[i + k] == positions[i] + k

The expectations below for the packed cases are copied from that document's
worked examples, so this is a test against the reference rather than against a
restatement of the implementation.

What this does NOT test, because none of it is ported: `MTPTransformerBlock`,
`MTPDecoder`, `MTPLoss`, or any of the MTP sharding. See models/common/mtp.py.
"""

import pytest
import torch

from models.common.mtp import roll_mtp_sequence
from train.loss import IGNORE_INDEX

# token-id fill for shifted INPUTS; labels use IGNORE_INDEX instead
FILL = 0


def test_unpacked_row_shifts_and_fills_the_tail():
    """MTP.md section 2.3, the unpacked table. Each extra depth loses one more
    target off the end -- there is no next row to borrow from."""
    tokens = torch.tensor([10, 11, 12, 13, 14, 15])

    assert roll_mtp_sequence(tokens, shift=1, fill_value=FILL).tolist() == [
        11, 12, 13, 14, 15, FILL,
    ]
    assert roll_mtp_sequence(tokens, shift=2, fill_value=FILL).tolist() == [
        12, 13, 14, 15, FILL, FILL,
    ]


def test_packed_row_does_not_cross_document_boundaries():
    """MTP.md section 2.3, the packed table, verbatim.

    tokens    = [A0, A1, A2, B0, B1, B2, B3, C0, C1, C2]
    positions = [ 0,  1,  2,  0,  1,  2,  3,  0,  1,  2]

    The slot after A2 must be filled even though a physical next token exists:
    it is B0, whose position resets to 0.
    """
    # 0..2 = doc A, 10..13 = doc B, 20..22 = doc C
    tokens = torch.tensor([0, 1, 2, 10, 11, 12, 13, 20, 21, 22])
    positions = torch.tensor([0, 1, 2, 0, 1, 2, 3, 0, 1, 2])

    shift1 = roll_mtp_sequence(tokens, shift=1, fill_value=FILL, positions=positions)
    assert shift1.tolist() == [1, 2, FILL, 11, 12, 13, FILL, 21, 22, FILL]

    shift2 = roll_mtp_sequence(tokens, shift=2, fill_value=FILL, positions=positions)
    assert shift2.tolist() == [2, FILL, FILL, 12, 13, FILL, FILL, 22, FILL, FILL]


def test_valid_mask_marks_exactly_the_real_sources():
    tokens = torch.tensor([0, 1, 2, 10, 11])
    positions = torch.tensor([0, 1, 2, 0, 1])

    rolled, valid = roll_mtp_sequence(
        tokens, shift=1, fill_value=FILL, positions=positions, return_valid_mask=True
    )
    assert rolled.tolist() == [1, 2, FILL, 11, FILL]
    assert valid.tolist() == [True, True, False, True, False]
    # the mask is what the MTP block uses to zero the previous hidden state, so
    # it has to agree with the fills rather than merely overlap them
    assert (rolled[~valid] == FILL).all()


def test_labels_fill_with_ignore_index_not_a_token_id():
    """Same helper, different fill. A label filled with 0 would train the model
    to predict token 0 at every boundary instead of skipping the position."""
    labels = torch.tensor([1, 2, 3, 11, 12])
    positions = torch.tensor([0, 1, 2, 0, 1])

    rolled = roll_mtp_sequence(
        labels, shift=1, fill_value=IGNORE_INDEX, positions=positions
    )
    assert rolled.tolist() == [2, 3, IGNORE_INDEX, 12, IGNORE_INDEX]
    assert IGNORE_INDEX not in labels.tolist(), "fill must not collide with real data"


def test_multimodal_padding_produces_no_mtp_targets():
    """Our own position convention, not upstream's.

    `Qwen35Model.get_attention_masks` documents that multimodal padding uses
    position 0 for EVERY padded token. The validity test is then `0 == 0 + k`,
    false throughout, so a padded run yields no MTP target -- which is what we
    want, and worth pinning because it falls out of the convention rather than
    being checked anywhere.
    """
    # one 3-token document, then 4 padded tokens all at position 0
    tokens = torch.tensor([5, 6, 7, 99, 99, 99, 99])
    positions = torch.tensor([0, 1, 2, 0, 0, 0, 0])

    rolled, valid = roll_mtp_sequence(
        tokens, shift=1, fill_value=FILL, positions=positions, return_valid_mask=True
    )
    assert rolled.tolist() == [6, 7, FILL, FILL, FILL, FILL, FILL]
    assert valid.tolist() == [True, True, False, False, False, False, False]
    assert not valid[3:].any(), "no MTP target may come out of padding"


def test_a_single_token_document_yields_nothing():
    """Degenerate but real: packing can leave 1-token documents, and a document
    shorter than the shift has no valid position at all."""
    tokens = torch.tensor([0, 10, 20, 30])
    positions = torch.tensor([0, 0, 0, 0])  # four 1-token documents

    rolled, valid = roll_mtp_sequence(
        tokens, shift=1, fill_value=FILL, positions=positions, return_valid_mask=True
    )
    assert rolled.tolist() == [FILL] * 4
    assert not valid.any()


def test_trailing_dimensions_are_carried_through():
    """MTP shifts token ids, but the same helper is documented as taking
    ``[T, ...]``; a 2-D input must shift rows, not elements."""
    seq = torch.arange(12).reshape(6, 2)
    rolled = roll_mtp_sequence(seq, shift=2, fill_value=-1)
    assert rolled.shape == seq.shape
    assert rolled[:4].tolist() == seq[2:].tolist()
    assert (rolled[4:] == -1).all()


@pytest.mark.parametrize("shift", [0, -1, 7])
def test_an_out_of_range_shift_raises(shift):
    """Guards the MTP-depth loop: depth > seq_len is a config error, and a
    wrapped-around or silently-empty row would be far harder to notice."""
    tokens = torch.arange(6)
    with pytest.raises(ValueError, match="shift must be in"):
        roll_mtp_sequence(tokens, shift=shift, fill_value=FILL)


def test_short_positions_raise():
    tokens = torch.arange(6)
    with pytest.raises(ValueError, match="positions need at least"):
        roll_mtp_sequence(
            tokens, shift=1, fill_value=FILL, positions=torch.arange(3)
        )


def test_shift_without_positions_ignores_packing():
    """Omitting `positions` is the unpacked contract, and it genuinely does cross
    documents -- so the MTP decoder must always pass positions for packed rows.
    Pinned so nobody reads the optional argument as merely an optimisation."""
    tokens = torch.tensor([0, 1, 2, 10, 11])
    crossed = roll_mtp_sequence(tokens, shift=1, fill_value=FILL)
    assert crossed.tolist() == [1, 2, 10, 11, FILL]
    assert crossed[2] == 10, "without positions, doc A reads doc B's first token"
