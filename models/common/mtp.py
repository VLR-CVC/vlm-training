# Vendored from torchtitan b21f7d43e: `roll_mtp_sequence` of
# torchtitan/models/deepseek_v3/mtp.py. The design is documented upstream in
# torchtitan/models/deepseek_v3/MTP.md.
# Copyright (c) Meta Platforms, Inc. and affiliates. BSD-style license, see
# https://github.com/pytorch/torchtitan/blob/b21f7d43e/LICENSE
#
# SCOPE: this is the shift kernel ONLY, not Multi-Token Prediction. Upstream MTP
# also needs `MTPTransformerBlock` (enorm/hnorm/eh_proj/mtp_norm fusing the
# shifted embedding with the previous depth's hidden state), `MTPDecoder`
# (preprocess_inputs + a prediction tuple per depth), `MTPLoss` (weighted
# auxiliary cross-entropy), `apply_fsdp_to_mtp_decoder`, and the TP/SP/EP
# sharding configs. None of those are here.
#
# Note also that upstream has MTP for deepseek_v3 and deepseek_v4 only -- NOT for
# qwen3_5. Our `configs/models/qwen3_5_*.json` carry `mtp_num_hidden_layers` and
# `mtp_use_dedicated_embeddings` because HF's config does, and
# `qwen35_config_from_hf` ignores both.
#
# This file is here because the shift is the one part of MTP with logic that is
# easy to get silently wrong (packed-document boundaries) and that can be tested
# without a model at all.

import spmd_types as spmd
import torch

__all__ = ["roll_mtp_sequence"]

def roll_mtp_sequence(
    sequence: torch.Tensor,
    *,
    shift: int,
    fill_value: int,
    positions: torch.Tensor | None = None,
    return_valid_mask: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Left-roll an MTP sequence while preserving packed-document boundaries.

    MTP depth ``k`` needs the token or label at ``i + k`` for each position
    ``i``. This helper builds that shifted view along the token axis
    (dimension 0). Tail positions, and positions that would cross a packed
    document boundary, are filled instead of wrapped around.

    Args:
        sequence: Tensor to shift, with shape ``[T, ...]``. Any trailing
            dimensions are carried along unchanged.
        shift: Future-token offset to use. ``shift=1`` maps each position to the
            next token, ``shift=2`` maps to the token after next, and so on.
            Must be positive and no larger than ``seq_len``.
        positions: Optional reset-style position IDs with shape ``[T]``. When
            present, a shifted source position is valid only if
            ``positions[i + shift] == positions[i] + shift``.
            This prevents MTP inputs or labels from crossing packed-document
            boundaries.
        fill_value: Value used for invalid positions. Use token id ``0`` for
            shifted input tokens and ``IGNORE_INDEX`` for shifted labels.
        return_valid_mask: If true, also return a boolean mask marking positions
            where the shifted value came from a valid source position.

    Returns:
        The shifted tensor. If ``return_valid_mask`` is true, returns
        ``(shifted, valid_mask)``.

    Example:
        ``sequence=[A0, A1, A2, B0, B1]`` and
        ``positions=[0, 1, 2, 0, 1]`` with ``shift=1`` returns
        ``[A1, A2, fill, B1, fill]``.

    One property that matters for THIS trainer and not for upstream's: our
    multimodal batches pad with position 0 on every padded token
    (``Qwen35Model.get_attention_masks``), so inside a padded run the validity
    test ``0 == 0 + shift`` is false everywhere and no MTP target is produced
    there. That is the behaviour we want, but it is a consequence of the
    position convention rather than something this function checks for.
    """
    seq_len = sequence.shape[0]
    if shift <= 0 or shift > seq_len:
        raise ValueError(f"MTP roll shift must be in [1, {seq_len}], got {shift}.")

    rolled = torch.full_like(sequence, fill_value)
    valid_mask = torch.zeros_like(sequence, dtype=torch.bool)

    source = sequence[shift:]
    if positions is None:
        rolled[: seq_len - shift] = source
        valid_mask[: seq_len - shift] = True
        if return_valid_mask:
            return rolled, valid_mask
        return rolled

    if positions.shape[0] < seq_len:
        raise ValueError(
            f"MTP positions need at least {seq_len} tokens, got {positions.shape[0]}."
        )
    valid_tokens = positions[shift:seq_len] == positions[: seq_len - shift] + shift
    # valid_tokens follows positions placement, while valid_mask intentionally
    # follows sequence placement for the following where.
    with spmd.no_typecheck():
        valid_mask[: seq_len - shift] = valid_tokens
    rolled[: seq_len - shift] = torch.where(
        valid_mask[: seq_len - shift],
        source,
        rolled[: seq_len - shift],
    )
    if return_valid_mask:
        return rolled, valid_mask
    return rolled
