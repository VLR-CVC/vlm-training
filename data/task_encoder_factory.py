"""The energon TaskEncoder for training: `PackedBatchEncoder`.
"""

from __future__ import annotations

from typing import Any

from data.energon_dataloader import PackedBatchEncoder
from train.config import Data as DataArgs


def build_task_encoder(
    data_args: DataArgs,
    processor,
    *,
    image_token_id: int,
    video_token_id: int,
    spatial_merge_size: int,
) -> tuple[Any, dict[str, Any]]:
    """Returns (task_encoder, extra kwargs for `get_train_dataset`)."""
    encoder = PackedBatchEncoder(
        processor,
        int(data_args.seq_len),
        data_args.microbatch_tokens,
        image_token_id=image_token_id,
        video_token_id=video_token_id,
        spatial_merge_size=spatial_merge_size,
    )
    return encoder, {"packing_buffer_size": data_args.packing_buffer_size}
