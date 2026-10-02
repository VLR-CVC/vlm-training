from __future__ import annotations

from typing import Any

from data.energon_dataloader import PackedBatchEncoder, PretrainBatchEncoder
from train.config import Data as DataArgs


def build_task_encoder(
    data_args: DataArgs,
    processor,
    *,
    image_token_id: int | None,
    video_token_id: int | None,
    spatial_merge_size: int | None,
) -> tuple[Any, dict[str, Any]]:
    cls = PretrainBatchEncoder if data_args.pretrain else PackedBatchEncoder
    encoder = cls(
        processor,
        int(data_args.seq_len),
        data_args.microbatch_tokens,
        image_token_id=-1 if image_token_id is None else image_token_id,
        video_token_id=-1 if video_token_id is None else video_token_id,
        spatial_merge_size=spatial_merge_size or 1,
    )
    return encoder, {"packing_buffer_size": data_args.packing_buffer_size}
