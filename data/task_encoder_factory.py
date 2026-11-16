from __future__ import annotations

from typing import Any

from data.energon_dataloader import PackedBatchEncoder, PretrainBatchEncoder
from train.config import Data as DataArgs
from train.logger import Color, logger


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
        max_image_size=data_args.max_image_size,
        loss_unit=data_args.loss_unit,
        loss_sqrt=data_args.loss_sqrt,
        assistant_prefix=data_args.assistant_prefix,
    )
    return encoder, {"packing_buffer_size": data_args.packing_buffer_size}


def apply_chat_template(processor, path: str) -> None:
    """Override the processor's chat template, and warn about the Qwen default."""
    if path and path != "NULL":
        with open(path) as f:
            processor.chat_template = f.read()
        logger.info(f"chat template overridden from {path}")
        return

    template = getattr(processor, "chat_template", None) or ""
    if "> ns.last_query_index" in template:
        c = Color()
        logger.warning(
            f"{c.red}chat template in {processor.__class__.__name__} drops "
            f"<think> spans from all but the last turn. Fine for inference, "
            f"wrong for SFT on reasoning data. Set "
            f"training.chat_template=assets/chat_template_sft.jinja if this "
            f"dataset has reasoning.{c.reset}"
        )
