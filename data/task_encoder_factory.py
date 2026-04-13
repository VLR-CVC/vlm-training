"""Select the right energon TaskEncoder for a given training run.

Dispatch:
    data.text_dataset == True                 -> QwenTextEncoder
    data.text_dataset == False, batch_size>0  -> SingleBatchEncoder
    data.text_dataset == False, batch_size==0 -> PackedBatchEncoder (online datapacking)

Returns (task_encoder, extra_dataset_kwargs) so the caller can splice
encoder-specific args (e.g. `packing_buffer_size`) into `get_train_dataset`.
"""

from __future__ import annotations

from typing import Any

from train.config import Data as DataArgs
from data.energon_dataloader import (
    PackedBatchEncoder,
    QwenTextEncoder,
    SingleBatchEncoder,
)


def build_task_encoder(
    data_args: DataArgs,
    tokenizer,
    processor,
) -> tuple[Any, dict[str, Any]]:
    seq_len = int(data_args.seq_len)

    if data_args.text_dataset:
        encoder = QwenTextEncoder(tokenizer=tokenizer, max_len=seq_len)
        return encoder, {}
    elif data_args.batch_size:
        encoder = SingleBatchEncoder(processor=processor, max_seq_len=seq_len)
        return encoder, {}
    elif not data_args.batch_size and data_args.packing_buffer_size:
        encoder = PackedBatchEncoder(processor, seq_len)
        return encoder, {"packing_buffer_size": data_args.packing_buffer_size}
    else:
        raise ValueError("Wrong data args, revise the config. Use `train/config.py` for guidence.")
