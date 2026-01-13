from time import sleep

import pdb

import io
import os
import json
import glob
import random
import logging
import re
import time
import itertools
import warnings
from dataclasses import dataclass
from typing import Dict, Sequence, List, Any
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset, get_worker_info
import torch.distributed as dist
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

import transformers

from . import data_list
from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100
IMAGE_TOKEN_INDEX = 151655
VIDEO_TOKEN_INDEX = 151656
DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_VIDEO_TOKEN = "<video>"

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def _make_abs_paths(base: Path, files: str) -> str:
    return f"{(base / files).resolve()}"


def _build_messages(item: Dict[str, Any], base_path: Path) -> List[Dict[str, Any]]:
    # Extract and normalize images and videos
    images = item.get("image") or []
    if isinstance(images, str):
        images = [images]

    videos = item.get("video") or []
    if isinstance(videos, str):
        videos = [videos]

    # Build media pools with absolute paths
    image_pool = [
        {"type": "image", "image": img} for img in images
    ]
    video_pool = [
        {"type": "video", "video": vid} for vid in videos
    ]

    messages = []
    for turn in item["conversations"]:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            # Split text by <image> or <video> placeholders while keeping delimiters
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg == "<video>":
                    if not video_pool:
                        raise ValueError(
                            "Number of <video> placeholders exceeds the number of provided videos"
                        )
                    content.append(video_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    # Check for unused media files
    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )
    if video_pool:
        raise ValueError(
            f"{len(video_pool)} video(s) remain unused (not consumed by placeholders)"
        )

    return messages


def preprocess_qwen_visual(
    sources,
    processor,
) -> Dict:
    if len(sources) != 1:
        raise ValueError(f"Expected 1 source, got {len(sources)}")

    source = sources[0]
    base_path = Path(source.get("data_path", ""))
    messages = _build_messages(source, base_path)

    # IMAGE PROCESSING HAPPENS HERE
    try:
        full_result = processor.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt"
        )
    except Exception as e:
        print(f'exception in apply_chat_template: {e}')
        return None

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids
    return full_result

class ParquetIterableDataset(IterableDataset):
    def __init__(self, processor, data_args):
        super().__init__()
        self.processor = processor
        self.data_args = data_args
        
        self.data_root = data_args.data_path

        self.parquet_files = sorted(glob.glob(os.path.join(self.data_root, "*.parquet")))

        self.seq_len = self.data_args.seq_len

        if len(self.parquet_files) == 0:
            raise ValueError(f"No parquet files found in {self.data_root}")

        print(f"Found {len(self.parquet_files)} parquet chunks.")

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1

        # files per GPU
        self.device_files = self.parquet_files[self.rank::self.world_size]
        self._total_samples = 0
        for f in self.device_files:
            meta = pq.read_metadata(f) 
            self._total_samples += meta.num_rows

        rank0_print(f"total samples in dataset: {self._total_samples}")

        self.model_type = data_args.model_type
        if data_args.model_type == "qwen3vl":
            self.get_rope_index = get_rope_index_3
        elif data_args.model_type == "qwen2.5vl":
            self.get_rope_index = get_rope_index_25
        elif data_args.model_type == "qwen2vl":
            self.get_rope_index = get_rope_index_2
        else:
            raise ValueError(f"model_type: {data_args.model_type} not supported")
            
        self.merge_size = getattr(processor.image_processor, "merge_size", 2)

    def __len__(self):
        return self._total_samples

    def _get_item(self, source):
        data_dict = preprocess_qwen_visual(
            [source],
            self.processor,
        )

        # if the data pre-processing fails we skip this sample
        if not data_dict:
            return None
        
        seq_len = data_dict["input_ids"][0].size(0)

        if seq_len > self.seq_len:
            return None

        if "image_grid_thw" in data_dict:
            grid_thw = data_dict.get("image_grid_thw")
            if not isinstance(grid_thw, Sequence):
                grid_thw = [grid_thw]
        else:
            grid_thw = None

        if "video_grid_thw" in data_dict:
            video_grid_thw = data_dict.get("video_grid_thw")
            if not isinstance(video_grid_thw, Sequence):
                video_grid_thw = [video_grid_thw]
            second_per_grid_ts = [
                self.processor.video_processor.temporal_patch_size
                / self.processor.video_processor.fps
            ] * len(video_grid_thw)
        else:
            video_grid_thw = None
            second_per_grid_ts = None

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
            video_grid_thw=(
                torch.cat(video_grid_thw, dim=0) if video_grid_thw else None
            ),
            second_per_grid_ts=second_per_grid_ts if second_per_grid_ts else None,
        )

        data_dict["position_ids"] = position_ids
        data_dict["attention_mask"] = torch.tensor([0, seq_len, self.seq_len], dtype=torch.int32)
        
        labels = data_dict["labels"][0]
        labels = [
            tid if tid != -100 else self.processor.tokenizer.pad_token_id
            for tid in labels
        ]
        
        return data_dict

    def __iter__(self):
        if dist.is_initialized():
            rank = dist.get_rank()
            world_size = dist.get_world_size()
        else:
            rank = 0
            world_size = 1
            
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_workers = world_size * num_workers
        global_worker_id = (rank * num_workers) + worker_id
        my_files = self.parquet_files[global_worker_id::total_workers]

        for parquet_path in my_files:
            df = pd.read_parquet(parquet_path)

            records = df.sample(frac=1).to_dict('records')

            for row in records:
                raw_imgs = row['images']

                if raw_imgs.ndim == 0:
                    image_byte_list = []
                else:
                    raw_imgs = raw_imgs.tolist()
                    image_bytes_list = [image['bytes'] for image in raw_imgs]

                output_conversations = []
                input_texts = row['texts']

                num_images = len(image_bytes_list)

                if num_images == 0:
                    warnings.warn(f'skipped samples with {num_images} images')
                    continue 

                for turn_idx, turn_data in enumerate(input_texts):
                    user_text = turn_data['user']
                    assistant_text = turn_data['assistant']

                    # removes any leftover <image> strings from the data
                    user_text = re.sub('<image>', '', user_text)

                    if turn_idx == 0:
                        if len(image_bytes_list):
                            image_tokens = "<image>" * num_images
                            user_text = image_tokens + user_text

                    output_conversations.append({
                        "from": "human",
                        "value": user_text
                    })
                    output_conversations.append({
                        "from": "assistant",
                        "value": assistant_text
                    })

                pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]
                source = {
                    "conversations": output_conversations,
                    "image": pil_images,
                    "data_path": parquet_path
                }
                
                processed_sample = self._get_item(source)
                if processed_sample is None:
                    warnings.warn('skipped sample due to large seq_len')
                    continue

                yield processed_sample
                    
def pad_and_cat(tensor_input: torch.Tensor | list, max_lenght=None, value=None):
    if value is None:
        value = 1

    if isinstance(tensor_input, torch.Tensor):
        if max_lenght is None:
            max_lenght = tensor_input.shape[-1]

        pad_length = max_lenght - tensor_input.shape[-1]

        if pad_length > 0:
            return torch.nn.functional.pad(tensor_input, (0, pad_length), "constant", value=value)
        return tensor_input

    if max_lenght is None:
        max_lenght = max(tensor.shape[-1] for tensor in tensor_input)

    padded_tensors = []
    for tensor in tensor_input:
        pad_length = max_lenght - tensor.shape[2]
        padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), "constant", value=value)
        padded_tensors.append(padded_tensor)

    stacked_tensor = torch.cat(padded_tensors, dim=1)

    return stacked_tensor


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    seq_len: int

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, position_ids = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "position_ids")
        )
        attention_mask = instances[0]['attention_mask']
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        position_ids = pad_and_cat(position_ids)
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        position_ids = position_ids[:, :, : self.tokenizer.model_max_length]

        seq_len = self.seq_len
        pad_token_id = self.tokenizer.pad_token_id

        input_ids = pad_and_cat(input_ids, max_lenght=seq_len, value=pad_token_id)
        labels = pad_and_cat(labels, max_lenght=seq_len, value=-100)
        position_ids = pad_and_cat(position_ids, max_lenght=seq_len)

        batch = dict(
            input_ids=input_ids,
            labels=labels,
        )
        images = list(
            instance["pixel_values"]
            for instance in instances
            if "pixel_values" in instance
        )
        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(images) != 0:
            concat_images = torch.cat([image for image in images], dim=0)
            grid_thw = [
                instance["image_grid_thw"]
                for instance in instances
                if "image_grid_thw" in instance
            ]
            grid_thw = torch.cat(grid_thw, dim=0)
        else:
            concat_images = None
            grid_thw = None

        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values"] = concat_images
        batch["image_grid_thw"] = grid_thw
        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw
        batch["position_ids"] = position_ids
        batch["attention_mask"] = attention_mask

        return batch

