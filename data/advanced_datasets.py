import io
import os
import re
import glob
import torch
import threading
import itertools
import warnings
import numpy as np
import pandas as pd

from pathlib import Path
from PIL import Image
from queue import Queue
from typing import Iterator, List, Dict, Any, Sequence
from torch.utils.data import IterableDataset, get_worker_info
import torch.distributed as dist
import pyarrow.parquet as pq

from transformers import Qwen2_5_VLProcessor

from .rope2d import get_rope_index_25, get_rope_index_2, get_rope_index_3

IGNORE_INDEX = -100

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

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt"
    )

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


class ShardedParquetSource(IterableDataset):
    def __init__(self, data_path: str, start_idx: int, end_idx: int):
        super().__init__()
        self.data_root = data_path
        
        all_files = sorted(glob.glob(os.path.join(self.data_root, "*.parquet")))

        self.parquet_files = []
        for p_path in all_files:
            fname = os.path.basename(p_path)

            digits = re.search(r'(\d+)', fname)
            file_id = int(digits.group(1))

            if start_idx <= file_id <= end_idx:
                self.parquet_files.append(p_path)
        
        print(f"Found {len(self.parquet_files)} parquet chunks in range: {start_idx, end_idx}")

        if dist.is_initialized():
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
            
        self._total_samples = 0
        device_files = self.parquet_files[self.rank::self.world_size]
        for f in device_files:
            try:
                self._total_samples += pq.read_metadata(f).num_rows
            except:
                pass

    def __len__(self):
        return self._total_samples

    @property
    def files(self):
        return self.parquet_files

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        total_workers = self.world_size * num_workers
        global_worker_id = (self.rank * num_workers) + worker_id
        
        # files for a specific GPU
        my_files = self.parquet_files[global_worker_id::total_workers]

        for parquet_path in my_files:
            try:
                df = pd.read_parquet(parquet_path)
                records = df.sample(frac=1).to_dict('records')

                for row in records:
                    raw_imgs = row['images']
                    image_bytes_list = []
                    
                    if hasattr(raw_imgs, 'ndim') and raw_imgs.ndim == 0:
                        pass
                    elif raw_imgs is not None:
                        raw_imgs_list = raw_imgs.tolist() if hasattr(raw_imgs, 'tolist') else raw_imgs
                        if isinstance(raw_imgs_list, list):
                            image_bytes_list = [img['bytes'] for img in raw_imgs_list]

                    num_images = len(image_bytes_list)

                    # SAMPLES WITHOUT IMAGES ARE SKIPPED
                    if num_images == 0:
                        continue 

                    input_texts = row['texts']
                    output_conversations = []
                    
                    for turn_idx, turn_data in enumerate(input_texts):
                        user_text = turn_data['user']
                        assistant_text = turn_data['assistant']
                        
                        # add <image> to user prompt
                        user_text = re.sub('<image>', '', user_text)
                        if turn_idx == 0:
                            image_tokens = "<image>" * num_images
                            user_text = image_tokens + user_text

                        output_conversations.append({"from": "human", "value": user_text})
                        output_conversations.append({"from": "assistant", "value": assistant_text})

                    pil_images = [Image.open(io.BytesIO(b)).convert("RGB") for b in image_bytes_list]

                    yield {
                        "conversations": output_conversations,
                        "images": pil_images,
                        "source_id": parquet_path 
                    }

            except Exception as e:
                warnings.warn(f"Error reading {parquet_path}: {e}")
                continue

class QwenPackedDataset(IterableDataset):
    def __init__(
        self,
        dataset,
        processor,
        data_args,
        queue_size: int = 16,
        infinite: bool = True,
        max_images_per_knapsack: int = 32,
    ):
        self.dataset = dataset
        self.processor = processor
        self.data_args = data_args
        self.max_seq_len = self.data_args.seq_len
        self.queue_size = self.data_args.queue_len
        self.infinite = infinite
        self.max_images_per_knapsack = max_images_per_knapsack
        
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
        self._sentinel = object()

    def __len__(self):
        return len(self.dataset)

    def __iter__(self) -> Iterator[dict]:

        def make_iter():
            return iter(self.dataset)

        queue = Queue(maxsize=self.queue_size)
        
        producer = threading.Thread(
            target=self._producer, args=(make_iter, queue), daemon=True
        )
        producer.start()

        while True:
            packed_batch = queue.get()
            if packed_batch is self._sentinel:
                break
            yield packed_batch

    def _process_single_item(self, source):
        """
        Runs Qwen preprocessing (tokenization + RoPE + Grid calculation) 
        on a single raw item.
        """
        source = {
            "conversations": source["conversations"],
            "image": source["images"],
            "data_path": source["source_id"],
        }

        try:
            data_dict = preprocess_qwen_visual([source], self.processor)
        except Exception as e:
            print(f'got exception {e} on preprocessing')
            return None

        seq_len = data_dict["input_ids"][0].size(0)

        grid_thw = None
        if "image_grid_thw" in data_dict:
            grid_thw = data_dict["image_grid_thw"]
            if not isinstance(grid_thw, Sequence): grid_thw = [grid_thw]

        position_ids, _ = self.get_rope_index(
            self.merge_size,
            data_dict["input_ids"],
            image_grid_thw=torch.cat(grid_thw, dim=0) if grid_thw else None,
        )

        return {
            "input_ids": data_dict["input_ids"][0], # 1D tensor
            "labels": data_dict["labels"][0],       # 1D tensor
            "position_ids": position_ids[0],        # 3D tensor [3, Seq_Len]
            "pixel_values": data_dict.get("pixel_values"),
            "image_grid_thw": grid_thw,
            "seq_len": seq_len
        }

    def _producer(self, make_iterator, queue: Queue):
        iterator = make_iterator()
        buffer = []
        
        while True:
            raw_item = next(iterator)
            processed_item = self._process_single_item(raw_item)

            # processing error
            if processed_item is None:
                continue
            
            # max len
            if processed_item["seq_len"] > self.max_seq_len:
                continue 

            # null images
            num_images = processed_item['image_grid_thw'][0].shape[0]
            if num_images == 0:
                continue

            buffer.append(processed_item)

            current_total_len = sum(x["seq_len"] for x in buffer)
            if current_total_len < self.max_seq_len:
                continue

            groups = self._balanced_greedy_knapsack(
                buffer, 
                self.max_seq_len, 
                max_images_per_knapsack=self.max_images_per_knapsack
            )

            used_indices = set()

            for group_indices in groups:
                selected_items = [buffer[i] for i in group_indices]
                batch_len = sum(x['seq_len'] for x in selected_items)

                if batch_len >= self.max_seq_len * .9:
                    packed_batch = self._pack_one_group(group_indices, buffer)
                    queue.put(packed_batch)
                    used_indices.update(group_indices)

            buffer = [item for i, item in enumerate(buffer) if i not in used_indices]
            if len(buffer) > self.queue_size * 4:
                pass

        if buffer:
             groups = [[i] for i in range(len(buffer))] 
             for g in groups:
                 queue.put(self._pack_one_group(g, buffer))
                 
        queue.put(self._sentinel)

    def _balanced_greedy_knapsack(self, buffer, L, delta=0, max_images_per_knapsack=None):
        lengths = [x["seq_len"] for x in buffer]
        image_counts = []
        for x in buffer:
            cnt = 0
            if x["pixel_values"] is not None:
                cnt += x["pixel_values"].shape[0]
            image_counts.append(cnt)

        items = sorted(
            enumerate(zip(lengths, image_counts)), 
            key=lambda x: x[1][0], 
            reverse=True
        )

        knapsack_groups = []
        knapsack_loads = []
        knapsack_imgs = []

        for idx, (item_len, item_img_cnt) in items:
            best_ks = -1
            
            for ks_id, (load, img_load) in enumerate(zip(knapsack_loads, knapsack_imgs)):
                if (load + item_len <= L):
                    best_ks = ks_id
                    break # Greedy first fit (since sorted)
            
            if best_ks != -1:
                knapsack_groups[best_ks].append(idx)
                knapsack_loads[best_ks] += item_len
                knapsack_imgs[best_ks] += item_img_cnt
            else:
                knapsack_groups.append([idx])
                knapsack_loads.append(item_len)
                knapsack_imgs.append(item_img_cnt)

        return knapsack_groups

    def _pack_one_group(self, group_indices, buffer) -> Dict[str, torch.Tensor]:
        """
        Concatenates items into a Flattened Varlen Batch.
        """
        selected_items = [buffer[i] for i in group_indices]
        
        input_ids = torch.cat([x["input_ids"] for x in selected_items], dim=0)
        labels = torch.cat([x["labels"] for x in selected_items], dim=0)
        
        position_ids = torch.cat([x["position_ids"] for x in selected_items], dim=1)

        seq_lens = [x["seq_len"] for x in selected_items]
        cu_seqlens = torch.tensor([0] + list(itertools.accumulate(seq_lens)) + [self.max_seq_len], dtype=torch.int32)

        pixel_values = []
        image_grid_thw = []
        pixel_values_videos = []
        video_grid_thw = []

        for x in selected_items:
            if x["pixel_values"] is not None:
                pixel_values.append(x["pixel_values"])
            if x["image_grid_thw"] is not None:
                image_grid_thw.extend(x["image_grid_thw"]) # extend list of tensors
            
        batch = {
            "input_ids": input_ids.unsqueeze(0),       # [1, Total_Len]
            "labels": labels.unsqueeze(0),             # [1, Total_Len]
            "position_ids": position_ids.unsqueeze(0), # [1, 3, Total_Len]
            "attention_mask": cu_seqlens,              # PASSED AS CU_SEQLENS
        }

        if pixel_values:
            batch["pixel_values"] = torch.cat(pixel_values, dim=0)
            batch["image_grid_thw"] = torch.cat(image_grid_thw, dim=0)
        
        return batch
