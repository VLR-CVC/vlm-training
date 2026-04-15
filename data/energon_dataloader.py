from megatron.energon import Batch, TaskEncoder, stateless, InterleavedSample, Cooker, CrudeSample, basic_sample_keys, SkipSample
from dataclasses import dataclass


import torch
import numpy as np

@dataclass
class TextRawBatch(Batch):
    text: list[str]

@dataclass
class TextBatch(Batch):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor

@dataclass
class EncodedTextSample:
    __key__: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    length: int


class QwenTextEncoder(TaskEncoder):
    """
    Text-only encoder. Produces the same batch layout as `SingleBatchEncoder`:
    a single packed sequence of length `max_len` with `cu_seqlens` for
    FlashAttention varlen. Labels are identical to `input_ids` (with -100
    on pad positions).
    """

    def __init__(self, tokenizer, max_len):
        super().__init__()
        self.tokenizer = tokenizer
        self.max_length = max_len
        self._batch_type = None

    def encode_sample(self, sample) -> EncodedTextSample:
        text = sample.text if hasattr(sample, "text") else sample["text"]
        tokenized = self.tokenizer(
            text=text,
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.max_length,
            add_special_tokens=True,
        )
        input_ids = tokenized["input_ids"][0]
        attention_mask = tokenized["attention_mask"][0]
        key = getattr(sample, "__key__", None) or "text"
        return EncodedTextSample(
            __key__=key,
            input_ids=input_ids,
            attention_mask=attention_mask,
            length=int(input_ids.size(0)),
        )

    def batch(self, samples: list[EncodedTextSample]) -> dict:
        packed_input_ids = torch.cat([s.input_ids for s in samples])
        packed_attention_mask = torch.cat([s.attention_mask for s in samples])
        cu_seqlens = torch.tensor(
            [0] + list(np.cumsum([s.length for s in samples])), dtype=torch.int32
        )

        pad_len = self.max_length - packed_input_ids.size(0)
        if pad_len > 0:
            packed_input_ids = torch.cat(
                [packed_input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id, dtype=packed_input_ids.dtype)]
            )
            packed_attention_mask = torch.cat(
                [packed_attention_mask, torch.zeros((pad_len,), dtype=packed_attention_mask.dtype)]
            )
            cu_seqlens = torch.cat([cu_seqlens, torch.tensor([self.max_length], dtype=torch.int32)])

        packed_labels = packed_input_ids.clone()
        packed_labels[packed_attention_mask == 0] = -100

        return {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "cu_seqlens": cu_seqlens,
            "labels": packed_labels,
        }


@stateless
def cooker_captioning(sample: dict, add_system_prompt: bool = True) -> InterleavedSample:
    role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant'}
    
    messages = []
    
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
        
    image_added = False
    
    for turn in sample['json']['conversations']:
        raw_role = turn.get('from', turn.get('role', 'user'))
        role = role_map.get(str(raw_role).lower(), 'user')
        
        text_val = turn.get('value', turn.get('content', ''))
        
        content = []
        
        if "<image>" in text_val or (role == 'user' and not image_added):
            content.append({"type": "image"})
            text_val = text_val.replace("<image>", "").strip()
            image_added = True
            
        if text_val:
            content.append({"type": "text", "text": text_val})
            
        if not content:
            content.append({"type": "text", "text": ""})
            
        messages.append({"role": role, "content": content})
    
    image = sample['jpg']

    sequence = [
        image,
        messages,
    ]

    return InterleavedSample(
        **basic_sample_keys(sample),
        sequence=sequence,
    )

@dataclass
class EncodedSample:
    __key__: str
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    length: int
    labels: torch.Tensor
    pixel_values: torch.Tensor
    image_grid_thw: torch.Tensor
    mm_token_type_ids: torch.Tensor


class SingleBatchEncoder(TaskEncoder):
    """
    Given a batch size it builds a sequence. The attention mask is created with cu_seqlens, built for FlashAttention varlen.
    Does NOT perform data packing. Use in datasets where the sample sequence size does not vary.

    - CrudeWebdataset Energon dataset as input
    - The token for "assistant" has to be a single token. This is the case for the Qwen3-VL and Qwen3.5 tokenizers.
    """
    def __init__(self, processor, max_seq_len):
        super().__init__()
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.max_length = max_seq_len
        self._batch_type = None

        self.assistant_token = self.tokenizer.encode("assistant")[0]
        self.EOS_token  = self.tokenizer.eos_token_id

    cookers = [
        # subflavors can be used here to distinguish datasets when using a Metadataset
        Cooker(cooker_captioning),
    ]

    # transform the RAW data, tokenize a single sample
    def encode_sample(self, sample: InterleavedSample) -> EncodedSample:
        text = self.processor.apply_chat_template(
            conversation=sample.sequence[1],
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(text=[text], images=[sample.sequence[0]], padding=False, return_tensors="pt")

        input_ids = inputs['input_ids']

        labels = torch.full_like(input_ids, -100)
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == self.assistant_token:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != self.EOS_token:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[
                        0, ans_start : ans_end + 2
                    ]
                    pos = ans_end
            pos += 1

        # all `[0]` are used as .squeeze()
        return EncodedSample(
            __key__=sample.__key__,
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            length=len(inputs["input_ids"][0]),
            labels=labels[0],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids")[0],
        )

# collate the batch into a single sample
    def batch(self, samples: list[EncodedSample]) -> dict:
        packed_input_ids = torch.cat([s.input_ids for s in samples])
        packed_labels = torch.cat([s.labels for s in samples])
        packed_attention_mask = torch.cat([s.attention_mask for s in samples])
        packed_mm_token_types = torch.cat([s.mm_token_type_ids for s in samples])
        cu_seqlens = torch.tensor([0] + list(np.cumsum([s.length for s in samples])), dtype=torch.int32)
        
        pad_len = self.max_length - packed_input_ids.size(0)
        if pad_len > 0:
            packed_input_ids = torch.cat([packed_input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            packed_attention_mask = torch.cat([packed_attention_mask, torch.zeros((pad_len,))])
            packed_labels = torch.cat([packed_labels, torch.full((pad_len,), -100)])
            packed_mm_token_types = torch.cat([packed_mm_token_types, torch.full((pad_len,), 0)])

            cu_seqlens = torch.cat([cu_seqlens, torch.tensor([self.max_length], dtype=torch.int32)])

        batch_out = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "cu_seqlens": cu_seqlens,
            "labels": packed_labels,
            "mm_token_type_ids": packed_mm_token_types,
        }

        valid_pixel_values = [s.pixel_values for s in samples if s.pixel_values is not None]
        if valid_pixel_values:
            batch_out["pixel_values"] = torch.cat(valid_pixel_values, dim=0)
            
        valid_grid_thw = [s.image_grid_thw for s in samples if s.image_grid_thw is not None]
        if valid_grid_thw:
            packed_grid_thw = torch.cat(valid_grid_thw, dim=0)
            batch_out["image_grid_thw"] = packed_grid_thw
            
            vision_lengths = packed_grid_thw[:, 0] * packed_grid_thw[:, 1] * packed_grid_thw[:, 2]
            batch_out["vision_cu_seqlens"] = torch.nn.functional.pad(
                vision_lengths.cumsum(dim=0, dtype=torch.int32), (1, 0)
            )
            batch_out["vision_max_seqlen"] = int(vision_lengths.max().item())
        else:
            batch_out["vision_cu_seqlens"] = torch.tensor([0], dtype=torch.int32)
            batch_out["vision_max_seqlen"] = 0
            
        return batch_out

class PackedBatchEncoder(TaskEncoder):
    def __init__(self, processor, max_seq_len):
        super().__init__()
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.max_length = max_seq_len
        self._batch_type = None

        self.assistant_token = self.tokenizer.encode("assistant")
        self.EOS_token  = self.processor.eos_token_id

    cookers = [
        # subflavors can be used to distinguish datasets when using a Metadataset
        Cooker(cooker_captioning),
    ]

    # transform the RAW data, tokenize a single sample
    @stateless(restore_seeds=True)
    def encode_sample(self, sample: InterleavedSample) -> EncodedSample:
        text = self.processor.apply_chat_template(
            conversation=sample.sequence[1],
            tokenize=False,
            add_generation_prompt=False,
        )
        inputs = self.processor(text=[text], images=[sample.sequence[0]], padding=False, return_tensors="pt")

        input_ids = inputs['input_ids']

        labels = torch.full_like(input_ids, -100)
        input_ids_flat = input_ids[0].tolist()
        L = len(input_ids_flat)
        pos = 0
        while pos < L:
            if input_ids_flat[pos] == self.assistant_token:
                ans_start = pos + 2
                ans_end = ans_start
                while ans_end < L and input_ids_flat[ans_end] != self.EOS_token:
                    ans_end += 1
                if ans_end < L:
                    labels[0, ans_start : ans_end + 2] = input_ids[
                        0, ans_start : ans_end + 2
                    ]
                    pos = ans_end
            pos += 1

        # all `[0]` are used like .squeeze()
        return EncodedSample(
            __key__=sample.__key__,
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            length=len(inputs["input_ids"][0]),
            labels=labels[0],
            pixel_values=inputs.get("pixel_values"),
            image_grid_thw=inputs.get("image_grid_thw"),
            mm_token_type_ids=inputs.get("mm_token_type_ids")[0],
        )
    
    def select_samples_to_pack(self, samples: list[EncodedSample]) -> list[list[EncodedSample]]:
        samples.sort(key=lambda x: x.length, reverse=True)
        groups = []
        while samples:
            current_group = [samples.pop(0)]
            current_len = current_group[0].length
            i = 0
            while i < len(samples):
                if current_len + samples[i].length <= self.max_length:
                    sample = samples.pop(i)
                    current_group.append(sample)
                    current_len += sample.length
                else:
                    i += 1
            groups.append(current_group)
        return groups

    # collate the batch into a single sample
    @stateless
    def pack_selected_samples(self, samples: list[EncodedSample]) -> dict:
        packed_input_ids = torch.cat([s.input_ids for s in samples])
        packed_labels = torch.cat([s.labels for s in samples])
        packed_attention_mask = torch.cat([s.attention_mask for s in samples])
        packed_mm_token_types = torch.cat([s.mm_token_type_ids for s in samples])
        cu_seqlens = torch.tensor([0] + list(np.cumsum([s.length for s in samples])), dtype=torch.int32)
        
        pad_len = self.max_length - packed_input_ids.size(0)
        if pad_len > 0:
            packed_input_ids = torch.cat([packed_input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            packed_attention_mask = torch.cat([packed_attention_mask, torch.zeros((pad_len,))])
            packed_labels = torch.cat([packed_labels, torch.full((pad_len,), -100)])
            packed_mm_token_types = torch.cat([packed_mm_token_types, torch.full((pad_len,), 0)])

            cu_seqlens = torch.cat([cu_seqlens, torch.tensor([self.max_length], dtype=torch.int32)])

        batch_out = {
            "input_ids": packed_input_ids,
            "attention_mask": packed_attention_mask,
            "cu_seqlens": cu_seqlens,
            "labels": packed_labels,
            "mm_token_type_ids": packed_mm_token_types,
        }

        valid_pixel_values = [s.pixel_values for s in samples if s.pixel_values is not None]
        if valid_pixel_values:
            batch_out["pixel_values"] = torch.cat(valid_pixel_values, dim=0)
            
        valid_grid_thw = [s.image_grid_thw for s in samples if s.image_grid_thw is not None]
        if valid_grid_thw:
            batch_out["image_grid_thw"] = torch.cat(valid_grid_thw, dim=0)
            
        return batch_out