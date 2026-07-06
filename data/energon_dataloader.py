from megatron.energon import Batch, TaskEncoder, stateless, InterleavedSample, Cooker, basic_sample_keys, SkipSample
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder
from dataclasses import dataclass

from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import Sample

import torch
import random
import numpy as np
from PIL import Image

MAX_IMAGE_SIZE = 1024

def cap_image_size(image, max_size: int = MAX_IMAGE_SIZE):
    """Downscale a PIL image so neither side exceeds `max_size`, preserving aspect
    ratio. Returns the image unchanged if it already fits (or is None)."""
    if image is None:
        return None
    w, h = image.size
    longest = max(w, h)
    if longest <= max_size:
        return image
    scale = max_size / longest
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return image.resize(new_size, Image.BICUBIC)

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


@edataclass
class EnergonSample(Sample):
    image: torch.Tensor | None
    messages: list

@stateless
def cooker_llava_recap(sample: dict) -> EnergonSample:

    caption = sample['json']['conversations'][1]['value']
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"}
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": caption}
        ]},
    ]

    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@stateless
def cooker_llava_imagenet(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"} 
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": sample['txt']}
        ]},
    ]
    
    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})
        
    image = sample['jpg']

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@stateless
def cooker_onevision_instruct(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    role_map = {'human': 'user', 'gpt': 'assistant', 'user': 'user', 'assistant': 'assistant'}

    has_image = sample.get('jpg') is not None

    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    image_added = False

    for turn in sample['json']['conversations']:
        raw_role = turn.get('from', turn.get('role', 'user'))
        role = role_map.get(str(raw_role).lower(), 'user')

        text_val = turn.get('value') or turn.get('content') or ''

        wants_image = has_image and not image_added and (
            "<image>" in text_val or role == 'user'
        )
        text_val = text_val.replace("<image>", "").strip()

        content = []

        if wants_image:
            content.append({"type": "image"})
            image_added = True

        if text_val:
            content.append({"type": "text", "text": text_val})

        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": role, "content": content})

    return EnergonSample(
        **basic_sample_keys(sample),
        image=sample['jpg'] if has_image else None,
        messages=messages,
    )


@stateless
def cooker_captioning(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
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

    return EnergonSample(
        **basic_sample_keys(sample),
        image=image,
        messages=messages,
    )

@edataclass
class EncodedSample(Sample):
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
    # Decode images as PIL (uint8 [0,255]). The energon default decoder ("torchrgb")
    # returns float [0,1] tensors, which the HF processor then re-rescales (do_rescale
    # ÷255) + normalizes, collapsing pixels to a near-constant ~-0.99 blank image -> the
    # model trains on destroyed images. PIL feeds the processor what it expects.
    # See utils/diff_image_preprocessing.py.
    decoder = SampleDecoder(image_decode="pil")

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
        images = [cap_image_size(sample.image)] if sample.image is not None else None
        inputs = self.processor(text=[text], images=images, padding=False, return_tensors="pt")

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
        # derive_from carries the source sample's __restore_key__ so the loader
        # state can be checkpointed/restored.
        return EncodedSample.derive_from(
            sample,
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
            batch_out["image_grid_thw"] = torch.cat(valid_grid_thw, dim=0)
            
        return batch_out

class PackedBatchEncoder(TaskEncoder):
    # PIL image decode (uint8 [0,255]); avoids the float[0,1] -> double-rescale ->
    # near-blank image bug with the HF processor. See utils/diff_image_preprocessing.py.
    decoder = SampleDecoder(image_decode="pil")

    def __init__(self, processor, max_seq_len):
        super().__init__()
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.max_length = max_seq_len
        self._batch_type = None

        self.assistant_token = self.tokenizer.encode("assistant")[0]
        self.EOS_token  = self.tokenizer.eos_token_id

    cookers = [
        # subflavors can be used to distinguish datasets when using a Metadataset
        Cooker(cooker_captioning, has_subflavors={"type_dataset": "synth"}),
        Cooker(cooker_llava_recap, has_subflavors={"type_dataset": "llava_recap"}),
        Cooker(cooker_llava_imagenet, has_subflavors={"type_dataset": "llava_recap_mn5"}),
        Cooker(cooker_onevision_instruct, has_subflavors={"type_dataset": "onevision_instruct"}),
    ]

    # transform the RAW data, tokenize a single sample
    @stateless(restore_seeds=True)
    def encode_sample(self, sample: EnergonSample) -> EncodedSample:
        text = self.processor.apply_chat_template(
            conversation=sample.messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Text-only samples carry no image -> pass images=None so the processor returns
        # no pixel_values / image_grid_thw / mm_token_type_ids. Cap oversized images first.
        images = [cap_image_size(sample.image)] if sample.image is not None else None
        inputs = self.processor(text=[text], images=images, padding=False, return_tensors="pt")

        input_ids = inputs['input_ids']

        if input_ids.shape[1] > self.max_length:
            raise SkipSample()

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

        # samples with no loss are skipped
        # does not affect training stability just logs `nan` loss
        if (labels[0] != -100).sum() == 0:
            raise SkipSample()

        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None and pixel_values.ndim > 1 and pixel_values.shape[0] == 1:
            pixel_values = pixel_values[0] # remove dummy batch dim

        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None and grid_thw.ndim == 3 and grid_thw.shape[0] == 1:
            grid_thw = grid_thw[0] # remove dummy batch dim -> (num_images, 3)

        # mm_token_type_ids is absent for text-only samples -> all-text (zeros).
        mm_token_type_ids = inputs.get("mm_token_type_ids")
        if mm_token_type_ids is not None:
            mm_token_type_ids = mm_token_type_ids[0]
        else:
            mm_token_type_ids = torch.zeros_like(inputs["input_ids"][0])

        # all `[0]` are used like .squeeze()
        # derive_from carries the source sample's __restore_key__ so the loader
        # state can be checkpointed/restored.
        return EncodedSample.derive_from(
            sample,
            input_ids=inputs["input_ids"][0],
            attention_mask=inputs["attention_mask"][0],
            length=len(inputs["input_ids"][0]),
            labels=labels[0],
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            mm_token_type_ids=mm_token_type_ids,
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

        random.shuffle(groups)
        return groups

    # collate the batch into a single sample
    @stateless
    def pack_selected_samples(self, samples: list[EncodedSample]) -> dict:
        packed_input_ids = torch.cat([s.input_ids for s in samples])
        packed_labels = torch.cat([s.labels for s in samples])
        packed_attention_mask = torch.cat([s.attention_mask for s in samples])
        packed_mm_token_types = torch.cat([s.mm_token_type_ids for s in samples])
        cu_seqlens = torch.tensor([0] + list(np.cumsum([s.length for s in samples])), dtype=torch.int32).squeeze()
        
        pad_len = self.max_length - packed_input_ids.size(0)
        if pad_len > 0:
            packed_input_ids = torch.cat([packed_input_ids, torch.full((pad_len,), self.tokenizer.pad_token_id)])
            packed_attention_mask = torch.cat([packed_attention_mask, torch.zeros((pad_len,))])
            packed_labels = torch.cat([packed_labels, torch.full((pad_len,), -100)])
            packed_mm_token_types = torch.cat([packed_mm_token_types, torch.full((pad_len,), 0)])

            cu_seqlens = torch.cat([cu_seqlens, torch.tensor([self.max_length], dtype=torch.int32)]).squeeze()

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
