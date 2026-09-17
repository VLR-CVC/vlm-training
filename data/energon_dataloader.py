from megatron.energon import Batch, TaskEncoder, stateless, Cooker, basic_sample_keys, SkipSample
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
    Text-only encoder for `eval/val.py` (HF models; not used for training):
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

@stateless
def cooker_ovevision_midtraining(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    messages = [
        {'role': 'user', 'content': [
            {"type": "image"} 
        ]},
        {'role': 'assistant', 'content': [
            {"type": "text", "text": sample['json']['caption']}
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
def cooker_olmo_ocr(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
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
def cooker_finevision(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
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
def cooker_idl(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
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
def cooker_nemotron(sample: dict, add_system_prompt: bool = True) -> EnergonSample:
    """Nemotron-VLM-Dataset v2/v3, repacked by `utils/prepare_nemotron_energon.py`.

    Unlike the other subsets here, the published `messages` are already
    role/content-list shaped, so this is a filter rather than a rebuild: the
    image part carries a filename and a metadata dict that the processor must
    not see. The packer guarantees exactly one image per sample, so a bare
    `{"type": "image"}` resolves unambiguously to `sample['png']`.

    Images stay PNG (the source is PNG; re-encoding to match the `jpg` key the
    other cookers use would be a lossy no-op), hence `sample['png']`.

    Assistant turns keep their `<think>...</think>` spans verbatim -- these are
    CoT subsets and the reasoning is the training signal.
    """
    return EnergonSample(
        **basic_sample_keys(sample),
        image=sample["png"],
        messages=nemotron_messages(sample["json"]["messages"], add_system_prompt),
    )


def nemotron_messages(turns: list, add_system_prompt: bool = True) -> list[dict]:
    """The message rebuild half of `cooker_nemotron`, split out so tooling that
    reads the shards directly (`utils/tokenizer_compare.py`) counts exactly the
    tokens training sees rather than a re-derivation that can drift from it."""
    messages = []

    if not add_system_prompt:
        messages.append({"role": "system", "content": [{"type": "text", "text": ""}]})

    for turn in turns:
        content = []
        for part in turn["content"]:
            # v3 writes text parts as bare strings, v2 as {"type": "text", ...}
            if isinstance(part, str):
                if part:
                    content.append({"type": "text", "text": part})
            elif part.get("type") == "image":
                content.append({"type": "image"})
            elif part.get("text"):
                content.append({"type": "text", "text": part["text"]})

        if not content:
            content.append({"type": "text", "text": ""})

        messages.append({"role": turn["role"], "content": content})

    return messages


@edataclass
class PackedSample(Sample):
    """One document, already in the model's per-token layout (see `pack_selected_samples`)."""
    input_ids: torch.Tensor        # (L,)
    labels: torch.Tensor           # (L,) the token to predict at t, -100 on the last token
    positions: torch.Tensor        # (L,) arange
    mrope_positions: torch.Tensor  # (L, 3)
    pixel_values: torch.Tensor | None
    image_grid_thw: torch.Tensor | None
    length: int


class PackedBatchEncoder(TaskEncoder):
    """Samples -> packed micro-batches in the input layout of `models/qwen3_5_tt` and
    `models/qwen3_vl_tt`, the only training path.

    Each sample becomes one document of at most `seq_len` tokens. Documents are
    first-fit-decreasing packed into rows of exactly `seq_len` tokens (right padding
    is a segment of its own), and `tokens_per_microbatch // seq_len` rows are joined
    into one micro-batch -- torchtitan's layout (TITAN_MIGRATION_v2.md S4).

    A micro-batch is a dict:
      input (T,), labels (T,) shifted with -100 on every document's last token,
      positions (T,) restarting at 0 per document (the model derives the varlen
      offsets from them), mrope_positions (T, 3), pixel_values / grid_thw when the
      row has images, and host ints num_valid_tokens (loss normaliser), num_tokens
      (non-padding) and num_samples.

    Per-sample work (labels, positions, MRoPE) runs here, in the loader workers.
    """

    # PIL image decode (uint8 [0,255]); avoids the float[0,1] -> double-rescale ->
    # near-blank image bug with the HF processor. See utils/diff_image_preprocessing.py.
    decoder = SampleDecoder(image_decode="pil")

    def __init__(
        self,
        processor,
        seq_len: int,
        tokens_per_microbatch: int,
        *,
        image_token_id: int,
        video_token_id: int,
        spatial_merge_size: int,
    ):
        super().__init__()
        if tokens_per_microbatch % seq_len:
            raise ValueError(f"tokens_per_microbatch {tokens_per_microbatch} is not a multiple of seq_len {seq_len}")
        self.processor = processor
        self.tokenizer = self.processor.tokenizer
        self.row_length = seq_len
        self.rows = tokens_per_microbatch // seq_len
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id
        self.spatial_merge_size = spatial_merge_size
        # The chat template opens every answer with `<|im_start|>assistant\n`. Matching
        # the whole prefix, not the `assistant` token alone: that token is also what
        # "assistant" at the start of a line in user text encodes to, and would
        # supervise the rest of that turn.
        self.assistant_prefix = self.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)
        self.EOS_token = self.tokenizer.eos_token_id

    cookers = [
        # subflavors can be used to distinguish datasets when using a Metadataset
        Cooker(cooker_captioning, has_subflavors={"type_dataset": "synth_cap"}),
        Cooker(cooker_captioning, has_subflavors={"type_dataset": "synth_finevision"}),
        Cooker(cooker_llava_recap, has_subflavors={"type_dataset": "llava_recap"}),
        Cooker(cooker_llava_imagenet, has_subflavors={"type_dataset": "llava_recap_mn5"}),
        Cooker(cooker_onevision_instruct, has_subflavors={"type_dataset": "onevision_instruct"}),
        Cooker(cooker_ovevision_midtraining, has_subflavors={"type_dataset": "onevision_midtraining"}),
        Cooker(cooker_olmo_ocr, has_subflavors={"type_dataset": "olmo_ocr"}),
        Cooker(cooker_finevision, has_subflavors={"type_dataset": "finevision"}),
        Cooker(cooker_idl, has_subflavors={"type_dataset": "idl_ocr"}),
        Cooker(cooker_nemotron, has_subflavors={"type_dataset": "plotqa_cot"}),
        Cooker(cooker_nemotron, has_subflavors={"type_dataset": "nemotron"}),
    ]

    def assistant_labels(self, ids: list[int]) -> torch.Tensor:
        """Unshifted labels: every assistant answer, through its `<|im_end|>` and the
        newline after it, as the id; everything else -100."""
        labels = torch.full((len(ids),), -100, dtype=torch.long)
        prefix, n = self.assistant_prefix, len(self.assistant_prefix)
        pos = 0
        while pos + n <= len(ids):
            if ids[pos : pos + n] != prefix:
                pos += 1
                continue
            start = end = pos + n
            while end < len(ids) and ids[end] != self.EOS_token:
                end += 1
            if end == len(ids):
                break  # unterminated answer: truncated, leave it unsupervised
            labels[start : end + 2] = torch.tensor(ids[start : end + 2])
            pos = end
        return labels

    @stateless(restore_seeds=True)
    def encode_sample(self, sample: EnergonSample) -> PackedSample:
        from data.model_batch import mrope_positions

        text = self.processor.apply_chat_template(
            conversation=sample.messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        # Text-only samples carry no image -> pass images=None so the processor returns
        # no pixel_values / image_grid_thw. Cap oversized images first.
        images = [cap_image_size(sample.image)] if sample.image is not None else None
        inputs = self.processor(text=[text], images=images, padding=False, return_tensors="pt")

        input_ids = inputs["input_ids"][0]
        length = input_ids.shape[0]
        if length > self.row_length:
            raise SkipSample()
        labels = self.assistant_labels(input_ids.tolist())
        if (labels != -100).sum() == 0:
            raise SkipSample()  # nothing to learn from

        shifted = torch.full_like(labels, -100)
        shifted[:-1] = labels[1:]

        grid_thw = inputs.get("image_grid_thw")
        if grid_thw is not None and grid_thw.ndim == 3:
            grid_thw = grid_thw[0]  # remove dummy batch dim -> (num_images, 3)
        pixel_values = inputs.get("pixel_values")
        if pixel_values is not None and pixel_values.ndim == 3:
            pixel_values = pixel_values[0]

        # derive_from carries the source sample's __restore_key__ so the loader
        # state can be checkpointed/restored.
        return PackedSample.derive_from(
            sample,
            input_ids=input_ids,
            labels=shifted,
            positions=torch.arange(length),
            mrope_positions=mrope_positions(
                input_ids,
                torch.tensor([0, length], dtype=torch.int32),
                grid_thw,
                image_token_id=self.image_token_id,
                video_token_id=self.video_token_id,
                spatial_merge_size=self.spatial_merge_size,
            ),
            pixel_values=pixel_values,
            image_grid_thw=grid_thw,
            length=length,
        )

    def select_samples_to_pack(self, samples: list[PackedSample]) -> list[list[PackedSample]]:
        samples.sort(key=lambda x: x.length, reverse=True)
        groups = []
        while samples:
            current_group = [samples.pop(0)]
            current_len = current_group[0].length
            i = 0
            while i < len(samples):
                if current_len + samples[i].length <= self.row_length:
                    sample = samples.pop(i)
                    current_group.append(sample)
                    current_len += sample.length
                else:
                    i += 1
            groups.append(current_group)

        random.shuffle(groups)
        # one micro-batch = `rows` bins, back to back; pack_selected_samples recovers
        # the row boundaries greedily (the longest prefix that fits a row is the first bin)
        return [sum(groups[i : i + self.rows], []) for i in range(0, len(groups), self.rows)]

    @stateless
    def pack_selected_samples(self, samples: list[PackedSample]) -> dict:
        rows, row, used = [], [], 0
        for sample in samples:
            if used + sample.length > self.row_length:
                rows.append(row)
                row, used = [], 0
            row.append(sample)
            used += sample.length
        rows.append(row)
        if len(rows) > self.rows:
            raise ValueError(f"{len(samples)} samples do not fit {self.rows} rows of {self.row_length}")
        rows += [[] for _ in range(self.rows - len(rows))]

        input_ids, labels, positions, mrope = [], [], [], []
        for row in rows:
            pad = self.row_length - sum(s.length for s in row)
            input_ids += [s.input_ids for s in row] + [torch.full((pad,), self.tokenizer.pad_token_id)]
            labels += [s.labels for s in row] + [torch.full((pad,), -100)]
            # padding is a document of its own: positions restart at 0
            positions += [s.positions for s in row] + [torch.arange(pad)]
            mrope += [s.mrope_positions for s in row] + [torch.arange(pad).unsqueeze(-1).expand(-1, 3)]

        labels = torch.cat(labels)
        batch = {
            "input": torch.cat(input_ids),
            "labels": labels,
            "positions": torch.cat(positions),
            "mrope_positions": torch.cat(mrope),
            "num_valid_tokens": int((labels != -100).sum()),
            "num_tokens": sum(s.length for s in samples),
            "num_samples": len(samples),
        }
        with_images = [s for s in samples if s.pixel_values is not None]
        if with_images:
            batch["pixel_values"] = torch.cat([s.pixel_values for s in with_images])
            batch["grid_thw"] = torch.cat([s.image_grid_thw for s in with_images])
        return batch

    @stateless
    def batch(self, samples: list[dict]) -> dict:
        # the loader runs with batch_size=1: pass the packed micro-batch through
        # uncollated, so no field gains a leading dimension
        return samples[0]
