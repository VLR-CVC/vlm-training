from megatron.energon import TaskEncoder, stateless, SkipSample
from megatron.energon.flavors.webdataset.sample_decoder import SampleDecoder

from megatron.energon.edataclass import edataclass
from megatron.energon.flavors.base_dataset import Sample

import torch
import random
from PIL import Image

from data.cookers import COOKERS, EnergonSample
from data.model_batch import mrope_positions

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
    """Samples -> packed micro-batches in the input layout of `models/qwen3_5` and
    `models/qwen3_vl`, the only training path.

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

    cookers = COOKERS

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
