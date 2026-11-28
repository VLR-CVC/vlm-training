"""Tests for online video decoding via Energon's AVDecoder."""

from __future__ import annotations

import io
from fractions import Fraction

import numpy as np
import pytest
import torch
from PIL import Image

av = pytest.importorskip("av")

from megatron.energon import SampleDecoder, SkipSample
from megatron.energon.av import AVDecoder
from megatron.energon.av import av_decoder as _av_decoder_mod

if not _av_decoder_mod.AV_DECODE_AVAILABLE:
    pytest.skip(f"megatron-energon AV decoding unavailable: {_av_decoder_mod.MISSING_DEPENDENCY}", allow_module_level=True)

from data.cookers import cooker_video
from data.energon_dataloader import PackedBatchEncoder
from data.video_decode import decode_video_frames


def _make_mp4_bytes(num_frames: int = 30, fps: int = 10, width: int = 160, height: int = 96) -> bytes:
    """Create a valid in-memory H.264 MP4 whose red channel ramps monotonically with frame index."""
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as container:
        stream = container.add_stream("mpeg4", rate=fps)
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.time_base = Fraction(1, fps)
        for i in range(num_frames):
            r = int(round(255 * i / max(1, num_frames - 1)))
            arr = np.zeros((height, width, 3), dtype=np.uint8)
            arr[:, :, 0] = r
            arr[:, :, 1] = 40
            arr[:, :, 2] = 255 - r
            frame = av.VideoFrame.from_ndarray(arr, format="rgb24")
            frame.pts = i
            for pkt in stream.encode(frame):
                container.mux(pkt)
        for pkt in stream.encode():
            container.mux(pkt)
    return buf.getvalue()


def test_sample_decoder_returns_avdecoder():
    raw = _make_mp4_bytes(num_frames=20, fps=10)
    sd = SampleDecoder(image_decode="pil", av_decode="AVDecoder", video_decode_audio=False)
    decoded = sd({"__key__": "clip0", "mp4": raw, "json": b"{}"})
    assert isinstance(decoded["mp4"], AVDecoder)


def test_decode_video_frames_basic_and_resize():
    # 3.0 s video at 10 fps (30 frames), 160x96
    raw = _make_mp4_bytes(num_frames=30, fps=10, width=160, height=96)
    dec = AVDecoder(io.BytesIO(raw))

    frames, eff_fps = decode_video_frames(dec, fps=2.0, max_frames=32, max_side=80)
    # 3.0s * 2.0 fps = 6 frames (already even)
    assert len(frames) == 6
    assert abs(eff_fps - 2.0) < 1e-3
    assert all(isinstance(f, Image.Image) for f in frames)
    # Downscaled preserving 160:96 aspect ratio -> 80x48
    assert frames[0].size == (80, 48)
    # Red channel increases monotonically over time
    reds = [float(np.asarray(f)[:, :, 0].mean()) for f in frames]
    assert reds == sorted(reds) and reds[-1] - reds[0] > 100


def test_decode_video_frames_max_frames_cap_and_even_align():
    # 4.0 s video at 10 fps
    raw = _make_mp4_bytes(num_frames=40, fps=10, width=96, height=64)
    dec = AVDecoder(io.BytesIO(raw))

    # 4.0s * 5.0 fps = 20, capped to max_frames=5 -> aligned down to 4
    frames, eff_fps = decode_video_frames(dec, fps=5.0, max_frames=5, max_side=0)
    assert len(frames) == 4
    assert abs(eff_fps - 1.0) < 1e-3
    assert frames[0].size == (96, 64)


def test_decode_video_frames_clip_subrange():
    # 4.0 s video; sample second half [2.0, 4.0] -> red channel should be high (>100)
    raw = _make_mp4_bytes(num_frames=40, fps=10, width=96, height=64)
    dec = AVDecoder(io.BytesIO(raw))

    frames, eff_fps = decode_video_frames(dec, fps=2.0, max_frames=16, max_side=64, start=2.0, end=4.0)
    assert len(frames) == 4
    assert abs(eff_fps - 2.0) < 1e-3
    first_red = float(np.asarray(frames[0])[:, :, 0].mean())
    assert first_red > 100


def test_cooker_video_polylithic_and_llava():
    raw = _make_mp4_bytes(num_frames=20, fps=10)
    dec = AVDecoder(io.BytesIO(raw))
    img = Image.new("RGB", (32, 32))

    # Polylithic with interleaved image + video + per-video overrides
    poly_sample = {
        "__key__": "p0",
        "__restore_key__": ("wds", 0),
        "__subflavors__": {"type_dataset": "video"},
        "v0.mp4": dec,
        "img.jpg": img,
        "json": {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "v0.mp4", "fps": 1.0, "start": 0.5, "end": 2.0},
                        {"type": "image", "image": "img.jpg"},
                        "Describe both.",
                    ],
                },
                {"role": "assistant", "content": ["A clip and an image."]},
            ]
        },
    }
    cooked = cooker_video(poly_sample)
    assert cooked.image is img
    assert len(cooked.videos) == 1
    assert cooked.videos[0]["decoder"] is dec
    assert cooked.videos[0]["fps"] == 1.0
    assert cooked.videos[0]["start"] == 0.5
    assert cooked.videos[0]["end"] == 2.0

    # LLaVA-Video / ShareGPT format
    llava_sample = {
        "__key__": "l0",
        "__restore_key__": ("wds", 1),
        "__subflavors__": {"type_dataset": "video"},
        "mp4": dec,
        "json": {
            "conversations": [
                {"from": "human", "value": "<video>\nWhat happens?"},
                {"from": "gpt", "value": "Colors shift."},
            ]
        },
    }
    cooked2 = cooker_video(llava_sample)
    assert len(cooked2.videos) == 1
    assert cooked2.videos[0]["decoder"] is dec
    assert cooked2.messages[0]["content"][0] == {"type": "video"}
    assert cooked2.messages[0]["content"][1] == {"type": "text", "text": "What happens?"}


def test_encode_sample_skips_corrupt_video():
    class _DummyTokenizer:
        eos_token_id = 2
        unk_token_id = 0
        pad_token_id = 0

        def encode(self, s, add_special_tokens=False):
            return [10, 11]

        def convert_tokens_to_ids(self, tok):
            return 2

    class _DummyProcessor:
        tokenizer = _DummyTokenizer()

        def apply_chat_template(self, conversation, tokenize=False, add_generation_prompt=False):
            return "dummy"

    enc = PackedBatchEncoder(
        _DummyProcessor(),
        seq_len=128,
        tokens_per_microbatch=128,
        image_token_id=100,
        video_token_id=101,
        spatial_merge_size=2,
        max_image_size=448,
        video_fps=2.0,
        video_max_frames=8,
        video_max_side=128,
    )
    bad_sample = cooker_video(
        {
            "__key__": "corrupt",
            "__restore_key__": ("wds", 2),
            "__subflavors__": {"type_dataset": "video"},
            "mp4": AVDecoder(io.BytesIO(b"not a valid video")),
            "json": {
                "conversations": [
                    {"from": "human", "value": "<video>\nhi"},
                    {"from": "gpt", "value": "hello"},
                ]
            },
        }
    )
    with pytest.raises(SkipSample):
        PackedBatchEncoder.encode_sample.__wrapped__(enc, bad_sample)
