from __future__ import annotations

import warnings
from typing import Any

from PIL import Image

VIDEO_EXTENSIONS = ("mp4", "avi", "mov", "webm", "mkv", "m4v", "flv")

def _target_frame_size(width: int | None, height: int | None, max_side: int) -> tuple[int, int] | None:
    if max_side <= 0 or not width or not height:
        return None
    longest = max(width, height)
    if longest <= max_side:
        return None
    scale = max_side / longest
    return (max(2, round(width * scale)), max(2, round(height * scale)))

def decode_video_frames(
    decoder: Any,
    *,
    fps: float = 2.0,
    max_frames: int = 32,
    max_side: int = 448,
    start: float | None = None,
    end: float | None = None,
    temporal_patch_size: int = 2,
) -> tuple[list[Image.Image], float]:
    """Decode uniformly sampled RGB PIL frames from an Energon ``AVDecoder``.

    Args:
        decoder: An ``AVDecoder`` instance (e.g. produced by Energon's
            ``SampleDecoder(av_decode="AVDecoder")``).
        fps: Target sampling rate in frames per second.
        max_frames: Maximum number of frames to decode from the clip.
        max_side: Maximum longest side in pixels; frames are downscaled in FFmpeg
            while preserving aspect ratio. ``<= 0`` keeps native resolution.
        start: Optional clip start time in seconds.
        end: Optional clip end time in seconds.
        temporal_patch_size: Align multi-frame counts to a multiple of this value
            (2 for Qwen3.5 / Qwen3-VL) to avoid trailing temporal padding.

    Returns:
        ``(frames, effective_fps)`` where ``frames`` is a list of RGB ``PIL.Image``
        and ``effective_fps`` is the sampling rate corresponding to
        ``frames_indices = [0, 1, ..., len(frames) - 1]``.
    """
    if fps <= 0:
        raise ValueError(f"fps must be positive, got {fps}")
    if max_frames <= 0:
        raise ValueError(f"max_frames must be positive, got {max_frames}")

    meta = decoder.get_metadata(
        get_video=True,
        get_video_duration=True,
        get_video_frame_count=False,
        get_video_frame_size=max_side > 0,
        get_audio=False,
    )
    if meta.video_duration is None or meta.video_duration <= 0:
        raise ValueError(f"video has no valid duration: {meta.video_duration!r}")
    if meta.video_fps is None or meta.video_fps <= 0:
        raise ValueError(f"video has no valid fps: {meta.video_fps!r}")

    t0 = max(0.0, float(start)) if start is not None else 0.0
    t1 = min(float(meta.video_duration), float(end)) if end is not None else float(meta.video_duration)
    clip_duration = t1 - t0
    if clip_duration <= 0:
        raise ValueError(f"empty video clip [{t0}, {t1}] for duration {meta.video_duration}")

    n = max(1, min(max_frames, int(round(clip_duration * fps))))
    if n > 1 and temporal_patch_size > 1:
        n -= n % temporal_patch_size
        n = max(1, n)

    spf = 1.0 / float(meta.video_fps)
    hi = max(t0, t1 - 0.5 * spf)
    if n == 1:
        eff_fps = float(fps)
        targets = [min(hi, t0 + 0.5 * clip_duration)]
    else:
        eff_fps = n / clip_duration
        targets = [min(hi, t0 + k / eff_fps) for k in range(n)]

    out_size = _target_frame_size(meta.video_width, meta.video_height, max_side)
    clip_ranges = [(t, t) for t in targets]

    # Energon's AVDecoder warns on every MP4 when seeking in seconds on a frame-indexed container.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Video container unit is seconds, but seeking only supports frames",
            category=RuntimeWarning,
        )
        clips, _timestamps = decoder.get_video_clips(
            video_clip_ranges=clip_ranges,
            video_unit="seconds",
            video_out_frame_size=out_size,
        )

    frames: list[Image.Image] = []
    for clip in clips:
        if clip.ndim == 4 and clip.shape[0] > 0:
            arr = clip[0].permute(1, 2, 0).contiguous().numpy()
            frames.append(Image.fromarray(arr))

    if len(frames) > 1 and temporal_patch_size > 1 and len(frames) % temporal_patch_size:
        frames = frames[: len(frames) - (len(frames) % temporal_patch_size)]
        eff_fps = len(frames) / clip_duration

    if not frames:
        raise ValueError("no frames decoded from video")

    return frames, float(eff_fps)
