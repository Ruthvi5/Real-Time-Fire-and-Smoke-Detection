"""Sequential video reading, time-based frame sampling and timestamps.

Frames are read strictly in order. Only frames that are due according to
video.sample_fps are decoded and sent to the detector; the rest are counted
as skipped. Nothing ever looks ahead at future frames.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mkv", ".mov", ".m4v", ".mpg", ".mpeg",
                    ".wmv", ".flv", ".webm", ".ts", ".3gp", ".dav", ".h264", ".264"}

# Give up on a video after this many consecutive frames that cannot be decoded.
MAX_CONSECUTIVE_DECODE_FAILURES = 50


class VideoOpenError(RuntimeError):
    pass


@dataclass
class SampledFrame:
    frame_index: int        # index of this frame among all frames read from the video (0-based)
    timestamp_s: float      # presentation time of this frame in seconds
    image: np.ndarray       # BGR image, original resolution


@dataclass
class ReadStats:
    frames_read: int = 0
    frames_processed: int = 0
    decode_failures: int = 0
    fps: float = 0.0
    fps_source: str = ""
    timestamp_source: str = "container"
    warnings: list = field(default_factory=list)

    @property
    def frames_skipped(self) -> int:
        return self.frames_read - self.frames_processed


def list_videos(input_path: str | Path) -> list[tuple[str, Path]]:
    """Return (source_id, path) pairs. A folder is searched recursively, in sorted order."""
    p = Path(input_path).expanduser()
    if p.is_file():
        return [(p.name, p)]
    if p.is_dir():
        files = sorted(f for f in p.rglob("*")
                       if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
                       and not f.name.startswith("."))
        return [(f.relative_to(p).as_posix(), f) for f in files]
    raise FileNotFoundError(f"input not found: {p}")


class VideoReader:
    """Iterate over the sampled frames of one video, in order."""

    def __init__(self, path: str | Path, sample_fps: float, fallback_fps: float = 25.0):
        self.path = str(path)
        self.sample_fps = float(sample_fps)
        self.fallback_fps = float(fallback_fps)
        self.stats = ReadStats()

        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise VideoOpenError(f"cannot open video: {self.path}")

        fps = self.cap.get(cv2.CAP_PROP_FPS)
        if fps is None or not math.isfinite(fps) or fps <= 0 or fps > 300:
            self.stats.fps = self.fallback_fps
            self.stats.fps_source = "fallback"
            self.stats.warnings.append(
                f"video reports invalid fps ({fps}); using fallback {self.fallback_fps}")
        else:
            self.stats.fps = float(fps)
            self.stats.fps_source = "container"

    def _timestamp(self, frame_index: int, last_ts: float | None) -> float:
        """Prefer the container timestamp; fall back to index / fps if it is missing or not increasing."""
        ts_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        ts = ts_ms / 1000.0 if ts_ms is not None and math.isfinite(ts_ms) else -1.0
        computed = frame_index / self.stats.fps
        if ts < 0 or (frame_index > 0 and ts <= 0) or (last_ts is not None and ts <= last_ts):
            if self.stats.timestamp_source == "container" and frame_index > 0:
                self.stats.timestamp_source = "mixed"
            return computed if last_ts is None or computed > last_ts else last_ts + 1.0 / self.stats.fps
        return ts

    def __iter__(self) -> Iterator[SampledFrame]:
        interval = 1.0 / self.sample_fps if self.sample_fps > 0 else 0.0
        eps = 1e-6
        next_due = 0.0
        last_ts = None
        frame_index = -1
        consecutive_failures = 0
        try:
            while True:
                ok = self.cap.grab()
                if not ok:
                    break
                frame_index += 1
                self.stats.frames_read += 1
                ts = self._timestamp(frame_index, last_ts)
                last_ts = ts

                due = interval == 0.0 or ts + eps >= next_due
                if not due:
                    continue

                ok, image = self.cap.retrieve()
                if not ok or image is None or image.size == 0:
                    self.stats.decode_failures += 1
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_DECODE_FAILURES:
                        self.stats.warnings.append(
                            f"stopped after {consecutive_failures} consecutive undecodable frames")
                        break
                    continue  # try the next frame instead; next_due is unchanged
                consecutive_failures = 0

                if interval > 0:
                    # Keep a regular sampling grid; jump ahead if the video has gaps.
                    next_due += interval
                    while next_due <= ts + eps:
                        next_due += interval

                self.stats.frames_processed += 1
                yield SampledFrame(frame_index=frame_index, timestamp_s=ts, image=image)
        finally:
            self.close()

        if self.stats.decode_failures:
            self.stats.warnings.append(f"{self.stats.decode_failures} sampled frames could not be decoded")

    def close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
