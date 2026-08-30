"""Stage 0 — shot boundary segmentation.

Broadcast video cuts constantly, and `sv.ByteTrack` identity only holds within
a single continuous shot (docs/01-architecture.md). This module finds the hard
cuts using frame-to-frame HSV histogram distance.

Per docs/03-pipeline-stages.md the threshold is deliberately biased toward
over-detecting cuts: a false positive just resets tracking a little early,
while a missed cut lets ByteTrack bridge two unrelated shots and silently
produce garbage tracks.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path

import cv2
import numpy as np

from pipeline.common import Shot

# Frames are downscaled before histogramming — cut detection does not need
# detail, and this keeps a full-quarter pass fast.
SIGNATURE_WIDTH = 160

DEFAULT_CUT_THRESHOLD = 0.30
# A hard cut cannot plausibly be followed by another one a few frames later at
# broadcast frame rates. Without this guard, strobing arena lights and camera
# flashes fragment a single shot into dozens of unusable stubs.
DEFAULT_MIN_SHOT_FRAMES = 8


def frame_signature(frame: np.ndarray) -> np.ndarray:
    """Normalized hue/saturation histogram of a frame."""
    height, width = frame.shape[:2]
    scale = SIGNATURE_WIDTH / max(width, 1)
    if scale < 1.0:
        frame = cv2.resize(
            frame, (SIGNATURE_WIDTH, max(int(height * scale), 1)),
            interpolation=cv2.INTER_AREA,
        )
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """0.0 for identical frames, →1.0 for completely unrelated ones."""
    correlation = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)
    return float(np.clip(1.0 - correlation, 0.0, 1.0))


def iter_frames(
    video_path: Path, start_frame: int = 0, end_frame: int | None = None
) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (absolute frame_idx, BGR frame) sequentially over an inclusive range.

    Sequential decoding, with a single seek to `start_frame`. Much cheaper than
    seeking per frame, which is why stages use this and only the UI seeks.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        if start_frame > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        frame_idx = start_frame
        while end_frame is None or frame_idx <= end_frame:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame_idx, frame
            frame_idx += 1
    finally:
        cap.release()


def segment_shots(
    video_path: Path,
    start_frame: int = 0,
    end_frame: int | None = None,
    threshold: float = DEFAULT_CUT_THRESHOLD,
    min_shot_frames: int = DEFAULT_MIN_SHOT_FRAMES,
    progress: Callable[[int], None] | None = None,
) -> tuple[list[Shot], np.ndarray]:
    """Split a video (or a frame range of one) into continuous camera shots.

    Returns the shots and the per-frame cut-strength signal, the latter aligned
    to `start_frame` and 0.0 at its first entry. The signal is kept so the UI
    can plot it against the threshold — tuning this number by eye is far easier
    than guessing at it.
    """
    distances: list[float] = [0.0]
    cut_frames: list[int] = []
    previous_signature: np.ndarray | None = None
    last_cut_frame = start_frame
    last_frame_seen = start_frame - 1

    for frame_idx, frame in iter_frames(video_path, start_frame, end_frame):
        signature = frame_signature(frame)
        if previous_signature is not None:
            distance = signature_distance(previous_signature, signature)
            distances.append(distance)
            if distance > threshold and frame_idx - last_cut_frame >= min_shot_frames:
                cut_frames.append(frame_idx)
                last_cut_frame = frame_idx
        previous_signature = signature
        last_frame_seen = frame_idx
        if progress is not None:
            progress(frame_idx)

    if last_frame_seen < start_frame:
        return [], np.zeros(0, dtype=np.float32)

    shots = shots_from_cuts(cut_frames, start_frame, last_frame_seen)
    return shots, np.asarray(distances, dtype=np.float32)


def shots_from_cuts(
    cut_frames: list[int], start_frame: int, last_frame: int
) -> list[Shot]:
    """Build inclusive-range shots from the frame indices that start a new shot."""
    starts = [start_frame, *cut_frames]
    shots = []
    for shot_id, start in enumerate(starts):
        end = starts[shot_id + 1] - 1 if shot_id + 1 < len(starts) else last_frame
        shots.append(Shot(shot_id=shot_id, start_frame=start, end_frame=end))
    return shots


def shot_id_for_frame(shots: list[Shot], frame_idx: int) -> int:
    """Shot containing a frame. Shots are contiguous, so this is a bisect."""
    lo, hi = 0, len(shots) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        shot = shots[mid]
        if frame_idx < shot.start_frame:
            hi = mid - 1
        elif frame_idx > shot.end_frame:
            lo = mid + 1
        else:
            return shot.shot_id
    raise ValueError(f"Frame {frame_idx} falls outside all shots")
