"""Pluggable tracker backends for Stage 2.

`sv.ByteTrack` is deprecated as of supervision 0.28 and is scheduled for
removal in 0.31; Roboflow moved tracking into the separate `trackers`
package. That package
also ships algorithms that target this project's actual failure mode better
than plain ByteTrack:

* **OC-SORT** — observation-centric recovery, for a player who disappears
  behind a screen and reappears a few frames later.
* **BoT-SORT** — camera motion compensation, which matters because a
  broadcast camera pans and zooms constantly and that corrupts the constant-
  velocity assumption every motion-only tracker relies on.

The two families do not share a parameter vocabulary, and the overlap is a
trap rather than a convenience: supervision's `minimum_matching_threshold`
is an IoU *distance* ceiling (0.8 means "accept matches with IoU above 0.2"),
while trackers' `minimum_iou_threshold` is a direct IoU *floor*. Passing 0.95
from one to the other would silently match almost nothing. So backends take
their own keyword arguments rather than a fabricated common schema, and each
one's defaults are its own.

None of these use appearance features — `trackers` dropped `ReIDModel` and
`DeepSORTTracker` in v2.1.0 and lists ReID as planned, not present. Two
players who genuinely swap positions mid-crossing can still swap ids under
any backend here.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

SUPERVISION_BYTETRACK = "sv-bytetrack"
DEFAULT_BACKEND = "bytetrack"
BACKEND_NAMES = (SUPERVISION_BYTETRACK, "bytetrack", "ocsort", "botsort")

# Per-backend defaults tuned for this project: 30fps broadcast, ~10 players,
# heavy mutual occlusion. `minimum_consecutive_frames` above 1 suppresses
# one-frame noise tracks; `lost_track_buffer` is generous because a player
# lost behind a screen should be recovered, not renamed.
BACKEND_DEFAULTS = {
    SUPERVISION_BYTETRACK: {
        "track_activation_threshold": 0.25,
        "lost_track_buffer": 30,
        "minimum_matching_threshold": 0.8,
        "frame_rate": 30,
    },
    "bytetrack": {
        "track_activation_threshold": 0.3,
        "lost_track_buffer": 60,
        "minimum_iou_threshold": 0.1,
        "minimum_consecutive_frames": 2,
        "frame_rate": 30.0,
    },
    "ocsort": {
        "lost_track_buffer": 60,
        "minimum_iou_threshold": 0.3,
        "minimum_consecutive_frames": 2,
        "frame_rate": 30.0,
    },
    "botsort": {
        "track_activation_threshold": 0.3,
        "lost_track_buffer": 60,
        "minimum_consecutive_frames": 2,
        "enable_cmc": True,
        "frame_rate": 30.0,
    },
}


class _SupervisionByteTrack:
    """The original backend, kept only so old commands and runs stay reproducible.

    `sv.ByteTrack` still exists (deprecated) in supervision 0.30 and is due
    for removal in 0.31. This guard means the day it goes, the failure is a
    clear message naming the replacement rather than an AttributeError —
    and every other backend keeps working, which is what let the
    `supervision<0.30` pin be lifted.
    """

    needs_frame = False

    def __init__(self, **kwargs):
        if not hasattr(sv, "ByteTrack"):
            raise RuntimeError(
                f"'{SUPERVISION_BYTETRACK}' needs a supervision that still "
                f"ships sv.ByteTrack, but supervision "
                f"{getattr(sv, '__version__', '?')} is installed and no longer "
                "does. Use --tracker bytetrack, which is the same algorithm "
                "from the `trackers` package."
            )
        self._tracker = sv.ByteTrack(**kwargs)

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, detections: sv.Detections, frame=None) -> sv.Detections:
        return self._tracker.update_with_detections(detections)


class _TrackersBackend:
    """Adapter for the `trackers` package's uniform `.update()` API."""

    def __init__(self, tracker, needs_frame: bool):
        self._tracker = tracker
        self.needs_frame = needs_frame

    def reset(self) -> None:
        self._tracker.reset()

    def update(self, detections: sv.Detections, frame=None) -> sv.Detections:
        # Passing a frame to a tracker that ignores it emits a UserWarning per
        # call, which would be thousands of lines over a shot.
        if self.needs_frame:
            return self._tracker.update(detections, frame)
        return self._tracker.update(detections)


def build_tracker(name: str, **kwargs):
    """Construct a backend by name, merging its defaults with `kwargs`."""
    if name not in BACKEND_NAMES:
        raise ValueError(f"Unknown tracker '{name}'. Choose from {BACKEND_NAMES}.")

    settings = {**BACKEND_DEFAULTS[name], **kwargs}

    if name == SUPERVISION_BYTETRACK:
        return _SupervisionByteTrack(**settings)

    from trackers import BoTSORTTracker, ByteTrackTracker, OCSORTTracker

    classes = {
        "bytetrack": (ByteTrackTracker, False),
        "ocsort": (OCSORTTracker, False),
        # Only BoT-SORT reads pixels, for camera motion compensation.
        "botsort": (BoTSORTTracker, True),
    }
    cls, needs_frame = classes[name]
    needs_frame = needs_frame and settings.get("enable_cmc", True)
    return _TrackersBackend(cls(**settings), needs_frame)


class FrameFeeder:
    """Streams frames forward from a video, handing out one index at a time.

    Only BoT-SORT's motion compensation needs pixels, and holding a shot's
    frames in memory is not an option — 465 frames of 1080p is ~2.9 GB. This
    keeps a single sequential read open and steps forward to each requested
    index, which is far cheaper than seeking per frame.

    Requested indices must be non-decreasing; Stage 2 iterates frames in order.
    """

    def __init__(self, video_path: Path, start_frame: int):
        self._cap = cv2.VideoCapture(str(video_path))
        if not self._cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        self._next_idx = start_frame
        self._current = None

    def get(self, frame_idx: int) -> np.ndarray | None:
        while self._next_idx <= frame_idx:
            ok, frame = self._cap.read()
            if not ok:
                return self._current
            self._current = frame
            self._next_idx += 1
        return self._current

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "FrameFeeder":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
