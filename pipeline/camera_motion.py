"""Carry one annotated frame's homography across a whole clip.

A homography describes one camera framing, and the broadcast camera pans,
tilts and zooms continuously — within a single shot, never mind between them.
Measured on the test clip: an identical fixed crop at frames 3097, 3300 and
3561 of *one* shot shows three different pieces of court. So the matrix fitted
to the annotated frame is only right near that frame.

The fix is standard structure-from-motion practice and the same idea BoT-SORT
uses for camera motion compensation: find where the camera has moved to by
matching image features, then compose that motion with the annotated
homography. Every frame gets `H_ref @ W`, where `W` maps its pixels back onto
the reference frame.

Two measured decisions shape the implementation:

**Match every frame straight to the reference, never neighbour to neighbour.**
Chaining always has overlap to work with, but accumulates error, and there is
no way to tell a drifted chain from a good one. Direct matching either
succeeds or visibly fails. It holds up across the whole test shot — 93 to 450
inliers at ±230 frames from the reference — so the chain buys nothing.

**Mask the broadcast overlay, or everything matches everything.** The scorebug
sits at identical screen coordinates in every frame of the clip, so it matches
itself across a cut and RANSAC reports that as the camera motion. Unmasked,
all 31 other shots of the test clip "matched" the reference with about a
hundred inliers each — including baseline closeups sharing no court pixels
at all. Masked, 24 of them are refused with 5-10 inliers while the 7 genuinely
shot from the same camera keep 92-219. That refusal is the safety property
this module rests on: a frame the reference cannot reach gets no homography
rather than a confident wrong one.

The overlay is found rather than configured, using the property that defines
it: match the reference against frames from *other* shots and keep the
matches that barely moved. Court content cannot do that across a cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline.common import walk_frames

# Matching runs at half resolution: same inlier counts, a third of the cost
# (0.08s versus 0.26s a frame), and marginally better alignment.
DEFAULT_SCALE = 0.5
MAX_FEATURES = 2000
RATIO_TEST = 0.75
RANSAC_PX = 3.0
# Below this a frame is refused. Genuine same-camera frames carry hundreds;
# the graphics-only false matches carry 5-10 once the overlay is masked.
MIN_INLIERS = 40

# Overlay detection. A match that moves less than this across a camera cut is
# screen-fixed, because no court feature survives a cut in place.
OVERLAY_STILL_PX = 2.0
OVERLAY_PROBES = 8
OVERLAY_RADIUS_PX = 28
# If this much of the frame looks screen-fixed, the detection is wrong — a
# clip of one continuous shot has no cuts to compare across, and everything
# looks still. Better to mask nothing and say so.
OVERLAY_MAX_COVERAGE = 0.25


@dataclass
class FrameMotion:
    """Where one frame sits relative to the reference frame."""

    frame_idx: int
    matrix: np.ndarray | None
    inliers: int
    matches: int
    rms_px: float

    @property
    def usable(self) -> bool:
        return self.matrix is not None and self.inliers >= MIN_INLIERS


_DETECTOR = None
_MATCHER = None


def _detector():
    """One detector for the whole run; rebuilding it per frame is wasted work."""
    global _DETECTOR
    if _DETECTOR is None:
        _DETECTOR = cv2.SIFT_create(nfeatures=MAX_FEATURES)
    return _DETECTOR


def _matcher():
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = cv2.BFMatcher()
    return _MATCHER


def describe(image: np.ndarray, mask: np.ndarray | None, scale: float):
    """Features of one frame, in full-resolution coordinates."""
    height, width = image.shape[:2]
    small = cv2.resize(image, (int(width * scale), int(height * scale)))
    grey = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    small_mask = None
    if mask is not None:
        small_mask = cv2.resize(mask, (grey.shape[1], grey.shape[0]))
        small_mask = np.where(small_mask > 0, 0, 255).astype(np.uint8)
    keypoints, descriptors = _detector().detectAndCompute(grey, small_mask)
    points = np.array([kp.pt for kp in keypoints], dtype=np.float32) / scale
    return points, descriptors


def match_points(points_a, desc_a, points_b, desc_b) -> tuple[np.ndarray, np.ndarray]:
    """Mutually plausible correspondences, by Lowe's ratio test."""
    if desc_a is None or desc_b is None or len(desc_a) < 2 or len(desc_b) < 2:
        return np.empty((0, 2)), np.empty((0, 2))
    pairs = _matcher().knnMatch(desc_a, desc_b, k=2)
    good = [
        m for m, n in (p for p in pairs if len(p) == 2)
        if m.distance < RATIO_TEST * n.distance
    ]
    if not good:
        return np.empty((0, 2)), np.empty((0, 2))
    return (
        np.array([points_a[m.queryIdx] for m in good]),
        np.array([points_b[m.trainIdx] for m in good]),
    )


def estimate_motion(
    frame: np.ndarray,
    reference_points,
    reference_desc,
    overlay: np.ndarray | None,
    frame_idx: int,
    scale: float = DEFAULT_SCALE,
) -> FrameMotion:
    """Homography mapping this frame's pixels onto the reference frame's."""
    points, descriptors = describe(frame, overlay, scale)
    src, dst = match_points(points, descriptors, reference_points, reference_desc)
    if len(src) < 12:
        return FrameMotion(frame_idx, None, 0, len(src), float("nan"))

    matrix, inlier_mask = cv2.findHomography(
        src.reshape(-1, 1, 2), dst.reshape(-1, 1, 2), cv2.RANSAC, RANSAC_PX,
        maxIters=4000,
    )
    if matrix is None:
        return FrameMotion(frame_idx, None, 0, len(src), float("nan"))

    keep = inlier_mask.ravel() == 1
    if not keep.any():
        # RANSAC can return a matrix with no inliers at all, and
        # perspectiveTransform hands back None for an empty input rather than
        # an empty array.
        return FrameMotion(frame_idx, None, 0, len(src), float("nan"))

    moved = cv2.perspectiveTransform(src[keep].reshape(-1, 1, 2), matrix)
    residuals = np.linalg.norm(moved.reshape(-1, 2) - dst[keep], axis=1)
    return FrameMotion(
        frame_idx,
        matrix,
        int(keep.sum()),
        len(src),
        float(np.sqrt((residuals**2).mean())),
    )


def detect_overlay(
    capture,
    probe_frames: list[int],
    reference: np.ndarray,
    scale: float = DEFAULT_SCALE,
) -> tuple[np.ndarray, list[dict]]:
    """Find the screen-fixed broadcast graphics. Returns (mask, regions).

    `probe_frames` must come from *other* camera shots — the detection works
    by keeping matches that survive a cut without moving, which only an
    overlay can do.
    """
    height, width = reference.shape[:2]
    reference_points, reference_desc = describe(reference, None, scale)
    mask = np.zeros((height, width), np.uint8)

    still = 0
    for frame_idx in probe_frames:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = capture.read()
        if not ok:
            continue
        points, descriptors = describe(frame, None, scale)
        src, dst = match_points(
            reference_points, reference_desc, points, descriptors
        )
        if not len(src):
            continue
        fixed = np.linalg.norm(src - dst, axis=1) <= OVERLAY_STILL_PX
        for x, y in src[fixed]:
            cv2.circle(mask, (int(x), int(y)), OVERLAY_RADIUS_PX, 255, -1)
        still += int(fixed.sum())

    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((45, 45), np.uint8))
    coverage = float(mask.mean() / 255)
    if coverage > OVERLAY_MAX_COVERAGE:
        return np.zeros((height, width), np.uint8), []

    regions = []
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if area < 0.0005 * height * width:
            continue
        regions.append(
            {
                "x": round(x / width, 4),
                "y": round(y / height, 4),
                "w": round(w / width, 4),
                "h": round(h / height, 4),
                "coverage": round(float(area) / (height * width), 4),
            }
        )
    return mask, regions


def propagate(
    capture,
    reference_frame: int,
    frames: list[int],
    overlay: np.ndarray | None,
    court_matrix: np.ndarray,
    scale: float = DEFAULT_SCALE,
    progress=None,
) -> list[tuple[FrameMotion, np.ndarray | None]]:
    """Court homography for each frame the reference can reach.

    Returns (motion, court matrix) per frame, the matrix being None wherever
    the frame could not be matched — which is the correct answer for a
    different camera angle, and is why this can run over a whole clip rather
    than a hand-picked shot.
    """
    capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
    ok, reference = capture.read()
    if not ok:
        raise IOError(f"Could not read reference frame {reference_frame}")
    reference_points, reference_desc = describe(reference, overlay, scale)

    wanted = sorted({int(f) for f in frames})
    results = []
    solved = set()
    for position, (frame_idx, frame) in enumerate(walk_frames(capture, wanted)):
        solved.add(frame_idx)
        motion = estimate_motion(
            frame, reference_points, reference_desc, overlay, frame_idx, scale
        )
        # H_ref maps reference pixels to court feet; motion maps this frame's
        # pixels to reference pixels. Composed, this frame maps to court feet.
        matrix = court_matrix @ motion.matrix if motion.usable else None
        results.append((motion, matrix))
        if progress is not None:
            progress(position + 1, len(wanted), motion)

    # A frame the decoder never handed back is as unsolved as one that failed
    # to match, and the caller counts both against what it attempted.
    for frame_idx in wanted:
        if frame_idx not in solved:
            results.append((FrameMotion(frame_idx, None, 0, 0, float("nan")), None))
    return results
