"""Render the landmark guide as a broadcast camera sees it.

`court_reference.py` draws the court from above, which is where the
coordinates live but not where the annotating happens. Reading a top-down
diagram and then finding the same corner in a side-on broadcast frame is the
step that actually confuses people — the lane stops looking like a rectangle,
and "left" stops meaning "on the left of the screen".

So this projects the same `COURT_LANDMARKS` through a homography chosen to
resemble a half-court broadcast shot. The perspective is computed, not drawn
by hand, so every label sits exactly where that landmark really would.

    python tools/court_camera_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import cv2
import numpy as np

from pipeline.court_geometry import (
    BASKET_FROM_BASELINE_FT,
    CORNER_THREE_FROM_SIDELINE_FT,
    COURT_LANDMARKS,
    COURT_WIDTH_FT,
    FREE_THROW_FROM_BASELINE_FT,
    THREE_POINT_RADIUS_FT,
)

OUTPUT = REPO_ROOT / "docs" / "court-landmarks-camera.png"
W, H = 1560, 880

WOOD = (196, 222, 240)
PAINT = (150, 96, 40)
LINE = (255, 255, 255)
INK = (44, 40, 36)
LEFT_SIDE = (70, 180, 70)
# BGR, so this is orange. Orange rather than blue on purpose: the lane
# underneath is painted blue, and a blue line on it would disappear.
RIGHT_SIDE = (40, 120, 240)
MARK = (40, 40, 40)

# Four correspondences that put the near basket on the left and the court
# receding to the right, the way a half-court broadcast shot looks. x runs
# across the baseline (0-50) and y down the length, so x=0 is the sideline
# nearest the camera and x=50 the far one.
ANCHORS_COURT = np.float32([[0, 0], [COURT_WIDTH_FT, 0], [0, 55], [COURT_WIDTH_FT, 55]])
ANCHORS_IMAGE = np.float32([[210, 742], [268, 300], [1416, 690], [1210, 330]])
HOMOGRAPHY = cv2.getPerspectiveTransform(ANCHORS_COURT, ANCHORS_IMAGE)


def project(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(pts, HOMOGRAPHY).reshape(-1, 2)


def poly(points, colour, fill=False, thickness=2):
    pts = project(points).astype(np.int32)
    if fill:
        cv2.fillPoly(img, [pts], colour)
    else:
        cv2.polylines(img, [pts], False, colour, thickness, cv2.LINE_AA)


img = np.full((H, W, 3), 250, np.uint8)

# Court surface out to half court, then the lane and its markings.
poly([[0, 0], [COURT_WIDTH_FT, 0], [COURT_WIDTH_FT, 55], [0, 55]], WOOD, fill=True)
poly([[0, 0], [COURT_WIDTH_FT, 0], [COURT_WIDTH_FT, 55], [0, 55]], LINE, thickness=3)

lane_left = COURT_LANDMARKS["lane_baseline_left"][0]
lane_right = COURT_LANDMARKS["lane_baseline_right"][0]
ft_y = FREE_THROW_FROM_BASELINE_FT
poly(
    [[lane_left, 0], [lane_right, 0], [lane_right, ft_y], [lane_left, ft_y]],
    PAINT, fill=True,
)
poly(
    [[lane_left, 0], [lane_right, 0], [lane_right, ft_y], [lane_left, ft_y]],
    LINE, thickness=3,
)

# Free-throw circle and the three-point line, both traced as dense polylines
# so perspective bends them the way the camera would.
angles = np.linspace(0, 2 * np.pi, 90)
poly(
    np.column_stack([
        COURT_WIDTH_FT / 2 + 6 * np.cos(angles),
        ft_y + 6 * np.sin(angles),
    ]),
    LINE, thickness=2,
)

dx = COURT_WIDTH_FT / 2 - CORNER_THREE_FROM_SIDELINE_FT
dy = float(np.sqrt(THREE_POINT_RADIUS_FT**2 - dx**2))
arc_end_y = BASKET_FROM_BASELINE_FT + dy
for x in (CORNER_THREE_FROM_SIDELINE_FT, COURT_WIDTH_FT - CORNER_THREE_FROM_SIDELINE_FT):
    poly([[x, 0], [x, arc_end_y]], LINE, thickness=2)
theta = np.linspace(np.arctan2(dy, -dx), np.arctan2(dy, dx), 60)
poly(
    np.column_stack([
        COURT_WIDTH_FT / 2 + THREE_POINT_RADIUS_FT * np.cos(theta),
        BASKET_FROM_BASELINE_FT + THREE_POINT_RADIUS_FT * np.sin(theta),
    ]),
    LINE, thickness=2,
)

# The two long sides of the lane, coloured — this is the rule that matters:
# both "left" names on one, both "right" names on the other.
poly([[lane_left, 0], [lane_left, ft_y]], LEFT_SIDE, thickness=6)
poly([[lane_right, 0], [lane_right, ft_y]], RIGHT_SIDE, thickness=6)

VISIBLE = [
    "baseline_left_corner", "corner_three_left", "lane_baseline_left",
    "basket", "lane_baseline_right", "corner_three_right",
    "baseline_right_corner", "free_throw_left", "free_throw_centre",
    "free_throw_right", "halfcourt_left", "halfcourt_right",
]
# Hand-placed so labels do not collide: the baseline landmarks stack within
# ~20px of each other in perspective, so they are pushed to opposite sides.
OFFSETS = {
    "baseline_left_corner": (-198, 8), "corner_three_left": (16, 26),
    "lane_baseline_left": (-24, 34), "basket": (-30, 40),
    "lane_baseline_right": (-30, -16), "corner_three_right": (16, -10),
    "baseline_right_corner": (-210, -6), "free_throw_left": (18, 32),
    "free_throw_centre": (18, 30), "free_throw_right": (18, -16),
    "halfcourt_left": (-30, 34), "halfcourt_right": (-30, -18),
}

for name in VISIBLE:
    x, y = project([COURT_LANDMARKS[name]])[0]
    px, py = int(x), int(y)
    side = LEFT_SIDE if name.endswith("_left") else (
        RIGHT_SIDE if name.endswith("_right") else MARK
    )
    cv2.drawMarker(img, (px, py), side, cv2.MARKER_CROSS, 22, 3)
    ox, oy = OFFSETS[name]
    for thickness, shade in ((4, (252, 252, 252)), (1, side)):
        cv2.putText(img, name, (px + ox, py + oy), cv2.FONT_HERSHEY_SIMPLEX,
                    0.52, shade, thickness, cv2.LINE_AA)


def text(s, org, scale=0.62, colour=INK, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, colour, thick, cv2.LINE_AA)


text("Landmarks as a side-on broadcast camera sees them", (40, 46), 0.86, INK, 2)
text("Near basket at the left, court receding right - the half-court shot you annotate.",
    (40, 76), 0.56, (110, 105, 100))

text("GREEN = the two 'left' names, on one long side of the lane", (40, H - 92),
     0.58, LEFT_SIDE, 2)
text("ORANGE = the two 'right' names, on the other side", (40, H - 62),
     0.58, RIGHT_SIDE, 2)
text("Which side you call left does not matter. Using the other one consistently "
     "mirrors the court, and distances are unchanged. Mixing the two is what breaks it.",
     (40, H - 26), 0.54, (110, 105, 100))

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
cv2.imwrite(str(OUTPUT), img)
print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(VISIBLE)} landmarks)")
