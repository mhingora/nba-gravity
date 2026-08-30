"""Stage 5 — mapping broadcast pixels to real court feet.

Gravity is a *distance* metric, so until this stage exists every number the
pipeline produces is in pixels — and a pixel is worth more feet at the far
end of the court than the near end, and changes meaning whenever the camera
zooms. Two defenders "50px away" in different frames are not comparably
close. A homography fixes that by projecting foot points onto a flat court
plane measured in feet.

Landmark coordinates below are the real NBA court, in feet, with the origin
at the left baseline corner as the stub specifies: **x runs along the
baseline (0-50, the court's width) and y runs down the length (0-94)**. That
ordering is worth stating because it is the opposite of the usual "x is the
long axis" instinct, and getting it backwards produces a homography that
fits beautifully and means nothing.

Only a handful of landmarks need to be visible in any given frame — four
non-collinear points are the minimum for a homography, and more is better —
so the catalogue is deliberately larger than any single camera angle can see.
"""

from __future__ import annotations

import numpy as np

COURT_LENGTH_FT = 94.0
COURT_WIDTH_FT = 50.0
LANE_WIDTH_FT = 16.0
# The free-throw line is 15ft from the backboard, and the backboard hangs 4ft
# inside the baseline, putting the line 19ft from the baseline itself. This
# trips people up constantly, hence the arithmetic in the open.
BACKBOARD_INSET_FT = 4.0
FREE_THROW_FROM_BACKBOARD_FT = 15.0
FREE_THROW_FROM_BASELINE_FT = BACKBOARD_INSET_FT + FREE_THROW_FROM_BACKBOARD_FT
BASKET_FROM_BASELINE_FT = 5.25
THREE_POINT_RADIUS_FT = 23.75
CORNER_THREE_FROM_SIDELINE_FT = 3.0

_LANE_LEFT = (COURT_WIDTH_FT - LANE_WIDTH_FT) / 2.0   # 17.0
_LANE_RIGHT = (COURT_WIDTH_FT + LANE_WIDTH_FT) / 2.0  # 33.0

COURT_LANDMARKS: dict[str, tuple[float, float]] = {
    # Baseline the camera is looking toward (near basket, y = 0)
    "baseline_left_corner": (0.0, 0.0),
    "baseline_right_corner": (COURT_WIDTH_FT, 0.0),
    "lane_baseline_left": (_LANE_LEFT, 0.0),
    "lane_baseline_right": (_LANE_RIGHT, 0.0),
    "free_throw_left": (_LANE_LEFT, FREE_THROW_FROM_BASELINE_FT),
    "free_throw_right": (_LANE_RIGHT, FREE_THROW_FROM_BASELINE_FT),
    "free_throw_centre": (COURT_WIDTH_FT / 2.0, FREE_THROW_FROM_BASELINE_FT),
    "basket": (COURT_WIDTH_FT / 2.0, BASKET_FROM_BASELINE_FT),
    # Where the corner-three line meets the baseline
    "corner_three_left": (CORNER_THREE_FROM_SIDELINE_FT, 0.0),
    "corner_three_right": (COURT_WIDTH_FT - CORNER_THREE_FROM_SIDELINE_FT, 0.0),
    # Half court
    "halfcourt_left": (0.0, COURT_LENGTH_FT / 2.0),
    "halfcourt_right": (COURT_WIDTH_FT, COURT_LENGTH_FT / 2.0),
    "centre_circle": (COURT_WIDTH_FT / 2.0, COURT_LENGTH_FT / 2.0),
    # Far end, rarely visible in a half-court broadcast shot but catalogued
    "far_baseline_left": (0.0, COURT_LENGTH_FT),
    "far_baseline_right": (COURT_WIDTH_FT, COURT_LENGTH_FT),
}

MIN_LANDMARKS = 4

# Pairs whose true separation is known, used to sanity-check a homography
# against something it was not fitted to explain.
KNOWN_DISTANCES: list[tuple[str, str, float]] = [
    ("lane_baseline_left", "lane_baseline_right", LANE_WIDTH_FT),
    ("free_throw_left", "free_throw_right", LANE_WIDTH_FT),
    ("lane_baseline_left", "free_throw_left", FREE_THROW_FROM_BASELINE_FT),
    ("lane_baseline_right", "free_throw_right", FREE_THROW_FROM_BASELINE_FT),
    ("baseline_left_corner", "baseline_right_corner", COURT_WIDTH_FT),
]


def parse_keypoints(mapping: dict) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Turn {landmark: [x_norm, y_norm]} into paired image/court arrays.

    Image coordinates are normalized 0-1 for the same reason the court
    polygon is: a calibration fitted on 1080p then stays valid for the same
    camera at 720p.
    """
    names, image_points, court_points = [], [], []
    for name, point in mapping.items():
        if name not in COURT_LANDMARKS:
            raise ValueError(
                f"Unknown landmark '{name}'. Known: {', '.join(sorted(COURT_LANDMARKS))}"
            )
        if len(point) != 2:
            raise ValueError(f"Landmark '{name}' needs exactly [x, y]; got {point}")
        names.append(name)
        image_points.append([float(point[0]), float(point[1])])
        court_points.append(list(COURT_LANDMARKS[name]))

    if len(names) < MIN_LANDMARKS:
        raise ValueError(
            f"Need at least {MIN_LANDMARKS} landmarks for a homography; got {len(names)}"
        )
    return (
        np.asarray(image_points, dtype=np.float64),
        np.asarray(court_points, dtype=np.float64),
        names,
    )


def compute_homography(
    image_points_norm: np.ndarray,
    court_points: np.ndarray,
    frame_width: float,
    frame_height: float,
):
    """Fit pixels -> court feet. Returns (matrix, per-point error, rms error).

    Errors are reported in **pixels**, by projecting the court points back
    into the image and measuring the miss there. Reporting them in feet would
    flatter distant landmarks, where a large ground error is a small pixel
    one — and pixels are what you can actually see when checking an overlay.
    """
    import cv2

    image_points = image_points_norm.copy()
    image_points[:, 0] *= frame_width
    image_points[:, 1] *= frame_height

    matrix, _ = cv2.findHomography(image_points, court_points, method=0)
    if matrix is None:
        raise ValueError(
            "findHomography failed — landmarks are probably collinear or "
            "duplicated. Spread them across the court, not along one line."
        )

    inverse = np.linalg.inv(matrix)
    reprojected = _apply(inverse, court_points)
    per_point = np.linalg.norm(reprojected - image_points, axis=1)
    rms = float(np.sqrt(np.mean(per_point**2)))
    return matrix, per_point, rms


def _apply(matrix: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points))])
    projected = homogeneous @ matrix.T
    # A point on the camera's horizon projects to w=0; guard rather than
    # silently producing infinities that poison every downstream mean.
    w = projected[:, 2:3]
    w = np.where(np.abs(w) < 1e-9, np.nan, w)
    return projected[:, :2] / w


def to_court_feet(matrix: np.ndarray, pixel_points: np.ndarray) -> np.ndarray:
    """Project image pixels onto the court plane, in feet."""
    return _apply(matrix, np.asarray(pixel_points, dtype=np.float64))


def known_distance_checks(
    matrix: np.ndarray,
    image_points_norm: np.ndarray,
    names: list[str],
    frame_width: float,
    frame_height: float,
) -> list[dict]:
    """Measure known court distances through the homography.

    This is the milestone's stated done-when: project real locations and see
    whether the distances between them come out near their true values. It is
    a weaker check than it looks — these landmarks were used to *fit* the
    homography, so it verifies internal consistency rather than accuracy — but
    a fit that cannot even reproduce the lane width is definitely wrong.
    """
    index = {name: i for i, name in enumerate(names)}
    pixels = image_points_norm.copy()
    pixels[:, 0] *= frame_width
    pixels[:, 1] *= frame_height
    projected = to_court_feet(matrix, pixels)

    results = []
    for start, end, truth in KNOWN_DISTANCES:
        if start not in index or end not in index:
            continue
        measured = float(
            np.linalg.norm(projected[index[start]] - projected[index[end]])
        )
        results.append(
            {
                "from": start,
                "to": end,
                "expected_ft": truth,
                "measured_ft": round(measured, 2),
                "error_ft": round(measured - truth, 2),
            }
        )
    return results
