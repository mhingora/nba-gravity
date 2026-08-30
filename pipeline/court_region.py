"""Court-region tests used to reject crowd, bench and referee detections.

A broadcast camera sees the court as a trapezoid, not a rectangle: the far
sideline is compressed toward the top of the frame while the near sideline
spreads across the bottom. That geometry is why the axis-aligned
`--court-roi` band can't separate crowd from court — spectators standing
behind the far baseline have foot points at the same `y` as players on the
far side of the court, so any band wide enough to keep those players also
keeps the front rows behind them.

A polygon tracks the actual playing surface, so the same foot-point test
becomes decisive. Coordinates are normalized 0-1 for the same reason the
rectangle is: they survive a change of source resolution, so a polygon
tuned on a 1080p clip still applies to a 720p one from the same camera.

Both this module and `02_track.py` need this logic, and so does `app.py`'s
polygon tuner — the numbered stage scripts can't be imported (a module name
can't start with a digit), so it lives here.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from pipeline.common import CALIBRATION_REF_DIR

MIN_POLYGON_POINTS = 3


def parse_polygon(values) -> np.ndarray | None:
    """Turn a flat ``[x1, y1, x2, y2, ...]`` sequence into an (N, 2) array.

    Returns None for an empty/absent value so callers can treat "no polygon"
    and "no filtering" identically.
    """
    if not values:
        return None

    flat = [float(v) for v in values]
    if len(flat) % 2 != 0:
        raise ValueError(
            f"--court-polygon needs an even number of values (x y pairs); got {len(flat)}"
        )

    points = np.asarray(flat, dtype=float).reshape(-1, 2)
    if len(points) < MIN_POLYGON_POINTS:
        raise ValueError(
            f"--court-polygon needs at least {MIN_POLYGON_POINTS} points; got {len(points)}"
        )
    if ((points < -0.5) | (points > 1.5)).any():
        raise ValueError(
            "--court-polygon values look like pixels; they must be normalized 0-1"
        )
    return points


def to_pixels(polygon: np.ndarray, frame_width: float, frame_height: float) -> np.ndarray:
    """Scale a normalized polygon to absolute pixel coordinates."""
    scaled = polygon.copy()
    scaled[:, 0] *= frame_width
    scaled[:, 1] *= frame_height
    return scaled


def points_in_polygon(
    xs: np.ndarray, ys: np.ndarray, polygon_px: np.ndarray
) -> np.ndarray:
    """Vectorized even-odd (ray casting) test for many points at once.

    `cv2.pointPolygonTest` handles one point per call, which is far too slow
    for ~100 detections across thousands of frames. This walks the polygon's
    edges instead, toggling a boolean per point for each edge its ray crosses.
    """
    xs = np.asarray(xs, dtype=float)
    ys = np.asarray(ys, dtype=float)
    inside = np.zeros(xs.shape, dtype=bool)

    n = len(polygon_px)
    j = n - 1
    for i in range(n):
        xi, yi = polygon_px[i]
        xj, yj = polygon_px[j]
        # Guard the horizontal-edge division; such edges never cross a
        # horizontal ray, and the straddle test already excludes them.
        straddles = (yi > ys) != (yj > ys)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_at_ray = (xj - xi) * (ys - yi) / np.where(yj == yi, np.nan, yj - yi) + xi
        inside ^= straddles & (xs < x_at_ray)
        j = i

    return inside


DEFAULT_BROADCAST_POLYGON = (
    0.00, 0.62,
    0.20, 0.34,
    0.86, 0.34,
    1.00, 0.58,
    1.00, 0.95,
    0.30, 0.95,
)
"""A starting trapezoid for a standard mid-court broadcast angle.

Not a universal constant — camera height and zoom differ per arena and per
broadcast. Use the polygon tuner in the viewer's Tracking tab to fit your
own, then pass the result to `--court-polygon`.
"""


# --------------------------------------------------------------------------
# Court profiles — the polygon and its tracker settings, stored per camera
# angle rather than per game.
#
# A polygon describes where the court sits in *this camera's* frame, so it is
# a property of the broadcast setup, not of one game: the same profile applies
# to every game shot from the same position. Storing it per game_id would mean
# re-tuning identical footage. `data/calibration/` is already specified as
# "court keypoint references, per camera angle" — this is that file.
#
# Tracker settings ride along because they are tuned against the same footage
# and are meaningless apart from it; a profile is "everything I learned about
# this camera angle", so one flag replaces a dozen.
# --------------------------------------------------------------------------

PROFILE_SUFFIX = ".json"


def profile_path(name: str) -> Path:
    return CALIBRATION_REF_DIR / f"{name}{PROFILE_SUFFIX}"


def list_profiles() -> list[str]:
    if not CALIBRATION_REF_DIR.is_dir():
        return []
    return sorted(p.stem for p in CALIBRATION_REF_DIR.glob(f"*{PROFILE_SUFFIX}"))


def load_profile(name: str) -> dict:
    """Read a court profile. Raises FileNotFoundError naming what exists."""
    path = profile_path(name)
    if not path.exists():
        available = ", ".join(list_profiles()) or "none"
        raise FileNotFoundError(
            f"No court profile '{name}' at {path}. Available: {available}. "
            "Create one in the viewer's Tracking tab (Court polygon tuner)."
        )

    profile = json.loads(path.read_text(encoding="utf-8"))
    # The polygon is optional. A detector trained on basketball already
    # ignores the crowd, so it needs no geometric filter — but the tracker
    # settings tuned alongside it are still worth carrying in a profile.
    points = profile.get("court_polygon")
    if points:
        # Stored as [[x, y], ...]; parse_polygon takes a flat sequence, and
        # validates range and point count for hand-edited files too.
        profile["court_polygon"] = parse_polygon([v for pt in points for v in pt])
    else:
        profile["court_polygon"] = None
    return profile


def save_profile(
    name: str,
    polygon: np.ndarray,
    tracker: dict | None = None,
    description: str = "",
    backend: str | None = None,
) -> Path:
    """Write a court profile, creating `data/calibration/` if needed.

    The polygon is written one `[x, y]` pair per line. `json.dumps(indent=2)`
    puts every single coordinate on its own line, which turns a ten-point
    polygon into forty lines and makes the file miserable to hand-edit — and
    hand-editing is the point of keeping it in `data/`.
    """
    CALIBRATION_REF_DIR.mkdir(parents=True, exist_ok=True)
    path = profile_path(name)

    pairs = (
        None
        if polygon is None
        else ",\n".join(
            f"    [{round(float(x), 4)}, {round(float(y), 4)}]" for x, y in polygon
        )
    )
    body = {"name": name, "description": description}
    if backend:
        # Tracker settings only mean anything alongside the backend they were
        # tuned for — the backends do not share a parameter vocabulary.
        body["backend"] = backend
    body["tracker"] = tracker or {}
    head = json.dumps(body, indent=2)[1:-1].rstrip().rstrip(",")

    polygon_block = (
        f'  "court_polygon": [\n{pairs}\n  ]\n' if pairs else '  "court_polygon": []\n'
    )
    path.write_text("{" + head + ",\n" + polygon_block + "}\n", encoding="utf-8")
    return path
