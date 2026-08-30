"""Render the court-landmark reference diagram used when annotating.

Stage 5 asks you to say which pixel each named landmark sits at. That is only
answerable if you know what the names mean, so this draws every entry in
`COURT_LANDMARKS` onto a scale court with its real coordinates.

Generated rather than hand-drawn so it cannot drift from the table it
documents: the positions come from `pipeline/court_geometry.py`, and the only
things hard-coded here are the court markings themselves (the arc, the
circles) which are decoration, not data.

    python tools/court_reference.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle, Rectangle

from pipeline.court_geometry import (
    BASKET_FROM_BASELINE_FT,
    COURT_LENGTH_FT,
    COURT_WIDTH_FT,
    COURT_LANDMARKS,
    CORNER_THREE_FROM_SIDELINE_FT,
    FREE_THROW_FROM_BASELINE_FT,
    THREE_POINT_RADIUS_FT,
)

OUTPUT = REPO_ROOT / "docs" / "court-landmarks.png"

LINE = "#3b3b3b"
ACCENT = "#c0392b"
WOOD = "#f5e6cf"
PAINT = "#cfd9e8"


def draw_court(ax) -> None:
    ax.add_patch(
        Rectangle((0, 0), COURT_WIDTH_FT, COURT_LENGTH_FT, facecolor=WOOD,
                  edgecolor=LINE, linewidth=2, zorder=0)
    )
    lane_left = COURT_LANDMARKS["lane_baseline_left"][0]
    lane_right = COURT_LANDMARKS["lane_baseline_right"][0]
    ax.add_patch(
        Rectangle((lane_left, 0), lane_right - lane_left,
                  FREE_THROW_FROM_BASELINE_FT, facecolor=PAINT,
                  edgecolor=LINE, linewidth=1.5, zorder=1)
    )

    centre_x = COURT_WIDTH_FT / 2.0
    # Free-throw circle, and the centre circle at half court.
    ax.add_patch(Arc((centre_x, FREE_THROW_FROM_BASELINE_FT), 12, 12, theta1=0,
                     theta2=360, edgecolor=LINE, linewidth=1.2, zorder=2))
    ax.add_patch(Arc((centre_x, COURT_LENGTH_FT / 2.0), 12, 12, theta1=0,
                     theta2=360, edgecolor=LINE, linewidth=1.2, zorder=2))
    ax.plot([0, COURT_WIDTH_FT], [COURT_LENGTH_FT / 2.0] * 2, color=LINE,
            linewidth=1.5, zorder=2)

    # Basket and backboard.
    ax.add_patch(Circle((centre_x, BASKET_FROM_BASELINE_FT), 0.75,
                        edgecolor=ACCENT, facecolor="none", linewidth=1.4,
                        zorder=3))
    ax.plot([centre_x - 3, centre_x + 3], [4.0, 4.0], color=LINE,
            linewidth=2, zorder=3)

    # Three-point line: straight corner segments, then the arc between them.
    # The corner line stops where it meets the arc, which is a right triangle
    # away from the basket: dy = sqrt(r^2 - dx^2).
    dx = centre_x - CORNER_THREE_FROM_SIDELINE_FT
    dy = float(np.sqrt(THREE_POINT_RADIUS_FT**2 - dx**2))
    corner_end_y = BASKET_FROM_BASELINE_FT + dy
    for x in (CORNER_THREE_FROM_SIDELINE_FT,
              COURT_WIDTH_FT - CORNER_THREE_FROM_SIDELINE_FT):
        ax.plot([x, x], [0, corner_end_y], color=LINE, linewidth=1.5, zorder=2)
    theta = float(np.degrees(np.arctan2(dy, -dx)))
    ax.add_patch(Arc((centre_x, BASKET_FROM_BASELINE_FT),
                     2 * THREE_POINT_RADIUS_FT, 2 * THREE_POINT_RADIUS_FT,
                     theta1=180 - theta, theta2=theta, edgecolor=LINE,
                     linewidth=1.5, zorder=2))


def main() -> int:
    order = [
        "baseline_left_corner", "corner_three_left", "lane_baseline_left",
        "basket", "lane_baseline_right", "corner_three_right",
        "baseline_right_corner", "free_throw_left", "free_throw_centre",
        "free_throw_right", "halfcourt_left", "centre_circle",
        "halfcourt_right", "far_baseline_left", "far_baseline_right",
    ]
    missing = set(COURT_LANDMARKS) - set(order)
    assert not missing, f"landmark(s) not in the diagram: {sorted(missing)}"

    fig, (ax, legend_ax) = plt.subplots(
        1, 2, figsize=(15.5, 10), gridspec_kw={"width_ratios": [1.05, 1]}
    )
    draw_court(ax)

    # Labels are nudged so they do not sit on top of the markings; the
    # numbered marker is always at the true coordinate.
    nudges = {
        "baseline_left_corner": (-1.5, -2.6, "right"),
        "baseline_right_corner": (1.5, -2.6, "left"),
        "corner_three_left": (-0.8, -2.6, "right"),
        "corner_three_right": (0.8, -2.6, "left"),
        "lane_baseline_left": (-1.2, -2.6, "right"),
        "lane_baseline_right": (1.2, -2.6, "left"),
        "basket": (2.0, -1.6, "left"),
        "free_throw_left": (-1.5, 0.8, "right"),
        "free_throw_right": (1.5, 0.8, "left"),
        "free_throw_centre": (0.0, 2.2, "center"),
        "halfcourt_left": (-1.5, 1.2, "right"),
        "halfcourt_right": (1.5, 1.2, "left"),
        "centre_circle": (0.0, -3.4, "center"),
        "far_baseline_left": (-1.5, 1.6, "right"),
        "far_baseline_right": (1.5, 1.6, "left"),
    }

    for i, name in enumerate(order, start=1):
        x, y = COURT_LANDMARKS[name]
        ax.plot(x, y, "o", color=ACCENT, markersize=8, zorder=5)
        ax.text(x, y, str(i), color="white", fontsize=7, ha="center",
                va="center", zorder=6, fontweight="bold")
        dx, dy, align = nudges[name]
        ax.text(x + dx, y + dy, str(i), color=ACCENT, fontsize=10,
                ha=align, va="center", fontweight="bold", zorder=6)

    ax.set_xlim(-7, COURT_WIDTH_FT + 7)
    ax.set_ylim(-7, COURT_LENGTH_FT + 7)
    ax.set_aspect("equal")
    ax.set_xlabel("x — across the baseline (feet)", fontsize=10)
    ax.set_ylabel("y — down the length of the court (feet)", fontsize=10)
    ax.set_title(
        "Court landmarks\norigin (0, 0) at the left corner of the baseline "
        "you are annotating",
        fontsize=12, fontweight="bold",
    )
    ax.grid(alpha=0.15, linestyle=":")

    legend_ax.axis("off")
    legend_ax.text(0.0, 1.0, "Landmark names and true coordinates",
                   fontsize=12, fontweight="bold", va="top")
    rows = [
        f"{i:>2}.  {name:<24}({COURT_LANDMARKS[name][0]:>5.2f} ft, "
        f"{COURT_LANDMARKS[name][1]:>5.2f} ft)"
        for i, name in enumerate(order, start=1)
    ]
    legend_ax.text(0.0, 0.94, "\n".join(rows), fontsize=10.5, va="top",
                   family="monospace", linespacing=1.75)

    notes = (
        "How to use this\n"
        "\n"
        "You supply only WHERE EACH POINT IS IN YOUR FRAME.\n"
        "The coordinates above are fixed by the rules of\n"
        "basketball and are already in the code.\n"
        "\n"
        "Annotate the basket end the camera is looking at,\n"
        "whichever end of the real court that is — the origin\n"
        "is that baseline's left corner from the camera's view.\n"
        "\n"
        "Pick 4 or more, spread out. All along the baseline is\n"
        "collinear and gets refused. A good starter set is\n"
        "3, 5, 8, 10 — the four lane corners.\n"
        "\n"
        "14 and 15 are the far baseline, 94 ft away. They are\n"
        "rarely visible in a half-court shot; skip them unless\n"
        "you can genuinely see them."
    )
    legend_ax.text(0.0, 0.40, notes, fontsize=10.5, va="top", linespacing=1.6)

    fig.tight_layout()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=110, facecolor="white")
    print(f"wrote {OUTPUT.relative_to(REPO_ROOT)} ({len(order)} landmarks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
