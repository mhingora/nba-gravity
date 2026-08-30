"""Stage 4 — who has the ball, per frame.

The spec in `03-pipeline-stages.md` assumes one ball position per frame and
goes straight to "distance from ball centroid to each player". Real detector
output is messier: on the test possession 243 of 398 ball-bearing frames
carry more than one candidate, up to five. So there is a selection step first
(`select_ball`), and only then the distance-plus-debounce logic the spec
describes.

Ball rows in `tracks.parquet` have no confidence column — Stage 2 does not
carry one through — so candidates are resolved by temporal continuity
instead: the ball is where it was a moment ago. That is a stronger signal
than confidence anyway for a small fast object a generic detector is unsure
about, and it needs no extra input file.

Distances are measured to the nearest point of a player's box rather than to
its centre, because a player reaching for the ball has a hand at the box edge
while their centre is half a metre away — the difference decides possession
right at the threshold.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.common import CLASS_BALL, CLASS_PLAYER

# Possession radius as a fraction of median player-box height, so the same
# setting works at any resolution. Player boxes run ~218px at 1080p, and a
# player is roughly 2m, making 0.35 about 0.75m — an arm's length, which is
# what the spec asks for.
DEFAULT_POSSESSION_FRAC = 0.35
DEFAULT_DEBOUNCE_FRAMES = 6
# How far the ball may move between frames, as a fraction of median player-box
# height. A player is roughly 2m, so 1.0 allows about 2m per frame — 60m/s,
# far beyond any pass, while still refusing the leap into the stands that a
# crowd false positive requires.
DEFAULT_MAX_JUMP_FRAC = 1.0

POSSESSION_COLUMNS = [
    "game_id",
    "shot_id",
    "frame_idx",
    "ball_handler_tracker_id",
    "ball_distance_px",
]


def box_centres(boxes: np.ndarray) -> np.ndarray:
    return np.column_stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2.0, (boxes[:, 1] + boxes[:, 3]) / 2.0)
    )


def point_to_box_distance(point: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Distance from one point to the nearest edge/corner of each box.

    Zero when the point is inside a box, which is the common case for a held
    ball and correctly reads as "closest possible".
    """
    dx = np.maximum.reduce([boxes[:, 0] - point[0], np.zeros(len(boxes)), point[0] - boxes[:, 2]])
    dy = np.maximum.reduce([boxes[:, 1] - point[1], np.zeros(len(boxes)), point[1] - boxes[:, 3]])
    return np.hypot(dx, dy)


def select_ball(
    ball_rows: pd.DataFrame,
    players_by_frame: dict[int, np.ndarray],
    max_jump_px: float,
    reset_after: int = 5,
) -> dict[int, np.ndarray]:
    """Reduce multiple ball candidates per frame to one position per frame.

    Forward pass over the shot: each frame keeps the candidate nearest the
    last accepted position, **but only within `max_jump_px`**. Without that
    gate a single false positive in the crowd captures the track and drags
    every subsequent frame with it — which is exactly what happened on the
    test possession, producing handlers 250-290px from a ball sitting among
    the photographers.

    A frame whose candidates all fail the gate simply has no ball, which is
    an honest answer. After `reset_after` such frames the anchor is dropped
    so the ball can be re-acquired somewhere else — otherwise a genuine long
    pass or shot would strand the tracker at a stale position forever.

    With no anchor, the fallback is the candidate closest to any player: at
    the start of a possession the ball is nearly always in someone's hands,
    and a stray detection in the stands is not.
    """
    chosen: dict[int, np.ndarray] = {}
    previous: np.ndarray | None = None
    missed = 0

    for frame_idx, group in ball_rows.groupby("frame_idx", sort=True):
        candidates = box_centres(group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float))

        if previous is not None:
            jumps = np.linalg.norm(candidates - previous, axis=1)
            nearest = int(np.argmin(jumps))
            if jumps[nearest] > max_jump_px:
                missed += 1
                if missed >= reset_after:
                    previous, missed = None, 0
                continue
            pick = candidates[nearest]
        else:
            boxes = players_by_frame.get(int(frame_idx))
            if boxes is None or len(boxes) == 0:
                pick = candidates[0]
            else:
                scores = [point_to_box_distance(c, boxes).min() for c in candidates]
                pick = candidates[int(np.argmin(scores))]

        chosen[int(frame_idx)] = pick
        previous = pick
        missed = 0

    return chosen


def assign_shot_possession(
    shot_tracks: pd.DataFrame,
    max_distance_px: float,
    debounce_frames: int,
    max_jump_px: float,
) -> list[dict]:
    """Per-frame handler for one shot, with debounced possession changes."""
    players = shot_tracks[shot_tracks["class"] == CLASS_PLAYER]
    balls = shot_tracks[shot_tracks["class"] == CLASS_BALL]
    if players.empty:
        return []

    players_by_frame = {
        int(frame_idx): group[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        for frame_idx, group in players.groupby("frame_idx", sort=True)
    }
    ids_by_frame = {
        int(frame_idx): group["tracker_id"].to_numpy(dtype=int)
        for frame_idx, group in players.groupby("frame_idx", sort=True)
    }
    ball_by_frame = (
        select_ball(balls, players_by_frame, max_jump_px) if not balls.empty else {}
    )

    confirmed: int | None = None
    challenger: int | None = None
    streak = 0
    rows: list[dict] = []

    for frame_idx in sorted(players_by_frame):
        ball = ball_by_frame.get(frame_idx)
        nearest_id: int | None = None
        nearest_distance = float("nan")

        if ball is not None:
            boxes = players_by_frame[frame_idx]
            distances = point_to_box_distance(ball, boxes)
            best = int(np.argmin(distances))
            if distances[best] <= max_distance_px:
                nearest_id = int(ids_by_frame[frame_idx][best])
                nearest_distance = float(distances[best])
            else:
                # Ball is in flight — correctly nobody's, and it must not be
                # allowed to reinforce the outgoing handler either.
                nearest_distance = float(distances[best])

        # Debounce. A candidate has to hold the nearest position for several
        # consecutive frames before it takes over, which is what stops the
        # handler flickering during a rebound scrum or a hand-off.
        if nearest_id is None:
            challenger, streak = None, 0
        elif nearest_id == confirmed:
            challenger, streak = None, 0
        elif nearest_id == challenger:
            streak += 1
            if streak >= debounce_frames:
                confirmed, challenger, streak = nearest_id, None, 0
        else:
            challenger, streak = nearest_id, 1

        # A frame only has a handler if someone is actually within reach this
        # frame. The spec is explicit that a ball in flight belongs to nobody,
        # and reporting the last holder through a pass would invent possession
        # that never happened — the debounced `confirmed` id exists to stop
        # flicker *between players*, not to paper over the ball's absence.
        rows.append(
            {
                "frame_idx": frame_idx,
                "ball_handler_tracker_id": (
                    confirmed if nearest_id is not None and confirmed is not None else None
                ),
                "ball_distance_px": nearest_distance,
            }
        )

    return rows


def possession_summary(possession: pd.DataFrame) -> pd.DataFrame:
    """Contiguous possession spans, for eyeballing against the video."""
    if possession.empty:
        return pd.DataFrame(columns=["shot_id", "ball_handler_tracker_id", "start_frame", "end_frame", "n_frames"])

    frame = possession.sort_values(["shot_id", "frame_idx"]).copy()
    handler = frame["ball_handler_tracker_id"]
    changed = (handler != handler.shift()) | (frame["shot_id"] != frame["shot_id"].shift())
    frame["span"] = changed.cumsum()

    spans = (
        frame.groupby("span")
        .agg(
            shot_id=("shot_id", "first"),
            ball_handler_tracker_id=("ball_handler_tracker_id", "first"),
            start_frame=("frame_idx", "min"),
            end_frame=("frame_idx", "max"),
            n_frames=("frame_idx", "size"),
        )
        .reset_index(drop=True)
    )
    return spans
