"""Stage 4 — Ball possession.

Input:  outputs/tracks/{game_id}.parquet (ball + player rows)
Output: outputs/possession/{game_id}.parquet  (see docs/02-data-schemas.md)

Spec: docs/03-pipeline-stages.md (Stage 4)

Example:
    python pipeline/04_ball_possession.py --game-id 0022500123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.common import (
    CLASS_PLAYER,
    POSSESSION_DIR,
    possession_path,
    tracks_path,
)
from pipeline.possession import (
    DEFAULT_DEBOUNCE_FRAMES,
    DEFAULT_MAX_JUMP_FRAC,
    DEFAULT_POSSESSION_FRAC,
    POSSESSION_COLUMNS,
    assign_shot_possession,
    possession_summary,
)


def assign_possession(
    game_id: str,
    possession_frac: float = DEFAULT_POSSESSION_FRAC,
    max_distance_px: float | None = None,
    debounce_frames: int = DEFAULT_DEBOUNCE_FRAMES,
    max_jump_frac: float = DEFAULT_MAX_JUMP_FRAC,
) -> tuple[pd.DataFrame, float]:
    """Assign ball_handler_tracker_id per frame. Returns rows and the radius used."""
    tracks = pd.read_parquet(tracks_path(game_id))

    players = tracks[tracks["class"] == CLASS_PLAYER]
    median_height = float((players["y2"] - players["y1"]).median())
    if max_distance_px is None:
        max_distance_px = possession_frac * median_height
    max_jump_px = max_jump_frac * median_height

    rows: list[dict] = []
    for shot_id, shot_tracks in tracks.groupby("shot_id", sort=True):
        for row in assign_shot_possession(
            shot_tracks, max_distance_px, debounce_frames, max_jump_px
        ):
            rows.append({"game_id": game_id, "shot_id": int(shot_id), **row})

    if not rows:
        return pd.DataFrame(columns=POSSESSION_COLUMNS), max_distance_px

    frame = pd.DataFrame(rows)
    # Nullable integer: a frame with nobody in possession is genuinely null,
    # not zero, and float ids would be a nasty surprise downstream.
    frame["ball_handler_tracker_id"] = frame["ball_handler_tracker_id"].astype("Int64")
    return frame[POSSESSION_COLUMNS], max_distance_px


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 4 — ball possession")
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--possession-frac",
        type=float,
        default=DEFAULT_POSSESSION_FRAC,
        help="Possession radius as a fraction of median player-box height, so "
        "the threshold scales with resolution instead of being re-tuned.",
    )
    parser.add_argument(
        "--max-distance-px",
        type=float,
        default=None,
        help="Absolute possession radius in pixels; overrides --possession-frac.",
    )
    parser.add_argument(
        "--debounce-frames",
        type=int,
        default=DEFAULT_DEBOUNCE_FRAMES,
        help="Consecutive frames a challenger must be nearest before taking "
        "possession. Stops flicker during rebounds and hand-offs.",
    )
    parser.add_argument(
        "--max-jump-frac",
        type=float,
        default=DEFAULT_MAX_JUMP_FRAC,
        help="Largest per-frame ball movement, as a fraction of median "
        "player-box height. Stops a crowd false positive from capturing the "
        "ball track.",
    )
    args = parser.parse_args()

    source = tracks_path(args.game_id)
    if not source.exists():
        print(f"[error] {source} not found — run 02_track.py first", file=sys.stderr)
        return 1

    possession, radius = assign_possession(
        args.game_id,
        args.possession_frac,
        args.max_distance_px,
        args.debounce_frames,
        args.max_jump_frac,
    )
    if possession.empty:
        print("[error] no player tracks to assign possession over", file=sys.stderr)
        return 1

    POSSESSION_DIR.mkdir(parents=True, exist_ok=True)
    destination = possession_path(args.game_id)
    possession.to_parquet(destination, index=False)

    held = possession["ball_handler_tracker_id"].notna()
    spans = possession_summary(possession)
    real_spans = spans[spans["ball_handler_tracker_id"].notna()]
    print(
        f"[stage 4] radius {radius:.0f}px, debounce {args.debounce_frames}f: "
        f"{held.sum()}/{len(possession)} frames have a handler "
        f"({100.0 * held.mean():.0f}%), "
        f"{len(real_spans)} possession span(s), "
        f"median span {real_spans['n_frames'].median() if len(real_spans) else 0:.0f}f",
        file=sys.stderr,
    )
    print(f"[stage 4] {len(possession)} row(s) -> {destination.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
