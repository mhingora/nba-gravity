"""Stage 3 — Identity resolution (team clustering + jersey OCR).

Input:  outputs/tracks/{game_id}.parquet, data/rosters/
Output: outputs/identity/{game_id}.parquet  (see docs/02-data-schemas.md)

Spec: docs/04-identity-resolution.md
Independent of stages 4-5; can be developed and validated in parallel.

Part A (team clustering) is implemented — it is Milestone 3, and gravity
needs to know which tracks are defenders before any distance means anything.
Part B (jersey OCR) is Milestone 6 and still a stub, so `jersey_number` and
`player_name` are written as null. That is an expected state, not a failure:
`02-data-schemas.md` says downstream geometry does not depend on the name,
and only the final per-player aggregation does.

Example:
    python pipeline/03_identify.py --game-id 0022500123
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.common import (
    IDENTITY_DIR,
    find_video,
    identity_path,
    tracks_path,
)
from pipeline.team_clustering import (
    TEAM_OTHER,
    assign_teams,
    track_features,
)

IDENTITY_COLUMNS = [
    "game_id",
    "shot_id",
    "tracker_id",
    "team_id",
    "jersey_number",
    "player_name",
    "identity_confidence",
]


def classify_teams(
    game_id: str,
    max_samples: int = 12,
    min_track_len: int = 10,
    ambiguity_ratio: float = 0.80,
) -> pd.DataFrame:
    """Part A — cluster tracked players into 2 teams by torso colour."""
    tracks = pd.read_parquet(tracks_path(game_id))
    features = track_features(
        find_video(game_id), tracks, max_samples, min_track_len
    )
    if features.empty:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)
    return assign_teams(features, ambiguity_ratio)


def resolve_jersey_numbers(game_id: str) -> None:
    """Part B — OCR jersey numbers with majority vote per track.

    Not implemented; arrives with Milestone 6. Measured constraint from the
    1080p footage: player boxes are ~218px tall, leaving roughly 76px of
    torso, so expect low yield and never guess — `04-identity-resolution.md`
    requires >= 3 agreeing reads before accepting a number.
    """
    raise NotImplementedError("Jersey OCR is Milestone 6")


def resolve_identity(clustered: pd.DataFrame, game_id: str) -> pd.DataFrame:
    """(team_id, jersey_number) -> roster lookup -> player_name.

    With Part B unimplemented, `jersey_number` is always null, so
    `player_name` is too — the spec is explicit that a null name is correct
    and that partial information must never be used to guess one.

    `identity_confidence` is the spec's formula with its OCR term absent:
    normally `frame_agreement * clipped_silhouette`, here just the clipped
    silhouette, scaled by how many frames actually produced a usable crop. It
    therefore describes confidence in the *team* label only, which is all
    that has been resolved.
    """
    result = clustered.copy()
    silhouette = result["silhouette"].clip(lower=0.0, upper=1.0)
    sample_weight = (result["n_samples"] / result["n_samples"].max()).clip(0.0, 1.0)

    result["game_id"] = game_id
    result["jersey_number"] = pd.NA
    result["player_name"] = pd.NA
    result["identity_confidence"] = (silhouette * sample_weight).astype(float)
    # An unresolved team is not a confident identity, whatever the clustering
    # score says about the tracks that did resolve.
    result.loc[result["team_id"] == TEAM_OTHER, "identity_confidence"] = 0.0
    return result[IDENTITY_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 — identity resolution")
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="Frames sampled per track for the colour feature.",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=10,
        help="Skip tracks shorter than this; too few frames to cluster on.",
    )
    parser.add_argument(
        "--ambiguity-ratio",
        type=float,
        default=0.80,
        help="A track becomes 'other' when its distance to the rival team's "
        "centroid is less than this multiple worse than to its own — i.e. the "
        "colour evidence does not favour either team. Raise it to keep more "
        "borderline tracks, lower it to be stricter.",
    )
    args = parser.parse_args()

    source = tracks_path(args.game_id)
    if not source.exists():
        print(f"[error] {source} not found — run 02_track.py first", file=sys.stderr)
        return 1

    clustered = classify_teams(
        args.game_id, args.max_samples, args.min_track_len, args.ambiguity_ratio
    )
    if clustered.empty:
        print(
            "[error] no tracks long enough to cluster — lower --min-track-len",
            file=sys.stderr,
        )
        return 1

    identity = resolve_identity(clustered, args.game_id)

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    destination = identity_path(args.game_id)
    identity.to_parquet(destination, index=False)

    counts = identity["team_id"].value_counts().to_dict()
    print(
        f"[stage 3] {len(identity)} track(s): "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f" | silhouette {clustered['silhouette'].iloc[0]:.2f}",
        file=sys.stderr,
    )
    print(f"[stage 3] {len(identity)} row(s) -> {destination.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
