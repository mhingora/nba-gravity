"""Stage 6 — Aggregation: the gravity metric.

Input:  tracks, identity, possession, calibration — joined on
        game_id / shot_id / frame_idx
Output: outputs/metrics/{game_id}_distances.parquet  (per-frame evidence)
        outputs/metrics/{game_id}_gravity.parquet    (the deliverable table)
        outputs/metrics/all_gravity.parquet          (with --all)

Spec: docs/03-pipeline-stages.md (Stage 6), docs/05-metrics-and-analysis.md

Two artifacts rather than one because the deliverable is an average of
averages, and a surprising number in it is untraceable without the rows it
came from. The per-frame table is also what the viewer's Gravity tab plots.

Example:
    python pipeline/06_aggregate.py --game-id 0022500123
    python pipeline/06_aggregate.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from pipeline.common import (
    METRICS_DIR,
    distances_path,
    identity_path,
    metrics_path,
    possession_path,
    tracks_path,
)
from pipeline.gravity import (
    EXPECTED_DEFENDERS,
    MIN_BUCKET_FRAMES,
    MIN_OFFENSE_SHARE,
    aggregate_players,
    combine_games,
    defender_distances,
    load_calibration,
    offensive_team,
)


def usable_shots(
    game_id: str,
    shot_ids,
    max_reprojection_px: float,
    all_shots: bool,
) -> tuple[dict, list[str]]:
    """Homographies for the shots worth projecting, plus why the rest are out.

    Stage 5 writes one matrix per shot, but every shot of a game gets the
    *same* matrix and the same reprojection error, because the error is the
    residual on the frame the landmarks were annotated on. So the error alone
    cannot say whether a given shot's camera matches — only the annotation
    record can, and by default a shot is aggregated when the landmarks came
    from it.
    """
    matrices: dict[int, np.ndarray] = {}
    skipped: list[str] = []

    for shot_id in sorted(int(s) for s in shot_ids):
        calibration = load_calibration(game_id, shot_id)
        if calibration is None:
            skipped.append(f"shot {shot_id}: no calibration — run 05_calibrate.py")
            continue

        error = float(calibration.get("reprojection_error_px", float("inf")))
        if error > max_reprojection_px:
            skipped.append(
                f"shot {shot_id}: reprojection error {error:.1f}px over the "
                f"{max_reprojection_px:.1f}px limit"
            )
            continue

        annotated = calibration.get("annotated_on") or {}
        annotated_shot = annotated.get("shot_id")
        if not all_shots and annotated_shot is not None and annotated_shot != shot_id:
            skipped.append(
                f"shot {shot_id}: homography was annotated on shot "
                f"{annotated_shot}, not this one"
            )
            continue
        if not all_shots and annotated_shot is None:
            skipped.append(
                f"shot {shot_id}: calibration does not record which frame it "
                "was annotated on — re-run 05_calibrate.py"
            )
            continue

        matrices[shot_id] = np.array(calibration["homography_matrix"], dtype=np.float64)

    return matrices, skipped


def aggregate_game(
    game_id: str,
    min_bucket_frames: int = MIN_BUCKET_FRAMES,
    min_identity_confidence: float = 0.0,
    max_reprojection_px: float = 15.0,
    defenders_required: int = EXPECTED_DEFENDERS,
    min_offense_share: float = MIN_OFFENSE_SHARE,
    all_shots: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Compute per-player gravity metrics for one game.

    Returns (per-frame distances, per-player metrics, notes about what was
    excluded). The notes are not decoration: on real footage most frames are
    excluded, and a table that does not say why is indistinguishable from one
    that found nothing.
    """
    tracks = pd.read_parquet(tracks_path(game_id))
    identity = pd.read_parquet(identity_path(game_id))
    possession = pd.read_parquet(possession_path(game_id))

    matrices, notes = usable_shots(
        game_id, tracks["shot_id"].unique(), max_reprojection_px, all_shots
    )

    offense = offensive_team(possession, identity, min_offense_share)
    for call in offense.values():
        if call.team_id is None:
            notes.append(f"shot {call.shot_id}: {call.reason}")
        elif call.shot_id in matrices:
            notes.append(
                f"shot {call.shot_id}: {call.team_id} on offence "
                f"({call.share:.0%} of {call.handler_frames} handler frames)"
            )

    distances = defender_distances(
        tracks, identity, possession, offense, matrices, defenders_required
    )
    metrics = aggregate_players(
        distances, identity, min_bucket_frames, min_identity_confidence
    )
    return distances, metrics, notes


def aggregate_all(min_bucket_frames: int = MIN_BUCKET_FRAMES) -> pd.DataFrame:
    """Roll per-game metrics up into outputs/metrics/all_gravity.parquet."""
    tables = []
    for path in sorted(METRICS_DIR.glob("*_gravity.parquet")):
        if path.name == "all_gravity.parquet":
            continue
        tables.append(pd.read_parquet(path))
    return combine_games(tables, min_bucket_frames)


def report(distances: pd.DataFrame, metrics: pd.DataFrame, notes: list[str]) -> None:
    """Say what was measured and what was thrown away, on stderr."""
    for note in notes:
        print(f"[stage 6] {note}", file=sys.stderr)

    if distances.empty:
        print(
            "[stage 6] no frame survived: nothing to average. The notes above "
            "say which stage to fix.",
            file=sys.stderr,
        )
        return

    frames = distances["frame_idx"].nunique()
    with_ball = int(distances["has_ball"].sum())
    print(
        f"[stage 6] {len(distances)} player-frame(s) over {frames} frame(s); "
        f"{with_ball} with the ball",
        file=sys.stderr,
    )
    print(
        f"[stage 6] defender distance median "
        f"{distances['avg_defender_distance_ft'].median():.1f}ft, nearest "
        f"{distances['nearest_defender_distance_ft'].median():.1f}ft",
        file=sys.stderr,
    )

    if metrics.empty:
        print(
            "[stage 6] no player row: no track carried an identity. Run "
            "03_identify.py --ocr to put numbers on tracks.",
            file=sys.stderr,
        )
        return

    resolved = int(metrics["gravity_delta"].notna().sum())
    print(
        f"[stage 6] {len(metrics)} player row(s), {resolved} with a "
        f"gravity_delta",
        file=sys.stderr,
    )
    if resolved < len(metrics):
        print(
            "[stage 6] the rest lack frames in one of the two buckets — a "
            "player who never holds the ball in the footage processed has no "
            "'with ball' average to subtract from.",
            file=sys.stderr,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 6 — gravity metric")
    parser.add_argument("--game-id")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Roll every per-game metrics table into all_gravity.parquet.",
    )
    parser.add_argument(
        "--min-bucket-frames",
        type=int,
        default=MIN_BUCKET_FRAMES,
        help="Frames required in both the with-ball and without-ball bucket "
        "before a gravity_delta is written. Below it the row still reports "
        "its averages and frame counts, with a null delta.",
    )
    parser.add_argument(
        "--min-identity-confidence",
        type=float,
        default=0.0,
        help="Drop tracks below this identity_confidence. The spec suggests "
        "0.6, which this footage cannot reach: the value is capped by the "
        "team-cluster silhouette (0.36 here), so set it relative to the "
        "silhouette the run actually achieved.",
    )
    parser.add_argument(
        "--max-reprojection-px",
        type=float,
        default=15.0,
        help="Skip shots whose homography fits worse than this.",
    )
    parser.add_argument(
        "--defenders",
        type=int,
        default=EXPECTED_DEFENDERS,
        help="Frames are used only when exactly this many defenders are "
        "tracked on court. Fewer means a missed player, more means one player "
        "tracked twice; either way the average is not over a defence.",
    )
    parser.add_argument(
        "--min-offense-share",
        type=float,
        default=MIN_OFFENSE_SHARE,
        help="Majority of a shot's handler frames needed to call one team the "
        "offence. Below it the shot is skipped rather than guessed at.",
    )
    parser.add_argument(
        "--all-shots",
        action="store_true",
        help="Aggregate every shot, not just the one its landmarks were "
        "annotated on. Every shot of a game carries the same homography, so "
        "this trusts it beyond where it was checked.",
    )
    args = parser.parse_args()

    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    if args.all:
        combined = aggregate_all(args.min_bucket_frames)
        destination = METRICS_DIR / "all_gravity.parquet"
        combined.to_parquet(destination, index=False)
        print(
            f"[stage 6] {len(combined)} player row(s) across games -> "
            f"{destination.name}",
            file=sys.stderr,
        )
        return 0

    if not args.game_id:
        print("[error] pass --game-id, or --all to roll up", file=sys.stderr)
        return 1

    for label, path in (
        ("tracks", tracks_path(args.game_id)),
        ("identity", identity_path(args.game_id)),
        ("possession", possession_path(args.game_id)),
    ):
        if not path.exists():
            print(
                f"[error] no {label} for {args.game_id} — {path} is missing",
                file=sys.stderr,
            )
            return 1

    distances, metrics, notes = aggregate_game(
        args.game_id,
        args.min_bucket_frames,
        args.min_identity_confidence,
        args.max_reprojection_px,
        args.defenders,
        args.min_offense_share,
        args.all_shots,
    )

    distances.to_parquet(distances_path(args.game_id), index=False)
    metrics.to_parquet(metrics_path(args.game_id), index=False)
    report(distances, metrics, notes)
    print(
        f"[stage 6] {len(metrics)} row(s) -> {metrics_path(args.game_id).name}, "
        f"{len(distances)} row(s) -> {distances_path(args.game_id).name}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
