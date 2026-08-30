"""Stage 6 — Aggregation: the gravity metric.

Input:  tracks, identity, possession, calibration — joined on
        game_id / shot_id / frame_idx
Output: outputs/metrics/{game_id}_gravity.parquet and the 'all' rollup

Spec: docs/03-pipeline-stages.md (Stage 6), docs/05-metrics-and-analysis.md
"""


def aggregate_game(game_id: str) -> None:
    """Compute per-player gravity metrics for one game.

    1. Project (foot_x, foot_y) through the shot's homography.
    2. Identify the 5 defenders (team clustering + which team has the ball).
    3. Bucket frames per target player: with ball vs. without ball; compute
       avg defender distance in each bucket. Also compute the secondary
       metrics: avg_defender_distance_overall and distance_nearest_defender.
    4. Filter before writing: min frames per bucket, identity_confidence
       threshold, reprojection_error_px threshold, exclude replay/closeup
       shots. Bad rows never reach the output table.
    """
    raise NotImplementedError


def aggregate_all() -> None:
    """Roll per-game metrics up into outputs/metrics/all_gravity.parquet."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
