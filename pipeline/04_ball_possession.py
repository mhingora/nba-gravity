"""Stage 4 — Ball possession.

Input:  outputs/tracks/{game_id}.parquet (ball + player rows)
Output: outputs/possession/{game_id}.parquet  (see docs/02-data-schemas.md)

Spec: docs/03-pipeline-stages.md (Stage 4)
"""


def assign_possession(game_id: str) -> None:
    """Assign ball_handler_tracker_id per frame.

    - Distance from ball centroid to nearest point of each player bbox
      (not bbox center — the reaching hand is at the bbox edge).
    - Handler = closest player under a tunable pixel threshold.
    - Debounce: same tracker_id must be closest for N consecutive frames
      (5-8 at broadcast fps) before confirming a possession change.
    - No player under threshold (ball in flight) => null, which is correct.
    """
    raise NotImplementedError


if __name__ == "__main__":
    pass
