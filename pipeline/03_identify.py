"""Stage 3 — Identity resolution (team clustering + jersey OCR).

Input:  outputs/tracks/{game_id}.parquet, data/rosters/
Output: outputs/identity/{game_id}.parquet  (see docs/02-data-schemas.md)

Spec: docs/04-identity-resolution.md
Independent of stages 4-5; can be developed and validated in parallel.
"""


def classify_teams(game_id: str) -> None:
    """Part A — cluster tracked players into 2 teams by torso color/embedding.

    - Crop upper ~40% of bbox across several frames of each track.
    - Cluster per shot, then anchor clusters to a per-game reference so
      team_id labels stay consistent across shots.
    - Referees/others fall out as a third outlier bucket.
    """
    raise NotImplementedError


def resolve_jersey_numbers(game_id: str) -> None:
    """Part B — OCR jersey numbers with majority vote per track.

    - Sample large, low-blur frames (bbox size + Laplacian variance filters).
    - Majority vote across the track; require >= 3 agreeing reads.
    - No majority => jersey_number stays null. Never guess.
    """
    raise NotImplementedError


def resolve_identity(game_id: str) -> None:
    """(team_id, jersey_number) -> roster lookup -> player_name.

    Either input null => player_name null. Compute identity_confidence
    (frame-agreement fraction * clipped silhouette score).
    """
    raise NotImplementedError


if __name__ == "__main__":
    pass
