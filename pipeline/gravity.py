"""Stage 6 — defender distances in feet, and the gravity metric built on them.

The metric itself is arithmetic (`05-metrics-and-analysis.md`): average how far
the defenders sit from a player when they have the ball, average it again when
they do not, subtract. Everything difficult is in deciding which rows are
allowed to enter that average, and each rule below exists because the test
possession showed what happens without it.

**Which team is on defence.** The only signal is the ball handler, and taken
frame by frame it is wrong. On the test possession the handler's team flips
four times in fifteen seconds, yet the broadcast shot clock counts 14 -> 9 ->
8 -> 7 without resetting, so San Antonio had the ball throughout: the flips
are a Knicks defender momentarily being the player nearest a contested ball.
Following them would invert who counts as a defender for those frames. So
offence is decided once per camera shot by majority of handler frames, and a
shot whose handler frames do not agree clearly is skipped rather than guessed
at — the same refusal the jersey vote makes.

**Fragmented tracks double-count defenders.** After projecting onto the court,
a third of frames show six to nine "defenders" on the floor. They are real
people in real positions; the surplus is one player carrying two tracker_ids
(Milestone 1's open problem), which weights that player twice in the average.
Frames are therefore kept only when exactly the expected number of defenders
is tracked, which on the test possession is 294 of 464 frames.

**One homography does not describe every shot.** Stage 5 writes the same
matrix to every shot of a game because they share a camera angle, and the
reprojection error it records is the fit residual from the annotated frame —
identical in the file for a shot the matrix was never checked against. By
default only the shot the landmarks were annotated on is aggregated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from pipeline.common import CLASS_PLAYER, calibration_path
from pipeline.court_geometry import COURT_LENGTH_FT, COURT_WIDTH_FT, to_court_feet

# Players stand a little outside the lines — inbounding, or a step over the
# baseline — so the court test is generous. It is here to drop the crowd and
# the bench, not to referee.
COURT_MARGIN_FT = 3.0

EXPECTED_DEFENDERS = 5
# Below this the handler frames disagree too much to call one team the
# offence. 0.6 keeps the test possession (70% dark) and would refuse a shot
# that genuinely spans a change of possession.
MIN_OFFENSE_SHARE = 0.60
# `05-metrics-and-analysis.md` asks for at least this many frames in a bucket
# before a bucket's average is worth quoting.
MIN_BUCKET_FRAMES = 100

DISTANCE_COLUMNS = [
    "game_id",
    "shot_id",
    "frame_idx",
    "tracker_id",
    "team_id",
    "court_x",
    "court_y",
    "has_ball",
    "n_defenders",
    "avg_defender_distance_ft",
    "nearest_defender_distance_ft",
]

GRAVITY_COLUMNS = [
    "player_label",
    "player_name",
    "team_id",
    "jersey_number",
    "games_included",
    "frames_with_possession",
    "frames_without_possession",
    "avg_defender_distance_with_ball",
    "avg_defender_distance_without_ball",
    "gravity_delta",
    "avg_defender_distance_overall",
    "nearest_defender_with_ball",
    "nearest_defender_without_ball",
    "nearest_defender_delta",
    "nearest_defender_overall",
]


@dataclass
class OffenseCall:
    """Which team had the ball in one shot, and how clearly."""

    shot_id: int
    team_id: str | None
    share: float
    handler_frames: int
    reason: str = ""


def load_calibration(game_id: str, shot_id: int) -> dict | None:
    path = calibration_path(game_id, shot_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def project_to_court(players: pd.DataFrame, matrix) -> pd.DataFrame:
    """Add court_x / court_y in feet, from each track's foot point.

    The foot point is used rather than the box centre because a homography
    maps the ground plane: a player's feet are on it, their head is six feet
    above it and would project metres away.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    court = to_court_feet(matrix, players[["foot_x", "foot_y"]].to_numpy())
    return players.assign(court_x=court[:, 0], court_y=court[:, 1])


def on_court(positions: pd.DataFrame, margin: float = COURT_MARGIN_FT) -> pd.Series:
    """Rows whose projected position is on the playing surface."""
    return positions["court_x"].between(-margin, COURT_WIDTH_FT + margin) & positions[
        "court_y"
    ].between(-margin, COURT_LENGTH_FT + margin)


def offensive_team(
    possession: pd.DataFrame,
    identity: pd.DataFrame,
    min_share: float = MIN_OFFENSE_SHARE,
) -> dict[int, OffenseCall]:
    """Decide, per camera shot, which team was attacking.

    Per-frame handler labels are too noisy to use directly — see the module
    docstring — so each shot takes the majority. `team_id` is None when the
    majority is not clear enough, which means the shot is not aggregated.
    """
    teams = identity.set_index(["shot_id", "tracker_id"])["team_id"].to_dict()
    calls: dict[int, OffenseCall] = {}

    for shot_id, group in possession.groupby("shot_id"):
        handlers = group.dropna(subset=["ball_handler_tracker_id"])
        labels = [
            teams.get((int(shot_id), int(tracker_id)))
            for tracker_id in handlers["ball_handler_tracker_id"]
        ]
        labels = [label for label in labels if label]
        shot_id = int(shot_id)
        if not labels:
            calls[shot_id] = OffenseCall(
                shot_id, None, 0.0, 0, "no frame has an identified ball handler"
            )
            continue
        counts = pd.Series(labels).value_counts()
        team, share = counts.index[0], counts.iloc[0] / counts.sum()
        if share < min_share:
            calls[shot_id] = OffenseCall(
                shot_id,
                None,
                float(share),
                int(counts.sum()),
                f"handler team is split {counts.to_dict()}, so which side is "
                "defending is unclear",
            )
            continue
        calls[shot_id] = OffenseCall(shot_id, str(team), float(share), int(counts.sum()))
    return calls


def defender_distances(
    tracks: pd.DataFrame,
    identity: pd.DataFrame,
    possession: pd.DataFrame,
    offense: dict[int, OffenseCall],
    matrices: dict[int, np.ndarray],
    defenders_required: int = EXPECTED_DEFENDERS,
) -> pd.DataFrame:
    """One row per (frame, offensive player): how far the defence is, in feet."""
    players = tracks[tracks["class"] == CLASS_PLAYER]
    teams = identity.set_index(["shot_id", "tracker_id"])["team_id"].to_dict()
    handler_of = (
        possession.set_index(["shot_id", "frame_idx"])["ball_handler_tracker_id"]
        .to_dict()
    )

    rows = []
    for shot_id, shot_players in players.groupby("shot_id"):
        shot_id = int(shot_id)
        call = offense.get(shot_id)
        if call is None or call.team_id is None or shot_id not in matrices:
            continue

        positioned = project_to_court(shot_players, matrices[shot_id])
        positioned = positioned[on_court(positioned)]
        positioned = positioned.assign(
            team_id=[
                teams.get((shot_id, int(tracker_id)))
                for tracker_id in positioned["tracker_id"]
            ]
        )

        for frame_idx, frame_rows in positioned.groupby("frame_idx"):
            defence = frame_rows[
                (frame_rows["team_id"].notna())
                & (frame_rows["team_id"] != call.team_id)
            ]
            # A defence of four is a missed player and a defence of eight is
            # one player tracked twice; neither average means what the metric
            # says it means.
            if len(defence) != defenders_required:
                continue
            spots = defence[["court_x", "court_y"]].to_numpy()

            handler = handler_of.get((shot_id, int(frame_idx)))
            attack = frame_rows[frame_rows["team_id"] == call.team_id]
            for row in attack.itertuples():
                gaps = np.hypot(spots[:, 0] - row.court_x, spots[:, 1] - row.court_y)
                rows.append(
                    {
                        "game_id": row.game_id,
                        "shot_id": shot_id,
                        "frame_idx": int(frame_idx),
                        "tracker_id": int(row.tracker_id),
                        "team_id": call.team_id,
                        "court_x": float(row.court_x),
                        "court_y": float(row.court_y),
                        "has_ball": bool(
                            handler is not None
                            and pd.notna(handler)
                            and int(handler) == int(row.tracker_id)
                        ),
                        "n_defenders": len(defence),
                        "avg_defender_distance_ft": float(gaps.mean()),
                        "nearest_defender_distance_ft": float(gaps.min()),
                    }
                )

    return pd.DataFrame(rows, columns=DISTANCE_COLUMNS)


def player_label(row) -> str | None:
    """How a track is named in the output, or None if it cannot be.

    A name is best, a team and number is still an identity a box score would
    recognise, and a bare tracker_id is neither — it cannot be matched to the
    same person in the next camera shot, let alone the next game, so it never
    becomes a player row.
    """
    if isinstance(row.get("player_name"), str) and row["player_name"]:
        return row["player_name"]
    number = row.get("jersey_number")
    if isinstance(number, str) and number:
        return f"{row.get('team_id') or '?'} #{number}"
    return None


def aggregate_players(
    distances: pd.DataFrame,
    identity: pd.DataFrame,
    min_bucket_frames: int = MIN_BUCKET_FRAMES,
    min_identity_confidence: float = 0.0,
) -> pd.DataFrame:
    """Per-player gravity metrics for one game.

    A player row is emitted whenever the player was identified and seen on
    court; `gravity_delta` is left null unless *both* buckets carry
    `min_bucket_frames`. Writing the row anyway is deliberate — the overall
    defender distance is a metric in its own right (`05-metrics-and-analysis`
    calls it raw gravity, against the delta's marginal gravity), and a null
    delta beside a visible frame count says exactly why it is missing.
    """
    if distances.empty:
        return pd.DataFrame(columns=GRAVITY_COLUMNS)

    known = identity[identity["identity_confidence"] >= min_identity_confidence]
    # team_id is deliberately not taken from identity here: the distances
    # table already carries the offence's label for the frame, and two columns
    # meaning almost the same thing is how they end up disagreeing.
    labelled = distances.merge(
        known[["shot_id", "tracker_id", "jersey_number", "player_name"]],
        on=["shot_id", "tracker_id"],
        how="inner",
    )
    if labelled.empty:
        return pd.DataFrame(columns=GRAVITY_COLUMNS)

    labelled["player_label"] = labelled.apply(player_label, axis=1)
    labelled = labelled[labelled["player_label"].notna()]
    if labelled.empty:
        return pd.DataFrame(columns=GRAVITY_COLUMNS)

    rows = []
    for label, group in labelled.groupby("player_label"):
        with_ball = group[group["has_ball"]]
        without = group[~group["has_ball"]]
        enough = (
            len(with_ball) >= min_bucket_frames and len(without) >= min_bucket_frames
        )

        def mean_or_none(frame: pd.DataFrame, column: str):
            return float(frame[column].mean()) if len(frame) else None

        avg_with = mean_or_none(with_ball, "avg_defender_distance_ft")
        avg_without = mean_or_none(without, "avg_defender_distance_ft")
        near_with = mean_or_none(with_ball, "nearest_defender_distance_ft")
        near_without = mean_or_none(without, "nearest_defender_distance_ft")

        first = group.iloc[0]
        rows.append(
            {
                "player_label": label,
                "player_name": first["player_name"]
                if isinstance(first["player_name"], str)
                else None,
                "team_id": first["team_id"],
                "jersey_number": first["jersey_number"]
                if isinstance(first["jersey_number"], str)
                else None,
                "games_included": int(group["game_id"].nunique()),
                "frames_with_possession": len(with_ball),
                "frames_without_possession": len(without),
                "avg_defender_distance_with_ball": avg_with,
                "avg_defender_distance_without_ball": avg_without,
                "gravity_delta": (avg_without - avg_with) if enough else None,
                "avg_defender_distance_overall": float(
                    group["avg_defender_distance_ft"].mean()
                ),
                "nearest_defender_with_ball": near_with,
                "nearest_defender_without_ball": near_without,
                "nearest_defender_delta": (near_without - near_with)
                if enough
                else None,
                "nearest_defender_overall": float(
                    group["nearest_defender_distance_ft"].mean()
                ),
            }
        )

    result = pd.DataFrame(rows, columns=GRAVITY_COLUMNS)
    return result.sort_values(
        "gravity_delta", ascending=False, na_position="last"
    ).reset_index(drop=True)


def combine_games(
    tables: list[pd.DataFrame], min_bucket_frames: int = MIN_BUCKET_FRAMES
) -> pd.DataFrame:
    """Roll per-game tables into one, weighting each average by its frames.

    A straight mean of per-game averages would give a game with 40 frames the
    same say as one with 400. The bucket-size rule is re-applied to the
    *totals*, which is the point of aggregating: 60 with-ball frames in each
    of two games is 120, and a delta neither game could support on its own
    becomes computable across both.
    """
    frames = [table for table in tables if not table.empty]
    if not frames:
        return pd.DataFrame(columns=GRAVITY_COLUMNS)

    stacked = pd.concat(frames, ignore_index=True)
    rows = []
    for label, group in stacked.groupby("player_label"):
        with_frames = group["frames_with_possession"].sum()
        without_frames = group["frames_without_possession"].sum()

        def weighted(column: str, weights: pd.Series) -> float | None:
            values = group[column]
            usable = values.notna() & (weights > 0)
            if not usable.any() or weights[usable].sum() == 0:
                return None
            return float(
                (values[usable] * weights[usable]).sum() / weights[usable].sum()
            )

        avg_with = weighted(
            "avg_defender_distance_with_ball", group["frames_with_possession"]
        )
        avg_without = weighted(
            "avg_defender_distance_without_ball", group["frames_without_possession"]
        )
        near_with = weighted(
            "nearest_defender_with_ball", group["frames_with_possession"]
        )
        near_without = weighted(
            "nearest_defender_without_ball", group["frames_without_possession"]
        )
        total = group["frames_with_possession"] + group["frames_without_possession"]
        has_delta = (
            with_frames >= min_bucket_frames and without_frames >= min_bucket_frames
        )

        rows.append(
            {
                "player_label": label,
                "player_name": group["player_name"].dropna().iloc[0]
                if group["player_name"].notna().any()
                else None,
                "team_id": group["team_id"].dropna().iloc[0]
                if group["team_id"].notna().any()
                else None,
                "jersey_number": group["jersey_number"].dropna().iloc[0]
                if group["jersey_number"].notna().any()
                else None,
                "games_included": int(group["games_included"].sum()),
                "frames_with_possession": int(with_frames),
                "frames_without_possession": int(without_frames),
                "avg_defender_distance_with_ball": avg_with,
                "avg_defender_distance_without_ball": avg_without,
                "gravity_delta": (avg_without - avg_with)
                if (has_delta and avg_with is not None and avg_without is not None)
                else None,
                "avg_defender_distance_overall": weighted(
                    "avg_defender_distance_overall", total
                ),
                "nearest_defender_with_ball": near_with,
                "nearest_defender_without_ball": near_without,
                "nearest_defender_delta": (near_without - near_with)
                if (has_delta and near_with is not None and near_without is not None)
                else None,
                "nearest_defender_overall": weighted(
                    "nearest_defender_overall", total
                ),
            }
        )

    return (
        pd.DataFrame(rows, columns=GRAVITY_COLUMNS)
        .sort_values("gravity_delta", ascending=False, na_position="last")
        .reset_index(drop=True)
    )
