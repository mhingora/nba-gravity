"""Run the pipeline end to end and check the output against known truth.

Two modes, because there are two different questions worth asking:

**Ground-truth mode** (default) regenerates the synthetic clip and runs every
implemented stage over it. That clip is built with known answers — three cuts
at fixed frames, ten players, five per kit colour — so the checks here are
real assertions, not smoke tests. If this fails, something is broken.

**Structural mode** (`--game-id X --structural`) runs the invariants that must
hold for *any* footage against output you already produced: schema columns,
no duplicate ids within a frame, possession referring to real tracks. It
cannot tell you the pipeline is *correct* on real video — no ground truth
exists there, which is what the viewer is for — but it catches the class of
bug that silently corrupts a table.

Exit code is 0 only if every check passes, so this is usable as a gate.

    python tools/verify.py
    python tools/verify.py --game-id S_N3_HD --structural
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

import numpy as np

from pipeline.common import (
    CLASS_BALL,
    CLASS_PLAYER,
    detections_path,
    identity_path,
    load_shots,
    ocr_reads_path,
    possession_path,
    tracks_path,
)
from pipeline.court_geometry import (
    COURT_LANDMARKS,
    compute_homography,
    known_distance_checks,
    parse_keypoints,
    to_court_feet,
)

SYNTHETIC_GAME = "TESTCLIP"
# Ground truth baked into tools/make_test_clip.py.
EXPECTED_SHOTS = [(0, 59), (60, 119), (120, 179)]
EXPECTED_PLAYERS_PER_FRAME = 10
EXPECTED_FRAMES_PER_SHOT = 60
# Ten players alternating between two kit colours.
EXPECTED_PER_TEAM_PER_SHOT = 5

GREEN, RED, YELLOW, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[0m"


class Checker:
    def __init__(self) -> None:
        self.passed = 0
        self.failed: list[str] = []

    def check(self, label: str, condition: bool, detail: str = "") -> None:
        if condition:
            self.passed += 1
            print(f"  {GREEN}PASS{RESET}  {label}")
        else:
            self.failed.append(label)
            print(f"  {RED}FAIL{RESET}  {label}" + (f"  ({detail})" if detail else ""))

    def note(self, message: str) -> None:
        print(f"  {YELLOW}note{RESET}  {message}")


def run(command: list[str], checker: Checker, label: str) -> bool:
    """Run a pipeline stage, surfacing its output only when it fails."""
    result = subprocess.run(
        [sys.executable, *command],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
    )
    ok = result.returncode == 0
    checker.check(label, ok, f"exit {result.returncode}")
    if not ok:
        tail = (result.stderr or result.stdout).strip().splitlines()[-6:]
        for line in tail:
            print(f"        {line}")
    return ok


def check_schema(checker: Checker, frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    checker.check(f"{name} has its documented columns", not missing, f"missing {missing}")


def structural_checks(checker: Checker, game_id: str) -> None:
    """Invariants that must hold for any footage, ground truth or not."""
    tracks_file = tracks_path(game_id)
    if not tracks_file.exists():
        checker.check(f"{game_id}: tracks exist", False, "run 02_track.py first")
        return

    tracks = pd.read_parquet(tracks_file)
    players = tracks[tracks["class"] == CLASS_PLAYER]

    check_schema(
        checker,
        tracks,
        ["game_id", "frame_idx", "shot_id", "tracker_id", "class", "foot_x", "foot_y", "track_len"],
        "tracks",
    )

    duplicates = players.groupby(["shot_id", "frame_idx", "tracker_id"]).size()
    checker.check(
        "no tracker_id appears twice in one frame",
        int((duplicates > 1).sum()) == 0,
        f"{int((duplicates > 1).sum())} duplicated",
    )
    # -1 is the ball sentinel; a player carrying it would collide with ball rows.
    checker.check(
        "no player track uses the ball's -1 sentinel",
        int((players["tracker_id"] < 0).sum()) == 0,
    )
    checker.check(
        "foot point is the bottom-centre of the box",
        bool(
            ((players["foot_y"] - players["y2"]).abs() < 1e-6).all()
            and ((players["foot_x"] - (players["x1"] + players["x2"]) / 2).abs() < 1e-6).all()
        ),
    )
    lengths = players.groupby(["shot_id", "tracker_id"])["track_len"].nunique()
    checker.check("track_len is constant within a track", int((lengths > 1).sum()) == 0)

    identity_file = identity_path(game_id)
    if identity_file.exists():
        identity = pd.read_parquet(identity_file)
        check_schema(
            checker,
            identity,
            ["game_id", "shot_id", "tracker_id", "team_id", "jersey_number", "player_name", "identity_confidence"],
            "identity",
        )
        checker.check(
            "one identity row per track, no duplicates",
            not identity.duplicated(["shot_id", "tracker_id"]).any(),
        )
        known = set(map(tuple, players[["shot_id", "tracker_id"]].drop_duplicates().values))
        unknown = [
            row for row in map(tuple, identity[["shot_id", "tracker_id"]].values)
            if row not in known
        ]
        checker.check("identity refers only to real tracks", not unknown, f"{len(unknown)} orphans")
        confidence = identity["identity_confidence"]
        checker.check(
            "identity_confidence is within 0-1",
            bool(confidence.between(0.0, 1.0).all()),
        )

        numbers = identity["jersey_number"].dropna()
        bad_numbers = [
            n for n in numbers
            if not (str(n).isdigit() and 1 <= len(str(n)) <= 2)
        ]
        checker.check(
            "every jersey_number is a plausible one (0-99 or 00)",
            not bad_numbers,
            f"got {bad_numbers[:5]}",
        )
        named = identity[identity["player_name"].notna()]
        checker.check(
            "no name without the number and team it was looked up from",
            bool(named["jersey_number"].notna().all())
            and bool(named["team_id"].notna().all()),
            f"{len(named)} named row(s)",
        )
        identity_ocr_checks(checker, game_id, identity)

    possession_file = possession_path(game_id)
    if possession_file.exists():
        possession = pd.read_parquet(possession_file)
        check_schema(
            checker,
            possession,
            ["game_id", "shot_id", "frame_idx", "ball_handler_tracker_id", "ball_distance_px"],
            "possession",
        )
        checker.check(
            "one possession row per frame",
            not possession.duplicated(["shot_id", "frame_idx"]).any(),
        )
        handlers = possession.dropna(subset=["ball_handler_tracker_id"])
        pairs = set(map(tuple, players[["shot_id", "tracker_id"]].drop_duplicates().values))
        bad = [
            (int(s), int(t))
            for s, t in zip(handlers["shot_id"], handlers["ball_handler_tracker_id"])
            if (int(s), int(t)) not in pairs
        ]
        checker.check("every handler is a real track", not bad, f"{len(bad)} phantom handlers")
        distances = possession["ball_distance_px"].dropna()
        checker.check("ball_distance_px is never negative", bool((distances >= 0).all()))

    metrics_checks(checker, game_id, players)


def metrics_checks(checker: Checker, game_id: str, players: pd.DataFrame) -> None:
    """Stage 6's two tables, and whether they agree with each other."""
    from pipeline.common import distances_path, metrics_path
    from pipeline.court_geometry import COURT_LENGTH_FT, COURT_WIDTH_FT
    from pipeline.gravity import COURT_MARGIN_FT

    distances_file = distances_path(game_id)
    if not distances_file.exists():
        checker.note(f"{game_id}: no gravity output — run 06_aggregate.py")
        return

    distances = pd.read_parquet(distances_file)
    check_schema(
        checker,
        distances,
        ["game_id", "shot_id", "frame_idx", "tracker_id", "team_id", "court_x",
         "court_y", "has_ball", "n_defenders", "avg_defender_distance_ft",
         "nearest_defender_distance_ft"],
        "distances",
    )
    if not distances.empty:
        checker.check(
            "every measured frame has the same defence size",
            distances["n_defenders"].nunique() == 1,
            f"sizes {sorted(distances['n_defenders'].unique())}",
        )
        checker.check(
            "the nearest defender is never further than the average",
            bool(
                (
                    distances["nearest_defender_distance_ft"]
                    <= distances["avg_defender_distance_ft"] + 1e-9
                ).all()
            ),
        )
        checker.check(
            "measured players are on the court",
            bool(
                distances["court_x"]
                .between(-COURT_MARGIN_FT, COURT_WIDTH_FT + COURT_MARGIN_FT)
                .all()
                and distances["court_y"]
                .between(-COURT_MARGIN_FT, COURT_LENGTH_FT + COURT_MARGIN_FT)
                .all()
            ),
        )
        known = set(map(tuple, players[["shot_id", "tracker_id"]].drop_duplicates().values))
        orphans = [
            row for row in map(tuple, distances[["shot_id", "tracker_id"]].values)
            if row not in known
        ]
        checker.check(
            "distances refer only to real tracks", not orphans, f"{len(orphans)} orphans"
        )
        checker.check(
            "one row per (frame, player), no duplicates",
            not distances.duplicated(["shot_id", "frame_idx", "tracker_id"]).any(),
        )

    metrics_file = metrics_path(game_id)
    if not metrics_file.exists():
        return
    metrics = pd.read_parquet(metrics_file)
    check_schema(
        checker,
        metrics,
        ["player_label", "player_name", "games_included",
         "frames_with_possession", "frames_without_possession",
         "avg_defender_distance_with_ball", "avg_defender_distance_without_ball",
         "gravity_delta", "avg_defender_distance_overall"],
        "gravity",
    )
    if metrics.empty:
        checker.note(f"{game_id}: no player row survived aggregation")
        return

    checker.check(
        "one row per player, no duplicates",
        not metrics["player_label"].duplicated().any(),
    )
    with_delta = metrics[metrics["gravity_delta"].notna()]
    if len(with_delta):
        recomputed = (
            with_delta["avg_defender_distance_without_ball"]
            - with_delta["avg_defender_distance_with_ball"]
        )
        checker.check(
            "gravity_delta is exactly without-ball minus with-ball",
            bool((with_delta["gravity_delta"] - recomputed).abs().max() < 1e-9),
        )
    else:
        checker.note(
            f"{game_id}: no player has frames in both buckets, so no "
            "gravity_delta — the footage processed is too short, not a bug"
        )
    checker.check(
        "no player row is built from zero frames",
        bool(
            (
                metrics["frames_with_possession"]
                + metrics["frames_without_possession"]
                > 0
            ).all()
        ),
    )
    total_frames = int(
        metrics["frames_with_possession"].sum()
        + metrics["frames_without_possession"].sum()
    )
    checker.check(
        "the metrics table counts no more frames than were measured",
        total_frames <= len(distances),
        f"{total_frames} counted vs {len(distances)} measured",
    )


def _gravity_fixture(defender_ys=(25, 30, 35, 40, 45), handler=1):
    """A frame with a target at (25,20) and defenders straight down court.

    Distances come out 5, 10, 15, 20, 25 ft, so the average is 15 and the
    nearest is 5 — numbers that can be checked by hand rather than by
    re-running the code that produced them.
    """
    rows = [
        {"game_id": "T", "shot_id": 0, "frame_idx": 0, "tracker_id": 1,
         "class": CLASS_PLAYER, "foot_x": 25.0, "foot_y": 20.0},
    ]
    for offset, y in enumerate(defender_ys):
        rows.append(
            {"game_id": "T", "shot_id": 0, "frame_idx": 0,
             "tracker_id": 100 + offset, "class": CLASS_PLAYER,
             "foot_x": 25.0, "foot_y": float(y)}
        )
    tracks = pd.DataFrame(rows)

    identity = pd.DataFrame(
        [{"shot_id": 0, "tracker_id": 1, "team_id": "dark",
          "jersey_number": "2", "player_name": "Tester",
          "identity_confidence": 0.5}]
        + [
            {"shot_id": 0, "tracker_id": 100 + i, "team_id": "light",
             "jersey_number": None, "player_name": None,
             "identity_confidence": 0.5}
            for i in range(len(defender_ys))
        ]
    )
    possession = pd.DataFrame(
        [{"game_id": "T", "shot_id": 0, "frame_idx": 0,
          "ball_handler_tracker_id": handler, "ball_distance_px": 1.0}]
    )
    return tracks, identity, possession


def gravity_checks(checker: Checker) -> None:
    """Stage 6's arithmetic, on positions whose answers are known by hand.

    The court projection is fed an identity homography here, so a foot point
    at (25, 20) is 25ft across and 20ft down the court. That keeps these
    checks about the metric rather than about the homography, which
    `homography_checks` already covers.
    """
    from pipeline.gravity import (
        aggregate_players,
        combine_games,
        defender_distances,
        offensive_team,
        player_label,
    )

    identity_matrix = np.eye(3)
    tracks, identity, possession = _gravity_fixture()

    offense = offensive_team(possession, identity)
    checker.check(
        "the handler's team is the offence",
        offense[0].team_id == "dark",
        f"got {offense[0].team_id}",
    )

    distances = defender_distances(
        tracks, identity, possession, offense, {0: identity_matrix}
    )
    checker.check(
        "one row per offensive player per frame",
        len(distances) == 1,
        f"got {len(distances)}",
    )
    if len(distances) == 1:
        row = distances.iloc[0]
        checker.check(
            "average defender distance is the mean of the five",
            abs(row["avg_defender_distance_ft"] - 15.0) < 1e-6,
            f"got {row['avg_defender_distance_ft']:.3f}",
        )
        checker.check(
            "nearest defender distance is the closest one",
            abs(row["nearest_defender_distance_ft"] - 5.0) < 1e-6,
            f"got {row['nearest_defender_distance_ft']:.3f}",
        )
        checker.check("the handler is flagged as having the ball", bool(row["has_ball"]))

    # A sixth defender means one player is tracked twice, which would weight
    # them double in the average.
    crowded_tracks, crowded_identity, crowded_possession = _gravity_fixture(
        defender_ys=(25, 30, 35, 40, 45, 50)
    )
    six = defender_distances(
        crowded_tracks,
        crowded_identity,
        crowded_possession,
        offensive_team(crowded_possession, crowded_identity),
        {0: identity_matrix},
    )
    checker.check("a frame with six defenders is dropped", six.empty, f"got {len(six)}")

    # Offence is a per-shot majority; a shot that cannot decide is refused.
    split = pd.DataFrame(
        [
            {"game_id": "T", "shot_id": 0, "frame_idx": i,
             "ball_handler_tracker_id": 1 if i % 2 else 100,
             "ball_distance_px": 1.0}
            for i in range(10)
        ]
    )
    checker.check(
        "a shot with a split handler team is refused, not guessed",
        offensive_team(split, identity)[0].team_id is None,
    )

    # Aggregation: 120 frames without the ball at 20ft, 120 with it at 10ft.
    frames = []
    for i in range(240):
        has_ball = i < 120
        frames.append(
            {"game_id": "T", "shot_id": 0, "frame_idx": i, "tracker_id": 1,
             "team_id": "dark", "court_x": 25.0, "court_y": 20.0,
             "has_ball": has_ball, "n_defenders": 5,
             "avg_defender_distance_ft": 10.0 if has_ball else 20.0,
             "nearest_defender_distance_ft": 4.0 if has_ball else 8.0}
        )
    table = aggregate_players(pd.DataFrame(frames), identity, min_bucket_frames=100)
    checker.check("an identified track becomes a player row", len(table) == 1)
    if len(table) == 1:
        got = table.iloc[0]
        checker.check(
            "gravity_delta is without-ball minus with-ball",
            abs(got["gravity_delta"] - 10.0) < 1e-6,
            f"got {got['gravity_delta']}",
        )
        checker.check(
            "the row is keyed by the roster name when there is one",
            got["player_label"] == "Tester",
            f"got {got['player_label']}",
        )

    thin = aggregate_players(
        pd.DataFrame(frames[:130]), identity, min_bucket_frames=100
    )
    checker.check(
        "a bucket under the frame floor leaves the delta null, not wrong",
        len(thin) == 1
        and pd.isna(thin.iloc[0]["gravity_delta"])
        and thin.iloc[0]["avg_defender_distance_overall"] is not None,
    )

    checker.check(
        "a number without a roster name still identifies a player",
        player_label({"player_name": None, "jersey_number": "24", "team_id": "dark"})
        == "dark #24",
    )
    checker.check(
        "a track with neither name nor number is not a player row",
        player_label({"player_name": None, "jersey_number": None, "team_id": "dark"})
        is None,
    )

    # Two games of 60 frames a bucket: neither supports a delta alone, both
    # together do, and the rollup must weight by frames rather than by game.
    def one_game(avg_with: float, frames_each: int) -> pd.DataFrame:
        return aggregate_players(
            pd.DataFrame(
                [
                    {"game_id": "T", "shot_id": 0, "frame_idx": i, "tracker_id": 1,
                     "team_id": "dark", "court_x": 25.0, "court_y": 20.0,
                     "has_ball": i < frames_each, "n_defenders": 5,
                     "avg_defender_distance_ft": avg_with if i < frames_each else 20.0,
                     "nearest_defender_distance_ft": 4.0 if i < frames_each else 8.0}
                    for i in range(frames_each * 2)
                ]
            ),
            identity,
            min_bucket_frames=100,
        )

    first, second = one_game(10.0, 60), one_game(14.0, 60)
    checker.check(
        "neither game alone supports a delta",
        pd.isna(first.iloc[0]["gravity_delta"])
        and pd.isna(second.iloc[0]["gravity_delta"]),
    )
    rolled = combine_games([first, second], min_bucket_frames=100)
    checker.check(
        "pooled frames across games do support one",
        len(rolled) == 1 and abs(rolled.iloc[0]["gravity_delta"] - 8.0) < 1e-6,
        f"got {rolled.iloc[0]['gravity_delta'] if len(rolled) else 'no row'}",
    )
    checker.check(
        "the rollup counts every frame from both games",
        len(rolled) == 1
        and rolled.iloc[0]["frames_with_possession"] == 120
        and rolled.iloc[0]["frames_without_possession"] == 120,
    )


def identity_ocr_checks(
    checker: Checker, game_id: str, identity: pd.DataFrame
) -> None:
    """The OCR sidecar has to tell the same story as the identity table.

    Only runs when `--ocr` was used. The viewer explains a track's number
    from this file while the metric is computed from the parquet, so the two
    disagreeing would mean the UI is vouching for something else's answer.
    """
    reads_file = ocr_reads_path(game_id)
    if not reads_file.exists():
        checker.note(
            f"{game_id}: no OCR evidence — stage 3 was run without --ocr, so "
            "jersey numbers are null by design"
        )
        return

    evidence = json.loads(reads_file.read_text(encoding="utf-8"))
    entries = evidence.get("tracks", [])
    min_agreement = evidence.get("settings", {}).get("min_agreement", 0)

    known = set(map(tuple, identity[["shot_id", "tracker_id"]].values))
    orphans = [
        (entry["shot_id"], entry["tracker_id"])
        for entry in entries
        if (entry["shot_id"], entry["tracker_id"]) not in known
    ]
    checker.check(
        "OCR evidence refers only to tracks in the identity table",
        not orphans,
        f"{len(orphans)} orphans",
    )

    from_parquet = {
        (int(row.shot_id), int(row.tracker_id)): row.jersey_number
        for row in identity.itertuples()
        if pd.notna(row.jersey_number)
    }
    from_json = {
        (entry["shot_id"], entry["tracker_id"]): entry["jersey_number"]
        for entry in entries
        if entry["jersey_number"]
    }
    checker.check(
        "the sidecar and the identity table agree on every number",
        from_parquet == from_json,
        f"{len(from_parquet)} in parquet vs {len(from_json)} in json",
    )
    under_threshold = [
        entry["tracker_id"]
        for entry in entries
        if entry["jersey_number"] and entry["agreeing"] < min_agreement
    ]
    checker.check(
        f"no number was accepted on fewer than {min_agreement} agreeing reads",
        not under_threshold,
        f"tracks {under_threshold[:5]}",
    )
    unsupported = [
        entry["tracker_id"]
        for entry in entries
        if entry["jersey_number"]
        and entry["counts"].get(entry["jersey_number"], 0) != entry["agreeing"]
    ]
    checker.check(
        "each accepted number's tally matches the reads behind it",
        not unsupported,
        f"tracks {unsupported[:5]}",
    )


def jersey_ocr_checks(checker: Checker) -> None:
    """The voting rules, asserted on the reads real footage actually produced.

    No video and no OCR model needed — this is the decision logic on its own,
    and each case here is a bug that was live at some point.
    """
    from collections import Counter

    from pipeline.jersey_ocr import (
        DEFAULT_MIN_AGREEMENT,
        build_team_map,
        normalise_read,
        vote,
    )

    checker.check("'00' survives as a number of its own", normalise_read("00") == "00")
    checker.check("'07' is read as 7", normalise_read("07") == "7")
    checker.check("'330' is rejected as impossible", normalise_read("330") is None)
    checker.check("empty text is rejected", normalise_read("") is None)

    # Track 5 of the test possession: Harper, who wears 2. An earlier version
    # folded the eight "2" reads into the six "24" reads and answered 24.
    number, agreeing, total = vote(Counter({"2": 8, "24": 6}))
    checker.check(
        "a plurality of single-digit reads wins over a two-digit rival",
        (number, agreeing, total) == ("2", 8, 14),
        f"got {number!r} with {agreeing}/{total}",
    )
    checker.check(
        "three agreeing reads are not enough",
        vote(Counter({"1": 3}))[0] is None,
    )
    checker.check(
        f"{DEFAULT_MIN_AGREEMENT} agreeing reads are enough",
        vote(Counter({"1": DEFAULT_MIN_AGREEMENT}))[0] == "1",
    )
    checker.check(
        "a tie resolves to nothing rather than a coin flip",
        vote(Counter({"1": 5, "0": 5}))[0] is None,
    )
    checker.check("no reads means no number", vote(Counter()) == (None, 0, 0))

    checker.check(
        "team mapping parses cluster=TEAM",
        build_team_map(["light=NYK", "dark=SAS"]) == {"light": "NYK", "dark": "SAS"},
    )
    malformed = False
    try:
        build_team_map(["NYK"])
    except ValueError:
        malformed = True
    checker.check("a malformed --team is refused", malformed)


def ground_truth_checks(checker: Checker) -> None:
    """Assertions only possible because the synthetic clip's answers are known."""
    shots = load_shots(SYNTHETIC_GAME)
    checker.check(
        f"stage 0 finds exactly {len(EXPECTED_SHOTS)} shots",
        len(shots) == len(EXPECTED_SHOTS),
        f"got {len(shots)}",
    )
    if len(shots) == len(EXPECTED_SHOTS):
        actual = [(s.start_frame, s.end_frame) for s in shots]
        checker.check(
            "cuts land exactly on the synthesised boundaries",
            actual == EXPECTED_SHOTS,
            f"got {actual}",
        )

    detections = pd.read_parquet(detections_path(SYNTHETIC_GAME))
    per_frame = detections[detections["class"] == CLASS_PLAYER].groupby("frame_idx").size()
    checker.check(
        f"stage 1 finds {EXPECTED_PLAYERS_PER_FRAME} players in every frame",
        bool((per_frame == EXPECTED_PLAYERS_PER_FRAME).all()),
        f"min {per_frame.min()}, max {per_frame.max()}",
    )
    checker.check(
        "stage 1 finds the ball at all",
        int((detections["class"] == CLASS_BALL).sum()) > 0,
    )

    tracks = pd.read_parquet(tracks_path(SYNTHETIC_GAME))
    players = tracks[tracks["class"] == CLASS_PLAYER]
    per_shot = players.groupby("shot_id")["tracker_id"].nunique()
    checker.check(
        f"stage 2 yields {EXPECTED_PLAYERS_PER_FRAME} tracks in every shot",
        bool((per_shot == EXPECTED_PLAYERS_PER_FRAME).all()),
        f"got {per_shot.to_dict()}",
    )
    spans = players.groupby(["shot_id", "tracker_id"]).size()
    # Allow one frame of slack: some backends need a frame to confirm a track.
    checker.check(
        "every track survives its whole shot",
        bool((spans >= EXPECTED_FRAMES_PER_SHOT - 1).all()),
        f"shortest {spans.min()} of {EXPECTED_FRAMES_PER_SHOT}",
    )

    identity = pd.read_parquet(identity_path(SYNTHETIC_GAME))
    counts = identity["team_id"].value_counts()
    expected_total = EXPECTED_PER_TEAM_PER_SHOT * len(EXPECTED_SHOTS)
    checker.check(
        f"stage 3 splits the two kit colours {expected_total}/{expected_total}",
        int(counts.get("light", 0)) == expected_total
        and int(counts.get("dark", 0)) == expected_total,
        f"got {counts.to_dict()}",
    )

    possession = pd.read_parquet(possession_path(SYNTHETIC_GAME))
    checker.check("stage 4 writes a row per tracked frame", len(possession) > 0)
    checker.note(
        "the synthetic ball flies on its own path and is not carried, so a low "
        "possession rate here is expected and is not checked"
    )


def homography_checks(checker: Checker) -> None:
    """Check the Stage 5 maths against a homography we constructed ourselves.

    Real footage has no ground truth for this — nobody knows the true camera
    matrix — so accuracy there can only be judged by eye in the viewer. What
    *can* be checked exactly is whether the solver recovers a transform that
    is known by construction: place the court landmarks through an invented
    perspective matrix, hand the resulting pixels back, and the fit should
    reproduce it to floating-point precision.

    That separates "the maths is wrong" from "the annotation is wrong", which
    otherwise look identical: both give a large reprojection error.
    """
    frame_width, frame_height = 1920.0, 1080.0
    names = [
        "baseline_left_corner",
        "baseline_right_corner",
        "free_throw_left",
        "free_throw_right",
        "halfcourt_left",
        "halfcourt_right",
    ]
    court = np.array([COURT_LANDMARKS[n] for n in names], dtype=np.float64)

    invented = np.array(
        [[6.0, 2.0, 300.0], [0.5, 3.5, 200.0], [0.0004, 0.0022, 1.0]]
    )
    homogeneous = np.column_stack([court, np.ones(len(court))]) @ invented.T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]

    keypoints = {
        name: (pixels[i, 0] / frame_width, pixels[i, 1] / frame_height)
        for i, name in enumerate(names)
    }
    image_points, court_points, parsed_names = parse_keypoints(keypoints)
    matrix, _, rms = compute_homography(
        image_points, court_points, frame_width, frame_height
    )

    checker.check(
        "homography recovers a known transform (rms < 0.01px)",
        rms < 0.01,
        f"rms {rms:.4f}px",
    )
    recovered = to_court_feet(matrix, pixels)
    checker.check(
        "projected landmarks land on their true court positions",
        bool(np.abs(recovered - court).max() < 0.01),
        f"max {np.abs(recovered - court).max():.4f}ft",
    )
    distances = known_distance_checks(
        matrix, image_points, parsed_names, frame_width, frame_height
    )
    checker.check(
        "known court distances measure correctly through the homography",
        bool(distances) and all(abs(d["error_ft"]) < 0.05 for d in distances),
        f"{[d['error_ft'] for d in distances]}",
    )

    # Degenerate input must be refused. A homography fitted to collinear
    # points is not merely inaccurate, it is meaningless — and it would
    # produce confident, wrong distances downstream.
    collinear = {
        "baseline_left_corner": (0.1, 0.5),
        "baseline_right_corner": (0.2, 0.5),
        "lane_baseline_left": (0.3, 0.5),
        "lane_baseline_right": (0.4, 0.5),
    }
    try:
        bad_image, bad_court, _ = parse_keypoints(collinear)
        compute_homography(bad_image, bad_court, frame_width, frame_height)
        rejected = False
    except ValueError:
        rejected = True
    checker.check("collinear landmarks are refused", rejected)

    try:
        parse_keypoints({"baseline_left_corner": (0.1, 0.2)})
        too_few_rejected = False
    except ValueError:
        too_few_rejected = True
    checker.check("fewer than four landmarks are refused", too_few_rejected)

    try:
        parse_keypoints({f"bogus_{i}": (0.1 * i, 0.2) for i in range(4)})
        unknown_rejected = False
    except ValueError:
        unknown_rejected = True
    checker.check("unknown landmark names are refused", unknown_rejected)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify pipeline output")
    parser.add_argument(
        "--game-id",
        default=SYNTHETIC_GAME,
        help="Game to check. Defaults to the synthetic clip.",
    )
    parser.add_argument(
        "--structural",
        action="store_true",
        help="Skip regenerating and re-running; check existing output for the "
        "invariants that hold on any footage. Use this on real games.",
    )
    args = parser.parse_args()

    checker = Checker()

    if not args.structural:
        if args.game_id != SYNTHETIC_GAME:
            print(
                f"[error] ground-truth mode only works on {SYNTHETIC_GAME}. "
                "Use --structural for real footage.",
                file=sys.stderr,
            )
            return 2

        print("Running the pipeline on the synthetic clip")
        stages = [
            (["tools/make_test_clip.py", "--game-id", SYNTHETIC_GAME], "make_test_clip"),
            (["pipeline/01_detect.py", "--game-id", SYNTHETIC_GAME, "--detector", "colorblob"], "stage 0+1 detect"),
            (["pipeline/02_track.py", "--game-id", SYNTHETIC_GAME], "stage 2 track"),
            (["pipeline/03_identify.py", "--game-id", SYNTHETIC_GAME], "stage 3 identify"),
            (["pipeline/04_ball_possession.py", "--game-id", SYNTHETIC_GAME], "stage 4 possession"),
            # The synthetic clip has no annotated court, so stage 6 finds
            # nothing to project and writes empty tables. Running it anyway is
            # the point: "nothing survived the filters" is the path most
            # likely to crash, and the one real footage hits most often.
            (["pipeline/06_aggregate.py", "--game-id", SYNTHETIC_GAME], "stage 6 aggregate"),
        ]
        for command, label in stages:
            if not run(command, checker, label):
                print("\nA stage failed; later checks would be meaningless.")
                return 1

        print("\nChecking against known ground truth")
        ground_truth_checks(checker)

        print("\nChecking stage 5 homography maths")
        homography_checks(checker)

        print("\nChecking stage 3 jersey-number voting")
        jersey_ocr_checks(checker)

        print("\nChecking stage 6 gravity arithmetic")
        gravity_checks(checker)

    print(f"\nChecking structural invariants for {args.game_id}")
    structural_checks(checker, args.game_id)

    total = checker.passed + len(checker.failed)
    print(f"\n{checker.passed}/{total} checks passed")
    if checker.failed:
        print(f"{RED}Failed:{RESET}")
        for label in checker.failed:
            print(f"  - {label}")
        return 1
    print(f"{GREEN}All checks passed.{RESET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
