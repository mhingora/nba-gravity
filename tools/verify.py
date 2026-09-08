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
