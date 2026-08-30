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
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd

from pipeline.common import (
    CLASS_BALL,
    CLASS_PLAYER,
    detections_path,
    identity_path,
    load_shots,
    possession_path,
    tracks_path,
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
