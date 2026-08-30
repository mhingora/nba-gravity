"""Stage 2 — Tracking.

Input:  outputs/detections/{game_id}.parquet
Output: outputs/tracks/{game_id}.parquet

Spec: docs/03-pipeline-stages.md (Stage 2)

Schema note: docs/02-data-schemas.md describes this table as one row per
tracked *player* per frame, but Stage 4 reads "ball + player rows" from it.
This writes both, distinguished by an extra `class` column, so possession can
be computed from a single file. Ball rows carry `tracker_id = -1` — the ball
is detected, never tracked, since ByteTrack's motion model is built for
person-sized boxes.

Example:
    python pipeline/02_track.py --game-id 0022500123
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import supervision as sv

from pipeline.common import (
    CLASS_BALL,
    CLASS_PLAYER,
    ParquetBatchWriter,
    detections_path,
    tracks_path,
)
from pipeline.common import find_video
from pipeline.tracker_backends import (
    BACKEND_NAMES,
    DEFAULT_BACKEND,
    SUPERVISION_BYTETRACK,
    FrameFeeder,
    build_tracker,
)
from pipeline.court_region import (
    load_profile,
    parse_polygon,
    points_in_polygon,
    to_pixels,
)
from pipeline.track_quality import format_shot_line, summarize_shot

BALL_TRACKER_ID = -1

TRACK_OUTPUT_COLUMNS = [
    "game_id",
    "frame_idx",
    "shot_id",
    "tracker_id",
    "class",
    "x1",
    "y1",
    "x2",
    "y2",
    "foot_x",
    "foot_y",
    "track_len",
]


def filter_detections(
    frame_detections: pd.DataFrame,
    min_confidence: float,
    min_height_frac: float,
    max_height_frac: float,
    min_aspect: float,
    max_aspect: float,
    frame_height: float,
    court_roi: tuple[float, float, float, float] | None,
    frame_width: float,
    court_polygon=None,
) -> pd.DataFrame:
    """Drop implausible player boxes before they reach the tracker.

    Garbage in, garbage tracked: a spurious box that survives a few frames
    becomes a phantom tracker_id that pollutes defender-distance averages
    downstream.
    """
    keep = frame_detections["confidence"] >= min_confidence

    height = frame_detections["y2"] - frame_detections["y1"]
    width = (frame_detections["x2"] - frame_detections["x1"]).replace(0, np.nan)
    keep &= height >= min_height_frac * frame_height
    keep &= height <= max_height_frac * frame_height

    # Players are taller than they are wide. Boxes that aren't are usually
    # two overlapping people merged into one detection, or bench/crowd clutter.
    aspect = height / width
    keep &= aspect.between(min_aspect, max_aspect)

    if court_roi is not None:
        # Test the foot point, not the whole box: a spectator's feet sit
        # outside the playing surface even when their head overlaps it.
        rx1, ry1, rx2, ry2 = court_roi
        foot_x = (frame_detections["x1"] + frame_detections["x2"]) / 2.0
        foot_y = frame_detections["y2"]
        keep &= foot_x.between(rx1 * frame_width, rx2 * frame_width)
        keep &= foot_y.between(ry1 * frame_height, ry2 * frame_height)

    if court_polygon is not None:
        # The trapezoid test the rectangle can't do: a spectator behind the
        # far baseline shares a `y` band with far-side players but falls
        # outside the playing surface itself.
        polygon_px = to_pixels(court_polygon, frame_width, frame_height)
        foot_x = (frame_detections["x1"] + frame_detections["x2"]) / 2.0
        foot_y = frame_detections["y2"]
        keep &= pd.Series(
            points_in_polygon(foot_x.to_numpy(), foot_y.to_numpy(), polygon_px),
            index=frame_detections.index,
        )

    return frame_detections[keep.fillna(False)]


def to_sv_detections(frame_detections: pd.DataFrame) -> sv.Detections:
    if frame_detections.empty:
        return sv.Detections.empty()
    return sv.Detections(
        xyxy=frame_detections[["x1", "y1", "x2", "y2"]].to_numpy(dtype=np.float32),
        confidence=frame_detections["confidence"].to_numpy(dtype=np.float32),
        class_id=np.zeros(len(frame_detections), dtype=int),
    )


def track_shot(
    game_id: str,
    shot_id: int,
    shot_detections: pd.DataFrame,
    tracker_args: dict,
    filter_args: dict,
    backend: str = SUPERVISION_BYTETRACK,
    video_path=None,
) -> list[dict]:
    """Track one continuous camera shot. Returns rows with `track_len` filled in."""
    # A fresh tracker per shot — never carry state across a cut, or the tracker
    # will happily bridge two unrelated camera angles.
    tracker = build_tracker(backend, **tracker_args)
    tracker.reset()

    # Only BoT-SORT's camera motion compensation reads pixels. Stage 2 stays a
    # pure parquet-to-parquet step for every other backend, which is why the
    # video is opened lazily rather than as a matter of course.
    feeder = None
    if tracker.needs_frame:
        if video_path is None:
            raise ValueError(
                f"backend '{backend}' needs video frames but none was provided"
            )
        feeder = FrameFeeder(video_path, int(shot_detections["frame_idx"].min()))

    players = shot_detections[shot_detections["class"] == CLASS_PLAYER]
    balls = shot_detections[shot_detections["class"] == CLASS_BALL]
    rows: list[dict] = []

    for frame_idx, frame_players in players.groupby("frame_idx", sort=True):
        kept = filter_detections(frame_players, **filter_args)
        frame = feeder.get(int(frame_idx)) if feeder is not None else None
        tracked = tracker.update(to_sv_detections(kept), frame)

        for i in range(len(tracked)):
            # The `trackers` backends return unconfirmed detections with
            # tracker_id -1, meaning "no id assigned yet"; sv.ByteTrack only
            # ever returned confirmed tracks. These are not tracks, and -1 is
            # this table's BALL_TRACKER_ID sentinel, so letting them through
            # both inflates track counts and collides with ball rows.
            if tracked.tracker_id is None or int(tracked.tracker_id[i]) < 0:
                continue

            x1, y1, x2, y2 = (float(v) for v in tracked.xyxy[i])
            rows.append(
                {
                    "game_id": game_id,
                    "frame_idx": int(frame_idx),
                    "shot_id": shot_id,
                    "tracker_id": int(tracked.tracker_id[i]),
                    "class": CLASS_PLAYER,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    # Bottom-center approximates where the player is standing,
                    # which is the point Stage 5 projects through the homography.
                    "foot_x": (x1 + x2) / 2.0,
                    "foot_y": y2,
                    "track_len": 0,
                }
            )

    for row in balls.itertuples():
        rows.append(
            {
                "game_id": game_id,
                "frame_idx": int(row.frame_idx),
                "shot_id": shot_id,
                "tracker_id": BALL_TRACKER_ID,
                "class": CLASS_BALL,
                "x1": float(row.x1),
                "y1": float(row.y1),
                "x2": float(row.x2),
                "y2": float(row.y2),
                "foot_x": (float(row.x1) + float(row.x2)) / 2.0,
                "foot_y": float(row.y2),
                "track_len": 0,
            }
        )

    # track_len is only knowable once the shot ends. Short tracks are usually
    # detection noise or an occlusion, so downstream stages filter on it.
    lengths = Counter(
        row["tracker_id"] for row in rows if row["class"] == CLASS_PLAYER
    )
    for row in rows:
        if row["class"] == CLASS_PLAYER:
            row["track_len"] = lengths[row["tracker_id"]]

    if feeder is not None:
        feeder.close()

    rows.sort(key=lambda r: (r["frame_idx"], r["class"], r["tracker_id"]))
    return rows


# Tracker defaults live here rather than in argparse so the three-way
# resolution below (explicit flag > court profile > default) has one source
# of truth. An argparse default would fire before a profile could be read,
# making "user said nothing" indistinguishable from "user asked for 30".
TRACKER_DEFAULTS = {
    "track_activation_threshold": 0.25,
    "lost_track_buffer": 30,
    "minimum_matching_threshold": 0.8,
    "frame_rate": 30,
}


def parse_tracker_args(pairs) -> dict:
    """Turn `KEY=VALUE` strings into typed kwargs (bool > int > float > str)."""
    parsed: dict = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--tracker-arg expects KEY=VALUE, got '{pair}'")
        key, _, raw = pair.partition("=")
        key, raw = key.strip(), raw.strip()
        if raw.lower() in {"true", "false"}:
            parsed[key] = raw.lower() == "true"
            continue
        for cast in (int, float):
            try:
                parsed[key] = cast(raw)
                break
            except ValueError:
                continue
        else:
            parsed[key] = raw
    return parsed


def resolve_tracker_args(args, profile: dict | None) -> dict:
    """Explicit CLI flag wins, then the court profile, then the default."""
    from_profile = (profile or {}).get("tracker", {})
    return {
        key: (
            getattr(args, key)
            if getattr(args, key) is not None
            else from_profile.get(key, fallback)
        )
        for key, fallback in TRACKER_DEFAULTS.items()
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 2 — tracking")
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.30,
        help="Confidence floor applied before tracking.",
    )
    parser.add_argument("--min-height-frac", type=float, default=0.03)
    parser.add_argument("--max-height-frac", type=float, default=0.60)
    parser.add_argument("--min-aspect", type=float, default=1.1)
    parser.add_argument("--max-aspect", type=float, default=6.0)
    parser.add_argument(
        "--court-roi",
        nargs=4,
        type=float,
        default=None,
        metavar=("X1", "Y1", "X2", "Y2"),
        help="Normalized 0-1 playing-surface bounds; foot points outside are dropped.",
    )
    parser.add_argument(
        "--court-polygon",
        nargs="+",
        type=float,
        default=None,
        metavar="X Y",
        help="Normalized 0-1 playing-surface polygon as x y x y ...; foot "
        "points outside are dropped. Fits a broadcast trapezoid, which "
        "--court-roi cannot. Tune it in the viewer's Tracking tab.",
    )
    parser.add_argument(
        "--tracker",
        choices=BACKEND_NAMES,
        default=None,
        help="Tracking backend. 'sv-bytetrack' is supervision's deprecated "
        "ByteTrack (the default, unchanged); the others come from the "
        "`trackers` package. 'botsort' adds camera motion compensation and "
        "is the only one that reads video frames.",
    )
    parser.add_argument(
        "--tracker-arg",
        action="append",
        default=None,
        metavar="KEY=VALUE",
        help="Backend-specific setting, repeatable, e.g. "
        "--tracker-arg lost_track_buffer=90. Backends do not share a "
        "parameter vocabulary, so these are passed through as given.",
    )
    parser.add_argument(
        "--court-profile",
        default=None,
        metavar="NAME",
        help="Load the court polygon and tracker settings from "
        "data/calibration/NAME.json. Any flag passed explicitly overrides "
        "the profile. Save one from the viewer's Court polygon tuner.",
    )
    parser.add_argument("--frame-width", type=float, default=None)
    parser.add_argument("--frame-height", type=float, default=None)
    # Default None, not the real value, so resolve_tracker_args() can tell
    # "user said nothing" from "user asked for the default".
    parser.add_argument("--track-activation-threshold", type=float, default=None)
    parser.add_argument("--lost-track-buffer", type=int, default=None)
    parser.add_argument("--minimum-matching-threshold", type=float, default=None)
    parser.add_argument("--frame-rate", type=int, default=None)
    args = parser.parse_args()

    source = detections_path(args.game_id)
    if not source.exists():
        print(f"[error] {source} not found — run 01_detect.py first", file=sys.stderr)
        return 1

    profile = None
    if args.court_profile:
        try:
            profile = load_profile(args.court_profile)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        polygon_note = (
            f"{len(profile['court_polygon'])}-point polygon"
            if profile.get("court_polygon") is not None
            else "no polygon"
        )
        print(
            f"[stage 2] court profile '{args.court_profile}': {polygon_note}"
            + (f", tracker {profile['tracker']}" if profile.get("tracker") else ""),
            file=sys.stderr,
        )

    detections = pd.read_parquet(source)
    if detections.empty:
        print("[error] detections file is empty", file=sys.stderr)
        return 1

    # Frame size is only needed for the relative-size filters; infer it from the
    # detections themselves so this stage stays a pure parquet-to-parquet step.
    frame_width = args.frame_width or float(detections["x2"].max())
    frame_height = args.frame_height or float(detections["y2"].max())

    filter_args = {
        "min_confidence": args.min_confidence,
        "min_height_frac": args.min_height_frac,
        "max_height_frac": args.max_height_frac,
        "min_aspect": args.min_aspect,
        "max_aspect": args.max_aspect,
        "frame_height": frame_height,
        "frame_width": frame_width,
        "court_roi": tuple(args.court_roi) if args.court_roi else None,
        # An explicit --court-polygon overrides the profile's, matching how
        # every other flag behaves.
        "court_polygon": (
            parse_polygon(args.court_polygon)
            if args.court_polygon
            else (profile or {}).get("court_polygon")
        ),
    }
    # Backend choice: explicit flag > profile > the historical default.
    backend = args.tracker or (profile or {}).get("backend") or DEFAULT_BACKEND

    if backend == SUPERVISION_BYTETRACK:
        # Preserve the original flag-driven path exactly.
        tracker_args = resolve_tracker_args(args, profile)
    else:
        # The trackers backends have their own vocabulary, so the legacy
        # per-flag resolution does not apply; profile settings and
        # --tracker-arg are merged over the backend's own defaults.
        try:
            tracker_args = {
                **(profile or {}).get("tracker", {}),
                **parse_tracker_args(args.tracker_arg),
            }
        except ValueError as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1

    video_path = None
    if backend == "botsort" and tracker_args.get("enable_cmc", True):
        try:
            video_path = find_video(args.game_id)
        except (FileNotFoundError, ValueError) as exc:
            print(f"[error] botsort needs the source video: {exc}", file=sys.stderr)
            return 1

    print(f"[stage 2] tracker '{backend}' {tracker_args}", file=sys.stderr)

    total_rows = 0
    total_players = 0
    with ParquetBatchWriter(tracks_path(args.game_id), TRACK_OUTPUT_COLUMNS) as writer:
        for shot_id, shot_detections in detections.groupby("shot_id", sort=True):
            rows = track_shot(
                args.game_id,
                int(shot_id),
                shot_detections,
                tracker_args,
                filter_args,
                backend,
                video_path,
            )
            writer.extend(rows)
            total_rows += len(rows)
            players = pd.DataFrame(
                [r for r in rows if r["class"] == CLASS_PLAYER],
                columns=TRACK_OUTPUT_COLUMNS,
            )
            total_players += len(players)
            if players.empty:
                print(f"  shot {shot_id}: no player tracks", file=sys.stderr)
            else:
                print(format_shot_line(summarize_shot(players)), file=sys.stderr)

    print(
        f"[stage 2] {total_rows} row(s) -> {tracks_path(args.game_id).name}",
        file=sys.stderr,
    )

    # A table with no player rows is not a usable result, and writing one
    # silently is worse than failing: every later stage reads it, and the
    # viewer cannot render it. Ball rows alone still count as empty, since
    # they pass through untracked and would mask the failure.
    if total_players == 0:
        print(
            "[error] no player tracks written — every detection was filtered "
            "out. Check --min-confidence against the detector's confidences, "
            "and any --court-polygon / --court-roi against the frame.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
