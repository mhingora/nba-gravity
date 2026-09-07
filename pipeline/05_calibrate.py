"""Stage 5 — Court calibration (homography).

Input:  data/calibration/{profile}.json court keypoint references
Output: outputs/calibration/{game_id}_{shot_id}.json  (see docs/02-data-schemas.md)

Spec: docs/03-pipeline-stages.md (Stage 5)
Independent of stage 3; can be developed and validated in parallel.

Keypoints are annotated by hand, per camera angle, in the viewer's Court
Calibration tab — the spec calls for exactly that, and it is the right call:
identifying a free-throw line intersection in a broadcast frame is a judgment
a person makes in seconds and a heuristic gets confidently wrong. Because
they live in a court profile keyed by camera angle, the annotation is done
once and reused across every shot and game from that camera, which is the
reuse the spec asks for.

Example:
    python pipeline/05_calibrate.py --game-id 0022500123 --court-profile msg_main
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import (
    CALIBRATION_DIR,
    calibration_path,
    find_video,
    load_shots,
    video_info,
)
from pipeline.court_geometry import (
    compute_homography,
    known_distance_checks,
    parse_keypoints,
)
from pipeline.court_region import load_profile

# The viewer stamps the profile description when landmarks are saved.
ANNOTATION_STAMP = re.compile(r"\[landmarks annotated on (\S+) frame (\d+)\]")


def calibrate_angle(
    keypoints: dict, frame_width: float, frame_height: float
) -> tuple[list[list[float]], float, list[dict], list[dict]]:
    """Fit one camera angle. Returns matrix, rms error, per-point and distance checks."""
    image_points, court_points, names = parse_keypoints(keypoints)
    matrix, per_point, rms = compute_homography(
        image_points, court_points, frame_width, frame_height
    )
    residuals = [
        {"landmark": name, "error_px": round(float(error), 2)}
        for name, error in zip(names, per_point)
    ]
    distances = known_distance_checks(
        matrix, image_points, names, frame_width, frame_height
    )
    return matrix.tolist(), rms, residuals, distances


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 5 — court calibration")
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--court-profile",
        required=True,
        metavar="NAME",
        help="Court profile holding this camera angle's annotated keypoints "
        "(data/calibration/NAME.json). Annotate them in the viewer.",
    )
    parser.add_argument(
        "--camera-angle",
        default="unlabeled",
        help="Angle tag recorded in the output. Shots are all 'unlabeled' "
        "until something classifies camera angles.",
    )
    parser.add_argument(
        "--allow-foreign-landmarks",
        action="store_true",
        help="Calibrate even when the landmarks were annotated on a different "
        "game. Only sensible when the two really do share a camera framing.",
    )
    parser.add_argument(
        "--max-error-px",
        type=float,
        default=None,
        help="Refuse to write a calibration whose rms reprojection error "
        "exceeds this. A bad homography is worse than none — it produces "
        "confident, wrong distances.",
    )
    args = parser.parse_args()

    try:
        profile = load_profile(args.court_profile)
    except (FileNotFoundError, ValueError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    keypoints = profile.get("court_keypoints") or {}
    if not keypoints:
        print(
            f"[error] profile '{args.court_profile}' has no court_keypoints. "
            "Annotate them in the viewer's Court Calibration tab.",
            file=sys.stderr,
        )
        return 1

    # A homography describes one camera framing. Tracker settings and the
    # court polygon travel between games happily; landmarks do not, because
    # pan and zoom move the court in the frame. Annotating for one clip and
    # calibrating another produces a confident, wrong homography — which is
    # exactly what happened once, and cost a session to diagnose.
    stamp = ANNOTATION_STAMP.search(profile.get("description", "") or "")
    if stamp and stamp.group(1) != args.game_id:
        message = (
            f"[{'warn' if args.allow_foreign_landmarks else 'error'}] profile "
            f"'{args.court_profile}' was annotated on {stamp.group(1)} frame "
            f"{stamp.group(2)}, not {args.game_id}. A homography only "
            "describes the framing it was annotated on."
        )
        print(message, file=sys.stderr)
        if not args.allow_foreign_landmarks:
            print(
                "        Re-annotate on this game in the viewer, or pass "
                "--allow-foreign-landmarks if the framing really is shared.",
                file=sys.stderr,
            )
            return 1
    elif not stamp:
        print(
            "[warn] this profile predates annotation stamping, so the frame "
            "its landmarks came from is unknown. Re-save them in the viewer "
            "to record it.",
            file=sys.stderr,
        )

    try:
        info = video_info(find_video(args.game_id))
    except (FileNotFoundError, ValueError, IOError) as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    try:
        matrix, rms, residuals, distances = calibrate_angle(
            keypoints, info.width, info.height
        )
    except ValueError as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1

    if args.max_error_px is not None and rms > args.max_error_px:
        print(
            f"[error] rms reprojection error {rms:.1f}px exceeds "
            f"--max-error-px {args.max_error_px:.1f}. Refusing to write; "
            "re-annotate the keypoints.",
            file=sys.stderr,
        )
        return 1

    shots = load_shots(args.game_id)
    if not shots:
        print("[error] no shots — run 01_detect.py first", file=sys.stderr)
        return 1

    CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
    # One file per shot, as the schema specifies, even though every shot from
    # the same camera shares a homography. Stage 6 then reads one predictable
    # path per shot rather than resolving angle tags itself, and a single shot
    # can later be re-fitted (a mid-possession zoom) without special-casing.
    for shot in shots:
        calibration_path(args.game_id, shot.shot_id).write_text(
            json.dumps(
                {
                    "game_id": args.game_id,
                    "shot_id": shot.shot_id,
                    "camera_angle": args.camera_angle,
                    "court_profile": args.court_profile,
                    "homography_matrix": matrix,
                    "reprojection_error_px": round(rms, 3),
                    "landmark_errors_px": residuals,
                    "known_distance_checks": distances,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    worst = max(distances, key=lambda d: abs(d["error_ft"])) if distances else None
    print(
        f"[stage 5] {len(keypoints)} landmark(s), rms reprojection error "
        f"{rms:.1f}px"
        + (
            f", worst known distance off by {worst['error_ft']:+.2f}ft "
            f"({worst['from']} to {worst['to']})"
            if worst
            else ""
        ),
        file=sys.stderr,
    )
    print(
        f"[stage 5] wrote {len(shots)} calibration file(s) -> "
        f"{CALIBRATION_DIR.name}/",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
