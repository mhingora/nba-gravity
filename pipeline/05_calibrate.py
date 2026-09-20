"""Stage 5 — Court calibration (homography).

Input:  data/calibration/{profile}.json court keypoint references
Output: outputs/calibration/{game_id}_{shot_id}.json  (see docs/02-data-schemas.md)
        outputs/calibration/{game_id}_homographies.parquet  (with --propagate)

Spec: docs/03-pipeline-stages.md (Stage 5)
Independent of stage 3; can be developed and validated in parallel.

Keypoints are annotated by hand, per camera angle, in the viewer's Court
Calibration tab — the spec calls for exactly that, and it is the right call:
identifying a free-throw line intersection in a broadcast frame is a judgment
a person makes in seconds and a heuristic gets confidently wrong. Because
they live in a court profile keyed by camera angle, the annotation is done
once and reused across every shot and game from that camera, which is the
reuse the spec asks for.

That reuse is where the spec's model runs out, though. One matrix per angle
assumes the camera holds still, and this one pans within a single shot: the
annotated matrix is out by 1.49 ft at the median across the very shot it was
fitted on. `--propagate` matches every frame back to the annotated frame and
composes the camera motion with the annotated homography (see
`pipeline/camera_motion.py`), which brings that to 0.03 ft and calibrates
other shots from the same camera without annotating them.

Example:
    python pipeline/05_calibrate.py --game-id 0022500123 --court-profile msg_main
    python pipeline/05_calibrate.py --game-id 0022500123 --court-profile msg_main         --propagate
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import pandas as pd

from pipeline.camera_motion import (
    DEFAULT_SCALE,
    MIN_INLIERS,
    OVERLAY_PROBES,
    RANSAC_PX,
    detect_overlay,
    propagate,
)
from pipeline.common import (
    CALIBRATION_DIR,
    calibration_path,
    find_video,
    homographies_path,
    load_shots,
    propagation_path,
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

# docs/02-data-schemas.md
HOMOGRAPHY_COLUMNS = (
    ["game_id", "shot_id", "frame_idx"]
    + [f"h{i}{j}" for i in range(3) for j in range(3)]
    + ["inliers", "matches", "motion_rms_px"]
)


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
    parser.add_argument(
        "--propagate",
        action="store_true",
        help="Carry the annotated frame's homography to every frame it can "
        "be matched to, following the camera instead of assuming it holds "
        "still. Writes one matrix per frame; costs about a minute per "
        "thousand frames.",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=DEFAULT_SCALE,
        help="Resolution multiplier for feature matching during propagation. "
        "Half size is the default: same inlier counts, a third of the cost.",
    )
    parser.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="Limit propagation to frames at or after this one.",
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Limit propagation to frames at or before this one.",
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

    # Which shot the landmarks were actually annotated on. Every shot gets the
    # same matrix, so without this the output cannot distinguish the shot the
    # homography was fitted to from the 31 it was merely copied to — and the
    # reprojection error is identical in both cases, so it gives no clue.
    annotated_frame = int(stamp.group(2)) if stamp else None
    annotated_shot = None
    if annotated_frame is not None:
        for shot in shots:
            if shot.start_frame <= annotated_frame <= shot.end_frame:
                annotated_shot = shot.shot_id
                break
    annotated_on = (
        {
            "game_id": stamp.group(1),
            "frame_idx": annotated_frame,
            "shot_id": annotated_shot,
        }
        if stamp
        else None
    )

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
                    "annotated_on": annotated_on,
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

    if args.propagate:
        if annotated_frame is None:
            print(
                "[error] --propagate needs to know which frame the landmarks "
                "were annotated on, and this profile has no stamp. Re-save "
                "the landmarks in the viewer.",
                file=sys.stderr,
            )
            return 1
        return propagate_homographies(
            args, shots, np.array(matrix, dtype=np.float64), annotated_frame
        )
    return 0


def propagate_homographies(args, shots, court_matrix, reference_frame) -> int:
    """Carry the annotated frame's homography to every frame it can reach.

    One matrix per shot is the spec's model and this footage breaks it: the
    camera pans within a single shot, so the fitted matrix is only right near
    the frame it was fitted on. Matching each frame back to that frame fixes
    both problems at once — it follows the pan, and it reaches other shots
    from the same camera without anyone annotating them.
    """
    video = find_video(args.game_id)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        print(f"[error] could not open {video}", file=sys.stderr)
        return 1

    try:
        reference_shot = next(
            (s for s in shots
             if s.start_frame <= reference_frame <= s.end_frame),
            None,
        )
        probes = [
            (shot.start_frame + shot.end_frame) // 2
            for shot in shots
            if reference_shot is None or shot.shot_id != reference_shot.shot_id
        ]
        overlay, regions = (np.zeros((1, 1), np.uint8), [])
        if len(probes) >= 2:
            step = max(1, len(probes) // OVERLAY_PROBES)
            capture.set(cv2.CAP_PROP_POS_FRAMES, reference_frame)
            ok, reference = capture.read()
            if not ok:
                print(
                    f"[error] could not read frame {reference_frame}",
                    file=sys.stderr,
                )
                return 1
            overlay, regions = detect_overlay(
                capture, probes[::step][:OVERLAY_PROBES], reference, args.scale
            )
        if not regions:
            print(
                "[warn] no broadcast overlay found, so nothing is masked out. "
                "A clip with no camera cuts gives this no way to tell a "
                "scorebug from the court, and frames from other angles may "
                "match on graphics alone.",
                file=sys.stderr,
            )
        else:
            covered = sum(r["coverage"] for r in regions)
            print(
                f"[stage 5] overlay: {len(regions)} region(s), {covered:.1%} "
                "of the frame, masked out of matching",
                file=sys.stderr,
            )

        frames = [
            frame_idx
            for shot in shots
            for frame_idx in range(shot.start_frame, shot.end_frame + 1)
        ]
        if args.start_frame is not None:
            frames = [f for f in frames if f >= args.start_frame]
        if args.end_frame is not None:
            frames = [f for f in frames if f <= args.end_frame]
        if not frames:
            print(
                "[error] --start-frame/--end-frame leave no frames to "
                "propagate over.",
                file=sys.stderr,
            )
            return 1

        print(
            f"[stage 5] propagating from frame {reference_frame} over "
            f"{len(frames)} frame(s)",
            file=sys.stderr,
        )

        def tick(done, total, motion):
            if done % 500 == 0 or done == total:
                print(f"[stage 5]   {done}/{total} frames", file=sys.stderr)

        results = propagate(
            capture,
            reference_frame,
            frames,
            overlay if regions else None,
            court_matrix,
            args.scale,
            tick,
        )
    finally:
        capture.release()

    shot_of = {}
    for shot in shots:
        for frame_idx in range(shot.start_frame, shot.end_frame + 1):
            shot_of[frame_idx] = shot.shot_id

    rows = []
    for motion, matrix in results:
        if matrix is None:
            continue
        flat = np.asarray(matrix, dtype=np.float64).ravel()
        rows.append(
            {
                "game_id": args.game_id,
                "shot_id": shot_of.get(motion.frame_idx, -1),
                "frame_idx": motion.frame_idx,
                **{f"h{i}{j}": float(flat[i * 3 + j])
                   for i in range(3) for j in range(3)},
                "inliers": motion.inliers,
                "matches": motion.matches,
                "motion_rms_px": motion.rms_px,
            }
        )

    table = pd.DataFrame(rows, columns=HOMOGRAPHY_COLUMNS)
    homographies_path(args.game_id).parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(homographies_path(args.game_id), index=False)

    reached = sorted(int(s) for s in table["shot_id"].unique()) if len(table) else []
    propagation_path(args.game_id).write_text(
        json.dumps(
            {
                "game_id": args.game_id,
                "reference_frame": reference_frame,
                "court_profile": args.court_profile,
                "settings": {
                    "scale": args.scale,
                    "min_inliers": MIN_INLIERS,
                    "ransac_px": RANSAC_PX,
                },
                "overlay_regions": regions,
                "frames_attempted": len(results),
                "frames_solved": len(table),
                "shots_reached": [int(s) for s in reached],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    print(
        f"[stage 5] {len(table)}/{len(results)} frame(s) solved "
        f"({len(table) / len(results):.0%}), reaching {len(reached)} shot(s): "
        f"{reached}",
        file=sys.stderr,
    )
    if len(table):
        print(
            f"[stage 5] inliers median {table['inliers'].median():.0f}, "
            f"motion rms median {table['motion_rms_px'].median():.2f}px "
            f"-> {homographies_path(args.game_id).name}",
            file=sys.stderr,
        )
    else:
        print(
            "[stage 5] nothing solved. The reference frame reaches no other "
            "frame, which usually means the overlay mask swallowed the image "
            "or the annotated frame is not from this video.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
