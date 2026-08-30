"""Stage 0 + Stage 1 — shot segmentation and per-frame detection.

Input:  data/raw_video/{game_id}/*.mp4
Output: outputs/detections/{game_id}.parquet
        outputs/detections/{game_id}_shots.json
        outputs/detections/{game_id}_shot_diffs.parquet

Spec: docs/03-pipeline-stages.md (Stages 0-1)

Examples
--------
Segment shots only (fast, no model — validates Milestone 2 on its own):
    python pipeline/01_detect.py --game-id 0022500123 --shots-only

Detect over one possession with stock COCO weights (Milestone 1):
    python pipeline/01_detect.py --game-id 0022500123 --start-frame 1200 --end-frame 1650
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.common import (
    DETECTION_COLUMNS,
    ParquetBatchWriter,
    Shot,
    detections_path,
    find_video,
    load_shots,
    save_shots,
    shot_diffs_path,
    shots_path,
    video_info,
)
from pipeline.detector import build_detector
from pipeline.shot_boundaries import (
    DEFAULT_CUT_THRESHOLD,
    DEFAULT_MIN_SHOT_FRAMES,
    iter_frames,
    segment_shots,
    shot_id_for_frame,
)


def _progress(label: str, every: int = 250):
    started = time.monotonic()
    state = {"last": -1}

    def report(frame_idx: int) -> None:
        if frame_idx - state["last"] < every:
            return
        state["last"] = frame_idx
        elapsed = time.monotonic() - started
        print(f"  {label}: frame {frame_idx} ({elapsed:.1f}s)", file=sys.stderr)

    return report


def run_segmentation(
    game_id: str,
    video_path: Path,
    start_frame: int,
    end_frame: int | None,
    threshold: float,
    min_shot_frames: int,
) -> list[Shot]:
    print(f"[stage 0] segmenting shots in {video_path.name}", file=sys.stderr)
    shots, diffs = segment_shots(
        video_path,
        start_frame=start_frame,
        end_frame=end_frame,
        threshold=threshold,
        min_shot_frames=min_shot_frames,
        progress=_progress("shot scan"),
    )
    save_shots(game_id, shots)

    diff_path = shot_diffs_path(game_id)
    diff_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame_idx": range(start_frame, start_frame + len(diffs)),
            "cut_distance": diffs,
        }
    ).to_parquet(diff_path, index=False)

    print(
        f"[stage 0] {len(shots)} shot(s) -> {shots_path(game_id).name}",
        file=sys.stderr,
    )
    return shots


def run_detection(
    game_id: str,
    video_path: Path,
    shots: list[Shot],
    detector,
    start_frame: int,
    end_frame: int | None,
    skip_angles: set[str],
    batch_size: int,
) -> int:
    """Detect players and the ball frame by frame, writing incrementally."""
    skipped_shot_ids = {s.shot_id for s in shots if s.camera_angle in skip_angles}
    if skipped_shot_ids:
        print(
            f"[stage 1] skipping {len(skipped_shot_ids)} shot(s) tagged "
            f"{sorted(skip_angles)}",
            file=sys.stderr,
        )

    report = _progress("detect")
    written = 0

    with ParquetBatchWriter(
        detections_path(game_id), DETECTION_COLUMNS, batch_size=batch_size
    ) as writer:
        for frame_idx, frame in iter_frames(video_path, start_frame, end_frame):
            shot_id = shot_id_for_frame(shots, frame_idx)
            # Burning detector compute on replays and tight isolation shots is
            # wasted — they get filtered out of the metric anyway.
            if shot_id in skipped_shot_ids:
                continue

            detections = detector.detect(frame)
            class_names = detections.data.get("class_name", [])
            for i in range(len(detections)):
                x1, y1, x2, y2 = detections.xyxy[i]
                writer.append(
                    {
                        "game_id": game_id,
                        "frame_idx": frame_idx,
                        "shot_id": shot_id,
                        "class": str(class_names[i]),
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "confidence": float(detections.confidence[i]),
                    }
                )
                written += 1
            report(frame_idx)

    print(
        f"[stage 1] {written} detection(s) -> {detections_path(game_id).name}",
        file=sys.stderr,
    )
    return written


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--start-frame", type=int, default=0, help="First frame to process (inclusive)"
    )
    parser.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Last frame to process (inclusive). Default: end of video.",
    )
    parser.add_argument(
        "--shots-only",
        action="store_true",
        help="Run shot segmentation and stop, without loading a detector.",
    )
    parser.add_argument(
        "--reuse-shots",
        action="store_true",
        help="Use the existing shots file instead of re-segmenting.",
    )
    parser.add_argument("--cut-threshold", type=float, default=DEFAULT_CUT_THRESHOLD)
    parser.add_argument("--min-shot-frames", type=int, default=DEFAULT_MIN_SHOT_FRAMES)
    parser.add_argument(
        "--detector",
        default="yolo",
        choices=["yolo", "colorblob"],
        help="'colorblob' is a smoke-test detector for the synthetic clip only.",
    )
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default=None, help="e.g. 'cuda:0' or 'cpu'")
    parser.add_argument(
        "--ball-model",
        default=None,
        help="Optional separate higher-resolution ball detector weights.",
    )
    parser.add_argument(
        "--skip-angles",
        nargs="*",
        default=["replay", "isolation_closeup"],
        help="Camera-angle tags to skip during detection.",
    )
    parser.add_argument("--batch-size", type=int, default=5000)
    args = parser.parse_args()

    video_path = find_video(args.game_id)
    info = video_info(video_path)
    print(
        f"[input] {video_path}  {info.width}x{info.height} @ {info.fps:.2f}fps, "
        f"{info.frame_count} frames",
        file=sys.stderr,
    )

    if args.reuse_shots and shots_path(args.game_id).exists():
        shots = load_shots(args.game_id)
        print(f"[stage 0] reusing {len(shots)} cached shot(s)", file=sys.stderr)
    else:
        shots = run_segmentation(
            args.game_id,
            video_path,
            args.start_frame,
            args.end_frame,
            args.cut_threshold,
            args.min_shot_frames,
        )

    if not shots:
        print("[error] no frames decoded — check the frame range", file=sys.stderr)
        return 1

    if args.shots_only:
        return 0

    detector_kwargs = {}
    if args.detector == "yolo":
        detector_kwargs = {
            "weights": args.model,
            "confidence": args.conf,
            "imgsz": args.imgsz,
            "device": args.device,
            "ball_weights": args.ball_model,
        }
    detector = build_detector(args.detector, **detector_kwargs)

    run_detection(
        args.game_id,
        video_path,
        shots,
        detector,
        args.start_frame,
        args.end_frame,
        set(args.skip_angles),
        args.batch_size,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
