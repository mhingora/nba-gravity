"""Generate a synthetic clip for smoke-testing the pipeline without real footage.

The clip has a known ground truth — a fixed number of camera cuts, ten
player-shaped blobs moving smoothly, and one small ball blob — so stage output
can be checked against it. Pair with `--detector colorblob` in `01_detect.py`.

This is scaffolding for exercising the plumbing, not a stand-in for broadcast
video. Detector and tracker quality can only be judged on real footage.

    python tools/make_test_clip.py --game-id TESTCLIP
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np

from pipeline.common import RAW_VIDEO_DIR

N_PLAYERS = 10
PLAYER_W, PLAYER_H = 18, 42
BALL_R = 5

# Distinct court tints per shot so the histogram cut detector has a real signal.
# Kept low-saturation like a real court surface, so the colorblob detector's
# saturation threshold separates players from the floor rather than seeing one
# giant blob.
SHOT_BACKGROUNDS = [
    (118, 134, 156),
    (150, 128, 110),
    (110, 140, 120),
]

TEAM_COLORS = [(60, 60, 220), (220, 180, 60)]


def draw_frame(width: int, height: int, background, t: float) -> np.ndarray:
    frame = np.full((height, width, 3), background, dtype=np.uint8)

    # Faint court lines — texture that stays constant within a shot, so it
    # doesn't create spurious cut signal.
    cv2.rectangle(frame, (30, 30), (width - 30, height - 30), (200, 200, 200), 1)
    cv2.line(frame, (width // 2, 30), (width // 2, height - 30), (200, 200, 200), 1)

    # Loose grid with small oscillation: players stay separated so each one
    # yields its own blob, keeping the ground-truth count unambiguous.
    for i in range(N_PLAYERS):
        col, row = i % 5, i // 5
        base_x = width * (0.14 + 0.18 * col)
        base_y = height * (0.32 + 0.30 * row)
        cx = int(base_x + 0.045 * width * math.sin(t * 0.9 + i))
        cy = int(base_y + 0.055 * height * math.cos(t * 0.7 + i * 1.3))
        color = TEAM_COLORS[i % 2]
        cv2.rectangle(
            frame,
            (cx - PLAYER_W // 2, cy - PLAYER_H // 2),
            (cx + PLAYER_W // 2, cy + PLAYER_H // 2),
            color,
            -1,
        )

    ball_x = int(width * (0.5 + 0.35 * math.sin(t * 1.7)))
    ball_y = int(height * (0.45 + 0.25 * math.cos(t * 2.1)))
    cv2.circle(frame, (ball_x, ball_y), BALL_R, (40, 140, 245), -1)

    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a synthetic test clip")
    parser.add_argument("--game-id", default="TESTCLIP")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--shot-frames",
        type=int,
        default=60,
        help="Frames per camera shot (3 shots are generated).",
    )
    args = parser.parse_args()

    out_dir = RAW_VIDEO_DIR / args.game_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.game_id}.mp4"

    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (args.width, args.height),
    )
    if not writer.isOpened():
        print(f"[error] could not open writer for {out_path}", file=sys.stderr)
        return 1

    try:
        for shot_index, background in enumerate(SHOT_BACKGROUNDS):
            for i in range(args.shot_frames):
                # Motion restarts each shot, as it would across a real cut.
                t = i / args.fps * 2.0
                writer.write(draw_frame(args.width, args.height, background, t))
    finally:
        writer.release()

    total = len(SHOT_BACKGROUNDS) * args.shot_frames
    cuts = [args.shot_frames * i for i in range(1, len(SHOT_BACKGROUNDS))]
    print(f"wrote {out_path}")
    print(f"  {total} frames, {len(SHOT_BACKGROUNDS)} shots, cuts at frames {cuts}")
    print(f"  ground truth: {N_PLAYERS} players + 1 ball per frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
