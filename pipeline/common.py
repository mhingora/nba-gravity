"""Paths, schemas, and video helpers shared by pipeline stages and the UI.

Every stage reads and writes only on-disk artifacts under `outputs/`, keyed by
`game_id` (see docs/01-architecture.md). This module is the single place that
knows where those artifacts live, so the UI can find them without importing
any stage logic.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent

RAW_VIDEO_DIR = REPO_ROOT / "data" / "raw_video"
ROSTERS_DIR = REPO_ROOT / "data" / "rosters"
CALIBRATION_REF_DIR = REPO_ROOT / "data" / "calibration"

OUTPUTS_DIR = REPO_ROOT / "outputs"
DETECTIONS_DIR = OUTPUTS_DIR / "detections"
TRACKS_DIR = OUTPUTS_DIR / "tracks"
IDENTITY_DIR = OUTPUTS_DIR / "identity"
POSSESSION_DIR = OUTPUTS_DIR / "possession"
CALIBRATION_DIR = OUTPUTS_DIR / "calibration"
METRICS_DIR = OUTPUTS_DIR / "metrics"

CLASS_PLAYER = "player"
CLASS_BALL = "ball"

# docs/02-data-schemas.md
DETECTION_COLUMNS = [
    "game_id",
    "frame_idx",
    "shot_id",
    "class",
    "x1",
    "y1",
    "x2",
    "y2",
    "confidence",
]

TRACK_COLUMNS = [
    "game_id",
    "frame_idx",
    "shot_id",
    "tracker_id",
    "x1",
    "y1",
    "x2",
    "y2",
    "foot_x",
    "foot_y",
    "track_len",
]


# --------------------------------------------------------------------------
# Artifact paths
# --------------------------------------------------------------------------


def detections_path(game_id: str) -> Path:
    return DETECTIONS_DIR / f"{game_id}.parquet"


def shots_path(game_id: str) -> Path:
    """Shot boundaries, written alongside detections by Stage 0."""
    return DETECTIONS_DIR / f"{game_id}_shots.json"


def shot_diffs_path(game_id: str) -> Path:
    """Per-frame cut-strength signal, for tuning the shot-boundary threshold."""
    return DETECTIONS_DIR / f"{game_id}_shot_diffs.parquet"


def tracks_path(game_id: str) -> Path:
    return TRACKS_DIR / f"{game_id}.parquet"


def identity_path(game_id: str) -> Path:
    return IDENTITY_DIR / f"{game_id}.parquet"


def ocr_reads_path(game_id: str) -> Path:
    """Per-crop OCR evidence, written alongside identity when `--ocr` runs.

    The parquet records what each track resolved to; this records why. The
    viewer needs the individual reads to show a failed track's crops next to
    what OCR made of them, and it may not re-run OCR itself.
    """
    return IDENTITY_DIR / f"{game_id}_ocr.json"


def possession_path(game_id: str) -> Path:
    return POSSESSION_DIR / f"{game_id}.parquet"


def calibration_path(game_id: str, shot_id: int) -> Path:
    return CALIBRATION_DIR / f"{game_id}_{shot_id}.json"


def metrics_path(game_id: str) -> Path:
    return METRICS_DIR / f"{game_id}_gravity.parquet"


# --------------------------------------------------------------------------
# Source video discovery
# --------------------------------------------------------------------------

VIDEO_SUFFIXES = (".mp4", ".mkv", ".mov", ".m4v")


def find_video(game_id: str) -> Path:
    """Locate the source video for a game.

    Supports both layouts: `data/raw_video/{game_id}/*.mp4` (the documented
    one-folder-per-game convention) and a flat `data/raw_video/{game_id}.mp4`.
    """
    game_dir = RAW_VIDEO_DIR / game_id
    if game_dir.is_dir():
        videos = sorted(
            p for p in game_dir.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES
        )
        if len(videos) > 1:
            raise ValueError(
                f"{game_dir} contains {len(videos)} videos; expected exactly one. "
                "Split multi-file games into separate game_ids."
            )
        if videos:
            return videos[0]

    for suffix in VIDEO_SUFFIXES:
        flat = RAW_VIDEO_DIR / f"{game_id}{suffix}"
        if flat.exists():
            return flat

    raise FileNotFoundError(
        f"No video found for game_id '{game_id}'. Expected "
        f"{game_dir}/<video>.mp4 or {RAW_VIDEO_DIR / (game_id + '.mp4')}"
    )


def list_games() -> list[str]:
    """Every game_id with a source video on disk, for UI selectors."""
    if not RAW_VIDEO_DIR.exists():
        return []
    games = set()
    for entry in RAW_VIDEO_DIR.iterdir():
        if entry.name.startswith("."):
            continue
        if entry.is_dir() and any(
            p.suffix.lower() in VIDEO_SUFFIXES for p in entry.iterdir()
        ):
            games.add(entry.name)
        elif entry.is_file() and entry.suffix.lower() in VIDEO_SUFFIXES:
            games.add(entry.stem)
    return sorted(games)


@dataclass(frozen=True)
class VideoInfo:
    path: Path
    frame_count: int
    fps: float
    width: int
    height: int


def video_info(video_path: Path) -> VideoInfo:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        return VideoInfo(
            path=video_path,
            frame_count=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)) or 30.0,
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()


def read_frame(video_path: Path, frame_idx: int):
    """Read a single frame by absolute index. Returns BGR ndarray or None.

    Used by the UI's frame scrubber. Seeking per frame is slower than a
    sequential read, which is why stages iterate instead of calling this.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"Could not open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


# --------------------------------------------------------------------------
# Shot boundaries
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Shot:
    """One continuous camera shot. `end_frame` is inclusive."""

    shot_id: int
    start_frame: int
    end_frame: int
    camera_angle: str = "unlabeled"

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    def contains(self, frame_idx: int) -> bool:
        return self.start_frame <= frame_idx <= self.end_frame


def save_shots(game_id: str, shots: list[Shot]) -> Path:
    path = shots_path(game_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"game_id": game_id, "shots": [asdict(s) for s in shots]}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load_shots(game_id: str) -> list[Shot]:
    path = shots_path(game_id)
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [Shot(**s) for s in payload["shots"]]


# --------------------------------------------------------------------------
# Incremental parquet writing
# --------------------------------------------------------------------------


class ParquetBatchWriter:
    """Append rows to a parquet file in batches.

    Stage 1 runs over a whole game, which is far too many detections to hold
    in memory (docs/03-pipeline-stages.md). Rows accumulate in a list and
    flush to a growing parquet file every `batch_size` rows.
    """

    def __init__(self, path: Path, columns: list[str], batch_size: int = 5000):
        import pyarrow as pa  # noqa: PLC0415 — optional until a stage runs

        self.path = path
        self.columns = columns
        self.batch_size = batch_size
        self._rows: list[dict] = []
        self._writer = None
        self._schema: pa.Schema | None = None
        self.rows_written = 0
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: dict) -> None:
        self._rows.append(row)
        if len(self._rows) >= self.batch_size:
            self.flush()

    def extend(self, rows: list[dict]) -> None:
        for row in rows:
            self.append(row)

    def flush(self) -> None:
        if not self._rows:
            return
        import pyarrow as pa
        import pyarrow.parquet as pq

        frame = pd.DataFrame(self._rows, columns=self.columns)
        table = pa.Table.from_pandas(frame, preserve_index=False)
        if self._writer is None:
            self._schema = table.schema
            self._writer = pq.ParquetWriter(self.path, self._schema)
        else:
            table = table.cast(self._schema)
        self._writer.write_table(table)
        self.rows_written += len(self._rows)
        self._rows.clear()

    def close(self) -> None:
        self.flush()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        elif self.rows_written == 0:
            # No rows at all — still write an empty file with the right schema
            # so downstream stages and the UI can read it without special-casing.
            pd.DataFrame(columns=self.columns).to_parquet(self.path, index=False)

    def __enter__(self) -> "ParquetBatchWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
