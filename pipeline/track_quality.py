"""Tracking-quality metrics — the numbers that answer "is Milestone 1 done?".

`06-roadmap.md` closes Milestone 1 when "all 10 players keep a consistent ID
for the whole possession". That is a visual check, but it has a numeric
shadow worth computing on every run, because the two ways it fails look
nothing alike in a table:

* **Recall failure** — fewer than 10 players tracked per frame. The detector
  is missing people.
* **Fragmentation / clutter** — 10+ per frame but far more than 10 distinct
  tracks, because IDs break mid-shot or because crowd and referees are being
  tracked as players.

A single "number of tracks" figure hides which one you have, so this reports
both sides. Shared by `02_track.py` (printed each run) and the viewer's
Tracking tab, so the CLI and the UI can never disagree.
"""

from __future__ import annotations

import pandas as pd

from pipeline.common import CLASS_PLAYER

EXPECTED_PLAYERS = 10
"""Players on court in a half-court possession — the target for per-frame counts."""

SUMMARY_COLUMNS = [
    "shot_id",
    "n_frames",
    "n_tracks",
    "median_per_frame",
    "pct_frames_full",
    "longest_track",
    "longest_pct",
    "tracks_over_half",
    "tracks_over_90pct",
    "verdict",
]


def summarize_shot(shot_players: pd.DataFrame, expected: int = EXPECTED_PLAYERS) -> dict:
    """Compute quality metrics for one shot's player rows."""
    if shot_players.empty:
        return dict.fromkeys(SUMMARY_COLUMNS, 0) | {"verdict": "no tracks"}

    first = int(shot_players["frame_idx"].min())
    last = int(shot_players["frame_idx"].max())
    n_frames = last - first + 1

    per_frame = shot_players.groupby("frame_idx").size()
    per_track = shot_players.groupby("tracker_id").size()

    longest = int(per_track.max())
    # Measured against the shot's span, not against the number of frames that
    # happen to carry a detection: a track covering every frame it *could* is
    # the thing Milestone 1 asks for.
    longest_pct = round(100.0 * longest / n_frames, 1) if n_frames else 0.0

    return {
        "shot_id": int(shot_players["shot_id"].iloc[0]),
        "n_frames": n_frames,
        "n_tracks": int(per_track.size),
        "median_per_frame": float(per_frame.median()),
        "pct_frames_full": round(100.0 * (per_frame >= expected).sum() / n_frames, 1)
        if n_frames
        else 0.0,
        "longest_track": longest,
        "longest_pct": longest_pct,
        "tracks_over_half": int((per_track >= n_frames * 0.5).sum()),
        "tracks_over_90pct": int((per_track >= n_frames * 0.9).sum()),
        "verdict": _verdict(per_frame.median(), int(per_track.size),
                            int((per_track >= n_frames * 0.9).sum()), expected),
    }


def _verdict(median_per_frame: float, n_tracks: int, stable: int, expected: int) -> str:
    """Name the dominant failure so the table says what to fix next.

    Track count alone can't separate clutter from fragmentation — both show
    up as "too many tracks". Boxes *per frame* is what tells them apart: more
    boxes than players on court means non-players are being detected, while
    the right number of boxes with too many track ids means the same players
    keep getting new ids. Those need opposite fixes (better detector or crowd
    filter, versus better association), so they get separate verdicts.
    """
    if median_per_frame < expected * 0.8:
        return "low recall - detector is missing players"
    if stable >= expected and n_tracks <= expected * 1.5:
        return f"PASS - {expected} stable tracks"
    if median_per_frame > expected * 1.3:
        return "clutter - more boxes per frame than players on court"
    if n_tracks > expected * 2:
        return "fragmenting - right count per frame, ids not persisting"
    return "partial - recall fine, ids not holding"


def summarize(tracks: pd.DataFrame, expected: int = EXPECTED_PLAYERS) -> pd.DataFrame:
    """One quality row per shot, for every shot present in a tracks table."""
    players = tracks[tracks["class"] == CLASS_PLAYER]
    if players.empty:
        return pd.DataFrame(columns=SUMMARY_COLUMNS)

    rows = [
        summarize_shot(shot_players, expected)
        for _, shot_players in players.groupby("shot_id", sort=True)
    ]
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def format_shot_line(metrics: dict, expected: int = EXPECTED_PLAYERS) -> str:
    """One-line CLI summary for a shot."""
    return (
        f"  shot {metrics['shot_id']}: {metrics['n_tracks']} track(s), "
        f"median {metrics['median_per_frame']:.0f}/frame "
        f"({metrics['pct_frames_full']:.0f}% of frames have {expected}+), "
        f"longest {metrics['longest_track']}f ({metrics['longest_pct']:.0f}%), "
        f"{metrics['tracks_over_90pct']} track(s) span the shot "
        f"- {metrics['verdict']}"
    )
