"""Stage 3, Part B — read jersey numbers, and only trust what repeats.

`04-identity-resolution.md` calls this the most failure-prone part of the
project, and the measurements agree: a player's torso is about 76px tall at
1080p, which is very little to read text from. A single frame's OCR is close
to worthless. Many frames voting is not.

Everything below is set by what a labelled possession from the test clip
actually produced. Seventeen tracks were sampled over their 24 largest frames,
and each track's crops were then read by eye to establish truth: five carried a
legible number (2 Harper, 32, 30 Champagnie, 00, 5) and the other twelve
carried none — obscured by arms, split across two players, or a person on the
sideline. Two more (11, Brunson) were legible to a human but never to OCR.

Three things that pass an eyeball test turned out to be wrong, and each one is
now a rule here:

* **A confident read is not a correct read.** EasyOCR is given a digit
  allowlist, so its decoder can only emit digits — a jersey wordmark or a piece
  of piping therefore comes back as a digit at confidence 1.0 rather than as
  nothing. Confidence alone accepted six wrong numbers. What separated signal
  from noise was *volume* of agreement: real numbers were read 4-17 times,
  noise 2-3 times.
* **Folding fragments into longer reads makes it worse.** OCR does often catch
  half a two-digit number, so an earlier version attributed a "2" to a "24"
  read on the same track. Track 5 reads "2" ten times and "24" six times — and
  the player is Harper, who wears **2**. That rule turned the one track with
  the clearest evidence into a wrong answer, and invented a number for a track
  with none. It is gone; a plain majority is both simpler and more accurate.
* **"00" is a number.** Stripping leading zeros to turn "07" into "7" also
  turned seventeen confident "00" reads into "0", which is a different player.
  Zeros are only stripped when a non-zero digit follows.

On that possession the rules here answer 5 tracks, all 5 correct, and stay
silent on the other 12 — including the two a human can read. That is the trade
the spec asks for: a wrong number attaches one player's name to another
player's movement, and every gravity value derived from it is quietly false.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
import pandas as pd

from pipeline.common import CLASS_PLAYER, ROSTERS_DIR

# Torso window as a fraction of the player box. Below the head, above the
# shorts: where a back or front number sits.
TORSO_TOP = 0.18
TORSO_BOTTOM = 0.52
TORSO_INSET = 0.18

UPSCALE = 4
MIN_SHARPNESS = 40.0
# Deliberately high. Below this, reads are mostly the decoder rendering
# non-digit texture as digits; above it, the wrong reads that survive are
# outvoted rather than argued with.
MIN_OCR_CONFIDENCE = 0.80
DEFAULT_SAMPLES_PER_TRACK = 24
# Four, not the spec's example of three: three agreeing reads accepted a "1"
# for a Spurs player whose number was never visible. No true number on the
# test possession was read fewer than four times.
DEFAULT_MIN_AGREEMENT = 4
# The winner must also be a strict majority of the reads kept, so a track that
# cannot make up its mind stays unresolved instead of answering by a hair.
MIN_WINNER_SHARE = 0.5


def torso_crop(frame: np.ndarray, box) -> np.ndarray | None:
    """The number-bearing region of a player box."""
    x1, y1, x2, y2 = box
    height, width = y2 - y1, x2 - x1
    top = int(y1 + TORSO_TOP * height)
    bottom = int(y1 + TORSO_BOTTOM * height)
    left = int(x1 + TORSO_INSET * width)
    right = int(x2 - TORSO_INSET * width)

    frame_h, frame_w = frame.shape[:2]
    top, left = max(0, top), max(0, left)
    bottom, right = min(frame_h, bottom), min(frame_w, right)
    if bottom - top < 12 or right - left < 8:
        return None
    return frame[top:bottom, left:right]


def sharpness(image: np.ndarray) -> float:
    """Laplacian variance — low means motion blur, which OCR cannot survive."""
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(grey, cv2.CV_64F).var())


def prepare(crop: np.ndarray) -> np.ndarray:
    """Enlarge before reading. Measured to triple the hit rate."""
    return cv2.resize(
        crop,
        (crop.shape[1] * UPSCALE, crop.shape[0] * UPSCALE),
        interpolation=cv2.INTER_CUBIC,
    )


def normalise_read(text: str) -> str | None:
    """A raw OCR string as a jersey number, or None if it cannot be one.

    NBA numbers run 0-99, so anything longer is a misread — "330" and "130"
    both showed up for a player wearing 30. A leading zero is how OCR renders
    a single digit it has clipped ("07" for a 7), *except* for "00", which is
    a number in its own right and belongs to a different player than "0".
    """
    text = text.strip()
    if not text.isdigit() or not 1 <= len(text) <= 2:
        return None
    if len(text) == 2 and text[0] == "0" and text[1] != "0":
        return text[1]
    return text


def read_digits(reader, image: np.ndarray) -> list[str]:
    """Jersey numbers OCR is confident about, in this one crop."""
    try:
        found = reader.readtext(image, allowlist="0123456789", detail=1)
    except Exception:
        return []
    out = []
    for _, text, confidence in found:
        if confidence < MIN_OCR_CONFIDENCE:
            continue
        number = normalise_read(text)
        if number is not None:
            out.append(number)
    return out


def vote(
    counts: Counter,
    min_agreement: int = DEFAULT_MIN_AGREEMENT,
    min_share: float = MIN_WINNER_SHARE,
):
    """Return (number, agreeing, total), or (None, agreeing, total) if unsure.

    A plain majority, with two ways to refuse: too few agreeing reads, or a
    winner that does not clear a majority of everything read. The share test
    is strict, so an exact tie refuses rather than picking whichever key the
    counter happens to yield first. Both refusals leave `jersey_number` null,
    which the spec calls the correct outcome.
    """
    total = sum(counts.values())
    if not total:
        return None, 0, 0
    number, agreeing = counts.most_common(1)[0]
    if agreeing < min_agreement or agreeing / total <= min_share:
        return None, agreeing, total
    return number, agreeing, total


@dataclass
class TrackReads:
    """What OCR saw for one track: the tally, and the evidence behind it.

    `samples` keeps every frame that was tried, including the ones that
    yielded nothing, because a track that failed to resolve is the case a
    person most needs to inspect — and "18 blurred, 2 read nothing" is a
    different diagnosis from "6 frames, 6 different answers".
    """

    counts: Counter = field(default_factory=Counter)
    samples: list[dict] = field(default_factory=list)


def read_track_numbers(
    video_path,
    tracks: pd.DataFrame,
    reader,
    samples_per_track: int = DEFAULT_SAMPLES_PER_TRACK,
    min_track_len: int = 30,
) -> dict[tuple[int, int], TrackReads]:
    """OCR each track's biggest, sharpest frames. Keyed by (shot, tracker)."""
    players = tracks[tracks["class"] == CLASS_PLAYER]
    players = players[players["track_len"] >= min_track_len]
    if players.empty:
        return {}

    # Choose frames first, then walk the video once in order: seeking is the
    # expensive part, and a track's samples are scattered through the shot.
    wanted: dict[int, list] = {}
    for (shot_id, tracker_id), group in players.groupby(["shot_id", "tracker_id"]):
        group = group.assign(height=group["y2"] - group["y1"])
        picks = group.sort_values("height", ascending=False).head(samples_per_track)
        for row in picks.itertuples():
            wanted.setdefault(int(row.frame_idx), []).append(
                (int(shot_id), int(tracker_id), (row.x1, row.y1, row.x2, row.y2))
            )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    results: dict[tuple[int, int], TrackReads] = {}
    try:
        for frame_idx in sorted(wanted):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue
            for shot_id, tracker_id, box in wanted[frame_idx]:
                track = results.setdefault((shot_id, tracker_id), TrackReads())
                crop = torso_crop(frame, box)
                if crop is None:
                    track.samples.append({"frame_idx": frame_idx, "status": "no_crop"})
                    continue
                blur = sharpness(crop)
                if blur < MIN_SHARPNESS:
                    track.samples.append(
                        {"frame_idx": frame_idx, "status": "blurred",
                         "sharpness": round(blur, 1)}
                    )
                    continue
                digits = read_digits(reader, prepare(crop))
                for number in digits:
                    track.counts[number] += 1
                track.samples.append(
                    {
                        "frame_idx": frame_idx,
                        "status": "read" if digits else "nothing_legible",
                        "sharpness": round(blur, 1),
                        "reads": digits,
                    }
                )
    finally:
        capture.release()

    return results


def load_roster(team_id: str) -> dict[str, str]:
    """Read `data/rosters/{team_id}.json` -> {number: player name}."""
    path = ROSTERS_DIR / f"{team_id}.json"
    if not path.exists():
        available = sorted(p.stem for p in ROSTERS_DIR.glob("*.json"))
        raise FileNotFoundError(
            f"No roster at {path}. Available: {', '.join(available) or 'none'}. "
            "Copy data/rosters/_TEMPLATE.json and fill in the squad."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    players = data.get("players")
    if not isinstance(players, dict):
        raise ValueError(f"{path} has no 'players' object")
    # Keys are matched against what OCR produces, so they go through the same
    # normalisation: a roster may write 7, "7" or "07" and still be found, and
    # "00" stays distinct from "0".
    roster = {}
    for number, name in players.items():
        key = normalise_read(str(number))
        if key is None:
            raise ValueError(
                f"{path}: '{number}' is not a jersey number (expected 0-99)"
            )
        roster[key] = name
    return roster


def build_team_map(pairs: list[str] | None) -> dict[str, str]:
    """Turn `light=NYK dark=SAS` into a cluster -> roster-team mapping.

    Stage 3A names its clusters by kit brightness because that is all colour
    can tell you. Which of them is the Knicks is a fact about the game, and a
    person supplies it — the same call as annotating court landmarks.
    """
    mapping: dict[str, str] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"--team expects cluster=TEAM, got '{pair}'")
        cluster, team = pair.split("=", 1)
        mapping[cluster.strip()] = team.strip()
    return mapping
