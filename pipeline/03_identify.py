"""Stage 3 — Identity resolution (team clustering + jersey OCR).

Input:  outputs/tracks/{game_id}.parquet, data/rosters/
Output: outputs/identity/{game_id}.parquet  (see docs/02-data-schemas.md)

Spec: docs/04-identity-resolution.md
Independent of stages 4-5; can be developed and validated in parallel.

Part A (team clustering) always runs — gravity needs to know which tracks are
defenders before any distance means anything. Part B (jersey OCR) is opt-in
behind `--ocr` because it is slow and, on broadcast footage, resolves a
minority of tracks. Every track it cannot resolve keeps a null
`jersey_number` and `player_name`, which `02-data-schemas.md` treats as an
expected state: downstream geometry does not depend on the name, only the
final per-player aggregation does.

Example:
    python pipeline/03_identify.py --game-id 0022500123
    python pipeline/03_identify.py --game-id 0022500123 --ocr \\
        --team light=NYK --team dark=SAS
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from pipeline.common import (
    IDENTITY_DIR,
    find_video,
    identity_path,
    ocr_reads_path,
    tracks_path,
)
from pipeline.jersey_ocr import (
    DEFAULT_MIN_AGREEMENT,
    DEFAULT_SAMPLES_PER_TRACK,
    MIN_OCR_CONFIDENCE,
    MIN_SHARPNESS,
    MIN_WINNER_SHARE,
    build_team_map,
    load_roster,
    read_track_numbers,
    vote,
)
from pipeline.team_clustering import (
    TEAM_OTHER,
    assign_teams,
    track_features,
)

IDENTITY_COLUMNS = [
    "game_id",
    "shot_id",
    "tracker_id",
    "team_id",
    "jersey_number",
    "player_name",
    "identity_confidence",
]


def classify_teams(
    game_id: str,
    max_samples: int = 12,
    min_track_len: int = 10,
    ambiguity_ratio: float = 0.80,
) -> pd.DataFrame:
    """Part A — cluster tracked players into 2 teams by torso colour."""
    tracks = pd.read_parquet(tracks_path(game_id))
    features = track_features(
        find_video(game_id), tracks, max_samples, min_track_len
    )
    if features.empty:
        return pd.DataFrame(columns=IDENTITY_COLUMNS)
    return assign_teams(features, ambiguity_ratio)


def resolve_jersey_numbers(
    game_id: str,
    samples_per_track: int = DEFAULT_SAMPLES_PER_TRACK,
    min_agreement: int = DEFAULT_MIN_AGREEMENT,
    min_track_len: int = 30,
) -> dict:
    """Part B — OCR jersey numbers with majority vote per track.

    Returns {(shot_id, tracker_id): (number|None, agreeing, total_reads)}.

    A single frame is not evidence here: a torso is about 76px tall at 1080p,
    and OCR will confidently read a digit off a jersey wordmark. Sampling a
    track's largest, sharpest frames and requiring several to agree is what
    turns that into an answer; tracks where they do not agree keep a null
    number, which the spec is explicit is the correct outcome. See
    `pipeline/jersey_ocr.py` for what the thresholds were measured against.
    """
    import easyocr  # heavy, and only needed when OCR is actually requested

    tracks = pd.read_parquet(tracks_path(game_id))
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    reads = read_track_numbers(
        find_video(game_id), tracks, reader, samples_per_track, min_track_len
    )
    results = {key: vote(track.counts, min_agreement) for key, track in reads.items()}
    write_ocr_reads(game_id, reads, results, samples_per_track, min_agreement)
    return results


def write_ocr_reads(
    game_id: str,
    reads: dict,
    results: dict,
    samples_per_track: int,
    min_agreement: int,
) -> None:
    """Save the per-crop evidence beside the identity table.

    The viewer is a read-only view over stage output and must never re-run
    OCR itself, so the reasoning has to be on disk for it to display: which
    frames were sampled, which were too blurred to try, and what each
    readable one produced.
    """
    payload = {
        "game_id": game_id,
        "settings": {
            "samples_per_track": samples_per_track,
            "min_agreement": min_agreement,
            "min_winner_share": MIN_WINNER_SHARE,
            "min_ocr_confidence": MIN_OCR_CONFIDENCE,
            "min_sharpness": MIN_SHARPNESS,
        },
        "tracks": [
            {
                "shot_id": shot_id,
                "tracker_id": tracker_id,
                "jersey_number": results[(shot_id, tracker_id)][0],
                "agreeing": results[(shot_id, tracker_id)][1],
                "total_reads": results[(shot_id, tracker_id)][2],
                "counts": dict(track.counts),
                "samples": track.samples,
            }
            for (shot_id, tracker_id), track in sorted(reads.items())
        ],
    }
    destination = ocr_reads_path(game_id)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def resolve_identity(
    clustered: pd.DataFrame,
    game_id: str,
    ocr: dict | None = None,
    team_map: dict | None = None,
    rosters: dict | None = None,
) -> pd.DataFrame:
    """(team_id, jersey_number) -> roster lookup -> player_name.

    Either input being null makes `player_name` null, and so does a number
    that no roster carries. The spec forbids filling that gap by inference —
    "probably the point guard" is exactly the silent wrong label that would
    corrupt the metric — so an unrecognised number stays unresolved.

    `identity_confidence` follows the spec: frame agreement multiplied by the
    clipped silhouette. Without OCR there is no agreement term, so the value
    describes confidence in the *team* label alone and is scaled by how many
    frames yielded a usable crop instead.
    """
    result = clustered.copy()
    silhouette = result["silhouette"].clip(lower=0.0, upper=1.0)
    sample_weight = (result["n_samples"] / result["n_samples"].max()).clip(0.0, 1.0)

    result["game_id"] = game_id
    result["jersey_number"] = pd.NA
    result["player_name"] = pd.NA
    result["identity_confidence"] = (silhouette * sample_weight).astype(float)

    if ocr:
        numbers, names, agreement = [], [], []
        for row in result.itertuples():
            number, agreeing, total = ocr.get(
                (int(row.shot_id), int(row.tracker_id)), (None, 0, 0)
            )
            numbers.append(number if number is not None else pd.NA)
            agreement.append(agreeing / total if total else 0.0)

            roster_team = (team_map or {}).get(row.team_id)
            squad = (rosters or {}).get(roster_team, {})
            names.append(squad.get(number, pd.NA) if number else pd.NA)

        result["jersey_number"] = numbers
        result["player_name"] = names
        result["identity_confidence"] = (
            silhouette * pd.Series(agreement, index=result.index)
        ).astype(float)

    # An unresolved team is not a confident identity, whatever the clustering
    # score says about the tracks that did resolve.
    result.loc[result["team_id"] == TEAM_OTHER, "identity_confidence"] = 0.0
    return result[IDENTITY_COLUMNS]


def main() -> int:
    parser = argparse.ArgumentParser(description="Stage 3 — identity resolution")
    parser.add_argument("--game-id", required=True)
    parser.add_argument(
        "--max-samples",
        type=int,
        default=12,
        help="Frames sampled per track for the colour feature.",
    )
    parser.add_argument(
        "--min-track-len",
        type=int,
        default=10,
        help="Skip tracks shorter than this; too few frames to cluster on.",
    )
    parser.add_argument(
        "--ambiguity-ratio",
        type=float,
        default=0.80,
        help="A track becomes 'other' when its distance to the rival team's "
        "centroid is less than this multiple worse than to its own — i.e. the "
        "colour evidence does not favour either team. Raise it to keep more "
        "borderline tracks, lower it to be stricter.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Read jersey numbers. Slow, and low-yield on broadcast footage; "
        "without it jersey_number and player_name stay null, which downstream "
        "geometry does not mind.",
    )
    parser.add_argument(
        "--team",
        action="append",
        metavar="CLUSTER=TEAM",
        help="Map a colour cluster to a roster, e.g. --team light=NYK "
        "--team dark=SAS. Clustering can only say which kit is brighter; "
        "which one is the Knicks is a fact a person supplies.",
    )
    parser.add_argument(
        "--ocr-samples",
        type=int,
        default=DEFAULT_SAMPLES_PER_TRACK,
        help="Frames sampled per track for OCR, largest and sharpest first.",
    )
    parser.add_argument(
        "--min-agreement",
        type=int,
        default=DEFAULT_MIN_AGREEMENT,
        help="Agreeing reads needed before a number is accepted.",
    )
    args = parser.parse_args()

    source = tracks_path(args.game_id)
    if not source.exists():
        print(f"[error] {source} not found — run 02_track.py first", file=sys.stderr)
        return 1

    clustered = classify_teams(
        args.game_id, args.max_samples, args.min_track_len, args.ambiguity_ratio
    )
    if clustered.empty:
        print(
            "[error] no tracks long enough to cluster — lower --min-track-len",
            file=sys.stderr,
        )
        return 1

    ocr_results = None
    team_map = None
    rosters = None
    if args.ocr:
        try:
            team_map = build_team_map(args.team)
            rosters = {team: load_roster(team) for team in set(team_map.values())}
        except (ValueError, FileNotFoundError) as exc:
            print(f"[error] {exc}", file=sys.stderr)
            return 1
        if not team_map:
            print(
                "[warn] no --team mapping given, so numbers will be read but "
                "no name can be looked up.",
                file=sys.stderr,
            )
        # A cluster name that this run never produced maps nothing, and would
        # otherwise fail silently as "no names matched".
        clusters = set(clustered["team_id"].unique())
        unknown = sorted(set(team_map) - clusters)
        if unknown:
            print(
                f"[error] --team names no cluster this run produced: "
                f"{', '.join(unknown)}. Clustering labelled: "
                f"{', '.join(sorted(clusters))}.",
                file=sys.stderr,
            )
            return 1
        ocr_results = resolve_jersey_numbers(
            args.game_id, args.ocr_samples, args.min_agreement
        )
        resolved = sum(1 for number, _, _ in ocr_results.values() if number)
        print(
            f"[stage 3] OCR read {len(ocr_results)} track(s), "
            f"{resolved} reached {args.min_agreement} agreeing reads",
            file=sys.stderr,
        )

    identity = resolve_identity(
        clustered, args.game_id, ocr_results, team_map, rosters
    )

    IDENTITY_DIR.mkdir(parents=True, exist_ok=True)
    destination = identity_path(args.game_id)
    identity.to_parquet(destination, index=False)

    counts = identity["team_id"].value_counts().to_dict()
    print(
        f"[stage 3] {len(identity)} track(s): "
        + ", ".join(f"{k} {v}" for k, v in sorted(counts.items()))
        + f" | silhouette {clustered['silhouette'].iloc[0]:.2f}",
        file=sys.stderr,
    )
    if args.ocr:
        numbered = identity[identity["jersey_number"].notna()]
        with_name = int(identity["player_name"].notna().sum())
        print(
            f"[stage 3] {len(numbered)}/{len(identity)} track(s) resolved a "
            f"jersey number, {with_name} matched a roster name",
            file=sys.stderr,
        )
        # A number nobody's roster carries is the one failure a person can fix
        # in a minute, so say exactly which entries are missing.
        gaps = sorted(
            {
                f"{team_map.get(row.team_id, row.team_id)} #{row.jersey_number}"
                for row in numbered.itertuples()
                if pd.isna(row.player_name)
            }
        )
        if gaps and team_map:
            print(
                f"[stage 3] no roster entry for: {', '.join(gaps)} — add them "
                "to data/rosters/ to get names for these tracks",
                file=sys.stderr,
            )

    print(f"[stage 3] {len(identity)} row(s) -> {destination.name}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
