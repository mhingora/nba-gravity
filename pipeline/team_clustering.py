"""Stage 3, Part A — sort tracked players into two teams by kit colour.

Gravity is measured against *defenders*, so before any distance means
anything the pipeline has to know which tracks are on the other team.

`04-identity-resolution.md` describes clustering each shot separately and
then anchoring the two clusters to a per-game reference so `team_id` stays
consistent across cuts. This does the equivalent with one fewer moving part:
features from every track in the game are clustered **once**, and each track
takes the label of its nearest centroid. Anchoring is then automatic rather
than a second matching step that can itself fail. It costs nothing here
because one game is one arena under one lighting rig; if per-shot lighting
ever varies enough to break it, the per-shot-plus-anchor scheme in the spec
is the fallback.

Labels are `"light"` and `"dark"` rather than arbitrary `team_a`/`team_b`,
assigned by cluster brightness. That makes them stable across runs — an
arbitrary label would flip whenever k-means seeded differently — and
readable when spot-checking, since basketball's home/away convention is
exactly light versus dark kit.
"""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

TEAM_LIGHT = "light"
TEAM_DARK = "dark"
TEAM_OTHER = "other"

# Torso only. Shorts and shoes carry more inter-team noise than the jersey,
# and the head region is mostly skin and background.
TORSO_TOP = 0.15
TORSO_BOTTOM = 0.55
# Horizontal inset: box edges catch background court and neighbouring
# players, which is exactly the contamination clustering is most sensitive to.
TORSO_INSET = 0.22

HUE_BINS = 12
SAT_BINS = 6
VAL_BINS = 6
FEATURE_LENGTH = HUE_BINS + SAT_BINS + VAL_BINS + 3


def torso_box(x1: float, y1: float, x2: float, y2: float) -> tuple[int, int, int, int]:
    """The torso sub-rectangle of a player box, as integer pixel bounds."""
    height = y2 - y1
    width = x2 - x1
    return (
        int(round(x1 + width * TORSO_INSET)),
        int(round(y1 + height * TORSO_TOP)),
        int(round(x2 - width * TORSO_INSET)),
        int(round(y1 + height * TORSO_BOTTOM)),
    )


def crop_torso(frame: np.ndarray, box) -> np.ndarray | None:
    """Extract the torso crop, or None if it falls outside the frame."""
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = torso_box(*box)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return frame[y1:y2, x1:x2]


def crop_feature(crop: np.ndarray) -> np.ndarray:
    """Colour signature of one torso crop.

    Hue alone cannot separate basketball kits: white and black jerseys are
    both nearly hueless, and their hue readings are noise. So the feature
    carries saturation and value histograms plus mean BGR as well — the
    light/dark axis lives in those, while hue handles genuinely coloured
    kits. Hue counts are weighted by saturation*value so that grey pixels
    contribute little to the hue histogram instead of voting randomly.
    """
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hue, sat, val = hsv[..., 0].ravel(), hsv[..., 1].ravel(), hsv[..., 2].ravel()

    weight = (sat.astype(np.float32) / 255.0) * (val.astype(np.float32) / 255.0)
    hue_hist = np.histogram(hue, bins=HUE_BINS, range=(0, 180), weights=weight)[0]
    sat_hist = np.histogram(sat, bins=SAT_BINS, range=(0, 256))[0].astype(np.float32)
    val_hist = np.histogram(val, bins=VAL_BINS, range=(0, 256))[0].astype(np.float32)

    for hist in (hue_hist, sat_hist, val_hist):
        total = hist.sum()
        if total > 0:
            hist /= total

    mean_bgr = crop.reshape(-1, 3).mean(axis=0).astype(np.float32) / 255.0
    return np.concatenate([hue_hist, sat_hist, val_hist, mean_bgr]).astype(np.float32)


def track_features(
    video_path,
    tracks: pd.DataFrame,
    max_samples: int = 12,
    min_track_len: int = 10,
) -> pd.DataFrame:
    """One colour feature per `(shot_id, tracker_id)`, sampled across its life.

    Frames are visited in ascending order in a single sequential pass: seeking
    per crop over thousands of frames is far slower than reading forward and
    picking off the frames that matter.

    The per-track feature is a **median** over sampled frames, not a mean, so
    one badly occluded frame — a defender's arm across the chest, a referee
    crossing in front — cannot drag the whole track's colour.
    """
    players = tracks[tracks["class"] == "player"]
    players = players[players["track_len"] >= min_track_len]
    if players.empty:
        return pd.DataFrame(columns=["shot_id", "tracker_id", "n_samples", "feature"])

    # Choose which frames to sample per track before touching the video.
    wanted: dict[int, list[tuple]] = {}
    for (shot_id, tracker_id), group in players.groupby(["shot_id", "tracker_id"]):
        group = group.sort_values("frame_idx")
        step = max(1, len(group) // max_samples)
        for row in group.iloc[::step].head(max_samples).itertuples():
            wanted.setdefault(int(row.frame_idx), []).append(
                (int(shot_id), int(tracker_id), (row.x1, row.y1, row.x2, row.y2))
            )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise IOError(f"Could not open video: {video_path}")

    collected: dict[tuple[int, int], list[np.ndarray]] = {}
    try:
        for frame_idx in sorted(wanted):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, frame = capture.read()
            if not ok:
                continue
            for shot_id, tracker_id, box in wanted[frame_idx]:
                crop = crop_torso(frame, box)
                if crop is None:
                    continue
                collected.setdefault((shot_id, tracker_id), []).append(
                    crop_feature(crop)
                )
    finally:
        capture.release()

    rows = [
        {
            "shot_id": shot_id,
            "tracker_id": tracker_id,
            "n_samples": len(features),
            "feature": np.median(np.stack(features), axis=0),
        }
        for (shot_id, tracker_id), features in collected.items()
    ]
    return pd.DataFrame(rows).sort_values(["shot_id", "tracker_id"]).reset_index(drop=True)


def assign_teams(features: pd.DataFrame, ambiguity_ratio: float = 0.80) -> pd.DataFrame:
    """Cluster track features into light/dark teams, with an `other` bucket.

    Returns the input frame plus `team_id`, `margin` (how much closer the
    track sits to its own centroid than to the other) and `silhouette` (one
    global score, reused per row for identity_confidence).

    Referees and anyone else the detector picked up do not form a clean third
    cluster — there are too few of them, and k-means with k=3 tends to split a
    team instead. So `other` is decided by *ambiguity*, not by distance rank:
    a track whose distance to the rival centroid is barely larger than to its
    own has no real colour evidence either way.

    An earlier version evicted a fixed top percentile of distances instead,
    which is not a confidence test at all — it always relabels the same
    fraction, so with clean input it throws away real players (spot-checking
    caught it discarding an unmistakable Spurs jersey), and with filthy input
    it keeps just as many. A ratio compares each track against its own
    alternative, so a clean game can legitimately yield no outliers.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    result = features.copy()
    if len(result) < 2:
        result["team_id"] = TEAM_OTHER
        result["margin"] = 0.0
        result["silhouette"] = 0.0
        return result

    matrix = np.stack(result["feature"].to_numpy())
    kmeans = KMeans(n_clusters=2, n_init=10, random_state=0).fit(matrix)
    labels = kmeans.labels_

    # Distance to both centroids, so ambiguity can be judged per track.
    to_both = np.linalg.norm(
        matrix[:, None, :] - kmeans.cluster_centers_[None, :, :], axis=2
    )
    own = to_both[np.arange(len(matrix)), labels]
    rival = to_both[np.arange(len(matrix)), 1 - labels]
    # 0 means "sits exactly on its centroid"; approaching 1 means "equally
    # close to the other team's colour", i.e. no evidence either way.
    ambiguity = np.divide(own, rival, out=np.ones_like(own), where=rival > 0)

    # Brightness is the last of the three mean-BGR components' average; using
    # the centroid rather than raw crops keeps the light/dark naming stable.
    brightness = kmeans.cluster_centers_[:, -3:].mean(axis=1)
    light_cluster = int(np.argmax(brightness))
    names = {
        light_cluster: TEAM_LIGHT,
        1 - light_cluster: TEAM_DARK,
    }

    result["team_id"] = [
        names[label] if ratio <= ambiguity_ratio else TEAM_OTHER
        for label, ratio in zip(labels, ambiguity)
    ]
    result["margin"] = 1.0 - ambiguity
    # Silhouette needs both clusters populated and more points than clusters.
    result["silhouette"] = (
        float(silhouette_score(matrix, labels)) if len(set(labels)) == 2 and len(result) > 2 else 0.0
    )
    return result
