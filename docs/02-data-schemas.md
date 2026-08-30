# Data Schemas

All tabular outputs are parquet, partitioned by `game_id`. Reference tables
(rosters, calibration points) are hand-maintained JSON/CSV.

> **Implementation status.** The detections, shots and tracks schemas below are
> implemented and written by `01_detect.py` / `02_track.py`. Identity,
> calibration, possession and metrics are specs only — no code writes them yet.
> Path helpers for every artifact live in `pipeline/common.py`, which is the
> single place that knows where things go; nothing else hardcodes a path.

## `data/rosters/{team_id}_{season}.json`

```json
{
  "team_id": "NYK",
  "season": "2025-26",
  "players": {
    "11": "Jalen Brunson",
    "9":  "OG Anunoby",
    "23": "Mikal Bridges"
  }
}
```

Jersey numbers are not globally unique — they're only unique *within a team*,
which is why identity resolution needs team classification before it can use
the roster lookup.

## `outputs/detections/{game_id}.parquet`

Raw per-frame detections, before tracking.

| column       | type  | notes                                   |
|--------------|-------|------------------------------------------|
| game_id      | str   |                                           |
| frame_idx    | int   | absolute frame number in the source mp4  |
| shot_id      | int   | which continuous camera shot this frame belongs to |
| class        | str   | `player` or `ball`                       |
| x1,y1,x2,y2  | float | pixel bbox                               |
| confidence   | float | detector confidence                      |

## `outputs/detections/{game_id}_shots.json`

Shot boundaries, written alongside detections by Stage 0. `end_frame` is
inclusive, and shots are contiguous and cover the whole processed range.

```json
{
  "game_id": "0022500123",
  "shots": [
    {"shot_id": 0, "start_frame": 0, "end_frame": 59, "camera_angle": "unlabeled"}
  ]
}
```

`camera_angle` defaults to `"unlabeled"`. Nothing populates it yet — angle
tagging is a later concern (see `01-architecture.md`), but the field exists so
`01_detect.py --skip-angles` can filter on it once it does.

## `outputs/detections/{game_id}_shot_diffs.parquet`

Per-frame cut-strength signal, kept purely for tuning the Stage 0 threshold.
The Tracking tab plots it so you can see peaks against the cut threshold
rather than guessing at the number.

| column        | type  | notes                                        |
|---------------|-------|-----------------------------------------------|
| frame_idx     | int   |                                                |
| cut_distance  | float | 1 − HSV histogram correlation vs. previous frame |

## `outputs/tracks/{game_id}.parquet`

Adds tracking on top of detections. One row per tracked player per frame,
**plus pass-through ball rows**.

| column       | type  | notes                                              |
|--------------|-------|------------------------------------------------------|
| game_id      | str   |                                                        |
| frame_idx    | int   |                                                        |
| shot_id      | int   |                                                        |
| tracker_id   | int   | unique only *within* a shot_id — resets every cut; `-1` for ball rows |
| class        | str   | `player` or `ball`                                     |
| x1,y1,x2,y2  | float | pixel bbox                                            |
| foot_x, foot_y | float | bottom-center of bbox, used for court projection    |
| track_len    | int   | total frames this tracker_id persisted, for confidence weighting; 0 for ball rows |

The `class` column resolves a conflict in the original spec: this table was
described as player rows only, but Stage 4 reads "ball + player rows" from it.
Carrying both keeps possession a single-file read. The ball is detected but
never tracked — ByteTrack's motion model is built for person-sized boxes — so
ball rows get the `-1` sentinel rather than a real tracker_id.

Because `tracker_id` only means anything within a shot, any cross-shot
grouping must key on `(shot_id, tracker_id)`. Counting `tracker_id` alone
silently collapses distinct tracks and hides fragmentation.

## `outputs/identity/{game_id}.parquet`

One row per `(shot_id, tracker_id)` — identity is resolved once per track, not
per frame.

| column        | type   | notes                                                |
|---------------|--------|-------------------------------------------------------|
| game_id       | str    |                                                         |
| shot_id       | int    |                                                         |
| tracker_id    | int    |                                                         |
| team_id       | str \| null | from color clustering; null if unresolved        |
| jersey_number | str \| null | majority-vote OCR read; null if no confident read |
| player_name   | str \| null | roster lookup of (team_id, jersey_number); null if either input is null |
| identity_confidence | float | 0-1, see `04-identity-resolution.md` for definition |

`player_name` being null is expected and fine — downstream geometry stages do
not depend on it. Only the final per-player aggregation needs it.

## `data/calibration/{name}.json` (hand-maintained)

A **court profile**: where the playing surface sits in one camera's frame,
plus the tracker settings tuned against that camera. Written by the viewer's
Court polygon tuner, read by `02_track.py --court-profile {name}`.

```json
{
  "name": "msg_main_wide",
  "description": "Madison Square Garden, main wide side camera.",
  "tracker": {
    "lost_track_buffer": 150,
    "minimum_matching_threshold": 0.95,
    "track_activation_threshold": 0.30
  },
  "court_polygon": [
    [0.03, 0.6], [0.18, 0.48], [0.42, 0.385], [0.7, 0.368], [1.0, 0.355],
    [1.0, 0.88], [0.7, 0.8], [0.45, 0.76], [0.2, 0.72], [0.05, 0.7]
  ]
}
```

`court_polygon` is normalized 0-1, clockwise, and tested against each
detection's **foot point** — a spectator's feet fall outside the playing
surface even when their head overlaps it. Normalized coordinates mean a
profile fitted on 1080p applies unchanged to 720p from the same camera.

Keyed by camera angle rather than `game_id` on purpose: the polygon describes
the broadcast setup, not the game, so one profile serves every game shot from
that position. `tracker` may be empty or omit keys; anything missing falls
back to the CLI default, and an explicitly passed flag overrides the profile.

## `outputs/calibration/{game_id}_{shot_id}.json`

Homography matrix for a shot, keyed by which angle-reference it was computed
against.

```json
{
  "game_id": "0022500123",
  "shot_id": 47,
  "camera_angle": "main_baseline",
  "homography_matrix": [[...], [...], [...]],
  "reprojection_error_px": 3.2
}
```

`reprojection_error_px` should be logged and used as a filter — throw out
distance calculations from shots where calibration was clearly bad rather
than silently trusting a warped homography.

## `outputs/possession/{game_id}.parquet`

| column      | type | notes                                       |
|-------------|------|-----------------------------------------------|
| game_id     | str  |                                                 |
| shot_id     | int  |                                                 |
| frame_idx   | int  |                                                 |
| ball_handler_tracker_id | int \| null | null if no player is within possession threshold |
| ball_distance_px | float | distance from ball centroid to handler bbox, for debugging threshold choices |

## `outputs/metrics/{game_id_or_'all'}_gravity.parquet`

Final aggregated output — this is the deliverable table.

| column                     | type  | notes                                    |
|-----------------------------|-------|--------------------------------------------|
| player_name                 | str   |                                              |
| games_included              | int   |                                              |
| frames_with_possession       | int   |                                              |
| frames_without_possession    | int   |                                              |
| avg_defender_distance_with_ball    | float | court units (feet)                    |
| avg_defender_distance_without_ball | float | court units (feet)                    |
| gravity_delta                | float | without_ball minus with_ball              |

Keep both the per-game and the `all`-aggregated version — per-game lets you
sanity-check for outlier games (bad calibration, garbage-time minutes, etc.)
before trusting the aggregate.
