# Data Schemas

All tabular outputs are parquet, partitioned by `game_id`. Reference tables
(rosters, calibration points) are hand-maintained JSON/CSV.

> **Implementation status.** Every table below is implemented and written by
> stages 1-6. Path helpers for every artifact live in `pipeline/common.py`,
> which is the single place that knows where things go; nothing else hardcodes
> a path.

## `data/rosters/{team_id}.json`

One file per team, with the season inside it rather than in the filename — a
run names a team on the command line (`--team dark=SAS`), and requiring the
season there would mean retyping a fact the file already states.

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

`jersey_number` is a **string**, not an integer: "00" and "0" are different
players, and casting to int would merge them.

`identity_confidence` answers "how sure are we *which player* this is", so a
track with no number scores 0 even when its team is obvious. Stage 6 should
select defenders on `team_id` and use this column only where a named player is
required — see `04-identity-resolution.md`.

## `outputs/identity/{game_id}_ocr.json`

Written alongside the identity table when stage 3 runs with `--ocr`. The
parquet says what each track resolved to; this says why, so the viewer can
show a failed track's crops next to what OCR made of them without re-running
OCR itself.

```json
{
  "game_id": "S_N3_HD",
  "settings": {
    "samples_per_track": 24, "min_agreement": 4, "min_winner_share": 0.5,
    "min_ocr_confidence": 0.8, "min_sharpness": 40.0
  },
  "tracks": [
    {
      "shot_id": 11, "tracker_id": 5,
      "jersey_number": "2", "agreeing": 8, "total_reads": 14,
      "counts": {"2": 8, "24": 6},
      "samples": [
        {"frame_idx": 3101, "status": "read", "sharpness": 900.7, "reads": ["24"]},
        {"frame_idx": 3140, "status": "blurred", "sharpness": 12.4}
      ]
    }
  ]
}
```

`samples` keeps every frame that was tried, including the ones that produced
nothing: "18 blurred, 2 read nothing" is a different diagnosis from "6 frames,
6 different answers". `status` is one of `read`, `nothing_legible`, `blurred`
or `no_crop`.

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
  "court_profile": "msg_main",
  "homography_matrix": [[...], [...], [...]],
  "reprojection_error_px": 3.2,
  "annotated_on": {"game_id": "0022500123", "frame_idx": 3329, "shot_id": 11},
  "landmark_errors_px": [{"landmark": "free_throw_left", "error_px": 1.34}],
  "known_distance_checks": [
    {"from": "lane_baseline_left", "to": "lane_baseline_right",
     "expected_ft": 16.0, "measured_ft": 16.0, "error_ft": -0.0}
  ]
}
```

`reprojection_error_px` should be logged and used as a filter — throw out
distance calculations from shots where calibration was clearly bad rather
than silently trusting a warped homography.

**`reprojection_error_px` cannot tell you the matrix fits *this* shot.** Every
shot of a game is written the same matrix and the same error, because the
error is the residual on the one frame a person annotated. `annotated_on`
records which frame and shot that was, and it is the only field that
distinguishes a fitted shot from a copied one. Stage 6 aggregates the
annotated shot by default for exactly this reason.

## `outputs/calibration/{game_id}_homographies.parquet`

One homography **per frame**, written by `05_calibrate.py --propagate`. A
single matrix per shot assumes the camera holds still, and it does not: the
annotated matrix is out by 1.49 ft at the median across its own shot, against
0.03 ft for the propagated one (measured by template-matching the annotated
landmarks, which knows nothing about either homography).

| column        | type  | notes                                            |
|---------------|-------|----------------------------------------------------|
| game_id       | str   |                                                      |
| shot_id       | int   |                                                      |
| frame_idx     | int   |                                                      |
| h00 … h22     | float | the 3x3 matrix, row-major; image pixels -> court feet |
| inliers       | int   | matched features that agreed on the camera motion     |
| matches       | int   | candidate matches before RANSAC                       |
| motion_rms_px | float | how well those inliers fit the estimated motion       |

**Frames are missing from this table by design.** A frame whose view cannot be
matched back to the annotated frame — a different camera, a replay, heavy
blur — gets no row, and Stage 6 leaves it unmeasured rather than projecting it
through a matrix that does not describe it. On the test clip 95% of the
annotated shot solves, and 7 of the other 31 shots solve because they are the
same camera; the remaining 24 are correctly absent.

## `outputs/calibration/{game_id}_homographies.json`

What produced the table beside it.

```json
{
  "game_id": "S_N3_HD",
  "reference_frame": 3329,
  "court_profile": "msg_main_ebard",
  "settings": {"scale": 0.5, "min_inliers": 40, "ransac_px": 3.0},
  "overlay_regions": [{"x": 0.228, "y": 0.865, "w": 0.131, "h": 0.135, "coverage": 0.0127}],
  "frames_attempted": 464,
  "frames_solved": 443,
  "shots_reached": [11]
}
```

`overlay_regions` are the broadcast graphics, found automatically rather than
configured: they are the only things that match across a camera cut without
moving. Masking them is not an optimisation — unmasked, every shot in a clip
matches every other one on the scorebug alone, and propagation would hand
Stage 6 a confident homography for a baseline closeup.

## `outputs/possession/{game_id}.parquet`

| column      | type | notes                                       |
|-------------|------|-----------------------------------------------|
| game_id     | str  |                                                 |
| shot_id     | int  |                                                 |
| frame_idx   | int  |                                                 |
| ball_handler_tracker_id | int \| null | null if no player is within possession threshold |
| ball_distance_px | float | distance from ball centroid to handler bbox, for debugging threshold choices |

## `outputs/metrics/{game_id}_distances.parquet`

Per-frame evidence, written by Stage 6 beside the metrics table. The gravity
table is an average of averages; this is what it averaged. One row per
offensive player per measured frame.

| column      | type | notes                                              |
|-------------|------|------------------------------------------------------|
| game_id     | str  |                                                        |
| shot_id     | int  |                                                        |
| frame_idx   | int  |                                                        |
| tracker_id  | int  |                                                        |
| team_id     | str  | the team on offence in this shot                       |
| court_x, court_y | float | feet, from the foot point through the homography |
| has_ball    | bool | this track is stage 4's handler for the frame          |
| n_defenders | int  | defenders tracked on court; constant by construction   |
| avg_defender_distance_ft     | float | mean over those defenders          |
| nearest_defender_distance_ft | float | the closest one                    |

A frame only appears when the defence was tracked at exactly the expected size
(five). Fewer is a missed player and more is one player carrying two
tracker_ids — neither average is over a defence, so the frame is dropped
rather than averaged. Gaps in this table are the normal state of real footage,
which is why the viewer plots it rather than summarising it away.

## `outputs/metrics/{game_id_or_'all'}_gravity.parquet`

Final aggregated output — this is the deliverable table.

| column                     | type  | notes                                    |
|-----------------------------|-------|--------------------------------------------|
| player_label                | str   | how the row is keyed: the roster name, else `{team} #{number}` |
| player_name                 | str \| null | null when the number matched no roster |
| team_id                     | str   |                                              |
| jersey_number               | str \| null |                                       |
| games_included              | int   |                                              |
| frames_with_possession       | int   |                                              |
| frames_without_possession    | int   |                                              |
| avg_defender_distance_with_ball    | float | court units (feet)                    |
| avg_defender_distance_without_ball | float | court units (feet)                    |
| gravity_delta                | float \| null | without_ball minus with_ball; null unless both buckets clear the frame floor |
| avg_defender_distance_overall      | float | the spec's raw-gravity measure   |
| nearest_defender_with_ball / _without_ball / _delta / _overall | float | the same four, over the single closest defender |

`player_label` exists because a jersey number *is* an identity — it is what a
box score uses — and requiring a roster match before a player can be measured
would throw away tracks whose only gap is a missing line in a JSON file. A
track with neither a name nor a number never becomes a row: it cannot be
matched to the same person in the next camera shot, let alone the next game.

A null `gravity_delta` beside visible frame counts is the designed answer for
a player who never holds the ball in the footage processed. The row still
carries its overall distances, which `05-metrics-and-analysis.md` treats as a
metric in its own right.

Keep both the per-game and the `all`-aggregated version — per-game lets you
sanity-check for outlier games (bad calibration, garbage-time minutes, etc.)
before trusting the aggregate. The rollup weights each game's averages by its
frame counts and re-applies the frame floor to the totals, so two games of 60
with-ball frames together support a delta neither supports alone.
