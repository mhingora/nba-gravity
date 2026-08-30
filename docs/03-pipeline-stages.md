# Pipeline Stages — Detailed Spec

Each stage below corresponds to one script in `pipeline/`. Inputs and outputs
reference the schemas in `02-data-schemas.md`.

Stages 0-2 are implemented; 3-6 are stubs. Shared code lives in three
importable modules beside the numbered scripts, which stay CLI entry points:

- `pipeline/common.py` — artifact paths, schema constants, video discovery and
  frame IO, the `Shot` type, and an incremental parquet writer.
- `pipeline/shot_boundaries.py` — Stage 0 cut detection.
- `pipeline/detector.py` — detector wrappers for Stage 1.

Every stage reads and writes only on-disk artifacts, so any one can be re-run
without redoing the expensive ones. Both implemented scripts take
`--start-frame` / `--end-frame` so you can work a single possession out of a
full-game file without trimming it first.

---

## Stage 0 — Shot boundary segmentation (pre-processing, run inside `01_detect.py`)

**Input:** raw mp4
**Output:** list of `(shot_id, start_frame, end_frame)` written alongside detections

Use frame-to-frame histogram difference or a simple pixel-diff threshold to
flag hard cuts. This doesn't need to be sophisticated — broadcast cuts are
usually abrupt, not gradual, so a naive threshold catches the vast majority.
False positives (flagging a cut that isn't one) are cheap — they just mean
tracking resets a bit more often than strictly necessary. False negatives
(missing a real cut) are worse — they let `sv.ByteTrack()` try to bridge two
unrelated shots and silently produce garbage tracks. Bias the threshold toward
over-detecting cuts.

**As implemented** (`pipeline/shot_boundaries.py`): frames are downscaled to
160px wide, converted to HSV, and reduced to a hue/saturation histogram; the
cut signal is `1 − correlation` against the previous frame. A cut is flagged
above `--cut-threshold` (default 0.30). One addition to the spec: a
`--min-shot-frames` floor (default 8) suppresses a second cut immediately
after the first, because camera flashes and strobing arena lights otherwise
fragment one shot into dozens of unusable stubs.

Segmentation needs no detector and no GPU, so it can run on its own:

```
python pipeline/01_detect.py --game-id GAME --shots-only
```

Then read the shot table in the viewer's Tracking tab to pick a clean
possession, and pass its frame range to the detection run with
`--reuse-shots`.

## Stage 1 — Detection (`01_detect.py`)

**Input:** raw mp4, shot boundaries
**Output:** `outputs/detections/{game_id}.parquet`

- Run your fine-tuned detector (`player`, `ball` classes) frame by frame using
  `sv.Detections`.
- Ball detection benefits from a separate, higher-resolution pass or crop-based
  detection since it's small and fast-moving — consider running it as its own
  model rather than a third class on the player detector if recall is poor.
- Write detections incrementally (e.g. batch every N frames) rather than
  holding a full game in memory.
- Skip frames tagged as `replay` or `isolation_closeup` shots at this stage if
  you've already angle-tagged — no need to burn compute on frames you'll
  discard later. If angle tagging happens after detection, filter downstream
  instead.

**As implemented** (`pipeline/detector.py`): `YoloDetector` wraps Ultralytics
and maps stock COCO classes onto ours — `person` → `player`, `sports ball` →
`ball` — so Milestone 1 runs against off-the-shelf weights before any
fine-tuning, exactly as `06-roadmap.md` advises. Point `--model` at custom
weights later and the same mapping picks up native `player` / `ball` classes
unchanged; anything unmapped is dropped. The optional separate ball pass is
wired up as `--ball-model` / `--ball-imgsz`.

Detections are written in batches (`--batch-size`, default 5000 rows) so a
full game never sits in memory. `--skip-angles` filters shots by their
`camera_angle` tag, which is inert until something populates that field.

A second detector, `--detector colorblob`, exists only to exercise the
pipeline against `tools/make_test_clip.py` without GPU weights. It finds
saturated blobs by HSV threshold and is useless on real footage.

```
python pipeline/01_detect.py --game-id GAME --start-frame 26100 --end-frame 26550 \
    --reuse-shots --model yolov8m.pt --device cuda:0
```

## Stage 2 — Tracking (`02_track.py`)

**Input:** `outputs/detections/{game_id}.parquet`
**Output:** `outputs/tracks/{game_id}.parquet`

- Instantiate a fresh `sv.ByteTrack()` per `shot_id` — never carry tracker
  state across a shot boundary.
- Filter obviously-spurious detections before tracking (e.g. bounding boxes
  outside plausible court-region bounds, confidence below a floor) — garbage
  in, garbage tracked.
- Compute `foot_x, foot_y` (bottom-center of bbox) here — this is the point
  you'll project through the homography later, since it approximates where
  the player is actually standing on the court, unlike the bbox center.
- Record `track_len` per tracker_id once the shot ends — short-lived tracks
  (a handful of frames) are usually detection noise or a player disappearing
  behind another player; downstream stages should be able to filter on this.

**As implemented** (`pipeline/02_track.py`): a fresh `sv.ByteTrack()` per
`shot_id`, which restores tracker_ids to 1..N on every cut as the schema
requires. Pre-tracking filters, all tunable from the CLI:

- `--min-confidence` (0.30) — confidence floor.
- `--min-height-frac` / `--max-height-frac` (0.03 / 0.60) — box height as a
  fraction of frame height, catching both specks and full-frame false hits.
- `--min-aspect` / `--max-aspect` (1.1 / 6.0) — players are taller than wide;
  boxes that aren't are usually two overlapping people merged into one.
- `--court-roi X1 Y1 X2 Y2` — normalized playing-surface bounds. This tests the
  *foot point*, not the whole box, because a spectator's feet sit outside the
  surface even when their head overlaps it. With stock COCO weights this is the
  main defence against crowd, bench and referee detections; something like
  `0.0 0.35 1.0 0.95` is a reasonable starting guess for a baseline angle.

Ball detections pass through untracked (see `02-data-schemas.md`).

```
python pipeline/02_track.py --game-id GAME --court-roi 0.0 0.35 1.0 0.95
```

## Stage 3 — Identity resolution (`03_identify.py`)

**Input:** `outputs/tracks/{game_id}.parquet`, rosters
**Output:** `outputs/identity/{game_id}.parquet`

See `04-identity-resolution.md` for the full method. Summary:
1. Cluster tracked players into two teams by jersey color (per shot, then
   reconciled across shots since lighting/angle can shift color slightly).
2. Crop each tracker_id's frames, run jersey-number OCR, take a majority vote
   across the whole track.
3. Join `(team_id, jersey_number)` against the roster to get `player_name`.
4. Anything that doesn't clear a confidence bar stays `null` — do not guess.

This stage does not block Stage 4 or Stage 5. It can be developed and
validated independently.

## Stage 4 — Ball possession (`04_ball_possession.py`)

**Input:** `outputs/tracks/{game_id}.parquet` (ball + player rows)
**Output:** `outputs/possession/{game_id}.parquet`

- Per frame, compute distance from ball centroid to the nearest point of each
  player bbox (not just bbox center — for a player reaching for the ball, the
  hand/edge of the bbox is closer than the centroid, which matters near the
  possession threshold).
- Assign `ball_handler_tracker_id` = closest player under a distance threshold
  (tune empirically — start around a value corresponding to roughly arm's
  length in pixel terms at that camera's typical scale, then adjust).
- Debounce: require the same tracker_id to be the closest player for at least
  N consecutive frames (e.g. 5-8 at typical broadcast frame rates) before
  confirming a possession change — this avoids flicker during rebounds,
  deflections, and passes where the ball is transiently near multiple players.
- Frames where no player is under threshold (ball in the air mid-pass/shot)
  get `ball_handler_tracker_id = null`, which is correct — nobody has
  possession in that instant.

## Stage 5 — Court calibration (`05_calibrate.py`)

**Input:** raw video frames (one representative frame per shot, or per angle type)
**Output:** `outputs/calibration/{game_id}_{shot_id}.json`

- Identify known court reference points (baseline corners, free-throw line
  intersections, three-point arc tangent points, half-court line ends) either
  by manual annotation per angle type, or a trained keypoint detector if you
  want to scale this beyond a handful of angles.
- Compute a homography with `cv2.findHomography` (`sv.ViewTransformer` was
  removed in supervision 0.30 and is no longer an option
  directly) mapping pixel coordinates to real-world court coordinates (use
  feet, with a fixed origin like one baseline corner).
- Store `reprojection_error_px` — reproject your reference points through the
  computed homography and measure how far off they land from ground truth.
  This is your calibration quality signal.
- Since camera angle repeats across shots (broadcasts cycle through a small
  set of physical camera positions), you often only need to calibrate once
  per angle type per game, not per shot — reuse the homography across all
  shots sharing that angle tag, and only recompute if reprojection error
  looks off for a specific shot (e.g. the broadcast zoomed slightly).

## Stage 6 — Aggregation (`06_aggregate.py`)

**Input:** tracks, identity, possession, calibration — all joined by
`game_id`/`shot_id`/`frame_idx`

**Output:** `outputs/metrics/{game_id}_gravity.parquet` and the `all`-aggregated version

1. Project every tracked player's `(foot_x, foot_y)` into court coordinates
   using the shot's homography.
2. For each frame, identify the 5 defenders — this requires knowing which
   team is on offense, which comes from identity resolution's team
   clustering plus possession (the team with the ball is on offense).
3. For each frame where a target player is on court:
   - if `ball_handler_tracker_id == target player's tracker_id`: bucket as
     "with ball," compute avg distance from the 5 defenders to the target.
   - else: bucket as "without ball," same computation.
4. Average within each bucket, per player, per game — then roll up across
   games for the final `gravity_delta`.
5. Apply filters before trusting a row: minimum frame count per bucket (a
   player with 12 total possession frames in a game shouldn't produce a
   headline number), minimum `identity_confidence`, and exclude
   shots/frames with poor `reprojection_error_px`.
