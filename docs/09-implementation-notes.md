# Implementation Notes

What has actually been built, where it deviates from the specs in `01`-`08`,
and why. Docs `01`-`08` describe the intended design; this one describes the
code as it stands.

## What exists

```
app.py                          Streamlit viewer, Tabs 1-2 live + polygon tuner
pipeline/common.py              paths, schema constants, video IO, parquet writer
pipeline/shot_boundaries.py     Stage 0 cut detection
pipeline/detector.py            Stage 1 detector wrappers
pipeline/court_region.py        court polygon test + camera-angle profiles
pipeline/track_quality.py       tracking-quality metrics (CLI + viewer share these)
pipeline/tracker_backends.py    pluggable tracking backends + frame streaming
pipeline/01_detect.py           Stage 0 + 1 CLI
pipeline/02_track.py            Stage 2 CLI
pipeline/03_..06_*.py           stubs — docstrings and NotImplementedError
tools/make_test_clip.py         synthetic clip with known ground truth
data/calibration/*.json         court profiles, one per camera angle
```

Stages 3-6 have never been run. Their viewer tabs render a placeholder naming
the milestone that unlocks them.

## Deviations from the specs, and why

**Shared modules beside the numbered scripts.** `01_detect.py` and friends
cannot be imported (a module name can't start with a digit), so anything shared
lives in `common.py`, `shot_boundaries.py` and `detector.py`. The numbered
files stay pure CLI entry points, matching the layout in the README. Both they
and `app.py` insert the repo root on `sys.path` before importing `pipeline.*`,
so they work regardless of the directory they're launched from.

**`class` column in `tracks.parquet`.** `02-data-schemas.md` described the
table as player rows only, while `03-pipeline-stages.md` had Stage 4 reading
"ball + player rows" from it. Both are now satisfied by carrying ball rows with
a `class` column and `tracker_id = -1`. Documented in `02-data-schemas.md`.

**Two artifacts the schema doc didn't mention.** Stage 0 writes
`{game_id}_shots.json` (the boundaries themselves) and
`{game_id}_shot_diffs.parquet` (the per-frame cut-strength signal). The latter
exists purely so the threshold can be tuned by eye in the viewer instead of
guessed at. Both are now specified in `02-data-schemas.md`.

**`--min-shot-frames` floor on Stage 0.** The spec says to bias toward
over-detecting cuts, which is right, but with no floor at all a camera flash or
strobing arena light shatters one shot into dozens of stubs too short to track.
A minimum of 8 frames between cuts keeps the intended bias without that
failure.

**Stock COCO classes as the Stage 1 starting point.** `person` → `player`,
`sports ball` → `ball`, per Milestone 1's instruction to validate the approach
before investing in a fine-tuned model. The mapping is a lookup table in
`detector.py`; custom weights with native `player` / `ball` classes need no
code change.

**The viewer renders clips as GIF, not mp4.** `08-ui-design.md` suggested
`sv.VideoSink`. Browser mp4 playback needs an H.264 encoder that OpenCV builds
don't reliably ship on Windows, whereas a GIF plays anywhere. Capped at 400
frames with a stride control.

**Shot-boundary table renders before tracking exists.** Originally it was
inside Tab 2's "tracks exist" branch, which was backwards: reading that table
is how you choose a possession to point Stage 1 at, and segmentation runs
without a detector. It now appears as soon as Stage 0 has run.

## Dependency constraints

**`supervision` is no longer pinned.** It was held below 0.30 because
`sv.ByteTrack` — which Stage 2 depended on — is deprecated and scheduled for
removal (now in 0.31). Tracking moved to the `trackers` package behind a
backend abstraction (`pipeline/tracker_backends.py`), so supervision is free
to float; the repo runs on 0.30.1. supervision still supplies `Detections`,
the annotators and colour lookups.

One casualty of the upgrade: **`sv.ViewTransformer` was removed in 0.30.**
No executing code used it, but Stage 5's spec named it; those references now
point at `cv2.findHomography`.

**`ultralytics` is only needed for Stage 1 detection.** Shot segmentation,
tracking, and the whole viewer run without it. Installing it pulls torch, which
is multiple GB, so it is worth keeping that dependency isolated.

**OpenCV is the headless build.** The viewer renders in a browser, so no native
windows are needed. Swap to `opencv-python` if you want `cv2.imshow` while
debugging.

## Verification performed

`tools/make_test_clip.py` generates a clip with known ground truth: three
camera cuts at fixed frames, ten separated player-shaped blobs, one ball.
Running Stage 0 → 1 → 2 against it with `--detector colorblob`:

- shot boundaries landed exactly on the real cuts (0-59, 60-119, 120-179)
- exactly 10 player detections on every frame
- exactly 10 tracks per shot, each persisting all 60 frames, with tracker_ids
  restarting at 1 in each shot
- foot points equal to bottom-center of each box

The viewer was driven through Streamlit's `AppTest` harness: widgets, shot
selection, frame stepping, clip rendering and both tabs run without exceptions.

**What this does not prove.** It exercises the plumbing — schemas, per-shot
tracker resets, filtering, the viewer — and nothing else. Detector recall,
tracker ID stability under occlusion, and cut detection on real broadcast
transitions can only be judged on actual footage. Milestones 1 and 2 are not
closed until their "done when" criteria are met on a real game.

## Verification on real footage

`tools/make_test_clip.py` remains the regression gate and passes on all four
tracking backends. Beyond it, one 15-second half-court possession from a
1080p broadcast has been run end to end through Stages 0-2, and re-run under
a detector A/B and a four-way tracker comparison. `pipeline/track_quality.py`
prints the resulting scorecard after every Stage 2 run and renders it in the
viewer, so the CLI and UI cannot disagree.

That scorecard has already caught two real defects: tracker ids colliding
with the `BALL_TRACKER_ID = -1` sentinel (the `trackers` backends emit
unconfirmed detections as -1, which `sv.ByteTrack` never did), and a track
reported as spanning 107% of its shot, which is what surfaced the collision.

## Known gaps

- Real footage has been processed, but only one possession (465 frames of
  7,786). Nothing has run at scale.
- Milestone 1 is blocked on appearance-based re-identification; every
  available tracker associates on geometry alone and plateaus at 6 of 10.
- Ball recall was poor with COCO's `sports ball` class (64% of frames); a
  basketball-trained detector raised it to 86%, which helps Milestone 4.
- `camera_angle` is always `"unlabeled"`. `--skip-angles` is wired up but inert
  until something populates the field.
- Referee/crowd rejection now has three options: `--court-roi`, the
  `--court-polygon` / `--court-profile` trapezoid, and — most effective — a
  basketball-trained detector that emits `referee` as its own class and
  ignores the crowd outright. The polygon should be skipped when using such a
  detector; it clips real players. The colour-cluster approach in
  `07-known-challenges.md` still arrives with Milestone 3.
- `.claude/launch.json` pins an absolute interpreter path, because `streamlit`
  is not on `PATH` on the machine this was built on and a second Python
  install shadows `python` in some contexts. Re-point it if that changes.
