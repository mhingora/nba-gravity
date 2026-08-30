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
pipeline/team_clustering.py     Stage 3 Part A — kit-colour clustering
pipeline/03_identify.py         Stage 3 CLI (Part A done, Part B stub)
pipeline/possession.py          Stage 4 — ball selection + possession logic
pipeline/04_ball_possession.py  Stage 4 CLI
pipeline/court_geometry.py      Stage 5 — court landmarks, homography, checks
pipeline/05_calibrate.py        Stage 5 CLI
pipeline/06_aggregate.py        stub — docstrings and NotImplementedError
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

### Stage 3 deviations

**Clustered once per game, not per shot.** `04-identity-resolution.md`
describes clustering each shot then anchoring the clusters to a per-game
reference. Features from every track are clustered together instead, so each
track takes its label from the nearest global centroid and cross-shot
consistency is automatic rather than a second matching step that can fail. One
game is one arena under one lighting rig; if that ever stops holding, the
spec's per-shot-plus-anchor scheme is the fallback.

**Teams are named `light` / `dark`, not `team_a` / `team_b`.** Assigned by
cluster brightness, so the label is stable across runs — an arbitrary label
flips whenever k-means seeds differently — and readable when spot-checking.

**`other` is decided by ambiguity, not distance rank.** The first version
evicted a fixed top percentile of centroid distances, which is not a
confidence test: it relabels the same fraction whatever the input. Spot-
checking the crop grid caught it discarding an unmistakable Spurs jersey. A
track is now `other` only when its distance to the rival centroid is barely
worse than to its own, so clean input can legitimately produce no outliers.

**`identity_confidence` carries only the team term.** The spec's formula is
`frame_agreement * clipped_silhouette`; with OCR unimplemented there is no
frame-agreement term, so the value is the clipped silhouette scaled by how
many frames yielded a usable crop, and describes confidence in the team label
alone.

### Stage 4 deviations

**A ball-selection step the spec does not mention.** The spec assumes one ball
position per frame. The detector supplies up to five: 243 of 398 ball-bearing
frames on the test possession carry more than one candidate. `select_ball`
resolves them by temporal continuity, since Stage 2 does not carry a
confidence column through for ball rows.

**A per-frame jump gate on the ball.** Continuity alone is not enough — a
single false positive in the crowd captures the ball track and drags every
later frame with it. Spot-checking produced handlers 250-290px from a "ball"
sitting among the photographers. Candidates further than
`--max-jump-frac` x median player height from the last accepted position are
refused, and after five consecutive refusals the anchor drops so a genuine
long pass can be re-acquired.

**Possession is not sticky.** An early version carried the last handler
through frames where the ball was missing or far away, which read as 96% of
frames having a handler. That contradicts the spec — a ball in flight belongs
to nobody — and would have invented possession that never happened. The
debounced id now suppresses flicker *between players* only; a frame with
nobody within reach reports null.

**The possession radius is a fraction of player-box height**, not a pixel
constant, so it scales with resolution. It also turns out barely to matter:
distances are strongly bimodal (median 0px, 75th percentile 13px, 90th
percentile 331px), so anything from 60px to 200px selects within three
percentage points of the same frames. The ball is either clearly held or
clearly in flight.

### Stage 5 deviations

**Keypoints live in the court profile, not a per-shot file.** The spec notes
that camera angles repeat across shots and suggests calibrating once per
angle. Since a profile is already keyed by camera angle and already holds the
court polygon, the annotated landmarks go there too. `05_calibrate.py` still
writes one output file per shot, as the schema specifies, so Stage 6 reads a
predictable path per shot and a single shot can later be re-fitted after a
mid-possession zoom without special-casing.

**Court axes run width-first.** Origin at a baseline corner as the stub
requires, with x along the baseline (0-50) and y down the length (0-94). This
is the opposite of the usual "x is the long axis" instinct, and getting it
backwards yields a homography that fits perfectly and means nothing, so it is
stated in the module docstring too.

**Reprojection error is reported in pixels, not feet.** Feet would flatter
distant landmarks, where a large ground error is a small pixel one — and
pixels are what you can actually see when checking the overlay.

**The known-distance check is weaker than it looks.** It measures real court
distances through the homography, which is the milestone's stated done-when,
but those landmarks were used to *fit* the homography, so it verifies internal
consistency rather than independent accuracy. It still catches a fit that
cannot reproduce the lane width, which is the failure that matters.

**`--max-error-px` refuses to write a bad calibration.** A wrong homography is
worse than none: it produces confident, wrong distances that look plausible
downstream.

## Known gaps

- Real footage has been processed, but only one possession (465 frames of
  7,786). Nothing has run at scale.
- No camera angle has been annotated for Stage 5, so no homography exists
  for real footage and every distance is still in pixels. The maths is
  verified against a constructed transform; the annotation is a human step.
- Stage 3 misassigns roughly 2 tracks in 25. Both observed failures were
  crops contaminated by background or an overlapping player rather than kit
  colour. The embedding-based feature in `04-identity-resolution.md` (SigLIP
  or similar) is the documented upgrade if colour histograms plateau.
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
