# Roadmap

Ordered so that each milestone produces something inspectable, rather than
building the whole pipeline blind before seeing any output.

## Current status

| Milestone | State |
|-----------|-------|
| 1 — Detection + tracking | **Open.** Runs on real 1080p broadcast footage; 6 of 10 players hold one id for a full possession. Blocked on appearance-based re-identification. |
| 2 — Shot boundary segmentation | **Open.** Runs over full clips (47 and 32 shots). Never checked by eye against the video, which is its "done when". |
| 3 — Team classification | **Implemented.** 23 of 25 tracks on the test possession land in the visually correct kit cluster; silhouette 0.36. Viewer Tab 3 renders the crop grid for spot-checking. |
| 4 — Ball possession | Not started (`04_ball_possession.py` is a stub). Ball detection now lands on 86% of frames, up from 64% with COCO weights. |
| 5 — Court calibration | Not started (`05_calibrate.py` is a stub). Needs a keypoint-annotation UI that does not exist yet. |
| 6 — Jersey OCR / identity | Not started. Measured constraint: player boxes are ~218px tall at 1080p, leaving roughly 76px of torso for OCR. |
| 7 — First end-to-end run | Blocked on 3-6. |
| 8-9 — Validation, scale-up | Blocked on 7. |

Coverage so far: shot segmentation has run over two full clips, but detection
and tracking have only ever run on **one hand-picked possession** — 465 of
7,786 frames, about 6% of a single clip. Nothing has been run at scale.

### What the real-footage runs established

Three findings, each measured rather than assumed:

1. **Source resolution dominated everything.** The same possession at 848x480
   versus 1920x1080 took frames-containing-10-tracked-players from 18% to 99%.
   No tracker tuning recovered what the detector missed at 480p.
2. **A basketball-trained detector beat a much larger generic one.** A
   basketball `yolov8n` outperformed COCO `yolov8x` — 20x its size — on every
   measure: detections per frame fell from 92 to 14 (the rest was crowd) and
   median confidence rose from 0.19 to 0.83. Referees and hoops come through
   as their own classes and are dropped automatically.
3. **Motion-only tracking has a ceiling here.** Four backends (supervision
   ByteTrack, trackers ByteTrack, OC-SORT, BoT-SORT), each swept over buffer
   and association thresholds, all plateau at 6 of 10 tracks spanning a
   possession. Every one associates on geometry alone. Two players who swap
   positions mid-crossing are indistinguishable without appearance features,
   and `trackers` dropped `ReIDModel` in v2.1.0 with ReID listed as planned,
   not present.

A court polygon (`--court-polygon` / `--court-profile`) was built to reject
crowd for the COCO detector and does so well. It is *counterproductive* with a
basketball detector, which already ignores the crowd — the polygon then clips
real players. Keep it for generic weights; skip it otherwise.

The synthetic-clip verification (`tools/make_test_clip.py`) still passes on
every backend and remains the regression gate: 10 tracks per shot, each
spanning all 60 frames.

## Milestone 1 — Detection + tracking on one clip
- Pick one half-court possession (10-15 seconds, minimal camera movement) from
  one of your downloaded games.
- Get a working detector (start with an off-the-shelf model or existing
  Roboflow Universe basketball dataset before investing in fine-tuning your
  own — validate the approach before optimizing the model).
- Run `sv.ByteTrack()` and visually confirm tracker_ids stay stable across the
  clip (annotate and export a video with `sv.BoxAnnotator` + `sv.LabelAnnotator`
  to eyeball it).
- **Done when:** you can watch an annotated clip and all 10 players keep a
  consistent ID for the whole possession.

## Milestone 2 — Shot boundary segmentation
- Implement the cut detector, run it against a full quarter.
- **Done when:** shot boundaries roughly match what you'd mark by eye
  scrubbing through the same quarter manually.

## Milestone 3 — Team classification
- Cluster tracked players into 2 teams by color/embedding on Milestone 1's
  clip.
- **Done when:** the two clusters visibly correspond to the two teams' jersey
  colors on a handful of spot-checked clips.

## Milestone 4 — Ball possession heuristic
- Add ball detection, implement the proximity + debounce logic.
- **Done when:** for a possession with a clear, unambiguous ball handler
  (isolation play, no quick passes), the pipeline correctly identifies who has
  it for the duration.

## Milestone 5 — Court calibration
- Manually annotate court reference points for your main camera angle,
  compute the homography, check `reprojection_error_px`.
- **Done when:** projecting a few known court locations (free-throw line,
  three-point arc) into court-space gives distances close to their known
  real-world values (free-throw line is 15 ft from backboard, etc. — use
  known court dimensions as ground truth).

## Milestone 6 — Jersey OCR + identity resolution
- Implement OCR + majority vote on Milestone 1's clip.
- **Done when:** a reasonable fraction (start with any nonzero signal — don't
  expect high accuracy immediately) of tracks resolve to the correct player
  name, spot-checked manually against the roster.

## Milestone 7 — First end-to-end run, single game
- Wire all stages together on one full game.
- **Done when:** you have a `metrics/{game_id}_gravity.parquet` with
  plausible-looking numbers for at least a handful of players with enough
  possession volume to trust.

## Milestone 8 — Validation pass
- Spot-check the pipeline's output against what you'd expect from watching
  the actual game — do the players you'd subjectively call "high gravity"
  (based on watching them play) show up with larger deltas than role players?
- This is the point to revisit filtering thresholds (identity_confidence,
  reprojection error, minimum frame counts) based on what you're seeing.

## Milestone 9 — Scale to multiple games
- Once single-game output looks trustworthy, run across more games from your
  League Pass downloads and start building the aggregated leaderboard.

## Later / v2 ideas (not required for a working v1)
- Court-zone bucketing to control for on-ball location (see
  `05-metrics-and-analysis.md`).
- Transition vs. half-court separation.
- Automated camera-angle classification instead of manual tagging.
- Positional-inference identity backfill for unresolved tracks.
- A small dashboard/notebook for browsing per-possession traces interactively.
