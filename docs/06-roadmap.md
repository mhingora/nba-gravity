# Roadmap

Ordered so that each milestone produces something inspectable, rather than
building the whole pipeline blind before seeing any output.

## Current status

| Milestone | State |
|-----------|-------|
| 1 — Detection + tracking | **Open.** Runs on real 1080p broadcast footage; 6 of 10 players hold one id for a full possession. Blocked on appearance-based re-identification. |
| 2 — Shot boundary segmentation | **Open.** Runs over full clips (47 and 32 shots). Never checked by eye against the video, which is its "done when". |
| 3 — Team classification | **Implemented.** 23 of 25 tracks on the test possession land in the visually correct kit cluster; silhouette 0.36. Viewer Tab 3 renders the crop grid for spot-checking. |
| 4 — Ball possession | **Implemented.** On the test possession the handler is visually correct wherever one is assigned; 46% of frames have a handler, and the limit is ball detection coverage, not the heuristic. |
| 5 — Court calibration | **Done for one camera.** Six landmarks annotated on S_N3_HD shot 11: rms reprojection error 1.3px, worst known distance off by 0.07ft, 52 of 52 projected players on court. `--propagate` then carries that frame's homography to every frame it can match, cutting median position error from 1.49ft to 0.03ft and calibrating other shots from the same camera with no extra annotation. A second camera angle still needs its own annotated frame. |
| 6 — Jersey OCR / identity | **Implemented.** 5 of 25 tracks on the test possession resolve a number; all 5 are correct, and the 12 tracks with no legible number are correctly left null. Two players a human can read (both #11) are missed. Viewer Tab 6 shows every crop with what OCR made of it. |
| 7 — First end-to-end run | **Runs end to end on a whole clip; not closed.** Propagation lifted coverage from one 15-second shot to 15 shots, and the chain now measures 914 frames and 3,578 player-frames at a median defender distance of 20.9ft, for 6 identified players. Deltas exist but only below the spec's 100-frame floor (at 20: #24 +6.3ft, Harper +0.0ft, #4 -2.8ft), so nothing here is trustworthy yet. Blocked on possession volume — 178 ball-frames across the clip. |
| 8-9 — Validation, scale-up | Blocked on 7. |

Coverage so far: the whole 4-minute test clip has now been detected and
tracked — 7,621 frames across 31 shots, 70,347 player rows — rather than the
one hand-picked possession earlier numbers were measured on. That is still one
clip of one game; nothing has run across games.

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
- **Done, then extended.** The spec's model is one matrix per camera angle.
  That is not enough for this footage: the camera pans *within* a shot, so the
  annotated matrix is out by 1.49ft at the median across the very shot it was
  fitted on. `05_calibrate.py --propagate` matches every frame back to the
  annotated frame and composes the camera motion, which brings that to 0.03ft
  and does not drift, because nothing is chained.
- The measurement is independent of the thing being measured: each annotated
  landmark is a patch of paint, so template-matching finds where it really is
  and compares that against where the homography predicted it.
- It also reaches 7 other shots of the test clip for free — every shot from
  the same camera — while refusing the 24 that are not. That refusal needed
  the broadcast overlay masked out: unmasked, the scorebug matches itself
  across any cut, and *every* shot "matched" the reference.

## Milestone 6 — Jersey OCR + identity resolution
- Implement OCR + majority vote on Milestone 1's clip.
- **Done when:** a reasonable fraction (start with any nonzero signal — don't
  expect high accuracy immediately) of tracks resolve to the correct player
  name, spot-checked manually against the roster.
- **Done.** `pipeline/jersey_ocr.py` behind `03_identify.py --ocr`. On the
  test possession: 5 numbers resolved, 5 correct (2 Harper, 32, 30
  Champagnie, 00, 5), 12 tracks with nothing legible correctly left null, 2
  legible-to-a-human #11s missed. Two of the five matched a roster name; the
  other three are Knicks whose name strips are not legible anywhere in the
  clip, so `data/rosters/NYK.json` deliberately does not list them.
- Every crop and its individual read is written to
  `outputs/identity/{game_id}_ocr.json` and rendered by viewer Tab 6, which
  is where the spot-check happens.
- What the numbers cost: three rules that each looked reasonable were wrong
  on real footage. See `04-identity-resolution.md` for the measurements.

## Milestone 7 — First end-to-end run, single game
- Wire all stages together on one full game.
- **Done when:** you have a `metrics/{game_id}_gravity.parquet` with
  plausible-looking numbers for at least a handful of players with enough
  possession volume to trust.
- **Built, not closed.** `pipeline/gravity.py` and `06_aggregate.py` run the
  whole chain and write both the deliverable table and the per-frame
  distances it averages. Viewer Tab 7 plots the trace with ball-possession
  frames shaded, which is the check that decides whether an aggregate
  deserves to be believed.
- What it produced on the test possession: 284 measured frames, 1,365
  player-frames, median defender distance 19.4ft and nearest 6.9ft. Two named
  players, whose raw gravity separates sensibly (Harper 16.4ft, Champagnie
  24.1ft).
- **Why it is still not closed.** Propagation removed the calibration
  bottleneck — the whole clip is now detected, tracked and identified, 15
  shots are calibrated, and 6 players are measured rather than 2. What is
  left is possession volume: only 178 frames in the clip have an identified
  handler on a measured frame, so the best-covered player has 47 with-ball
  frames against the spec's floor of 100. At a floor of 20 the deltas are
  +6.3ft, +0.0ft and -2.8ft — real numbers from the real pipeline, on far too
  little evidence to believe.
- Closing it needs more footage, not more code. The same chain run over a few
  more clips would put several players past the floor.
- Two bugs that only a whole clip could expose: the `other` cluster — tracks
  whose colour favoured neither kit — was voting for which team was on
  offence, and was then being counted among the defenders. Both are fixed;
  one shot's offence call went from 79% to 100% once `other` stopped voting.

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
