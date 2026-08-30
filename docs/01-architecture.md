# Architecture

> **Status.** The caching convention, shot segmentation and per-shot tracking
> described here are implemented (Stages 0-2). Identity resolution, calibration,
> possession and aggregation are still design only. One correction: `sv.ByteTrack`
> is deprecated and due for removal in supervision 0.31. Tracking now comes
> from the `trackers` package (`--tracker`, default `bytetrack`), so
> supervision is no longer pinned. See `09-implementation-notes.md`.

## Goals and non-goals

**Goal:** produce, per player, per game (and aggregated across games), the average
court-space distance of the 5 defenders when that player has the ball vs. when
they don't.

**Non-goals (v1):**
- Real-time processing. This is an offline batch pipeline.
- Perfect identity resolution on every frame. Gaps are acceptable; wrong labels
  are not.
- Handling every camera angle a broadcast might use. Start with the dominant
  "main baseline" angle and expand later.

## High-level data flow

```
raw_video/*.mp4
      │
      ▼
┌─────────────┐
│ 01_detect   │  per-frame player + ball bounding boxes
└──────┬──────┘
       ▼
┌─────────────┐
│ 02_track    │  ByteTrack within each continuous camera shot
└──────┬──────┘
       ▼
┌───────────────────────────┐
│ 03_identify  │ 05_calibrate│  (independent, both consume tracks)
│ jersey OCR + │ homography  │
│ team cluster │ per angle   │
└──────┬───────┴──────┬──────┘
       ▼               ▼
   tracker_id →    pixel coords →
   player_name      court coords
       │               │
       └───────┬───────┘
               ▼
       ┌───────────────┐
       │ 04_ball_       │  possession intervals per tracker_id
       │ possession     │
       └───────┬────────┘
               ▼
       ┌───────────────┐
       │ 06_aggregate   │  final gravity metric
       └───────────────┘
```

Note that identity resolution and court calibration are **independent of each
other** and both depend only on tracking output — they can be built and tested
in parallel, and either can fail without breaking the other. This is
intentional: geometry (where is everyone, are they close together) should never
be gated on knowing *who* everyone is.

## Why cache every stage to disk

Detection and tracking are the GPU-expensive steps. Everything downstream
(identity resolution, calibration, possession logic, aggregation) is iterated
on constantly while you tune heuristics. If stage 1–2 outputs aren't cached,
every tuning pass re-runs the expensive part for no reason.

Convention: every stage reads one or more parquet/JSON files and writes one or
more parquet/JSON files under `outputs/`, keyed by `game_id`. No stage should
require another stage's in-memory objects — only its on-disk output. This
also makes it trivial to re-run just one stage, inspect intermediate output by
hand, or swap an implementation (e.g. try a different OCR model) without
touching the rest of the pipeline.

## Shot-boundary segmentation

Broadcast video cuts angles constantly — every possession change, replay,
timeout, and foul call triggers a new camera shot. `tracker_id` continuity
from `sv.ByteTrack()` only holds *within* a shot, not across cuts.

Before tracking, run a shot-boundary detector (frame-to-frame histogram or
pixel-difference threshold is sufficient — no need for anything fancy) and
split the video into shot segments. Track each segment independently. This
means the same physical player will get a *new* `tracker_id` every time the
camera cuts — identity resolution (jersey OCR + team color) is what stitches
those back into a consistent player identity across cuts, not tracking itself.

## Camera angle tagging

Tag each shot segment with a rough camera-angle label (e.g. `main_baseline`,
`sideline`, `isolation_closeup`, `replay`). This can start as a manual lookup
or a simple classifier later. Angle tagging matters because:
- Court homography is computed per angle type, not per shot.
- Isolation/closeup shots usually don't contain enough of the court or enough
  players to be useful for the gravity metric — filter them out early rather
  than letting them inject noise.

## Failure isolation principle

Every stage should degrade to "no data for this frame/tracker" rather than a
wrong guess propagating downstream silently. See `07-known-challenges.md` for
the specific failure modes this addresses.
