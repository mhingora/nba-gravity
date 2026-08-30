# UI Design — Testing & Debugging Viewer

## Purpose

This UI exists primarily so you can **see** what each pipeline stage is doing
before trusting its output. Given the roadmap in `06-roadmap.md`, every
milestone ends with some version of "done when you can visually confirm X" —
that confirmation needs a viewer, not a notebook full of printed coordinates.
Treat this as a debugging instrument first, and a "cool results dashboard"
second. The exploration/leaderboard features from the earlier discussion are
real but lower priority than just being able to look at what Stage 1 and
Stage 2 produced.

**Recommended tool: Streamlit.** Single-user, local, no auth, minimal
boilerplate, and it can render annotated video frames and interactive charts
without you writing any frontend code. Revisit this choice only if the
project later needs multi-user access or hosted deployment — neither applies
yet.

## Build status

Implemented in `app.py`. Run it with:

```
python -m streamlit run app.py
```

| Tab | State |
|-----|-------|
| 1 — Detection Viewer | Built |
| 2 — Tracking Viewer | Built |
| 3-7 | Placeholder naming the milestone that unlocks them |

Two things landed differently from the sketch below, both noted in place:
the annotated clip renders as a **GIF rather than mp4**, and the shot-boundary
table lives in Tab 2 but renders as soon as Stage 0 has run, before tracking
exists — that table is how you pick a possession to point Stage 1 at.

## Design principle: one tab per pipeline stage, not one tab per "feature"

Because you'll be building and validating stages in the order laid out in
the roadmap, the UI should let you inspect **each stage's output in
isolation**, independent of whether later stages exist yet. Don't build a
single "results" view that assumes the whole pipeline is done — build it so
Milestone 1 is already testable through this UI before Stage 3 exists.

```
app.py
├── tab: Detection Viewer         # usable after Milestone 1
├── tab: Tracking Viewer          # usable after Milestone 1
├── tab: Team Classification      # usable after Milestone 3
├── tab: Ball Possession          # usable after Milestone 4
├── tab: Court Calibration        # usable after Milestone 5
├── tab: Identity Resolution      # usable after Milestone 6
└── tab: Gravity Results          # usable after Milestone 7+
```

Each tab reads directly from the corresponding `outputs/*.parquet` file (see
`02-data-schemas.md`) — it never talks to pipeline internals, never re-runs
processing itself. The UI is a read-only viewer over cached stage output.
This keeps it fully decoupled: you can run pipeline stages from the command
line and just refresh the UI to see results, with no risk of the UI and the
pipeline getting out of sync on logic.

## Tab 1 — Detection Viewer (build this first)

**What it needs to answer:** are the detector's boxes actually landing on
players and the ball, frame by frame?

- Video/game/shot selector.
- Frame scrubber (slider), or "step forward/back N frames" buttons — you
  need to move frame-by-frame, not just scrub coarsely, to catch things like
  a missed ball detection during a pass.
- Render the current frame with `sv.BoxAnnotator` drawing raw detection boxes
  from `outputs/detections/{game_id}.parquet`, color-coded by class
  (`player` vs `ball`).
- Show confidence score as a label on each box — low-confidence boxes are
  exactly what you're hunting for when tuning the detector.
- A simple frame-level stat line: "8 player detections, 1 ball detection,
  min confidence 0.41" — lets you spot systematically bad frames quickly
  without eyeballing every single one.

## Tab 2 — Tracking Viewer

**What it needs to answer:** do tracker_ids stay stable across a shot, and
where do they break?

- Same frame scrubber, but boxes are now colored/labeled by `tracker_id`
  instead of class — consistent color per ID across frames is the whole
  point, since an ID flickering color (or two players swapping IDs) is
  exactly the failure mode you're checking for.
- **Play as video, not just frame-by-frame** — ID stability is much easier to
  judge watching motion than stepping through stills. Render the annotated
  shot as a short mp4/gif on demand (using `sv.VideoSink` or similar) and let
  it play back in the UI.
- Shot boundary markers on a timeline so you can jump directly to the start
  of any shot rather than scrubbing through an entire quarter to find cuts.
- A per-shot table: `tracker_id`, `track_len`, first/last frame — sortable,
  so you can quickly spot suspiciously short-lived tracks (likely noise) or
  a shot with way more tracker_ids than 10 (likely ID switching/fragmenting).

## Tab 3 — Team Classification Viewer

- For a selected shot, show a grid of cropped torso images (the same crops
  used for clustering), grouped visually by assigned cluster/team.
- **Done-when signal for this tab:** you glance at the two groups and they
  obviously correspond to the two teams' actual jersey colors — this is a
  five-second visual check, not something you need charts for.
- Flag anything landing in an "unclustered/noise" bucket (referees, etc.) so
  you can confirm the noise filter is working rather than silently
  mis-assigning a referee to a team.

## Tab 4 — Ball Possession Viewer

- Render the annotated clip (as in Tab 2) but overlay which `tracker_id` is
  currently flagged as `ball_handler_tracker_id`, plus a small live-updating
  number showing `ball_distance_px` — watching this number spike and drop as
  passes happen is the fastest way to sanity-check your threshold and
  debounce values.
- A possession timeline strip beneath the video: colored segments showing
  who had the ball across the whole shot, so you can confirm possession
  changes line up with what you see happening in the video (a pass, a
  rebound) rather than flickering randomly.

## Tab 5 — Court Calibration Viewer

- Show the reference frame for a camera angle with your annotated keypoints
  overlaid, next to a top-down court diagram showing those same points
  projected through the computed homography.
- Display `reprojection_error_px` prominently — this is the single number
  that tells you whether to trust this angle's calibration at all.
- Optional but very useful: overlay a top-down "radar view" of all tracked
  players' projected court positions for a given frame, next to the raw
  broadcast frame — this is the clearest possible visual check that your
  homography is behaving (do the dots roughly match where players actually
  are on the court?).

## Tab 6 — Identity Resolution Viewer

- Per track: show the sampled OCR crops, each crop's individual OCR read,
  and the final majority-vote result plus `identity_confidence`.
- This directly exposes *why* a track resolved (or failed to resolve) to a
  given player — critical for tuning the OCR/voting approach rather than
  treating identity resolution as a black box.
- A simple accuracy-checking workflow: let yourself mark a track's resolved
  name as "correct" / "wrong" / "should be null" while browsing — even a
  crude manual tally here gives you a real precision estimate to decide if
  the `identity_confidence` threshold needs adjusting.

## Tab 7 — Gravity Results

This is the one from the earlier discussion — player selector, headline
numbers (`gravity_delta`, `avg_defender_distance_overall`, etc.), the
defender-distance time-series chart with possession segments highlighted,
and a leaderboard view across processed games. Lowest priority to build,
since it depends on every earlier stage actually working, but the schema for
it (`outputs/metrics/*.parquet`) is already defined in `02-data-schemas.md`
so there's nothing new to design here beyond the display itself.

## What NOT to build into this

- No authentication, no multi-user support, no deployment/hosting concerns —
  this runs on your machine while you develop.
- No video upload handling inside the UI — point it at files already in
  `data/raw_video/`, keep ingestion a filesystem operation, not a UI feature.
- No pipeline-triggering from within the UI for v1 — run stages from the
  command line, refresh the UI to inspect results. Wiring "click a button to
  re-run Stage 3" is a nice-to-have once the stages themselves are stable,
  not before.

## Suggested build order for the UI itself

Build Tab 1 and Tab 2 first, in parallel with Milestone 1 of the main
roadmap — you need them immediately to judge whether your detector and
tracker are any good. Add each subsequent tab exactly when its corresponding
milestone starts, not before. This means the UI grows alongside the pipeline
rather than being a separate large upfront build.
