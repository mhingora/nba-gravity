# NBA Player Gravity Tracker

A computer-vision project that measures **offensive gravity** — how much defensive
attention a player commands simply by being on the court — using broadcast game
footage, `supervision`, and a custom detection/tracking pipeline.

## Core research question

> When Player X has the ball, how tightly do the 5 defenders collapse toward them,
> compared to when Player X does *not* have the ball?

The gap between those two numbers is a proxy for "gravity": stars who pull
defenders even off the ball, or who force heavy attention on catch, should show
a bigger delta than role players.

## Results so far

Measured on one 15-second half-court possession from a 1080p broadcast, using
the scorecard that `pipeline/track_quality.py` prints after every run.

**A basketball-trained detector beat a generic one 20x its size.** Swapping
COCO `yolov8x` (68M params) for a basketball-trained `yolov8n` (3.15M):

| | COCO yolov8x | basketball yolov8n |
|---|---|---|
| Player detections per frame | 92 | **14** |
| Median confidence | 0.19 | **0.83** |
| Frames with the ball detected | 64% | **86%** |

92 boxes per frame was the detector finding the entire arena. 10 players were
on court.

**Source resolution mattered more than any parameter.** The same possession,
same settings, at 848x480 versus 1920x1080:

| | 480p | 1080p |
|---|---|---|
| Frames with 10+ players tracked | 18% | **99%** |
| Longest unbroken track | 79% of shot | **99%** |

**Motion-based tracking has a ceiling here, and it is not a tuning problem.**
Four trackers — supervision ByteTrack, and ByteTrack / OC-SORT / BoT-SORT from
the `trackers` package — each swept over buffer and association thresholds,
all plateau at **6 of 10** players holding a single id for a full possession.
Every one of them associates on motion and geometry alone. Two players who
swap positions mid-crossing are indistinguishable without appearance
features, which is what closing the remaining gap requires.

Counter-intuitively, the court-polygon crowd filter built for the COCO
detector *hurts* with a basketball detector: it was compensating for crowd
detections that no longer happen, and it clips real players instead. Kept for
generic weights, skipped otherwise.

## Status

**No milestone is closed yet.** Stages 0-4 run on real broadcast footage —
shot segmentation, detection, tracking, team classification and ball
possession. Stages 5-6 are stubs, so there is no gravity number yet and the
research question above is unanswered.

Detection and tracking have run on **one hand-picked possession** — 465 of
7,786 frames, about 6% of a single clip. Shot segmentation has run over full
clips. Nothing has run at scale.

See `docs/06-roadmap.md` for per-milestone state and `docs/09-implementation-notes.md`
for what was built and why.

## Repo structure

```
nba-gravity/
├── README.md                    # you are here
├── app.py                       # Streamlit debugging viewer (docs/08)
├── docs/
│   ├── 01-architecture.md       # system design, pipeline stages, data flow
│   ├── 02-data-schemas.md       # on-disk formats for every stage's output
│   ├── 03-pipeline-stages.md    # detailed spec for each processing stage
│   ├── 04-identity-resolution.md# jersey OCR + team clustering deep dive
│   ├── 05-metrics-and-analysis.md # gravity metric definition + stats notes
│   ├── 06-roadmap.md            # build order / milestones
│   ├── 07-known-challenges.md   # broadcast-specific failure modes
│   ├── 08-ui-design.md          # Streamlit testing/debugging viewer, per stage
│   └── 09-implementation-notes.md # what's built, deviations from spec, verification
├── data/
│   ├── raw_video/                # source mp4s, one folder per game_id
│   ├── rosters/                  # team_id -> {jersey_number: player_name}
│   └── calibration/               # court profiles + keypoint refs, per camera angle
├── pipeline/
│   ├── common.py                 # paths, schemas, video IO (shared)
│   ├── shot_boundaries.py        # Stage 0 cut detection
│   ├── detector.py               # detector wrappers for Stage 1
│   ├── 01_detect.py             # ✅ implemented
│   ├── 02_track.py              # ✅ implemented
│   ├── 03_identify.py           # stub
│   ├── 04_ball_possession.py    # stub
│   ├── 05_calibrate.py          # stub
│   └── 06_aggregate.py          # stub
├── tools/
│   └── make_test_clip.py         # synthetic clip for smoke-testing
├── outputs/
│   ├── detections/
│   ├── tracks/
│   └── metrics/
└── analysis/                     # notebooks for exploring results
```

## Build order (short version — see `docs/06-roadmap.md` for full detail)

1. Detection + tracking on one game, half-court possessions only.
2. Team classification (which tracked players are on defense).
3. Ball possession heuristic.
4. Court homography / calibration.
5. Jersey-number identity resolution.
6. Gravity metric aggregation + first results.
7. Scale to more games, tighten edge cases.

## Running it

```bash
pip install -r requirements.txt
```

The repo is location-independent — every path resolves from the repo root, so
it can be moved or cloned anywhere. Two things are machine-specific: the Python
environment you run it with (if `streamlit` is not on your `PATH`, use
`python -m streamlit run app.py`), and the interpreter path pinned in
`.claude/launch.json`.

Put a video at `data/raw_video/{game_id}/{game_id}.mp4`, then run the stages.
Each one reads and writes only on-disk artifacts, so you can re-run any single
stage without redoing the expensive ones.

Segment camera shots without loading a detector (fast, CPU-only):

```bash
python pipeline/01_detect.py --game-id 0022500123 --shots-only
```

Detect over one possession, then track. Stock COCO weights (`person` →
`player`, `sports ball` → `ball`) are the Milestone 1 starting point:

```bash
python pipeline/01_detect.py --game-id 0022500123 --start-frame 1200 --end-frame 1650
```

```bash
python pipeline/02_track.py --game-id 0022500123
```

Broadcast footage puts thousands of spectators in frame, and a stock COCO
detector finds them. Reject them with a court polygon — the playing surface as
a trapezoid, matching what the camera actually sees. Fit it interactively in
the viewer's Tracking tab ("Court polygon tuner"), which prints the exact flag
to paste:

```bash
python pipeline/02_track.py --game-id 0022500123 --court-profile msg_main_wide
```

A **court profile** (`data/calibration/{name}.json`) holds that polygon plus
the tracking backend and settings tuned against the same footage. It is keyed
by camera angle, not by game: the same profile applies to every game shot from
that camera position, so you fit it once. Any flag passed explicitly still
overrides the profile.

### Detector and tracker choice

A basketball-trained detector beats stock COCO weights by a wide margin — a
`yolov8n` trained on basketball outperformed COCO `yolov8x`, a model 20x its
size, on every measure. Point `--model` at one; no code change is needed,
since `detector.py` maps class names through a lookup table.

Model weights are not in this repo (they are downloadable artifacts, and
`yolov8x.pt` exceeds GitHub's file size limit). Fetch the basketball detector
with:

```bash
curl -L -o models/BODD_yolov8n_0001.pt https://huggingface.co/GabrieleGiudici/E-BARD-detection-models/resolve/main/BODD_yolov8n_0001.pt
```

Then pass `--model models/BODD_yolov8n_0001.pt`. Stock COCO weights download
themselves on first use. The E-BARD model is CC-BY-4.0; it emits `player`,
`referee`, `basketball` and `hoop`, and `detector.py` keeps the first and
third while dropping the rest.

`--tracker` selects the tracking backend:

| backend | notes |
|---------|-------|
| `sv-bytetrack` | supervision's `sv.ByteTrack`. The default, deprecated upstream |
| `bytetrack` | the `trackers` package port. Best on broadcast footage here |
| `ocsort` | observation-centric recovery; lower recall on this footage |
| `botsort` | camera motion compensation; the only backend that reads frames |

Backends do not share a parameter vocabulary — supervision's
`minimum_matching_threshold` is an IoU *distance* ceiling while trackers'
`minimum_iou_threshold` is an IoU *floor*, so the same number means opposite
things. Pass backend-specific settings with repeatable `--tracker-arg
KEY=VALUE`, or store them in a profile.

Every run prints a per-shot quality line — track count, players per frame,
longest track, and how many tracks span the whole shot — so you can tell a
detector recall problem from an ID-fragmentation one without opening anything.
The same numbers appear as a table in the viewer's Tracking tab.

Then inspect the results in the viewer:

```bash
streamlit run app.py
```

Sort the tracked players into teams (Milestone 3) — gravity is measured
against defenders, so this has to happen before any distance means anything:

```bash
python pipeline/03_identify.py --game-id 0022500123
```

Clusters are named `light` / `dark` by kit brightness rather than an arbitrary
`team_a` / `team_b`, so labels stay stable across runs. Tracks whose colour
favours neither team become `other` — referees, coaches, anyone the detector
picked up. Check the result in the viewer's Team Classification tab: the two
crop groups should obviously be the two kits.

Then work out who has the ball each frame (Milestone 4):

```bash
python pipeline/04_ball_possession.py --game-id 0022500123
```

Frames where nobody is within reach report a null handler, which is the
correct answer for a ball in flight rather than a failure. Check it in the
viewer's Ball Possession tab — the highlighted player should be the one
holding the ball, and possession changes should line up with passes.

No footage yet? Generate a synthetic clip with known ground truth (3 camera
cuts, 10 players, 1 ball) and run the whole path against it:

```bash
python tools/make_test_clip.py --game-id TESTCLIP && python pipeline/01_detect.py --game-id TESTCLIP --detector colorblob && python pipeline/02_track.py --game-id TESTCLIP
```

## Footage and licensing

No video is included in this repository, and `.gitignore` blocks
`data/raw_video/` so none is committed by accident. Broadcast footage is
licensed content — supply your own, and treat it as local-only. The pipeline
reads whatever you put in `data/raw_video/`; nothing about it is specific to
a particular game or broadcaster.

Model weights are likewise excluded (see above for how to fetch them). The
code in this repo is MIT licensed; the models and any footage you use carry
their own terms.

## Dependencies

See `requirements.txt`. `ultralytics` is only needed for Stage 1 detection —
shot segmentation, tracking, and the viewer run without it.
