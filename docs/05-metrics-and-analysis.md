# Metrics and Analysis

## Primary metric: gravity delta

For a given player P, across all valid frames where P is on court:

```
avg_dist_with_ball    = mean(avg distance of 5 defenders to P | P has ball)
avg_dist_without_ball = mean(avg distance of 5 defenders to P | P does not have ball)

gravity_delta = avg_dist_without_ball - avg_dist_with_ball
```

Interpretation: a larger positive delta means defenders sit further away when
P doesn't have the ball, and collapse in tighter when P gets it — i.e. P's
presence with the ball meaningfully changes defensive shape. A small delta
means defenders treat P about the same whether they have the ball or not
(could mean low gravity, or could mean defenders are already playing tight to
P off-ball because *of* their gravity — see caveats below).

## Important caveat: this metric alone can't distinguish two different things

A player who is **so dangerous off-ball** that defenders never leave them,
ball or not, will show a *small* gravity_delta even though they clearly have
high gravity — because the "without ball" distance is already low. This
metric as defined captures the *marginal* effect of catching the ball, not
total defensive attention. Worth computing a secondary, complementary metric
alongside it:

```
avg_defender_distance_overall = mean(avg distance of 5 defenders to P), regardless of possession
```

Comparing `avg_defender_distance_overall` across players tells you who
defenders generally play tight to (raw gravity), while `gravity_delta` tells
you who specifically changes defensive shape *by catching the ball* (marginal
gravity). Both are legitimate and answer different questions — report both
rather than picking one.

## "Nearest defender" as a secondary metric

Average distance across all 5 defenders can wash out a real effect: help
defense collapsing from the weak side while the primary defender stays put
would show up faintly in a 5-defender average but strongly in "distance to
single nearest defender." Consider tracking both:

- `avg_distance_all_5_defenders` — captures whole-defense collapse (help
  defense, weak-side tagging)
- `distance_nearest_defender` — captures primary, on-ball defensive pressure

These can diverge meaningfully and both are informative.

## Filtering before trusting a number

Apply these filters at aggregation time, not analysis time — bad rows
shouldn't even reach the output table:

- Minimum frames per bucket per player per game (e.g. require ≥ 100 frames in
  both the "with ball" and "without ball" bucket before including a
  player-game row — a player with 8 possession frames from garbage time
  shouldn't produce a headline stat).
- `identity_confidence` above threshold (see `04-identity-resolution.md`).
- `reprojection_error_px` below threshold for the shot's homography.
- Exclude frames from `replay` or `isolation_closeup` shot types.
- Consider excluding clear fast-break / transition frames if you want the
  metric to reflect half-court defensive shape specifically — transition
  defense spacing is a different phenomenon and mixing it in adds noise.
  This can be a v2 refinement rather than a v1 requirement.

## Normalization considerations

- **Court position matters independent of gravity.** A player standing in the
  corner naturally has fewer defenders near them than a player standing at
  the elbow, regardless of gravity — the corner is farther from the help-side
  defenders geometrically. Consider whether you want to control for on-ball
  location (e.g. bucket by court zone) before comparing raw distances across
  players who occupy different areas of the floor.
- **Minutes and possession volume differ hugely across players.** Always
  report frame counts alongside averages, and consider a minimum-sample
  threshold for any "leaderboard" style output.
- **Game context** (score differential, garbage time, opponent defensive
  scheme) will add noise you likely can't fully control for in v1 — flag it
  as a known limitation rather than trying to solve it immediately.

## What the implementation measured

Stage 6 is built (`pipeline/gravity.py`, run by `06_aggregate.py`). Three
decisions the spec leaves open turned out to matter more than the arithmetic,
and each was settled against the test possession rather than by argument.

**Which team is defending.** The only available signal is the ball handler,
and taken frame by frame it is wrong: on the test possession the handler's
team flips four times in fifteen seconds. The broadcast scorebug settles it —
the shot clock counts 14 → 9 → 8 → 7 with no reset, so San Antonio had the
ball throughout, and the flips are a Knicks defender momentarily being the
player nearest a contested ball. Following them would have inverted who counts
as a defender for those frames. Offence is therefore decided once per camera
shot, by majority of handler frames (70/30 here), and a shot whose handler
frames do not reach a clear majority is skipped rather than guessed at.

**How many defenders.** After projection, a third of frames show six to nine
defenders on the floor. They are real people in real positions; the surplus is
one player carrying two tracker_ids, which weights that player twice in the
average. Only frames with exactly five tracked defenders are measured — 294 of
464 on the test possession. This is a filter, not a fix: the fix is Milestone
1's re-identification problem.

**Which shots can be projected at all.** Stage 5 writes one homography per
shot, but every shot of a game gets the same matrix, so its
`reprojection_error_px` says nothing about a shot the landmarks were not
annotated on. Only the annotated shot is aggregated unless `--all-shots` says
otherwise.

**Referees are not defenders.** Stage 3 labels a track whose colour favours
neither kit as `other`. Treating "not the offence" as "the defence" quietly
counted those tracks among the five defenders, and let a frame whose *handler*
was an `other` track vote on which team was attacking — on a whole clip that
produced a shot reported as "other on offence", which makes both teams
defenders. Offence votes and defender sets are now restricted to the two real
team labels.

### What the clip produces

Over the whole 4-minute clip — 7,621 tracked frames, 417 tracks, 126 jersey
numbers read, 15 shots calibrated by propagation — the chain measures 914
frames and 3,578 player-frames, at a median defender distance of 20.9 ft and a
median nearest defender of 7.3 ft. Six players are identified well enough to
carry a row.

Three of them have frames in both buckets:

| player | with ball | without ball | gravity_delta | nearest-defender delta |
|---|---|---|---|---|
| #24 (SAS) | 23 | 613 | +6.27 ft | −2.68 ft |
| Harper (#2) | 25 | 482 | +0.01 ft | +1.65 ft |
| #4 (SAS) | 47 | 63 | −2.75 ft | −2.18 ft |

Every one of those is below the 100-frame floor this document asks for, so the
default run writes `gravity_delta` as null and reports the counts; the table
above comes from `--min-bucket-frames 20`. Treat it as a demonstration that
the arithmetic reaches the end, not as a measurement of anybody's gravity.

Harper is the case this document predicted: his five-defender average barely
moves when he catches the ball, while his nearest defender closes by 1.7 ft.
Whole-defence collapse and on-ball pressure are different things, and reporting
only the first would have shown nothing at all.

The limit is now possession volume, not calibration: only 178 frames in the
clip have an identified handler on a measured frame. More clips through the
same chain is what pushes players past the floor.

## Suggested output views

1. **Per-player leaderboard** — `gravity_delta` and
   `avg_defender_distance_overall`, sorted, with frame counts shown alongside
   so low-sample rows are visibly less trustworthy rather than hidden.
2. **Per-player, per-game table** — for sanity-checking whether a player's
   number is stable across games or driven by one outlier game with bad
   calibration/identity resolution.
3. **Distance-over-time trace for a single possession** — a debugging/
   validation view: plot the 5 defender distances frame-by-frame across one
   possession, overlaid with when the target player has the ball. Useful for
   confirming the pipeline is doing something sensible before trusting
   aggregate numbers.
