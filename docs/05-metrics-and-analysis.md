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
