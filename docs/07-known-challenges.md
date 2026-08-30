# Known Challenges

Failure modes specific to using broadcast footage for this project, and how
the design above addresses (or deliberately doesn't yet address) each one.

## Camera cuts break tracking identity
**Problem:** every possession change, foul, replay, or timeout cuts to a new
angle, resetting `tracker_id`.
**Mitigation:** shot boundary segmentation resets tracking per-shot rather
than letting ByteTrack try to bridge an impossible gap; identity resolution
re-establishes player identity per-shot instead of relying on tracker
continuity across the whole game.

## Camera zoom/pan changes scale mid-shot
**Problem:** even within one shot, broadcasts often push in on a player
during a big moment, changing pixel-to-court-distance scale.
**Mitigation:** `reprojection_error_px` per shot is your signal — if it's
poor, exclude the shot rather than trust a stale homography. This is a real
limitation of v1; a per-frame dynamic homography is a plausible v2 upgrade if
static-camera assumption proves too limiting.

## Player occlusion in the paint
**Problem:** five offensive + five defensive players clustering near the rim
causes heavy bounding-box overlap, which both degrades detection confidence
and confuses tracking (ID switches between two overlapping players).
**Mitigation:** `track_len` filtering catches the worst cases (a short,
choppy track is a signal something went wrong); no full fix in v1 — expect
noisier data specifically for possessions with heavy paint congestion, and
consider it a known limitation worth stating explicitly if you publish
results.

## Jersey number visibility
**Problem:** numbers are frequently not visible (facing away from camera,
occluded by arms/ball, too small at wide-shot distance).
**Mitigation:** majority-vote OCR across a whole track, explicit `null`
fallback rather than guessing — see `04-identity-resolution.md`. Expect
partial coverage, not full coverage, and design the aggregation stage to
tolerate that gracefully.

## Ball detection reliability
**Problem:** the ball is small, moves fast, and blurs heavily during passes
and shots — the hardest object in the frame to detect reliably.
**Mitigation:** consider a dedicated ball detector (possibly at higher
resolution or with temporal smoothing/interpolation across a few frames where
detection drops out entirely) rather than treating it as a third class on the
same model tuned for player-sized boxes.

## Referees and non-player people getting detected
**Problem:** referees wear distinct uniforms but are still "person"-shaped
and can get picked up by a general player detector; ball boys, coaches near
the sideline are similar risks.
**Mitigation:** filter by color cluster (referees' uniform color should form
a distinct third cluster from both teams — treat anything that doesn't
cleanly join one of the two team clusters as noise) and by court-region
bounds (people consistently outside the playing surface bounds aren't
players).

## Copyright / usage
**Problem:** this is licensed broadcast content.
**Mitigation:** not a technical problem — keep processing and any derived
outputs for personal research/learning use, consistent with your League Pass
terms, rather than redistributing raw clips or full-broadcast derivative
video.
