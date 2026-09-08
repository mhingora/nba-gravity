# Identity Resolution

This is the hardest, most failure-prone part of the whole project. The goal
is: given a `(shot_id, tracker_id)`, figure out which real player it is,
*or* correctly conclude "unknown" rather than guess wrong.

## Why this is hard

- Jersey numbers are only visible from certain angles (back number is often
  hidden, front number is small and frequently occluded by arms/ball).
- Motion blur during fast breaks degrades OCR badly.
- Two teams can share the same jersey number, so number alone is ambiguous.
- The same physical player gets a *new* `tracker_id` every camera cut, so
  identity has to be re-resolved constantly rather than solved once per game.

## Two-part approach

### Part A — Team classification (color clustering)

1. For each tracked player, crop the torso region (upper ~40% of the bbox,
   avoiding shoes/shorts which have more inter-team color noise) across
   several frames of the track.
2. Extract a color representation — a simple approach is dominant-color via
   k-means on pixel colors within the crop; a more robust approach (closer to
   what Roboflow's soccer example does) is to embed crops with a pretrained
   vision model (e.g. SigLIP) and cluster the embeddings, which is more
   robust to lighting/shadow variation than raw color histograms.
3. Cluster into exactly 2 groups (+ a "referee/other" outlier bucket if
   referees are getting picked up by your player detector — filter these by
   uniform color if they consistently stand out from both team clusters).
4. Because clustering runs per-shot but you need one consistent `team_id`
   label across the whole game, anchor each shot's two clusters to a known
   reference: pick one frame per team where you're confident of identity
   (e.g. a clear scoreboard graphic frame, or manual spot-check on game 1),
   and match new clusters to the nearest reference by color-embedding
   distance rather than trusting arbitrary cluster order.

### Part B — Jersey number OCR

1. Within each tracker_id's track, sample frames where the player is
   reasonably large in frame and not heavily motion-blurred (filter by bbox
   size and a simple blur metric like Laplacian variance).
2. Crop the number region. If you have a keypoint/pose model, use shoulder
   position to crop the back-number area specifically; otherwise a generic
   torso crop with OCR run on the whole thing works reasonably as a starting
   point.
3. Run OCR (EasyOCR is a reasonable default) on each sampled crop.
4. **Majority vote across all sampled frames in the track** — do not trust a
   single frame's read. Require a minimum number of agreeing reads (e.g. at
   least 3 frames agreeing) before accepting a number.
5. If no majority emerges, leave `jersey_number = null` for that track. This
   will happen often — that's fine, see below.

### Combining A + B

`(team_id, jersey_number)` → roster lookup → `player_name`. If either input is
null, `player_name` is null. Do not fall back to guessing based on partial
information (e.g. "probably the point guard because of court position") —
that kind of inference is exactly the sort of silent wrong-label that
corrupts the final metric.

## `identity_confidence` definition

A simple starting formula:

```
identity_confidence = (fraction of sampled frames agreeing on jersey number)
                       * (team cluster silhouette score, clipped to [0,1])
```

Use this as a filter in Stage 6 aggregation — e.g. only include a track's
frames in the final metric if `identity_confidence > 0.6`. Tune this
threshold empirically by spot-checking a sample of accepted vs. rejected
tracks against the actual broadcast.

## What the implementation measured

Part B is built (`pipeline/jersey_ocr.py`, run with `03_identify.py --ocr`).
The thresholds above were written as guesses; these are what replaced them
after checking against footage. Method: sample each track's 24 largest
frames, then read every crop by eye to establish truth before looking at what
OCR said. Of 17 tracks on the test possession, 5 carried a legible number, 12
carried none, and 2 more were legible to a human but never to OCR.

| Setting | Spec's guess | Measured | Why |
|---------|--------------|----------|-----|
| Agreeing reads | 3 | **4** | Three agreeing reads accepted a "1" for a player whose number was never visible. No true number was read fewer than four times. |
| OCR confidence | not specified | **0.80** | EasyOCR is given a digit allowlist, so its decoder *cannot* return "not a digit" — a jersey wordmark comes back as a digit at confidence 1.0. |
| Winner's share | not specified | **> 50%** of reads kept | Refuses a track that cannot make up its mind, instead of answering by a hair. |
| Upscale before reading | not specified | **4x cubic** | Raised the per-crop hit rate from 8% to 28%. |

Three plausible-looking rules that turned out to be wrong:

1. **Folding fragments into longer reads.** OCR often catches half a
   two-digit number, so single-digit reads were attributed to a two-digit
   read containing them. Track 5 reads "2" ten times and "24" six times — and
   the player is Harper, who wears **2**. The rule turned the clearest
   evidence on the clip into a wrong answer and invented a number for a track
   that had none. Removed; a plain majority is both simpler and more accurate.
2. **Stripping leading zeros.** Turning "07" into "7" also turned seventeen
   confident "00" reads into "0" — a different player. Zeros are now stripped
   only when a non-zero digit follows.
3. **Trusting confidence over volume.** Confidence alone accepted six wrong
   numbers, several at 0.99+. What separates signal from noise is how many
   frames agree, not how sure any one frame is.

Accuracy on the test possession: **5 answered, 5 correct, 12 correctly
refused, 2 missed.** That is the trade this spec asks for. A wrong number
attaches one player's name to another player's movement, and every gravity
value derived from it is quietly false, so refusing is the cheaper error.

Both missed tracks are the same player (#11, whose "11" reads as a single
"1"), tracked twice because the tracker lost him mid-possession. Repeated
digits are the known weak spot.

### `identity_confidence` in practice

Two things to know before using it as a filter:

- With `--ocr`, a track that resolved no number scores 0, because the formula
  multiplies by frame agreement. Only 5 of 25 tracks on the test possession
  score above zero. **Gravity geometry needs the team label, not the name**,
  so Stage 6 must not gate defender selection on this column — gate it on
  `team_id` instead, and use `identity_confidence` only where a *named*
  player is required.
- Without `--ocr` there is no agreement term at all, so the column reports a
  team-only proxy instead. The number is therefore not comparable between the
  two modes.
- The spec suggests filtering at `> 0.6`. That is unreachable on this
  footage: the team-cluster silhouette is 0.36, which caps the product at
  0.36 even for a unanimous OCR read. Any threshold has to be set relative to
  the silhouette the run actually achieved.

## Filling gaps across a game

Because identity resolves per-track (i.e. per camera shot) rather than once
per player per game, the same player will be resolved dozens of times over a
broadcast. This is actually useful: if track #47 for tracker_id 12 resolves
confidently to "Jalen Brunson" but track #52 for a different tracker_id in
the same shot cluster comes back null, you can optionally back-fill using
positional continuity between adjacent shots (same team, court position
consistent with a player who was on court a few seconds ago) — but treat this
as a secondary, lower-confidence signal, not a primary resolution method, and
tag it distinctly (e.g. `identity_source: "positional_inference"` vs.
`identity_source: "ocr_direct"`) so you can exclude inferred labels from the
final metric if they turn out to be unreliable.

## What "good enough" looks like for v1

You do not need 100% of frames identity-resolved. A reasonable target for a
usable dataset: **50-70% of on-court frames** have a confident player_name,
with the rest correctly left null rather than wrong. The gravity metric
aggregates over many frames and many games — partial coverage with high
precision beats full coverage with unknown accuracy.
