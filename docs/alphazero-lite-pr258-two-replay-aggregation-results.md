# PR #258 Two-Replay Aggregation Results

**Classification:** `aggregation_reduces_variance_only`

## Frozen Contract

- Five preregistered primary replay pairs: 53/54, 55/56, 57/58, 59/60, and
  61/62.
- Each replay used 700 ordinary reused-tree self-play games with 384
  simulations/move, `c_puct=1.25`, zero FPU, unnormalized values, visit-count
  policy targets, Kalah v3, and the established exploration and temperature
  schedule. No separate target search or shadow Q was used.
- Every candidate started from the frozen A16 step-16 model and Adam state.
  Training was policy-adapter-only at beta .95, Adam LR `1e-5`, grad clip 1.0,
  16 optimizer steps, and exactly 8,192 unique training examples.
- `single_a` and `single_b` each used 8,192 rows from one replay. `pair_mix`
  used the first frozen 4,096 rows from each corresponding single ordering,
  with every batch containing 256 rows from each replay.

## Invariants And Sealing

- The consumed-suite registry was repaired through U before generation. Newly
  sealed V/W/X suites used seeds 22042/23042/24042 and SHA-256 values
  `b08ef373f8a958b7279afa3dde481e582a465451cc520160b737f5b4b90195e2`,
  `f6c016c67e51f5e829ee70e93be5614dbbef312cd1db9f95908b3363520d5712`, and
  `00a93262cbbf2ca9dc0b97b5275337c80badb1aae338f25854d98628527184a0`.
- Preflight found zero final-state, prefix, and training-replay-state overlap
  for every V/W/X suite. V/W/X are now consumed in the authoritative registry.
- All 15 step-16 model hashes were unique and frozen before suite generation.
  Repeated-lane and reverse-order checks passed; the pristine Adam SHA was
  unchanged.
- All candidates improved beta-.95 CE: `+7.0131e-6` to `+1.0318e-5`.
  Ordinary reused-target CE also improved for every candidate.

## Primary 1200:1200 Arena

All arenas used deterministic ordinary 1200:1200 PUCT, zero FPU, unnormalized
values, seat swapping, arena seed 42, and matched P1-vs-P1 controls on V/W/X.
Each pair's delta is pair-mix effect minus the opening-aligned mean of its two
single-replay effects.

| Pair | V | W | X | Pooled delta |
| --- | ---: | ---: | ---: | ---: |
| 53/54 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 55/56 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| 57/58 | -0.018555 | -0.020508 | -0.026367 | -0.021810 |
| 59/60 | -0.037109 | -0.041016 | -0.052734 | -0.043620 |
| 61/62 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |

- Mean delta: `-0.013086`; median: `0`; SD: `0.019507`; range:
  `[-0.043620, 0]`; positive pairs: `0/5`.
- Hierarchical pair -> suite -> opening 95% CI: `[-0.030474, 0]`.
- The primary success rule fails mean, CI, sign-consistency, and no-negative-pair
  conditions.

## Secondary 384:256 Arena

The frozen candidates were evaluated on the same V/W/X suites only after the
primary result was written. Mean pair delta was `+0.002995` (one positive pair,
range `[0, +0.014974]`), so the preregistered material shallow-harm condition
did not trigger.

## Replay And Gradient Diagnostics

- State Jaccard overlap was .0374-.0406; exact state overlap was 1,686-1,810;
  trajectory-hash overlap was 10-20 games per pair.
- Full eligible-replay adapter-gradient cosine was .9543-.9899 (mean .9769),
  indicating strongly aligned single-replay update directions.
- The actual pair-mix first-step update had cosine .8382-.8654 with negative
  mean replay gradient.
- Individual single-replay primary-effect variance was `.00044396`; pair-mix
  primary-effect variance was `0`. This is variance reduction, not a positive
  mean-strength gain.

## Compute And Decision

- Single replay: 700 self-play games. Pair aggregation: 1,400 games, or about
  2x self-play search compute at the unchanged 384 simulations/move.
- SGD compute was identical: 8,192 examples and 16 optimizer steps per model.
- Wall-clock time: 15,772 seconds (about 4.38 hours).

Two ordinary replays reduced observed replay-sampling strength variance but did
not improve expected strength at fixed SGD compute. Close simple replay-size
expansion. The recommended next experiment is increased training iterations on
an aggregated buffer without changing search targets.

## Frozen Artifacts

- Frozen manifest SHA-256:
  `febe346ed947534a5e75d76d7a55301abcd6e451870f1b19c34c0158473ddfad`
- Preflight SHA-256:
  `e98733752c2791b2959600f9cc7a1ae5ef4752a1524afe474cab4fe4102c1e6a`
- Candidate model-set SHA-256:
  `20aa53af1d05daec4dca0e02980e1c4a5dda3b1141a85541d74405f3c174c4f8`
