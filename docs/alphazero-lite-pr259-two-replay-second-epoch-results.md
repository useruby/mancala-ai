# PR #259 Two-Replay Second-Epoch Results

**Classification:** `second_epoch_reduces_underfit_but_not_strength`

## Contract And Invariants

- Reused the exact PR #258 frozen replays for seeds 53 through 62, including all
  committed replay and state/outcome hashes.
- Reused the committed `single_a`, `single_b`, and `pair_mix` sample plans. Each
  epoch was the exact 16 batches x 512 ordering; batches 17 through 32 repeated
  batches 1 through 16 without reshuffling.
- Every lane started from the original A16 model and Adam state and trained
  continuously for 32 steps at policy-adapter-only, beta .95, Adam LR `1e-5`,
  grad clip 1.0, and batch size 512. No Adam reset or target/search change was
  made.
- All regenerated step-16 model and optimizer hashes matched PR #258, including
  adapter tensor equality with the serialized PR #258 checkpoints. The repeated
  pair1 `pair_mix` step-32 lane also matched exactly.
- All 30 step-16 and step-32 model hashes were frozen before evaluation.
- New self-play compute was zero. Each candidate retained 8,192 unique rows,
  with 8,192 exposures at step 16 and 16,384 at step 32.

## Sealed Evaluation

- The authoritative registry was verified through X, including committed V/W/X
  SHAs, before new suite selection.
- Y/Z/AA used seeds 25042/26042/27042 with SHAs
  `f6247d0e59b78af776270f80ecd6a00fc84c8fbcb69a462cbf07d5ecee49a6a9`,
  `2b004a0e1a52f9a2776d1dfc459181d1bf89b45c591ac0f9a4fd39245ee2e996`,
  and `629e997fadca2f220cab61684ca6bc77116154e42c7371ebaf91b83b09093375`.
- Preflight found zero final-state, prefix, and training-replay-state overlap
  against canonical through X and among Y/Z/AA. All three suites are consumed.
- Primary arenas used deterministic ordinary 1200:1200 PUCT, `c_puct=1.25`,
  zero FPU, unnormalized values, seat swaps, arena seed 42, and matched P1-vs-P1
  controls. Y/Z/AA were not used by PR #258's primary result.

## Training And Validation

- All 15 lanes improved own beta-.95 CE from initial to step 32 by
  `+1.19e-5` to `+1.65e-5`; the second epoch contributed `+4.78e-6` to
  `+6.17e-6`.
- Held-out beta-.95 CE improved from step 16 to step 32 for every lane by
  `+4.67e-6` to `+8.18e-6`. Repeating the frozen rows did not show this simple
  validation-overfitting signature.
- The pair-mix second epoch remained aligned with its first epoch: cosine of
  second epoch versus `d16` was `.9202` to `.9490` (mean `.9303`). Its cosine
  versus the negative full-replay mean gradient was `.7227` to `.8174` (mean
  `.7800`). The continuation did not rotate to a new direction.

## Primary 1200:1200 Arena

`delta` is pair-mix effect minus the opening-aligned mean single-replay effect.

| Pair | Delta step16 | Delta step32 | Interaction |
| --- | ---: | ---: | ---: |
| 53/54 | 0.000000 | 0.000000 | 0.000000 |
| 55/56 | 0.000000 | +0.008464 | +0.008464 |
| 57/58 | -0.021810 | -0.003581 | +0.018229 |
| 59/60 | -0.043620 | 0.000000 | +0.043620 |
| 61/62 | 0.000000 | -0.008464 | -0.008464 |

- Step32 mean delta: `-0.000716`; median: `0`; SD: `.006195`; range:
  `[-.008464, +.008464]`; positive pairs: `1/5`.
- Step32 hierarchical replay-pair -> suite -> opening 95% CI:
  `[-.006250, +.005078]`.
- Mean interaction: `+.012370`; median: `+.008464`; SD: `.020082`; range:
  `[-.008464, +.043620]`; positive pairs: `3/5`.
- Interaction hierarchical 95% CI: `[-.002344, +.030273]`.
- The step32 success rule fails A, B, C, E, and F. It passes D and G only.

## Duration And Variance

- Mean pair-mix duration effect was `-.005856`, 95% CI `[-.010547, -.001563]`.
- Mean single duration effect was `-.018342`, 95% CI `[-.037109, -.002734]`.
  Both directions lost absolute strength; pair mixing did not rescue duration.
- Replay-pair effect variance was `.000381` for single means and `0` for
  pair-mix at step16, versus `.0000155` and `.0000176` at step32. This is not a
  strength improvement.

## Secondary 384:256 Arena

- Step32 mean delta was `+.006250` with one pair below `-.02` (`-.031250`).
  The preregistered material shallow-harm gate did not trigger.

## Compute

- Final reported arena pass: 46,080 candidate games plus 1,536 matched control
  games at each budget. This is 114,278,400 PUCT simulations at 1200:1200 and
  30,474,240 at 384:256.
- A post-arena reporting-key failure required one deterministic rerun over the
  same frozen models and already-consumed suites. It did not create new
  self-play, samples, checkpoints, or opening suites; physical arena compute was
  therefore twice the final reported pass.

## Decision

The second epoch reduced target underfit on both training and held-out replay
rows, but did not improve game strength. Close the replay-aggregation
under-optimization hypothesis. The next experiment should not optimize this CE
duration further; test model capacity or a prospective full AlphaZero generation.

## Frozen Artifacts

- Source PR #258 manifest SHA-256:
  `febe346ed947534a5e75d76d7a55301abcd6e451870f1b19c34c0158473ddfad`
- PR #259 candidate model-set SHA-256:
  `7c173a923b47bd029aea9d52d00bac9d93a62ba6df359c26812ba725e79a28bd`
- Full machine-readable result:
  `/tmp/azlite_pr259_two_replay_second_epoch/summary.json`
