# PR #242 Target Entropy Factorization

**Classification:** `target_factorization_not_reproducible`

## Fixed Contract

- Frozen `ordinary_onpolicy.jsonl`: SHA-256
  `6671e248af4a4c82e1155c798cb7490cd66cd80dc10b203c97d89dced94527f2`;
  27,642 rows, with the exact PR #242 exclusions retaining 27,350 rows.
- Canonical suite: SHA-256
  `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`.
- The fresh noisy/denoised reconstruction reproduced PR #242's target diagnostics:
  L1 `0.0759297`, JS `0.0081920`, top-1 disagreement `0.0408775`, and
  `H(denoised) - H(noisy) = -0.0356288`.
- Searches retained the PR #242 A16 evaluator and deterministic target seeds,
  384 simulations, c_puct 1.25, zero FPU, no normalization, no subtree reuse,
  and the 1.0/0.1 policy-temperature schedule. No trajectories were generated.
- Every lane started from a deep copy of the same A16 model and Adam state.
  This fixes the prior loader's optimizer-state aliasing, which let earlier lanes
  mutate moments used by later lanes.

## Target Construction

The entropy matcher uses deterministic float64 bisection over a convex path to
uniform legal mass when increasing entropy and to the deterministic source top
move when decreasing it. It reaches entropy within `1e-8`, never puts mass on
an illegal move, and retains source ordering and top-1.

| Lane | Policy target SHA-256 |
| --- | --- |
| fresh denoised | `ac08cd03fbe66857da2ca1bebeb6b741b6c0be876cac5621ca0fc01611b325a2` |
| denoised order, noisy entropy | `2446e4d7c1f70189b1c7a42639ced1dae948447692f135fa1be8246b2f428f4b` |
| noisy order, denoised entropy | `f979d24ad832b1d3792eb73d8df2cde078ed08fbd2118701d42fb8b48a2bd011` |
| fresh noisy | `e1a2a4af07be3ca371f966803d1aee36cf309e4c191bd451c5905f4dfd6a60a0` |

All views share batch-plan SHA-256
`39c0663ff682a30d01e0dfc05d40448f4b22a6051aa3d2fe2464a0d50cc57789`.
All non-policy labels remain identical. The entropy and top-1 invariants were
also checked on all 1,118 noisy/denoised top-1-disagreement rows; noise-disabled
rows remain unchanged. The runner records all six pairwise L1, JS, entropy,
top-1, phase, noise-mode, and legal-move-count strata in its summary.

## Training

All inherited parameters remained bit-identical and all step-16 lanes met the
meaningful-fit criterion.

| Lane | Step-16 CE(target) | CE(P1) | CE(beta095) | Fit | L1 vs P1 | JS vs P1 | Top-1 | Adapter norm | Delta A16 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| fresh denoised | 0.95653 | 0.94546 | 0.94601 | 1.0000 | 0.00341 | 2.53e-06 | 0.00260 | 0.00469 | 0.00232 |
| denoised order, noisy entropy | 1.00017 | 0.94546 | 0.94819 | 1.0000 | 0.00335 | 2.45e-06 | 0.00263 | 0.00469 | 0.00228 |
| noisy order, denoised entropy | 0.99561 | 0.94545 | 0.94796 | 1.0000 | 0.00318 | 2.20e-06 | 0.00252 | 0.00456 | 0.00221 |
| fresh noisy | 1.03288 | 0.94546 | 0.94983 | 1.0000 | 0.00328 | 2.35e-06 | 0.00267 | 0.00470 | 0.00227 |

Steps 1 and 4 and cross-CE against every target family were also recorded by
the runner; no auxiliary loss or non-policy label changed.

## Ordinary-PUCT Arena

All rows use the fixed 128-opening suite, seat swap, seed 42, matched P1-vs-P1
controls, and 10,000-opening bootstrap intervals.

| Lane | 384:256 effect / CI | 1200:1200 effect / CI |
| --- | --- | --- |
| fresh denoised | -0.0195 [-0.0391, -0.0039] | -0.0195 [-0.0391, -0.0039] |
| denoised order, noisy entropy | -0.0195 [-0.0391, -0.0039] | -0.0195 [-0.0391, -0.0039] |
| noisy order, denoised entropy | -0.0195 [-0.0391, -0.0039] | -0.0195 [-0.0391, -0.0039] |
| fresh noisy | -0.0195 [-0.0391, -0.0039] | -0.0195 [-0.0391, -0.0039] |

Each lane had the same seat effects, P0 `-0.0391` and P1 `0.0`, and the same
candidate W/D/L: 384:256 `390/38/84`; 1200:1200 `180/132/200`.

Both prespecified, opening-aligned contrasts are exactly zero with bootstrap
CI `[0.0, 0.0]` at both budgets:

- denoised-order/noisy-entropy minus fresh-denoised
- noisy-order/denoised-entropy minus fresh-noisy

## Interpretation

The four entropy/order variants are meaningful supervised fits, but none
reproduced PR #242's reported fresh-noisy 1200:1200 advantage once each lane
received an independent frozen Adam state. Therefore neither entropy nor
action-specific noisy-MCTS direction can be credited with that earlier gain.
No candidate is promoted.
