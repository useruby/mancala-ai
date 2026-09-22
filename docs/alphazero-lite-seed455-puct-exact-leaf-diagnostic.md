# Seed455 PUCT Exact Leaf-Value Diagnostic

Primary classification: `puct_exact_endgame_repairs_tactic_but_hurts_globally`.

This evaluation is read-only. It does not train, generate self-play, alter
replay, run canonical holdouts, or change the rejected promotion decision for
`seed48-nextgen-s455-default-value`.

Candidate checkpoint SHA256: `c18beeaa16ebaa038cd383cfd222705c636e2b6398c7270e6674d33231aa9dd1`.
Candidate weights SHA256: `f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`.
Incumbent weights SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`.

## Search Policy

At a nonterminal PUCT evaluation with active pit stones at or below the
configured threshold, the normal network evaluator is invoked first. Its
policy priors are preserved exactly. Its value alone is replaced with the
current-player exact tablebase W/D/L value mapped from `[0, 1]` to `[-1, 1]`.
An eligible state without an exact value fails closed as
`puct_exact_solve_coverage_gap`.

Reusable configurations were `disabled`, `exact_threshold_8`,
`exact_threshold_10`, and `exact_threshold_12`. All use 384 simulations,
`c_puct=1.25`, zero FPU, deterministic roots, no subtree reuse, no value
normalization, zero tactical root bias, and no Dirichlet noise.

## Decisive Row

For `sparse_endgame-023`, seed455 selected losing move 5 with exact solving
disabled and at threshold 8. Thresholds 10 and 12 selected WDL-optimal draw
move 4. All qualifying lookups succeeded; no coverage gaps occurred.

| Threshold | Seed455 selected move | Visits: 1 / 4 / 5 |
| --- | ---: | --- |
| Disabled | 5 | 45 / 29 / 310 |
| 8 | 5 | 72 / 110 / 202 |
| 10 | 4 | 80 / 172 / 132 |
| 12 | 4 | 48 / 214 / 122 |

## Exact Safety

Threshold 10 was the smallest threshold that repaired the decisive row.
Seed455 sparse WDL regression fell from `0.0417` to `0.0000`; its
213-row exact-covered WDL regression fell from `0.0610` to `0.0563`.
The frozen PR #340 200-state exact corpus nevertheless regressed:

| Threshold | Optimal mass | Expected regret | Top-1 optimal | Top-1 regret |
| --- | ---: | ---: | ---: | ---: |
| Disabled | 0.7047 | 1.5296 | 0.740 | 1.230 |
| 8 | 0.6912 | 1.6004 | 0.730 | 1.290 |
| 10 | 0.6878 | 1.6358 | 0.730 | 1.320 |
| 12 | 0.6913 | 1.6142 | 0.750 | 1.160 |

Threshold 10 therefore worsened seed455 expected regret by `0.1063` and
reduced optimal mass by `0.0169`. It fails the pre-registered PR #340 exact
search safety condition and is not adopted.

## Diagnostic Arena

The non-promotion equal-threshold arena used the frozen 128-opening suite
SHA256 `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`.
Both models used threshold 10, 384 simulations, and deterministic roots.
Seed455 scored `0.7227` over 256 games (176 wins, 62 losses, 18 draws).
This result is secondary evidence and was not used to select a threshold.

## Result

No threshold is adopted. Keep seed48 as incumbent and investigate PUCT visit
allocation and exact-value propagation on `sparse_endgame-023` before any
retraining experiment.
