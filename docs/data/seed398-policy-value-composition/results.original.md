# Seed398 policy/value composition diagnostic

All challenger treatments faced the unchanged seed455 opponent. Component outputs were composed through their own source evaluators; no weights were spliced.

| Treatment | Score | Wins | Draws | Losses | Seat 0 score | Seat 1 score | Games |
|---|---:|---:|---:|---:|---:|---:|---:|
| SS | 0.5000 | 446 | 132 | 446 | 0.5664 | 0.4336 | 1,024 |
| FS | 0.4927 | 444 | 121 | 459 | 0.5557 | 0.4297 | 1,024 |
| SF | 0.5010 | 464 | 98 | 462 | 0.5508 | 0.4512 | 1,024 |
| FF | 0.5107 | 468 | 110 | 446 | 0.5605 | 0.4609 | 1,024 |

## Contrasts

| Contrast | Mean | Interval | Level | Classification |
|---|---:|---:|---:|---|
| policy_FS_minus_SS | -0.0073 | [-0.0312, +0.0166] | 97.5% | uncertain |
| value_SF_minus_SS | +0.0010 | [-0.0229, +0.0254] | 97.5% | uncertain |
| FF_minus_SS | +0.0107 | [-0.0112, +0.0322] | 95% descriptive | uncertain |
| interaction_FF_minus_FS_minus_SF_plus_SS | +0.0171 | [-0.0132, +0.0474] | 95% descriptive | uncertain |

SS seat-score difference (seat 0 minus seat 1): +0.1328. Seat 0 scored higher in all four treatment arms; seat scores remain paired in every opening and both seats are retained.
Primary components merit a follow-up training study only if the estimated effect is at least +0.03 and the 97.5% interval lower bound exceeds zero. Wide intervals are uncertainty, not equivalence.

The complete validated per-game ledger is `validated-outcome-ledger.jsonl`; the 512-opening score matrix is `four-treatment-opening-score-matrix.json`. Recompute the bootstrap by rerunning `python -m ml.alphazero_lite.seed398_composition_diagnostic analyze` against the hash-bound registration and ledgers.
