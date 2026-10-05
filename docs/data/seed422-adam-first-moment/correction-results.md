# Seed422 seat-accounting correction

This append-only correction uses each seat's actual 512-game denominator.
The opening matrix, overall scores, paired estimate, bootstrap intervals,
thresholds, and decision are unchanged. The original `analysis.json` and
`results.md` remain available byte-for-byte.

| Lane | Seat 0 (512 games) | Seat 1 (512 games) | Overall score |
|---|---:|---:|---:|
| A | 0.5322265625 | 0.4335937500 | 0.4829101562 |
| B | 0.5332031250 | 0.4658203125 | 0.4995117188 |

**Decision unchanged:** `retain_baseline_close_fixed_beta1_intervention`.
