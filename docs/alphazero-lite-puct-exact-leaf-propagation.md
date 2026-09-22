# PUCT Exact Leaf Propagation Diagnostic

Primary classification: `exact_wdl_margin_collapse_primary`.

This is a read-only diagnostic. It does not change a runtime threshold, train a
model, generate self-play, run a promotion holdout, or promote seed455.

## Aggregate

| Metric | Disabled | Threshold10 | Delta |
| --- | ---: | ---: | ---: |
| Optimal mass | 0.7047 | 0.6878 | -0.0169 |
| Expected regret | 1.5296 | 1.6358 | +0.1063 |
| Top-1 optimal | 0.740 | 0.730 | -0.010 |
| Top-1 regret | 1.230 | 1.320 | +0.090 |
| WDL-optimal top-1 | 0.990 | 0.990 | 0.000 |

The replay exactly reproduces PR #358, so this is not
`puct_exact_propagation_baseline_drift`.

## Changed Selections

| Change class | Count |
| --- | ---: |
| Outcome improvement | 0 |
| Outcome regression | 0 |
| Same-outcome margin improvement | 5 |
| Same-outcome margin regression | 6 |
| Exact tie | 2 |

Expected-regret delta decomposes as `+0.02677` from changed selections and
`+0.07948` from visit redistribution at unchanged selections. There are no
root WDL regressions.

## Decisive Row

For `sparse_endgame-023`, threshold 10 first backs up an exact leaf through
move 5 at simulation 16 and through move 4 at simulation 111. At 384
simulations, exact backup counts for moves 1/4/5 are 18/28/103 and visits are
80/172/132, versus disabled visits 45/29/310. Exact loss evidence removes
move 5's prior-driven allocation early enough for WDL-optimal draw move 4 to
win the deterministic root selection.

## Saturation Evidence

Within exact leaf WDL classes, neural values correlate with exact final score
margin: `0.448` across 13,243 wins and `0.538` across 9,348 losses. Replacing
all such leaves with `+1` or `-1` discards this ranking information. Final
root-child Q remains margin-correlated (disabled `0.871`, threshold10
`0.864`), indicating a finite-budget allocation effect rather than a complete
root-Q inversion.

## Boundary And Control

Seed455's network mean is `0.2247` at 11 stones and `0.2272` at 10 stones;
the threshold's mean exact-minus-network delta at 10 is `0.0368`. This does
not support a large seed455 boundary discontinuity as the primary mechanism.

Seed48 shows the same pattern: zero WDL changed selections, 16 same-outcome
margin regressions, and expected regret rising from `1.5078` to `1.5513`.
The effect is therefore not specific to seed455.

## Next Experiment

The next isolated ablation should test a generic exact leaf value that keeps
exact WDL ordering while preserving exact score-margin information. Do not
implement it as a runtime policy or combine it with a search-allocation change
in this diagnostic branch.
