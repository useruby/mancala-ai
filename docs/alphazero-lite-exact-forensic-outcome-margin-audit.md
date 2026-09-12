# Exact Forensic Outcome-vs-Margin Audit

## Semantics

Native exact action values are player-zero final stone margins. Exact training values and PUCT terminal values are root-player forced win/draw/loss utilities (+1/0/-1). This read-only audit preserves margin regret and separately evaluates the latter objective.

## Oracle Comparison

All 213 exact rows passed max-outcome-utility == exact_root_value.

| Scope | Margin set equals outcome set | Outcome set larger | Mean extra equivalent actions | Multi-action outcome set |
| --- | ---: | ---: | ---: | ---: |
| overall | 0.169 | 0.831 | 2.488 | 0.869 |
| capture_available | 0.042 | 0.958 | 2.667 | 1.000 |
| early_extra_turn | 0.042 | 0.958 | 4.042 | 0.958 |
| high_imbalance | 0.125 | 0.875 | 2.917 | 0.875 |
| high_value_swing | 0.250 | 0.750 | 1.750 | 0.792 |
| incumbent_proxy_disagreement | 0.000 | 1.000 | 2.906 | 1.000 |
| opening_plies_1_8 | 0.243 | 0.757 | 2.351 | 0.811 |
| sparse_endgame | 0.292 | 0.708 | 1.292 | 0.833 |
| starvation_pressure | 0.375 | 0.625 | 1.917 | 0.667 |

## Decision-Critical Tables

### capture_available-002

| Action | P0 margin | Root utility | Margin-optimal | Outcome-optimal | Control/44/45/46 selected |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | 2 | -1 | False | False |  |
| 1 | -4 | +1 | False | True | uniform44/uniform45 |
| 2 | -16 | +1 | True | True | control/uniform46 |
| 3 | -10 | +1 | False | True |  |
| 4 | -6 | +1 | False | True |  |

### capture_available-020

| Action | P0 margin | Root utility | Margin-optimal | Outcome-optimal | Control/44/45/46 selected |
| ---: | ---: | ---: | --- | --- | --- |
| 0 | -6 | +1 | False | True |  |
| 1 | -10 | +1 | False | True | control/uniform44/uniform45 |
| 3 | -16 | +1 | True | True |  |
| 4 | -12 | +1 | False | True | uniform46 |

## PR #287 Outcome Replay

| Seed | Outcome accuracy delta | Outcome regret delta | Outcome blunder delta | Margin accuracy delta | Margin regret delta | Same-outcome margin delta | Old / outcome shadow |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 44 | +0.042 | -0.085 | -0.042 | +0.085 | -1.023 | -0.042 | fail / fail |
| 45 | +0.047 | -0.085 | -0.047 | +0.127 | -1.249 | -0.080 | fail / fail |
| 46 | +0.047 | -0.075 | -0.047 | +0.113 | -1.014 | -0.066 | fail / pass |

## Margin Gate Failures

The decision-critical failures are outcome-optimal; separately listed rows below are genuine W/D/L regressions.

| Seed | State | Selected | Margin regret | Outcome regret | Classification |
| ---: | --- | ---: | ---: | ---: | --- |
| 44 | capture_available-002 | 1 | 12 | 0 | same-outcome margin regression |
| 44 | capture_available-020 | 1 | 6 | 0 | same-outcome margin regression |
| 44 | capture_available-025 | 0 | 4 | 1 | true outcome regression |
| 45 | capture_available-002 | 1 | 12 | 0 | same-outcome margin regression |
| 45 | capture_available-012 | 1 | 2 | 0 | same-outcome margin regression |
| 45 | capture_available-018 | 1 | 8 | 2 | true outcome regression |
| 45 | capture_available-020 | 1 | 6 | 0 | same-outcome margin regression |

## PR #300 Q/Visit Reinterpretation

| Seed/family | State | Q margin/outcome sufficient | Visit margin/outcome sufficient | Outcome error type |
| --- | --- | --- | --- | --- |
| 44:control | capture_available-002 | True/True | True/True | fully optimal |
| 44:control | capture_available-020 | False/True | True/True | margin-only Q |
| 44:uniform1200 | capture_available-002 | False/True | False/True | margin-only Q, margin-only visit |
| 44:uniform1200 | capture_available-020 | False/True | False/True | margin-only Q, margin-only visit |
| 45:control | capture_available-002 | True/True | True/True | fully optimal |
| 45:control | capture_available-020 | False/True | False/True | margin-only Q, margin-only visit |
| 45:uniform1200 | capture_available-002 | True/True | False/True | margin-only visit |
| 45:uniform1200 | capture_available-020 | False/True | False/True | margin-only Q, margin-only visit |
| 46:control | capture_available-002 | True/True | True/True | fully optimal |
| 46:control | capture_available-020 | True/True | False/True | margin-only visit |
| 46:uniform1200 | capture_available-002 | True/True | True/True | fully optimal |
| 46:uniform1200 | capture_available-020 | False/True | False/True | margin-only Q, margin-only visit |

## Classification

`uniform1200_true_outcome_regression_confirmed`

## One Next Experiment

Exactly one next experiment: perform the subtree evaluator/backup attribution audit on ONLY the true-outcome-regression states.
