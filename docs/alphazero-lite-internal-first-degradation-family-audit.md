# First Repeated Internal Outcome-Degradation Audit

## Inherited Inputs

Inherited classification: `true_outcome_subtree_policy_failure_primary`.
PR #302 audit SHA: `c691bd6a5b825ca8a9d312bc86de91125161ab7c98899fbe89e4d9582d1e200f`.
Label-manifest SHA: `5b1f6bac3d0289c62d74cccfa104fd3b31a41dc17eb7cfb24b457573df9f477c`.

## Deterministic Selection

Selected family: `win_to_draw|neither|extra_turn|5-6`.
Primary-root occurrences: `45:uniform1200:capture_available-018`.
This family is root-specific: its mechanically highest count did not occur in both failing roots.

## Family Table

| Family | Cohort | Events | Opportunities | Rate | Roots |
| --- | --- | ---: | ---: | ---: | ---: |
| `win_to_draw|capture|extra_turn+neither|3-4` | failing_uniform | 5 | 140 | 0.036 | 1 |
| `win_to_draw|capture|extra_turn+neither|3-4` | matched_control | 1 | 13 | 0.077 | 1 |
| `win_to_draw|capture|extra_turn+neither|3-4` | negative_control | 7 | 273 | 0.026 | 1 |
| `win_to_draw|capture|neither|3-4` | failing_uniform | 2 | 12 | 0.167 | 1 |
| `win_to_draw|capture|neither|3-4` | negative_control | 5 | 63 | 0.079 | 1 |
| `win_to_draw|extra_turn|extra_turn|3-4` | parent | 1 | 5 | 0.200 | 1 |
| `win_to_draw|extra_turn|extra_turn|5-6` | failing_uniform | 3 | 117 | 0.026 | 1 |
| `win_to_draw|extra_turn|extra_turn|5-6` | matched_control | 13 | 514 | 0.025 | 1 |
| `win_to_draw|extra_turn|extra_turn|5-6` | negative_control | 9 | 225 | 0.040 | 2 |
| `win_to_draw|extra_turn|extra_turn|5-6` | parent | 1 | 13 | 0.077 | 1 |
| `win_to_draw|extra_turn|neither|5-6` | failing_uniform | 10 | 160 | 0.062 | 1 |
| `win_to_draw|extra_turn|neither|5-6` | matched_control | 5 | 80 | 0.062 | 1 |
| `win_to_draw|extra_turn|neither|5-6` | negative_control | 20 | 590 | 0.034 | 2 |
| `win_to_draw|extra_turn|neither|5-6` | parent | 7 | 91 | 0.077 | 1 |
| `win_to_draw|neither|capture+extra_turn|5-6` | matched_control | 2 | 8 | 0.250 | 1 |
| `win_to_draw|neither|capture+extra_turn|5-6` | negative_control | 1 | 2 | 0.500 | 1 |
| `win_to_draw|neither|capture+neither|3-4` | negative_control | 1 | 16 | 0.062 | 1 |
| `win_to_draw|neither|capture+neither|5-6` | failing_uniform | 2 | 20 | 0.100 | 1 |
| `win_to_draw|neither|capture+neither|5-6` | matched_control | 2 | 4 | 0.500 | 1 |
| `win_to_draw|neither|capture+neither|5-6` | negative_control | 6 | 44 | 0.136 | 2 |
| `win_to_draw|neither|capture|5-6` | failing_uniform | 1 | 4 | 0.250 | 1 |
| `win_to_draw|neither|capture|5-6` | negative_control | 1 | 4 | 0.250 | 1 |
| `win_to_draw|neither|extra_turn+neither|3-4` | negative_control | 4 | 66 | 0.061 | 2 |
| `win_to_draw|neither|extra_turn+neither|3-4` | parent | 1 | 1 | 1.000 | 1 |
| `win_to_draw|neither|extra_turn+neither|5-6` | failing_uniform | 3 | 16 | 0.188 | 2 |
| `win_to_draw|neither|extra_turn+neither|5-6` | matched_control | 14 | 97 | 0.144 | 1 |
| `win_to_draw|neither|extra_turn+neither|5-6` | negative_control | 11 | 74 | 0.149 | 3 |
| `win_to_draw|neither|extra_turn+neither|5-6` | parent | 1 | 4 | 0.250 | 1 |
| `win_to_draw|neither|extra_turn|3-4` | failing_uniform | 1 | 7 | 0.143 | 1 |
| `win_to_draw|neither|extra_turn|3-4` | matched_control | 15 | 645 | 0.023 | 1 |
| `win_to_draw|neither|extra_turn|3-4` | negative_control | 16 | 570 | 0.028 | 2 |
| `win_to_draw|neither|extra_turn|3-4` | parent | 7 | 85 | 0.082 | 2 |
| `win_to_draw|neither|extra_turn|5-6` | failing_uniform | 69 | 4974 | 0.014 | 1 |
| `win_to_draw|neither|extra_turn|5-6` | matched_control | 33 | 873 | 0.038 | 2 |
| `win_to_draw|neither|extra_turn|5-6` | negative_control | 42 | 1620 | 0.026 | 4 |
| `win_to_draw|neither|extra_turn|5-6` | parent | 73 | 6171 | 0.012 | 2 |
| `win_to_draw|neither|neither|3-4` | failing_uniform | 5 | 52 | 0.096 | 1 |
| `win_to_draw|neither|neither|3-4` | matched_control | 5 | 121 | 0.041 | 2 |
| `win_to_draw|neither|neither|3-4` | negative_control | 10 | 215 | 0.047 | 3 |
| `win_to_draw|neither|neither|3-4` | parent | 35 | 8114 | 0.004 | 1 |
| `win_to_draw|neither|neither|5-6` | failing_uniform | 28 | 1347 | 0.021 | 1 |
| `win_to_draw|neither|neither|5-6` | matched_control | 47 | 2660 | 0.018 | 1 |
| `win_to_draw|neither|neither|5-6` | negative_control | 122 | 9672 | 0.013 | 2 |
| `win_to_draw|neither|neither|5-6` | parent | 108 | 25588 | 0.004 | 1 |
| `win_to_loss|capture|neither|3-4` | matched_control | 2 | 8 | 0.250 | 1 |
| `win_to_loss|capture|neither|3-4` | negative_control | 6 | 57 | 0.105 | 2 |
| `win_to_loss|extra_turn|extra_turn|5-6` | matched_control | 10 | 469 | 0.021 | 1 |
| `win_to_loss|extra_turn|extra_turn|5-6` | negative_control | 9 | 268 | 0.034 | 2 |
| `win_to_loss|extra_turn|extra_turn|5-6` | parent | 5 | 150 | 0.033 | 1 |
| `win_to_loss|neither|capture+extra_turn|5-6` | negative_control | 1 | 2 | 0.500 | 1 |
| `win_to_loss|neither|capture+extra_turn|5-6` | parent | 4 | 28 | 0.143 | 1 |
| `win_to_loss|neither|capture+neither|5-6` | negative_control | 3 | 18 | 0.167 | 2 |
| `win_to_loss|neither|capture|5-6` | failing_uniform | 8 | 42 | 0.190 | 1 |
| `win_to_loss|neither|capture|5-6` | negative_control | 1 | 4 | 0.250 | 1 |
| `win_to_loss|neither|extra_turn+neither|3-4` | negative_control | 5 | 118 | 0.042 | 1 |
| `win_to_loss|neither|extra_turn+neither|3-4` | parent | 2 | 4 | 0.500 | 1 |
| `win_to_loss|neither|extra_turn+neither|5-6` | failing_uniform | 14 | 306 | 0.046 | 1 |
| `win_to_loss|neither|extra_turn+neither|5-6` | matched_control | 40 | 712 | 0.056 | 2 |
| `win_to_loss|neither|extra_turn+neither|5-6` | negative_control | 41 | 830 | 0.049 | 4 |
| `win_to_loss|neither|extra_turn+neither|5-6` | parent | 2 | 10 | 0.200 | 1 |
| `win_to_loss|neither|extra_turn|3-4` | failing_uniform | 5 | 44 | 0.114 | 1 |
| `win_to_loss|neither|extra_turn|3-4` | matched_control | 16 | 306 | 0.052 | 1 |
| `win_to_loss|neither|extra_turn|3-4` | negative_control | 28 | 884 | 0.032 | 2 |
| `win_to_loss|neither|extra_turn|3-4` | parent | 7 | 75 | 0.093 | 1 |
| `win_to_loss|neither|extra_turn|5-6` | failing_uniform | 58 | 1825 | 0.032 | 2 |
| `win_to_loss|neither|extra_turn|5-6` | matched_control | 148 | 4081 | 0.036 | 2 |
| `win_to_loss|neither|extra_turn|5-6` | negative_control | 167 | 5045 | 0.033 | 4 |
| `win_to_loss|neither|extra_turn|5-6` | parent | 110 | 3744 | 0.029 | 2 |
| `win_to_loss|neither|neither|3-4` | failing_uniform | 12 | 767 | 0.016 | 2 |
| `win_to_loss|neither|neither|3-4` | matched_control | 8 | 1055 | 0.008 | 2 |
| `win_to_loss|neither|neither|3-4` | negative_control | 19 | 1690 | 0.011 | 4 |
| `win_to_loss|neither|neither|3-4` | parent | 10 | 183 | 0.055 | 1 |
| `win_to_loss|neither|neither|5-6` | failing_uniform | 11 | 130 | 0.085 | 2 |
| `win_to_loss|neither|neither|5-6` | matched_control | 47 | 5132 | 0.009 | 2 |
| `win_to_loss|neither|neither|5-6` | negative_control | 67 | 2475 | 0.027 | 4 |
| `win_to_loss|neither|neither|5-6` | parent | 7 | 56 | 0.125 | 2 |

## First-Degradation Depth

{"ge5": 0.09282700421940929, "le2": 0.35443037974683544, "le3": 0.8987341772151899, "le4": 0.9071729957805907, "median": 3, "min": 2, "p25": 2.0, "p75": 3.0}

## Local Attribution

Selected-family events: 217; raw-policy non-optimal top actions: 183; raw-bad search repairs: 0.
Raw policies, masked probabilities, local PUCT child statistics, exact alternatives, and replay exact-state lookup are retained per event in the machine artifact.

## Classification

Family mechanism: `internal_policy_prior_failure`.
Hard classification: `repeated_internal_prior_failure_identified`.

## Next Experiment

Exactly one next experiment: run a family-specific replay/target provenance audit to determine why the raw policy learned the degrading move; do not change search.
