# Uniform1200 Exact Population Audit

Classification: `forensic_exact_reference_mismatch_robust`.

## Combined Oracle Coverage

| Radius | Exact solved | Total | Unresolved |
| ---: | ---: | ---: | ---: |
| 0 | 36 | 38 | 2 |
| 1 | 196 | 199 | 3 |
| 2 | 936 | 953 | 17 |

## Frozen Reference Versus Exact

Observed mismatches: 25/38; adversarial bounds: 0.658 to 0.711; `forensic_exact_reference_mismatch_robust`.

## Missingness And Local Bounds

All rates retain each registered neighborhood denominator. The machine-readable result includes state characteristics, per-anchor coverage, raw network evaluations, state classifications, and per-seed regret/value deltas.

| Anchor | N | S | U | Classification |
| --- | ---: | ---: | ---: | --- |
| capture_available-002 | 30 | 30 | 0 | `robust_anchor_specific_regression` |
| capture_available-020 | 25 | 25 | 0 | `robust_anchor_specific_regression` |
| early_extra_turn-013 | 39 | 39 | 0 | `robust_anchor_specific_regression` |
| early_extra_turn-014 | 37 | 37 | 0 | `robust_anchor_specific_regression` |
| early_extra_turn-015 | 39 | 39 | 0 | `robust_anchor_specific_regression` |
| early_extra_turn-017 | 39 | 39 | 0 | `robust_anchor_specific_regression` |
| early_extra_turn-018 | 37 | 37 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-001 | 25 | 25 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-010 | 26 | 26 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-011 | 25 | 25 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-014 | 31 | 31 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-019 | 35 | 35 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-020 | 36 | 36 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-021 | 34 | 34 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-023 | 31 | 31 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-025 | 31 | 31 | 0 | `robust_anchor_specific_regression` |
| high_imbalance-026 | 38 | 38 | 0 | `robust_anchor_specific_regression` |
| high_value_swing-003 | 29 | 29 | 0 | `robust_anchor_specific_regression` |
| high_value_swing-008 | 30 | 30 | 0 | `robust_anchor_specific_regression` |
| high_value_swing-011 | 32 | 32 | 0 | `robust_anchor_specific_regression` |
| high_value_swing-017 | 30 | 30 | 0 | `robust_anchor_specific_regression` |
| high_value_swing-023 | 30 | 30 | 0 | `robust_anchor_specific_regression` |
| high_value_swing-025 | 28 | 28 | 0 | `robust_anchor_specific_regression` |
| incumbent_proxy_disagreement-028 | 38 | 38 | 0 | `robust_anchor_specific_regression` |
| incumbent_proxy_disagreement-032 | 32 | 32 | 0 | `robust_anchor_specific_regression` |
| opening_plies_1_8-009 | 34 | 26 | 8 | `missingness_or_model_mixed` |
| opening_plies_1_8-025 | 37 | 37 | 0 | `robust_anchor_specific_regression` |
| opening_plies_1_8-064 | 35 | 23 | 12 | `missingness_or_model_mixed` |
| opening_plies_1_8-070 | 32 | 32 | 0 | `robust_anchor_specific_regression` |
| opening_plies_1_8-071 | 33 | 32 | 1 | `robust_anchor_specific_regression` |
| sparse_endgame-001 | 11 | 11 | 0 | `robust_anchor_specific_regression` |
| sparse_endgame-009 | 20 | 20 | 0 | `robust_anchor_specific_regression` |
| starvation_pressure-002 | 33 | 33 | 0 | `robust_anchor_specific_regression` |
| starvation_pressure-003 | 33 | 33 | 0 | `robust_anchor_specific_regression` |
| starvation_pressure-007 | 30 | 29 | 1 | `robust_anchor_specific_regression` |
| starvation_pressure-009 | 29 | 29 | 0 | `robust_anchor_specific_regression` |
| starvation_pressure-010 | 27 | 27 | 0 | `robust_anchor_specific_regression` |
| starvation_pressure-026 | 35 | 35 | 0 | `robust_anchor_specific_regression` |

## Missingness Map

The JSON records solved/unresolved counts by radius, anchor ID, anchor family, bucket, active stones, legal actions, phase/move-index availability, extra-turn availability, and capture availability. These descriptors do not redefine the cohort.

## Raw-Network Exact Results

Each exact-solved state and matched seed has its legal-masked policy, selected action, network value, optimal-set correctness, integer exact regret, absolute exact-value error, and control-to-uniform deltas in the JSON. Value perspective is the existing root-training perspective.

## Improvement Controls

Equal-anchor weighting is primary; state weighting is secondary.

| Seed | Regression primary lower/upper | Improvement primary lower/upper |
| ---: | ---: | ---: |
| 44 | 0.165/0.182 | 0.000/0.979 |
| 45 | 0.129/0.146 | 0.003/0.982 |
| 46 | 0.134/0.151 | 0.001/0.980 |

## Replay And Intervention Diagnostics

No robust stable regression family was identified. Since the frozen reference is robustly invalid, model-vs-reference failures are not interpreted as ground truth; replay supervision, teacher-student inversion, and PR #290 repair diagnostics remain intentionally uninterpreted.

## Hard Classification

`forensic_exact_reference_mismatch_robust`

## One Recommended Next Experiment

rebuild/revalidate forensic references against exact play before another training intervention.
