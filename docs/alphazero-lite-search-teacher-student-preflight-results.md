# AlphaZero-Lite Search-Teacher Student Preflight Results

**Classification**: `teacher_distillation_not_enough`

## Current Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Teacher Runtime Profile

- profile: `{"artifact": "model-artifact/current", "c_puct_schedule": {"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}, "default_c_puct": 1.25, "root_policy_mode": "deterministic", "root_prior_transform": null, "tactical_root_bias": 0.0, "value_transform": null}`

## Dataset Audit

- row count: `50000`
- unique state count: `30447`
- duplicate state rate: `0.3911`
- phase distribution: `{"late": 102, "mid": 33057, "opening": 16841}`
- seat distribution: `{"challenger_player_0": 25166, "challenger_player_1": 24834}`
- budget-context distribution: `{"1200:1200": 12503, "1200:256": 12495, "384:256": 12502, "768:768": 12500}`
- teacher raw-vs-PUCT top-1 disagreement rate: `0.7372`
- overlap with PR #149 failure states: `{"matching_rows": 6, "matching_unique_states": 4, "state_hash_prefixes_available": 28}`

## Model Architecture Table

| Candidate | Kind | Model | Trunk | Blocks | Weights SHA256 |
|---|---|---|---|---|---|
| current_ref | reference | residual_v3 | 96 | 3 | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a |
| residual_v3_96x3 | student | residual_v3 | 96 | 3 | 14dadd747382afc8f9a6609ad035b8db9e90b8fa42ece093191575a5919ee878 |

## Training Loss Table

| Candidate | Epoch | Policy loss | Value loss | Total loss | Grad norm |
|---|---|---|---|---|---|

## Probe Imitation Table

| Candidate | Top-1 | KL | Baseline KL | Legal failures | Value MAE | Sign acc | Entropy | Aborted |
|---|---|---|---|---|---|---|---|---|
| current_ref | +0.5156 | +0.3571 | +1.1988 | 0 | +0.1507 | +0.6503 | +1.6702 | True |
| residual_v3_96x3 | +0.3895 | +0.4764 | +1.1988 | 0 | +0.0221 | +0.5041 | +2.0929 | True |

## Aborted-Candidate Table

| Candidate | Reasons |
|---|---|
| current_ref | teacher top-1 agreement < 60% |
| residual_v3_96x3 | teacher top-1 agreement < 60% |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |
| residual_v3_96x3 | +0.1719 | +0.1719 | +0.0885 | +0.0234 | -0.0078 | +0.0000 |

## Held-Out Mean/Worst-Suite DS Table

| Candidate | Held-out mean 384:256 | DS delta vs current | Held-out worst-suite 384:256 |
|---|---|---|---|
| current_ref | -0.3084 | +0.0000 | -0.3255 |
| residual_v3_96x3 | +0.1762 | +0.4846 | +0.1719 |

## Bootstrap CI For DS

- Note: DS is the seat-split difference `P0 score - P1 score`. It tracks seat asymmetry, not overall disadvantaged-seat strength.

| Candidate | Budget | Orientation | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| residual_v3_96x3 | 384:256 | candidate_minus_current | +0.4846 | +0.4492 | +0.5195 |
| residual_v3_96x3 | 384:256 | current_minus_candidate | -0.4846 | -0.5195 | -0.4492 |
| residual_v3_96x3 | 768:768 | candidate_minus_current | -0.5894 | -0.6137 | -0.5649 |
| residual_v3_96x3 | 768:768 | current_minus_candidate | +0.5894 | +0.5649 | +0.6137 |
| residual_v3_96x3 | 1200:1200 | candidate_minus_current | -0.2509 | -0.2852 | -0.2166 |
| residual_v3_96x3 | 1200:1200 | current_minus_candidate | +0.2509 | +0.2166 | +0.2852 |
| residual_v3_96x3 | 1200:256 | candidate_minus_current | +0.1146 | +0.0955 | +0.1337 |
| residual_v3_96x3 | 1200:256 | current_minus_candidate | -0.1146 | -0.1337 | -0.0955 |

## Bootstrap CI For Disadvantaged-Seat Score

- Note: disadvantaged-seat score is the weaker-seat score itself. It is the promotion-style robustness metric and can move differently from DS.

| Candidate | Budget | Orientation | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| residual_v3_96x3 | 384:256 | candidate_minus_current | -0.9240 | -0.9336 | -0.9138 |
| residual_v3_96x3 | 384:256 | current_minus_candidate | +0.9240 | +0.9138 | +0.9336 |
| residual_v3_96x3 | 768:768 | candidate_minus_current | -0.1617 | -0.1745 | -0.1495 |
| residual_v3_96x3 | 768:768 | current_minus_candidate | +0.1617 | +0.1495 | +0.1745 |
| residual_v3_96x3 | 1200:1200 | candidate_minus_current | -0.3639 | -0.3813 | -0.3468 |
| residual_v3_96x3 | 1200:1200 | current_minus_candidate | +0.3639 | +0.3468 | +0.3813 |
| residual_v3_96x3 | 1200:256 | candidate_minus_current | -0.8032 | -0.8188 | -0.7878 |
| residual_v3_96x3 | 1200:256 | current_minus_candidate | +0.8032 | +0.7878 | +0.8188 |

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| current_ref | +0.6157 | +0.9240 | +0.3084 |
| residual_v3_96x3 | +0.1762 | +0.0000 | -0.1762 |

## Duplicate Trajectory Count

| Candidate | Mean duplicates |
|---|---|
| current_ref | +1536.0000 |
| residual_v3_96x3 | +1536.0000 |

## Runtime Cost Comparison

| Candidate | Mean move latency ms | Mean p95 latency ms | Relative slowdown |
|---|---|---|---|
| current_ref | n/a | n/a | n/a |
| residual_v3_96x3 | n/a | n/a | n/a |

## Gate Classification If Run

- residual_v3_96x3: `not_run` reason=`aborted after probe`
- current_ref: `not_run` reason=`skipped because no student survived the probe gate`
