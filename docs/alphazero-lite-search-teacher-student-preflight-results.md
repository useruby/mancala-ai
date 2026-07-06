# AlphaZero-Lite Search-Teacher Student Preflight Results

**Classification**: `distillation_pipeline_broken,architecture_change_not_justified`

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
| residual_v3_128x4 | student | residual_v3 | 128 | 4 | 1459f32bb29037fb00a7a906d4c829cafe245081c8e0c2623909862d7f47f0cf |

## Training Loss Table

| Candidate | Epoch | Policy loss | Value loss | Total loss | Grad norm |
|---|---|---|---|---|---|

## Probe Imitation Table

| Candidate | Top-1 | KL | Baseline KL | Legal failures | Value MAE | Sign acc | Entropy | Aborted |
|---|---|---|---|---|---|---|---|---|
| current_ref | +0.5156 | +0.3571 | +1.1988 | 0 | +0.1507 | +0.6503 | +1.6702 | True |
| residual_v3_96x3 | +0.3895 | +0.4764 | +1.1988 | 0 | +0.0221 | +0.5041 | +2.0929 | True |
| residual_v3_128x4 | +0.4143 | +0.4470 | +1.1988 | 0 | +0.0234 | +0.7461 | +2.0644 | True |

## Aborted-Candidate Table

| Candidate | Reasons |
|---|---|
| current_ref | teacher top-1 agreement < 55% |
| residual_v3_96x3 | teacher top-1 agreement < 55%; value sign accuracy < 55% |
| residual_v3_128x4 | teacher top-1 agreement < 55% |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | n/a | n/a | n/a | n/a | n/a | n/a |
| residual_v3_96x3 | n/a | n/a | n/a | n/a | n/a | n/a |
| residual_v3_128x4 | n/a | n/a | n/a | n/a | n/a | n/a |

## Held-Out Mean/Worst-Suite DS Table

| Candidate | Held-out mean 384:256 | Delta vs current | Held-out worst-suite 384:256 |
|---|---|---|---|
| current_ref | n/a | n/a | n/a |
| residual_v3_96x3 | n/a | n/a | n/a |
| residual_v3_128x4 | n/a | n/a | n/a |

## Bootstrap CI For Candidate Minus Current

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|
| none | n/a | n/a | n/a |

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| current_ref | n/a | n/a | n/a |
| residual_v3_96x3 | n/a | n/a | n/a |
| residual_v3_128x4 | n/a | n/a | n/a |

## Duplicate Trajectory Count

| Candidate | Mean duplicates |
|---|---|
| current_ref | n/a |
| residual_v3_96x3 | n/a |
| residual_v3_128x4 | n/a |

## Runtime Cost Comparison

| Candidate | Mean move latency ms | Mean p95 latency ms | Relative slowdown |
|---|---|---|---|
| current_ref | n/a | n/a | n/a |
| residual_v3_96x3 | n/a | n/a | n/a |
| residual_v3_128x4 | n/a | n/a | n/a |

## Gate Classification If Run

- current_ref: `not_run` reason=`skipped because no student survived the probe gate`
- residual_v3_96x3: `not_run` reason=`aborted after probe`
- residual_v3_128x4: `not_run` reason=`aborted after probe`

