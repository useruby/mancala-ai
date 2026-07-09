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
| residual_v3_96x3 | student | residual_v3 | 96 | 3 | 0e3a1200cbf46a50fe11038379ddd8ee3f8858fa7937ee1e371084f383d06f8d |

## Training Loss Table

| Candidate | Epoch | Policy loss | Value loss | Total loss | Grad norm |
|---|---|---|---|---|---|

## Probe Imitation Table

| Candidate | Top-1 | KL | Baseline KL | Legal failures | Value MAE | Sign acc | Entropy | Aborted |
|---|---|---|---|---|---|---|---|---|
| current_ref | +0.5156 | +0.3571 | +1.1988 | 0 | +0.1507 | +0.6503 | +1.6702 | True |
| residual_v3_96x3 | +0.6726 | +0.2205 | +1.1988 | 0 | +0.0195 | +0.5235 | +1.5308 | False |

## Aborted-Candidate Table

| Candidate | Reasons |
|---|---|
| current_ref | teacher top-1 agreement < 60% |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |
| residual_v3_96x3 | +0.0234 | +0.0651 | +0.4089 | +0.2943 | +0.2370 | +0.1068 |

## Held-Out Mean/Worst-Suite DS Table

| Candidate | Held-out mean 384:256 | Delta vs current | Held-out worst-suite 384:256 |
|---|---|---|---|
| current_ref | -0.3051 | +0.0000 | -0.3112 |
| residual_v3_96x3 | -0.0026 | +0.3025 | -0.0195 |

## Bootstrap CI For Candidate Minus Current

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|
| residual_v3_96x3_minus_current_ref_1200_1200 | -0.1706 | -0.1988 | -0.1415 |
| residual_v3_96x3_minus_current_ref_1200_256 | -0.6697 | -0.6957 | -0.6424 |
| residual_v3_96x3_minus_current_ref_384_256 | -0.6124 | -0.6411 | -0.5829 |
| residual_v3_96x3_minus_current_ref_768_768 | +0.0087 | -0.0156 | +0.0334 |

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| current_ref | +0.6137 | +0.9188 | +0.3051 |
| residual_v3_96x3 | +0.3038 | +0.3064 | +0.0026 |

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

- current_ref: `not_run` reason=`n/a`
- residual_v3_96x3: `not_run` reason=`did not clear post-probe held-out robustness gate for explicit gate run`

