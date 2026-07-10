# AlphaZero-Lite Search-Teacher Distillation Value Ablation Results

**Classification**: `metric_reporting_bug`

## Current Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## PR #151 Artifact And Dataset Hashes

- PR #151 student artifact SHA256: `14dadd747382afc8f9a6609ad035b8db9e90b8fa42ece093191575a5919ee878`
- teacher dataset SHA256: `8274ee91379945826a80912c6db40e9c609bf1bedacc3f5215b82ee5b70b56d2`
- teacher dataset audit SHA256: `dba3c149f55c43c3c7af865ceb6b93ef564d37ca52d334409f5ef0d1a1d4be03`

## DS Orientation Audit

- audit passed: `False`
- inferred existing report orientation: `mixed_metrics_or_inverted_labels`
- warning: `Existing PR #151 report mixes DS deltas with disadvantaged-seat bootstrap signs; do not use it for promotion reasoning.`

## Value Target Audit

- stored value target distribution: `{"count": 50000, "max": 1.0, "mean": 0.00076, "median": 0.0, "min": -1.0, "negative": 205, "p25": 0.0, "p75": 0.0, "positive": 243, "zero": 49552}`
- teacher root value distribution: `{"count": 50000, "max": 0.7658968567848206, "mean": 0.04061829488074116, "median": 0.05013531446456909, "min": -0.9766600131988525, "negative": 17865, "p25": -0.041000060737133026, "p75": 0.14890959858894348, "positive": 32135, "zero": 0}`
- final outcome distribution: `{"count": 50000, "max": 1.0, "mean": 0.00076, "median": 0.0, "min": -1.0, "negative": 205, "p25": 0.0, "p75": 0.0, "positive": 243, "zero": 49552}`
- corr(stored, root): `+0.1321`
- corr(stored, outcome): `+1.0000`

## Candidate Lane Table

| Candidate | Kind | Model | Trunk | Blocks | Weights SHA256 |
|---|---|---|---|---|---|
| current_ref | reference | residual_v3 | 96 | 3 | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a |
| pr151_student_ref | reference | residual_v3 | 96 | 3 | 14dadd747382afc8f9a6609ad035b8db9e90b8fa42ece093191575a5919ee878 |
| scratch_96x3_outcome_value_repro | student | residual_v3 | 96 | 3 | 86b8d5c3c6a9b3a4aa99c72e26db73a422046c6a0fb4fdca575ca10b134a4b20 |
| scratch_96x3_teacher_root_value_e2 | student | residual_v3 | 96 | 3 | ab235ab0cd845d932947124702030c3af989d20fba6be0232f1426733543cc20 |
| scratch_96x3_blend_root75_outcome25_e2 | student | residual_v3 | 96 | 3 | 782ed53c0063a8e1c451d39c9370b86a2e1b04e490ee87b6e6197e2b764c0b58 |
| current_init_96x3_policy_only_e2 | student | residual_v3 | 96 | 3 | 58f2ed0ddb4a36e763e796806f03fe8c1e94d7336570f6f9d7bb1308224db4b7 |

## Training Loss Table

| Candidate | Epoch | Policy loss | Value loss | Total loss | Grad norm |
|---|---|---|---|---|---|
| scratch_96x3_outcome_value_repro | 1 | +1.5057 | +0.0045 | +1.5102 | +0.0673 |
| scratch_96x3_outcome_value_repro | 2 | +1.4655 | +0.0045 | +1.4700 | +0.1909 |
| scratch_96x3_teacher_root_value_e2 | 1 | +1.5118 | +0.0202 | +1.5320 | +0.0968 |
| scratch_96x3_teacher_root_value_e2 | 2 | +1.4756 | +0.0158 | +1.4914 | +0.1463 |
| scratch_96x3_blend_root75_outcome25_e2 | 1 | +1.5098 | +0.0102 | +1.5200 | +0.0797 |
| scratch_96x3_blend_root75_outcome25_e2 | 2 | +1.4726 | +0.0095 | +1.4822 | +0.1660 |
| current_init_96x3_policy_only_e2 | 1 | +1.2937 | +0.0203 | +1.3140 | +0.1515 |
| current_init_96x3_policy_only_e2 | 2 | +1.2813 | +0.0203 | +1.3016 | +0.1287 |

## Probe Metrics Table

| Candidate | Top-1 | KL | Entropy | Outcome MAE | Root MAE | Search agree | Legal fails |
|---|---|---|---|---|---|---|---|
| current_ref | +0.5156 | +0.3571 | +1.6702 | +0.1507 | +0.0000 | +1.0000 | 0 |
| pr151_student_ref | +0.3895 | +0.4764 | +2.0929 | +0.0221 | +0.1446 | +0.3916 | 0 |
| scratch_96x3_outcome_value_repro | +0.3902 | +0.4721 | +2.0948 | +0.0210 | +0.1428 | +0.3950 | 0 |
| scratch_96x3_teacher_root_value_e2 | +0.3833 | +0.4764 | +2.0988 | +0.0590 | +0.1292 | +0.4058 | 0 |
| scratch_96x3_blend_root75_outcome25_e2 | +0.3941 | +0.4729 | +2.0950 | +0.0420 | +0.1335 | +0.3862 | 0 |
| current_init_96x3_policy_only_e2 | +0.5362 | +0.3126 | +1.8243 | +0.1507 | +0.0000 | +0.8079 | 0 |

## Aborted-Candidate Table

| Candidate | Reasons |
|---|---|
| scratch_96x3_outcome_value_repro | teacher top-1 agreement < 0.60; value root MAE is worse than current_ref by more than 10%; 384:256 top-1 agreement < 0.50; 768:768 top-1 agreement < 0.50; 1200:1200 top-1 agreement < 0.50; 1200:256 top-1 agreement < 0.50 |
| scratch_96x3_teacher_root_value_e2 | teacher top-1 agreement < 0.60; KL is worse than PR #151 student; value root MAE is worse than current_ref by more than 10%; 384:256 top-1 agreement < 0.50; 768:768 top-1 agreement < 0.50; 1200:1200 top-1 agreement < 0.50; 1200:256 top-1 agreement < 0.50 |
| scratch_96x3_blend_root75_outcome25_e2 | teacher top-1 agreement < 0.60; value root MAE is worse than current_ref by more than 10%; 384:256 top-1 agreement < 0.50; 768:768 top-1 agreement < 0.50; 1200:1200 top-1 agreement < 0.50; 1200:256 top-1 agreement < 0.50 |
| current_init_96x3_policy_only_e2 | teacher top-1 agreement < 0.60 |

## Fixed Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |
| pr151_student_ref | +0.1719 | +0.1719 | +0.0885 | +0.0234 | -0.0078 | +0.0000 |

## Held-Out Mean/Worst-Suite DS Table

| Candidate | Mean 384:256 | Worst 384:256 | Mean 768:768 | Mean 1200:1200 | Mean 1200:256 |
|---|---|---|---|---|---|
| current_ref | -0.3084 | -0.3255 | +0.6766 | +0.2721 | -0.1107 |
| pr151_student_ref | +0.1762 | +0.1719 | +0.0872 | +0.0213 | +0.0039 |

## Bootstrap CIs

| Comparison | Cand-Cur mean | Lower | Upper | Cur-Cand mean | Lower | Upper |
|---|---|---|---|---|---|---|
| pr151_student_ref 384:256 | +0.4846 | +0.4492 | +0.5195 | -0.4846 | -0.5195 | -0.4492 |
| pr151_student_ref 768:768 | -0.5894 | -0.6137 | -0.5649 | +0.5894 | +0.5649 | +0.6137 |
| pr151_student_ref 1200:1200 | -0.2509 | -0.2852 | -0.2166 | +0.2509 | +0.2166 | +0.2852 |
| pr151_student_ref 1200:256 | +0.1146 | +0.0955 | +0.1337 | -0.1146 | -0.1337 | -0.0955 |

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| current_ref | +0.6157 | +0.9240 | +0.3084 |
| pr151_student_ref | +0.1762 | +0.0000 | -0.1762 |

## Duplicate Trajectory Count

| Candidate | Mean duplicates |
|---|---|
| current_ref | +1536.0000 |
| pr151_student_ref | +1536.0000 |

## Gate Classification If Run

- pr151_student_ref: `not_run` reason=`did not clear audited held-out robustness gate for explicit gate run`

## Final Recommendation

- result: `metric_reporting_bug`
