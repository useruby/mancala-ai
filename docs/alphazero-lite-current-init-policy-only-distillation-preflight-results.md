# AlphaZero-Lite Current-Init Policy-Only Distillation Preflight Results

**Classification**: `reporting_pipeline_fixed`

## Current Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## PR #152 Artifact Hashes

- PR #152 current-init policy-only e2 weights SHA256: `58f2ed0ddb4a36e763e796806f03fe8c1e94d7336570f6f9d7bb1308224db4b7`
- PR #152 current-init policy-only e2 checkpoint SHA256: `636d3c4a8e37c4464a8308571bace78e446aaf072e8c897a7963372e96c943d0`

## DS Orientation Audit

- audit passed: `True`
- reused PR #152 DS audit helper: `True`
- bootstrap orientations are explicit: `True`

## Candidate Table

| Candidate | Kind | Model | Trunk | Blocks | Weights SHA256 |
|---|---|---|---|---|---|
| current_ref | reference | residual_v3 | 96 | 3 | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a |
| pr152_current_init_policy_only_e2 | student | residual_v3 | 96 | 3 | 58f2ed0ddb4a36e763e796806f03fe8c1e94d7336570f6f9d7bb1308224db4b7 |
| current_init_policy_only_e1_repro | student | residual_v3 | 96 | 3 | e80f0e906ff08661d230334df76b37e5497ae14d3e98c2ddb457d95ec868480b |
| current_init_policy_only_e2_repro | student | residual_v3 | 96 | 3 | 55b0a78184d84ce7fd19f0ba2ebea35558d3437044d24baa4d4610251fae1ee6 |

## Training/Provenance Table

| Candidate | Provenance | Epochs | Trainable scope | Init checkpoint |
|---|---|---|---|---|
| current_ref | checked_in_current | None | None | None |
| pr152_current_init_policy_only_e2 | reused_pr152_artifact | 2 | policy_head | model-artifact/current |
| current_init_policy_only_e1_repro | trained_in_this_run | 1 | policy_head | /tmp/azlite_current_init_policy_only_distillation/current_materialized_checkpoint.npz |
| current_init_policy_only_e2_repro | trained_in_this_run | 2 | policy_head | /tmp/azlite_current_init_policy_only_distillation/current_materialized_checkpoint.npz |

## Training Loss Table

| Candidate | Epoch | Policy loss | Value loss | Total loss | Grad norm |
|---|---|---|---|---|---|
| none |  |  |  |  |  |

## Policy Probe Table

| Candidate | Teacher PUCT top-1 | KL | Entropy | Legal fails | Changed raw top-1 vs current |
|---|---|---|---|---|---|
| current_ref | +0.5278 | +0.3550 | +1.6803 | 0 | +0.0000 |
| pr152_current_init_policy_only_e2 | +0.5518 | +0.3112 | +1.8327 | 0 | +0.1598 |
| current_init_policy_only_e1_repro | +0.5448 | +0.3163 | +1.8288 | 0 | +0.1570 |
| current_init_policy_only_e2_repro | +0.5514 | +0.3112 | +1.8379 | 0 | +0.1535 |

## Value/Trunk Preservation Table

| Candidate | Max value diff | Mean value diff | Root MAE vs current | Preserved | Changed keys |
|---|---|---|---|---|---|
| current_ref | +0.0000 | +0.0000 | +0.0000 | True | none |
| pr152_current_init_policy_only_e2 | +0.0000 | +0.0000 | +0.0000 | True | none |
| current_init_policy_only_e1_repro | +0.0000 | +0.0000 | +0.0000 | True | none |
| current_init_policy_only_e2_repro | +0.0000 | +0.0000 | +0.0000 | True | none |

## Search-Aware Probe Table

| Candidate | Search agree with teacher | Changed search move rate | Root visit KL vs current | Selected visit-share delta | Probe pass |
|---|---|---|---|---|---|
| current_ref | +1.0000 | +0.0000 | +0.0000 | +0.0000 | True |
| pr152_current_init_policy_only_e2 | +0.8117 | +0.1883 | +0.0981 | -0.0109 | False |
| current_init_policy_only_e1_repro | +0.8215 | +0.1785 | +0.0901 | -0.0130 | False |
| current_init_policy_only_e2_repro | +0.8126 | +0.1874 | +0.0913 | -0.0123 | False |

## Probe Gate Result Table

| Candidate | Passed | Reasons |
|---|---|---|
| current_ref | True | passed |
| pr152_current_init_policy_only_e2 | False | 768:768 changed search move rate > 0.12; 1200:1200 changed search move rate > 0.12 |
| current_init_policy_only_e1_repro | False | 768:768 changed search move rate > 0.12; 1200:1200 changed search move rate > 0.12 |
| current_init_policy_only_e2_repro | False | 768:768 changed search move rate > 0.12; 1200:1200 changed search move rate > 0.12 |

## Medium DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | -0.2344 | -0.3750 | +0.6484 | +0.3906 | -0.1367 | -0.3750 |

## Fixed-Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| current_ref | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |

## Held-Out Mean/Worst-Suite DS Table

| Candidate | Mean 384:256 | Worst 384:256 | Mean 768:768 | Mean 1200:1200 | Mean 1200:256 |
|---|---|---|---|---|---|
| not_run |  |  |  |  |  |

## Bootstrap CIs

Every bootstrap row below names its orientation explicitly.

| Candidate | Budget | Orientation | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| not_run |  |  |  |  |  |

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| not_run |  |  |  |

## Duplicate Trajectory Count

| Candidate | Mean duplicates |
|---|---|
| not_run |  |

## Runtime Cost

| Candidate | Mean move latency ms | Mean p95 latency ms |
|---|---|---|
| not_run |  |  |

## Gate Result

- gate not run

## Final Classification

- result: `reporting_pipeline_fixed`
