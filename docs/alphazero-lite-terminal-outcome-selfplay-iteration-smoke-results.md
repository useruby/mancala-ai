# AlphaZero-Lite Terminal-Outcome Self-Play Iteration Smoke Results

**Date**: 2026-07-12

**Classification**: `selfplay_training_pipeline_not_ready`

## Current Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Promoted Runtime Profile Confirmation

- default runtime schedule: `{"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}`
- self-play runtime profiles: `{"384:384": {"c_puct": 1.25, "hash": "a452d2bc62f56048c9dc173aff149793a0ee84f0041f854b5608629ac49d14da", "kind": "self_play", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"}, "768:768": {"c_puct": 0.9, "hash": "aed239f4bd4aa59665ae660abf88e0333264c4198199925c54f3e92e17e5c5d9", "kind": "self_play", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"}}`

## Self-Play Replay Audit

- replay rows: `40871`
- games completed: `1024`
- unique states: `12116`
- duplicate state rate: `0.7036`
- invalid policies: `0`
- invalid legal masks: `0`
- outcome distribution: `{"draw": 3204, "loss": 17883, "win": 19784}`
- phase distribution: `{"late": 14799, "mid": 16856, "opening": 9216}`
- seat distribution: `{"player_0": 21479, "player_1": 19392}`

## Root-Value Vs Terminal-Outcome Calibration

- correlation: `+0.7065`
- root value mean: `+0.0174`
- terminal outcome mean: `+0.0465`

## Candidate Table

| Candidate | Scope | Epochs | Artifact SHA | Probe pass |
|---|---|---|---|---|
| current_ref | none | 0 | 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a | True |
| value_head_only_e1 | value_head_only | 1 | bb22400009b43581d12552278135c8f5652403e0735514749ed92c99e136a4a5 | False |
| policy_value_heads_e1 | policy_value_heads | 1 | b4ba7e7d78987cbb9b90ca6fd60b8339482f3d23b9b84f8df372effca9078486 | False |
| value_head_only_e2 | value_head_only | 2 | 69f56d96e86970a4aca16c23f4215bde04db199baf85f7e58271dc191de1ef8f | False |
| value_head_only_e2_low_lr | value_head_only | 2 | 403ce0a57ffe806428b2bf144371e7036e0c8598fe40153427f702612be00622 | False |
| policy_value_heads_e2 | policy_value_heads | 2 | None | False |
| value_head_only_best_probe | value_head_only | 2 | 69f56d96e86970a4aca16c23f4215bde04db199baf85f7e58271dc191de1ef8f | False |

## Training Loss Table

| Candidate | Policy loss | Value loss | Total loss | Grad norm |
|---|---|---|---|---|
| current_ref | +0.0000 | +0.0000 | +0.0000 | n/a |
| value_head_only_e1 | +1.0659 | +0.3158 | +1.5396 | n/a |
| policy_value_heads_e1 | +1.0602 | +0.3234 | +1.2542 | n/a |
| value_head_only_e2 | +1.0638 | +0.3024 | +1.4418 | n/a |
| value_head_only_e2_low_lr | +1.0659 | +0.3055 | +1.3714 | n/a |
| policy_value_heads_e2 | +0.0000 | +0.0000 | +0.0000 | n/a |
| value_head_only_best_probe | +1.0638 | +0.3024 | +1.4418 | n/a |

## Probe Table

| Candidate | Legal fails | Policy KL | Top-1 change | Value MAE | Sign acc | Value corr | Passed |
|---|---|---|---|---|---|---|---|
| current_ref | 0 | +0.3397 | +0.0000 | +0.7655 | +0.6577 | +0.5272 | True |
| value_head_only_e1 | 0 | +0.3397 | +0.0000 | +0.7391 | +0.6768 | +0.5526 | False |
| policy_value_heads_e1 | 0 | +0.3345 | +0.0166 | +0.7609 | +0.6646 | +0.5331 | False |
| value_head_only_e2 | 0 | +0.3397 | +0.0000 | +0.7278 | +0.6792 | +0.5631 | False |
| value_head_only_e2_low_lr | 0 | +0.3397 | +0.0000 | +0.7331 | +0.6782 | +0.5584 | False |
| policy_value_heads_e2 | 0 | +0.3345 | +0.0166 | +0.7609 | +0.6646 | +0.5331 | False |
| value_head_only_best_probe | 0 | +0.3397 | +0.0000 | +0.7278 | +0.6792 | +0.5631 | False |

## Search Drift Diagnostics

| Candidate | Mean |dV| | Changed-row |dV| | 768 |dV| | 768 changed-row |dV| | 768 changed rate |
|---|---|---|---|---|---|
| current_ref | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 |
| value_head_only_e1 | +0.0443 | +0.0622 | +0.0455 | +0.0651 | +0.0957 |
| policy_value_heads_e1 | +0.0106 | +0.0168 | +0.0119 | +0.0193 | +0.0625 |
| value_head_only_e2 | +0.0553 | +0.0791 | +0.0557 | +0.0830 | +0.1006 |
| value_head_only_e2_low_lr | +0.0486 | +0.0718 | +0.0491 | +0.0739 | +0.1045 |
| policy_value_heads_e2 | +0.0106 | +0.0168 | +0.0119 | +0.0193 | +0.0625 |
| value_head_only_best_probe | +0.0553 | +0.0791 | +0.0557 | +0.0830 | +0.1006 |

## Value-Search Drift Cases

- candidate: `value_head_only_best_probe`
- budget: `768:768`
- changed rows: `103` / `1024`
- mean |dV|: `+0.0557`
- changed-row mean |dV|: `+0.0830`

| Game | Ply | Phase | Seat | Current move | Cand move | Current V | Cand V | Delta V |
|---|---|---|---|---|---|---|---|---|
| 948 | 31 | late | player_0 | 4 | 5 | +0.3481 | +0.6047 | +0.2566 |
| 1004 | 31 | late | player_0 | 4 | 5 | +0.3481 | +0.6047 | +0.2566 |
| 894 | 27 | mid | player_0 | 4 | 1 | +0.3198 | +0.5461 | +0.2263 |
| 877 | 15 | mid | player_0 | 2 | 0 | +0.5391 | +0.3259 | -0.2132 |
| 947 | 39 | late | player_1 | 1 | 3 | +0.4125 | +0.6165 | +0.2040 |
| 975 | 38 | late | player_1 | 0 | 4 | -0.3369 | -0.4970 | -0.1600 |
| 888 | 2 | opening | player_1 | 5 | 1 | -0.0783 | -0.2320 | -0.1536 |
| 892 | 2 | opening | player_1 | 5 | 1 | -0.0783 | -0.2320 | -0.1536 |
| 906 | 2 | opening | player_1 | 5 | 1 | -0.0783 | -0.2320 | -0.1536 |
| 913 | 2 | opening | player_1 | 5 | 1 | -0.0783 | -0.2320 | -0.1536 |

## Aborted-Candidate Table

| Candidate | Reasons |
|---|---|
| value_head_only_e1 | value sign accuracy gain < +0.03 vs current_ref |
| policy_value_heads_e1 | value sign accuracy gain < +0.03 vs current_ref |
| value_head_only_e2 | value sign accuracy gain < +0.03 vs current_ref; 768:768 search changed rate > 0.10 |
| value_head_only_e2_low_lr | value sign accuracy gain < +0.03 vs current_ref; 768:768 search changed rate > 0.10 |
| policy_value_heads_e2 | policy_value_heads_e1 probes were not stable; e2 not trained |
| value_head_only_best_probe | value sign accuracy gain < +0.03 vs current_ref; 768:768 search changed rate > 0.10 |

## Medium DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| not_run | n/a | n/a | n/a | n/a | n/a | n/a |

## Fixed-Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| not_run | n/a | n/a | n/a | n/a | n/a | n/a |

## Bootstrap CIs

- orientation: `candidate_minus_current`

| Candidate | Budget | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|
| not_run |  |  |  |  |

## P0/P1 Split For 384:256

| Candidate | P0 changed | P1 changed |
|---|---|---|
| current_ref | +0.0000 | +0.0000 |
| policy_value_heads_e1 | +0.0409 | +0.0372 |
| policy_value_heads_e2 | +0.0409 | +0.0372 |
| value_head_only_best_probe | +0.0877 | +0.0763 |
| value_head_only_e1 | +0.0604 | +0.0587 |
| value_head_only_e2 | +0.0877 | +0.0763 |
| value_head_only_e2_low_lr | +0.0819 | +0.0607 |

## Duplicate Trajectory Count

| Candidate | 384:256 duplicates |
|---|---|
| not_run | n/a |

## Runtime Cost

| Phase | Seconds |
|---|---|
| selfplay_generation | +163.72 |
| suite_eval_skipped_after_probe | +1.00 |
| training_initial | +8.41 |
| training_value_e2 | +37.97 |
| training_value_e2_low_lr | +38.36 |

## Gate Result

- gate run: `False`
- gate passed: `False`
- gate candidate: `None`

## Diagnosis And Next Experiment

- outcome-grounded value learning is real: the best value-only lane improved sign accuracy to `+0.6851` and root/outcome correlation to `+0.5633`.
- the blocking failure mode is search/value integration sensitivity, not raw policy drift: value-head-only lanes keep raw policy top-1 unchanged while `768:768` search flips rise once mean root-value deltas reach roughly `+0.05` and changed-row deltas reach roughly `+0.08`.
- focused `768:768` integration diagnostics show move flips are driven by Q/selection-score reordering under materially shifted root values, with repeated changed rows where priors stay similar but backed-up values reorder the best move.
- `normalize_values=True` is not a safe runtime fix for this candidate: it sharply improves low-search budgets like `384:256`, but consistently regresses `768:768` and `1200:1200` across fixed-large and held-out suites.
- recommended next experiment: keep the current strict gates, treat `normalize_values` as diagnostic-only, and run a narrower search-control experiment that targets equal-search sensitivity without globally reshaping all budgets.

## Final Classification

- result: `selfplay_training_pipeline_not_ready`
