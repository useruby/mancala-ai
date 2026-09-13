# Internal Family Replay Provenance Audit

Inherited PR #303 classification: `repeated_internal_prior_failure_identified`; family: `win_to_draw|neither|extra_turn|5-6`.
PR #303 SHA: `f683f5b86f4195432ab649a7e1a2e57664b26c794b26e13abfbe82627f82e7a6`; labels SHA: `5b1f6bac3d0289c62d74cccfa104fd3b31a41dc17eb7cfb24b457573df9f477c`.

## Primary States

Primary case: `45:uniform1200:capture_available-018`. Canonical ordered-state SHA: `6ee595c97df4d02627e26af354469dc39cf40f0b2b8ccf6e4070376a50101eb0`.

| state | events | opportunities | rate | lineage | mechanism |
| --- | ---: | ---: | ---: | --- | --- |
| `{"current_player":0,"opponent_pits":[1,7,0,2,2,7],"opponent_store":3,"player_pits":[2,4,1,7,7,1],"player_store":4}` | 1 | 10 | 0.100 | `both_descendants_repair` | `internal_prior_failure_inherited` |
| `{"current_player":0,"opponent_pits":[2,0,7,7,1,6],"opponent_store":2,"player_pits":[6,2,0,6,6,0],"player_store":3}` | 68 | 73 | 0.932 | `other` | `internal_prior_failure_heterogeneous` |

## Checkpoint Lineage

| state | parent top | control top | uniform top | parent/control/uniform optimal mass |
| --- | ---: | ---: | ---: | --- |
| `{"current_player":0,"opponent_pits":[1,7,0,2,2,7],"opponent_store":3,"player_pits":[2,4,1,7,7,1],"player_store":4}` | 1 | 5 | 5 | 0.119/0.342/0.788 |
| `{"current_player":0,"opponent_pits":[2,0,7,7,1,6],"opponent_store":2,"player_pits":[6,2,0,6,6,0],"player_store":3}` | 0 | 3 | 3 | 0.400/0.270/0.060 |

## Training Sources

| lane | source | SHA256 | raw rows | weight | effective rows | verified fraction |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| control | `dynamic_current_self_play` | `d914fba65da942ed17e9da706893a53653977d644d5cb4e08441dffff76278d0` | 64108 | 1 | 64108 | 1.000 |
| control | `family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl` | `787cca93c7db454c2daebe1fdab001e93df0b9f80a1e7005598a0324774c427b` | 20 | 1 | 20 | 0.000 |
| control | `guard_safe_controls_only.jsonl` | `ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5` | 5 | 2 | 10 | 0.000 |
| uniform1200 | `dynamic_current_self_play` | `514913e29d646f3a011347ec31a13ac33a2105be48c482e0816e53f92724ba57` | 70975 | 1 | 70975 | 1.000 |
| uniform1200 | `family_leave_one_out_without_opening_missed_extra_turn_continuation.jsonl` | `787cca93c7db454c2daebe1fdab001e93df0b9f80a1e7005598a0324774c427b` | 20 | 1 | 20 | 0.000 |
| uniform1200 | `guard_safe_controls_only.jsonl` | `ffac24dc8fb773fddd4dc14f397317d7c5e0a973631ca71adb216b5e68b444a5` | 5 | 2 | 10 | 0.000 |

Both lanes have the same non-search training recipe and source list. Their only non-search difference is none: control self-play uses 192 with an opening 384 minimum for 8 plies; uniform uses 1200 throughout. `train.py` realizes weights by tiling source-row indexes, so effective examples are `raw_rows * replay_weight`.

## Exact And One-Ply Coverage

All primary exact states are `both_exact_state_absent` in the verified control and uniform mixtures. This is not evidence of a uniform coverage deficit. Per-source exact rows, one-ply parents, legal children, tactical types, and row-weighted child coverage are retained in the machine result.

## Target Quality And Teacher Gap

Because all exact primary states are absent, exact-state stored-target quality, paired dynamic target comparison, source contribution, and aggregate teacher-to-student target rows are null rather than imputed. The report therefore makes no target-quality or retention claim.

## Control Repair And Uniform Introduction

The control-repair subgroup is empty. The parent-correct, uniform-degrading state is retained as `other` because the matched control is also degrading; it is not misclassified as a uniform-introduced failure.

## Broad-Family Negative Control

A SHA-ordered context-only sample contains 2 states. It does not alter the primary cohort.

## Evidence Scope

All three seed-45 sources were byte-verified against the historical manifest SHA records. Dynamic rows are labeled `dynamic_only`; all three-source conclusions are `full_verified_mixture`. No unavailable source was converted to zero coverage.

## Classification

`internal_prior_failure_heterogeneous`

## Next Experiment

Exactly one next experiment: take the highest-contribution canonical state cluster inside capture_available-018 and run a state-cluster-specific provenance audit.
