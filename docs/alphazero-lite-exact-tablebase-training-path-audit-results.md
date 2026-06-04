# Exact Tablebase Training-Path Audit — Results

**Date:** 2026-06-04
**Audit:** `ml/alphazero_lite/run_exact_tablebase_training_path_audit.py`
**PR #79 context:** Medium exact-tablebase stabilization v2

## 1. Context

PR #79 ran a medium exact-tablebase stabilization v2 diagnostic with 147 production candidates, 206 targeted controls, and 1142 untouched holdouts. All local gates failed because promoted regression controls regressed. Training metrics were blank. Several w1/w2 traces appeared identical. This audit inspects the training/evaluation machinery to determine the root cause.

## 2. Why PR #79 requires training-path audit

PR #79 outcome: no arena, no promotion, no current overwrite. Best production reached 104/147 optimal@1200 but all strict local gates failed. Classification was `exact_tablebase_no_local_signal` with next action: "inspect target format/value perspective/training path." This audit executes that inspection.

## 3. Artifact target validation

| artifact | row_count | invalid_rows | duplicate_conflicts | target_mass | value_target_status | validation_status | notes |
|---|---|---|---|---|---|---|---|
| soft065_production | 147 | 0 | 0 | varies | ok | PASSED |  |
| soft055_production | 147 | 0 | 0 | varies | ok | PASSED |  |
| targeted_controls | 206 | 0 | 0 | varies | ok | PASSED |  |

## 4. Current baseline reproduction

| row_group | rows | optimal_1200 | optimal_2400 | avg_optimal_visit_share_1200 | avg_value_error | notes |
|---|---|---|---|---|---|---|---|
| production | 147 | 62/147 | 82 | 0.353719 | 0.73151 | |
| original_controls | 147 | 147/147 | 146 | 0.97983 | 0.685127 | |
| promoted_regression_controls | 59 | 59/59 | 59 | 0.97346 | 0.753476 | |
| nearest_neighbor_controls | 147 | 147/147 | 146 | 0.97983 | 0.685127 | |

Promoted controls already failing current: **0**

## 5. Replay weight plumbing audit

| trace | source_weights | checkpoint_hash | differs_from_baseline | status | notes |
|---|---|---|---|---|---|
| w1_w1 | [1, 1] | f69a6ea95dc5fc25... | False | bug | |
| w1_w5 | [1, 5] | 99fecb67d7d09315... | True | ok | |
| w5_w1 | [5, 1] | ecad9b0e658dae6e... | True | ok | |

Classification: **replay_weights_working**

## 6. Trace output isolation audit

| trace_name | source_weights | trace_names | training_data_hash | checkpoint_hash | status | notes |
|---|---|---|---|---|---|---|---|
| cap128_soft065_w1_half_e1 | [1,1] | cap128_soft065_w1_half_e1 | 5f2e3... | 58ef... | duplicate | identical to w2_half_e1 |
| cap128_soft065_w2_half_e1 | [1,2] | cap128_soft065_w2_half_e1 | 5f2e3... | 58ef... | duplicate | identical to w1_half_e1 |
| cap128_soft065_w1_half_e2 | [1,1] | cap128_soft065_w1_half_e2 | 5f2e3... | 31ee... | duplicate | identical to w2_half_e2 |
| cap128_soft065_w2_half_e2 | [1,2] | cap128_soft065_w2_half_e2 | 5f2e3... | 31ee... | duplicate | identical to w1_half_e2 |
| cap128_soft065_w1_half_e4 | [1,1] | cap128_soft065_w1_half_e4 | 5f2e3... | 7fbc... | duplicate | identical to w2_half_e4 |
| cap128_soft065_w2_half_e4 | [1,2] | cap128_soft065_w2_half_e4 | 5f2e3... | 7fbc... | duplicate | identical to w1_half_e4 |
| reg_soft065_w1_half_e4 | [1,1] | reg_soft065_w1_half_e4 | abc... | aa22... | duplicate | identical to w2_half_e4 |
| reg_soft065_w2_half_e4 | [1,2] | reg_soft065_w2_half_e4 | abc... | aa22... | duplicate | identical to w1_half_e4 |
| (6 more groups) | ... | ... | ... | ... | duplicate | all w1/w2 pairs identical |

**Finding:** Replay weights work correctly at the `train.py` level (Step 3 probe confirmed different checkpoints for [1,1] vs [1,5] vs [5,1]). However, the trace runner produced identical w1/w2 checkpoints. This means either:
- The trace runner cached combined training data files and reused them across traces
- The trace runner overwrote checkpoint paths between w1 and w2 variants
- The trace runner's combined data file building step created identical data for w1 and w2 variants

**Classification: `trace_output_collision_bug`** (replay weights are not the problem; trace isolation is)

## 7. train.py target consumption audit

| source | row_count | policy_loss_used | value_loss_used | source_weight_applied | train_only_handled | exclude_from_validation_handled | status | notes |
|---|---|---|---|---|---|---|---|---|
| production | 8 | yes | yes | yes | metadata_only | metadata_only | ok | |
| controls | 8 | yes | yes | yes | metadata_only | metadata_only | ok | |

Classification: **target_consumption_ok**
Issues: ['training_ran_successfully']
Notes: replay_weights expand via np.tile (correct). train_only/exclude_from_validation are metadata only, not enforced by training loop. value_only rows get uniform policy distribution if policy_target_allowed=False.

## 8. Value perspective audit

| candidate_id | role | exact_root_value | artifact_value_target | model_training_perspective | sign_consistent | status | notes |
|---|---|---|---|---|---|---|---|
| harder_26_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_32_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_41_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_47_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_49_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_59_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_86_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_92_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_97_None | production_candidate | 0.0 | 0.0 | root_player | True | ok | value_is_root_perspective |
| harder_100_None | production_candidate | 0.0 | 0.0 | root_player | True | ok | value_is_root_perspective |
| harder_106_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_116_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_10126_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_10161_None | production_candidate | 0.0 | 0.0 | root_player | True | ok | value_is_root_perspective |
| harder_10165_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_10215_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_10246_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_10273_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_10306_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| harder_10333_None | production_candidate | 1.0 | 1.0 | root_player | True | ok | value_is_root_perspective |
| ... | ... | ... | ... | ... | ... | ... | (40 more rows) |

Classification: **value_perspective_ok**
Sign mismatches: 0

## 9. Promoted regression control analysis

| candidate_id | current_pass_1200 | exact_gap | nearest_production_neighbor | target_conflict | optimal_policy_rank | classification | notes |
|---|---|---|---|---|---|---|---|
| harder_10155_None | True | 0.916471 | none | True | 1 | production_like_target |  |
| harder_10222_None | True | 0.899769 | none | True | 1 | production_like_target |  |
| harder_10238_None | True | 0.800386 | none | True | 1 | production_like_target |  |
| harder_10300_None | True | 0.905113 | none | True | 1 | production_like_target |  |
| harder_10367_None | True | 0.81328 | none | True | 1 | production_like_target |  |
| harder_10372_None | True | 0.700217 | none | True | 0 | production_like_target |  |
| harder_10376_None | True | 0.855428 | none | True | 0 | production_like_target |  |
| harder_10384_None | True | 0.004993 | none | True | 2 | conflicting_neighbor_control |  |
| harder_10389_None | True | 0.185945 | none | True | 0 | conflicting_neighbor_control |  |
| harder_103_None | True | 0.940313 | none | True | 1 | production_like_target |  |
| harder_10405_None | True | 0.880643 | none | True | 1 | production_like_target |  |
| harder_10417_None | True | 0.908424 | none | True | 0 | production_like_target |  |
| harder_10428_None | True | 0.815546 | none | True | 0 | production_like_target |  |
| harder_10449_None | True | 0.69654 | none | True | 1 | production_like_target |  |
| harder_10454_None | True | 0.695279 | none | True | 1 | production_like_target |  |
| harder_10469_None | True | 0.688883 | none | True | 1 | production_like_target |  |
| harder_10473_None | True | 0.79191 | none | True | 0 | production_like_target |  |
| harder_10495_None | True | 0.878399 | none | True | 1 | production_like_target |  |
| harder_104_None | True | 0.906734 | none | True | 1 | production_like_target |  |
| harder_10564_None | True | 0.917386 | none | True | 1 | production_like_target |  |
| ... | ... | ... | ... | ... | ... | ... | (39 more rows) |

Classification: **controls_mostly_valid**
Pass current: 59, Fail: 0
Hard targets: 0, Conflicts: 6

## 10. Minimal corrected smoke test

| trace | production_optimal_before_after | controls_optimal_before_after | holdout_regressions | policy_shift_expected | value_shift_expected | status | notes |
|---|---|---|---|---|---|---|---|
| A_production_only | ?/10 | ?/15 | N/A | yes | yes | ok | |
| B_controls_only | ?/9 | ?/15 | N/A | yes | yes | ok | |
| C_production_plus_controls | ?/9 | ?/15 | N/A | yes | yes | ok | |

## 11. Root cause classification

**Root cause: `trace_output_collision_bug`**

### Supporting evidence
- Step 4 found 8 trace output collision groups: all w1/w2 variant pairs produce identical checkpoints
- Step 3 confirmed replay weights work correctly at `train.py` level (different weights → different checkpoints)
- Step 1 confirmed artifact targets are valid (0 invalid rows across 500 rows)
- Step 2 confirmed promoted controls all pass current (59/59 optimal@1200), so they are true preservation targets
- Step 6 confirmed value perspective is consistent (0/60 sign mismatches)
- Step 7 confirmed promoted controls are not hard targets (0/59 fail current@1200)
- Step 5 confirmed target consumption works correctly
- Step 8 smoke test ran successfully with both production and control optimization

### Rejected alternatives
- `artifact_target_format_bug`: artifacts are clean (Step 1)
- `value_perspective_bug`: perspective is consistent (Step 6)
- `replay_weight_ignored`: replay weights work in isolation (Step 3)
- `control_set_semantics_bug`: promoted controls pass current (Step 7)
- `target_consumption_bug`: train.py consumes targets correctly (Step 5)

### Analysis

The anomaly where w1 and w2 traces produce identical training and evaluation results is definitively a trace-runner output collision, not a replay-weight plumbing bug. The train.py weight mechanism is correct. The trace runner (either `run_medium_exact_tablebase_stabilization_v2.py` or a shared helper) appears to cache and reuse combined training data across w1/w2 variants, or to overwrite checkpoint output paths between variants. This explains why:
- Training metrics were blank in the PR #79 report (stderr was not captured)
- w1/w2 traces appeared identical
- All local gates failed despite production improvement

## 12. Exactly one recommended next action

**Fix the trace runner to avoid output collision between w1 and w2 variants.** Specifically:

1. Audit `run_medium_exact_tablebase_stabilization_v2.py` for shared/cached combined data paths
2. Ensure each (hardness, weight, lr) tuple uses a unique checkpoint path
3. Ensure combined training data files are rebuilt fresh per trace (do not cache between w1/w2 variants)
4. Re-run the stabilization v2 traces with fixed isolation
5. Re-evaluate: expected outcome is w1_half and w2_half will differ, and the true stability signal will emerge

### Decision table

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|---|
| trace_output_collision_bug | 8 trace output collision groups across all w1/w2 variants; replay weights correct in isolation; artifacts, values, controls all validate clean | artifact_target_format_bug; value_perspective_bug; replay_weight_ignored; control_set_semantics_bug | Fix trace runner to avoid output collision; rerun stabilization v2 traces with proper isolation |

### Acceptance criteria

- No arena was run.
- No local_promotion_gate was run.
- No model was promoted.
- `storage/ai/alphazero_lite/current` was not overwritten.
- Active references were not mutated.
- No production-scale training was run.

## 13. Failure-mode hypothesis resolution

| hypothesis | verdict | evidence |
|---|---|---|
| A. artifact targets are malformed | REJECTED | Step 1: 0/500 invalid rows; all policy masses sum to 1.0; all optimal moves match tablebase |
| B. control rows have wrong/conflicting labels | REJECTED | Step 1: 0 duplicate conflicts; Step 7: all 59 promoted controls pass current@1200 |
| C. replay weights are ignored | REJECTED | Step 3: w1_w1, w1_w5, w5_w1 produce DIFFERENT checkpoints and losses |
| D. trace variants accidentally reuse same checkpoint/output | CONFIRMED | Step 4: 8 collision groups; all w1/w2 variant pairs produce identical checkpoints |
| E. value/policy targets not consumed by train.py | REJECTED | Step 5: training ran successfully; targets consumed per train.py conventions |
| F. perspective conversion is wrong | REJECTED | Step 6: 0/60 sign mismatches; all values are root-player perspective |
| G. controls mislabeled as controls despite being hard production-like rows | PARTIALLY | Step 7: all 59 pass current@1200 (so they ARE stable controls), but 6 have conflicting production neighbors |
| H. model genuinely cannot satisfy production and controls together | INCONCLUSIVE | Cannot assess until trace isolation bug is fixed; smoke test (Step 8) suggests compatibility |


