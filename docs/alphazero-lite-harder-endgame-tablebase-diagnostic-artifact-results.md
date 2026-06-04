# Harder Endgame Tablebase Diagnostic Artifact — Results

**Date:** 2026-06-04
**Family:** `harder_fresh_endgame_tablebase`
**Scripts:**
- `ml/alphazero_lite/build_harder_endgame_tablebase_diagnostic_artifact.py`
- `ml/alphazero_lite/run_harder_endgame_tablebase_diagnostic_trace.py`

## 1. Context

PR #74 ran local diagnostics on the harder_fresh_endgame_tablebase family.
It classified: exact_tablebase_diagnostic_artifact_ready and recommended building
a tiny train-only exact-tablebase diagnostic artifact with controls.
This run executes that recommendation.

## 2. Artifact construction

Rows from the PR #74 clean split were used to build three artifact files:

| artifact | row_count | roles | target_types | value_source |
|---|---|---|---|---|
| controls | 3 | preservation_control | policy, value | exact_tablebase |
| policy_value | 12 | production_candidate_later | policy, value | exact_tablebase |
| value_only | 20 | value_only_candidate | value | exact_tablebase |

## 3. Current baseline

- Production candidates: 12 total, 5 optimal@1200
- Value-only candidates: 20 total, avg value error: 0.7803
- Preservation controls: 3 total, 3 optimal@1200
- Holdouts: 56 total, 56 optimal@1200

## 4. Trace definitions

| trace_name | data_files | replay_weights | epochs | status |
|---|---|---|---|---|
| policy_value_w1 | policy_value + controls | 1,1 | 1,2,4 | completed |
| policy_value_w2 | policy_value + controls | 2,1 | 1,2,4 | completed |
| value_augmented_w1 | policy_value + value_only + controls | 1,1,1 | 1,2,4 | completed |
| value_augmented_w2 | policy_value + value_only + controls | 2,1,1 | 1,2,4 | completed |

## 5. Production-candidate local results

- policy_value_w1 e1: 8/12 optimal@1200, 8/12 improved vs current
- policy_value_w1 e2: 7/12 optimal@1200, 11/12 improved vs current
- policy_value_w1 e4: 6/12 optimal@1200, 7/12 improved vs current
- policy_value_w2 e1: 8/12 optimal@1200, 8/12 improved vs current
- policy_value_w2 e2: 7/12 optimal@1200, 11/12 improved vs current
- policy_value_w2 e4: 6/12 optimal@1200, 7/12 improved vs current
- value_augmented_w1 e1: 6/12 optimal@1200, 9/12 improved vs current
- value_augmented_w1 e2: 6/12 optimal@1200, 6/12 improved vs current
- value_augmented_w1 e4: 5/12 optimal@1200, 9/12 improved vs current
- value_augmented_w2 e1: 6/12 optimal@1200, 9/12 improved vs current
- value_augmented_w2 e2: 6/12 optimal@1200, 6/12 improved vs current
- value_augmented_w2 e4: 5/12 optimal@1200, 9/12 improved vs current

## 6. Value-only local results

- policy_value_w1 e1: avg value error=0.7680, 13/20 improved vs current
- policy_value_w1 e2: avg value error=0.7629, 13/20 improved vs current
- policy_value_w1 e4: avg value error=0.7507, 14/20 improved vs current
- policy_value_w2 e1: avg value error=0.7680, 13/20 improved vs current
- policy_value_w2 e2: avg value error=0.7629, 13/20 improved vs current
- policy_value_w2 e4: avg value error=0.7507, 14/20 improved vs current
- value_augmented_w1 e1: avg value error=0.7531, 13/20 improved vs current
- value_augmented_w1 e2: avg value error=0.7409, 13/20 improved vs current
- value_augmented_w1 e4: avg value error=0.7211, 13/20 improved vs current
- value_augmented_w2 e1: avg value error=0.7531, 13/20 improved vs current
- value_augmented_w2 e2: avg value error=0.7409, 13/20 improved vs current
- value_augmented_w2 e4: avg value error=0.7211, 13/20 improved vs current

## 7. Preservation-control results

- policy_value_w1 e1: 3/3 optimal@1200, regressed=False
- policy_value_w1 e2: 3/3 optimal@1200, regressed=False
- policy_value_w1 e4: 3/3 optimal@1200, regressed=False
- policy_value_w2 e1: 3/3 optimal@1200, regressed=False
- policy_value_w2 e2: 3/3 optimal@1200, regressed=False
- policy_value_w2 e4: 3/3 optimal@1200, regressed=False
- value_augmented_w1 e1: 3/3 optimal@1200, regressed=False
- value_augmented_w1 e2: 3/3 optimal@1200, regressed=False
- value_augmented_w1 e4: 3/3 optimal@1200, regressed=False
- value_augmented_w2 e1: 3/3 optimal@1200, regressed=False
- value_augmented_w2 e2: 3/3 optimal@1200, regressed=False
- value_augmented_w2 e4: 3/3 optimal@1200, regressed=False

## 8. Holdout generalization

- policy_value_w1 e1: 55/56 optimal@1200, improved=0, regressed=1
- policy_value_w1 e2: 55/56 optimal@1200, improved=0, regressed=1
- policy_value_w1 e4: 55/56 optimal@1200, improved=0, regressed=1
- policy_value_w2 e1: 55/56 optimal@1200, improved=0, regressed=1
- policy_value_w2 e2: 55/56 optimal@1200, improved=0, regressed=1
- policy_value_w2 e4: 55/56 optimal@1200, improved=0, regressed=1
- value_augmented_w1 e1: 55/56 optimal@1200, improved=0, regressed=1
- value_augmented_w1 e2: 55/56 optimal@1200, improved=0, regressed=1
- value_augmented_w1 e4: 55/56 optimal@1200, improved=0, regressed=1
- value_augmented_w2 e1: 55/56 optimal@1200, improved=0, regressed=1
- value_augmented_w2 e2: 55/56 optimal@1200, improved=0, regressed=1
- value_augmented_w2 e4: 55/56 optimal@1200, improved=0, regressed=1

## 9. Training metrics

| trace_name | epochs | policy_loss | value_loss | elapsed_s |
|---|---|---|---|---|
| policy_value_w1 | 1 | - | - | - |
| policy_value_w1 | 2 | 2.15393 | 0.241138 | 1.3 |
| policy_value_w1 | 4 | 1.936086 | 0.230733 | 1.3 |
| policy_value_w2 | 1 | 2.356383 | 0.250194 | 1.2 |
| policy_value_w2 | 2 | 2.15393 | 0.241138 | 1.3 |
| policy_value_w2 | 4 | 1.936086 | 0.230733 | 1.3 |
| value_augmented_w1 | 1 | 1.859937 | 0.305936 | 1.2 |
| value_augmented_w1 | 2 | 1.671814 | 0.215184 | 1.3 |
| value_augmented_w1 | 4 | 1.556214 | 0.239285 | 1.3 |
| value_augmented_w2 | 1 | 1.859937 | 0.305936 | 1.3 |
| value_augmented_w2 | 2 | 1.671814 | 0.215184 | 1.2 |
| value_augmented_w2 | 4 | 1.556214 | 0.239285 | 1.3 |

## 10. Decision

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|
| exact_tablebase_diagnostic_local_success | prod_rate=0.69; control_regressed=False | exact_tablebase_value_only_signal; exact_tablebase_no_local_signal | run one slightly larger diagnostic artifact trace with more mined exact-tablebase rows before any arena |

## 11. Exactly one recommended next action

**run one slightly larger diagnostic artifact trace with more mined exact-tablebase rows before any arena**

### Acceptance criteria

- No arena was run.
- No model was promoted.
- `storage/ai/alphazero_lite/current` was not overwritten.
- Holdout rows were not used for training.
- Exhausted families were excluded.
- Exact tablebase labels were used with documented perspective.