# Larger Harder Endgame Tablebase Diagnostic — Results

**Date:** 2026-06-04
**Family:** `harder_fresh_endgame_tablebase`
**Scripts:**
- `ml/alphazero_lite/build_larger_harder_endgame_tablebase_diagnostic_artifact.py`
- `ml/alphazero_lite/run_larger_harder_endgame_tablebase_diagnostic_trace.py`

## 1. Context

PR #75 built a tiny exact-tablebase diagnostic artifact with 12 production candidates, 20 value-only candidates, 3 controls, and 56 holdouts. The best local trace improved production optimal@1200 from 5/12 to 8/12 with no control regression. The next-action recommendation was: run one slightly larger diagnostic artifact trace with more mined exact-tablebase rows before any arena.

This run executes that recommendation by generating approximately 500 additional adversarial endgame candidates, re-running PUCT baselines and neural value rank scans, merging with the existing clean split, and constructing a larger artifact. Goal row counts: >=30 production, >=40 value-only, >=8 controls, >=50 holdouts.

## 2. Why PR #75 justified a larger diagnostic

PR #75 classified: exact_tablebase_diagnostic_local_success with production improvement rate 0.69, no control regression, and measurable value-error reduction. The primary risk identified was that the 12-row artifact was too small to distinguish representation improvement from data-order luck. A larger artifact tests whether the training signal generalizes to more diverse exact-tablebase positions.

## 3. Row mining / reuse

Existing rows from the PR #74 clean split (91 rows total) were reused: 12 production, 20 value-only, 3 controls, 56 holdouts. Approximately 500 new adversarial near-threshold endgame candidates were generated via random state enumeration within tablebase seed range (2-14 seeds). Candidates were deduplicated against the existing set, the forensic suite, and exhausted family inventories. PUCT baseline (budgets 64/384/1200, plus 2400 for failures) was run on new candidates only. Neural value rank scans identified value rank errors. New candidates were classified into target/control/holdout buckets and merged with the existing split.

## 4. Artifact construction

Three artifact files were built from the merged clean split:

| artifact | row_count | roles | target_types | policy_target_mass | value_source | validation_status | notes |
|---|---|---|---|---|---|---|---|
| policy_value | 32 | production_candidate_later | policy, value | 0.85 | exact_tablebase | PASSED | |
| value_only | 106 | value_only_candidate | value | - | exact_tablebase | PASSED | |
| controls | 11 | preservation_control | policy, value | 0.85 | exact_tablebase | PASSED | |

## 5. Static validation

Static validation checked:
- Policy targets sum to 1.0
- Exact optimal move receives highest policy mass
- Value targets in [-1.0, 1.0] range
- No duplicate canonical states with conflicting targets
- No exhausted-family overlap
- No holdout rows in training artifacts
- All metadata includes exact tablebase source
- Validation result: PASSED

## 6. Current baseline

- Production candidates: 32 total, 12 optimal@1200
- Value-only candidates: 106 total, avg value error: 0.7272
- Preservation controls: 11 total, 11 optimal@1200
- Holdouts: 320 total, 320 optimal@1200

## 7. Trace definitions

| trace_name | data_files | replay_weights | epochs | status | notes |
|---|---|---|---|---|---|
| policy_value_w1_short | policy_value + controls | 1,1 | 1,2,4 | completed |  |
| policy_value_w2_short | policy_value + controls | 2,1 | 1,2,4 | completed |  |
| value_augmented_w1_short | policy_value + value_only + controls | 1,1,1 | 1,2,4 | completed |  |

## 8. Production-candidate local results

- policy_value_w1_short e1: 18/32 optimal@1200, 20/32 improved vs current
- policy_value_w1_short e2: 21/32 optimal@1200, 22/32 improved vs current
- policy_value_w1_short e4: 21/32 optimal@1200, 20/32 improved vs current
- policy_value_w2_short e1: 18/32 optimal@1200, 20/32 improved vs current
- policy_value_w2_short e2: 21/32 optimal@1200, 22/32 improved vs current
- policy_value_w2_short e4: 21/32 optimal@1200, 20/32 improved vs current
- value_augmented_w1_short e1: 17/32 optimal@1200, 19/32 improved vs current
- value_augmented_w1_short e2: 20/32 optimal@1200, 21/32 improved vs current
- value_augmented_w1_short e4: 18/32 optimal@1200, 19/32 improved vs current

## 9. Value-only local results

- policy_value_w1_short e1: avg value error=0.6850, 90/106 improved vs current
- policy_value_w1_short e2: avg value error=0.6667, 90/106 improved vs current
- policy_value_w1_short e4: avg value error=0.6339, 87/106 improved vs current
- policy_value_w2_short e1: avg value error=0.6850, 90/106 improved vs current
- policy_value_w2_short e2: avg value error=0.6667, 90/106 improved vs current
- policy_value_w2_short e4: avg value error=0.6339, 87/106 improved vs current
- value_augmented_w1_short e1: avg value error=0.6313, 83/106 improved vs current
- value_augmented_w1_short e2: avg value error=0.5982, 84/106 improved vs current
- value_augmented_w1_short e4: avg value error=0.5311, 91/106 improved vs current

## 10. Preservation-control results

- policy_value_w1_short e1: 11/11 optimal@1200, regressed=False
- policy_value_w1_short e2: 11/11 optimal@1200, regressed=False
- policy_value_w1_short e4: 11/11 optimal@1200, regressed=False
- policy_value_w2_short e1: 11/11 optimal@1200, regressed=False
- policy_value_w2_short e2: 11/11 optimal@1200, regressed=False
- policy_value_w2_short e4: 11/11 optimal@1200, regressed=False
- value_augmented_w1_short e1: 11/11 optimal@1200, regressed=False
- value_augmented_w1_short e2: 11/11 optimal@1200, regressed=True
- value_augmented_w1_short e4: 11/11 optimal@1200, regressed=False

## 11. Holdout generalization

- policy_value_w1_short e1: 317/320 optimal@1200, improved=0, regressed=3
- policy_value_w1_short e2: 317/320 optimal@1200, improved=0, regressed=3
- policy_value_w1_short e4: 317/320 optimal@1200, improved=0, regressed=3
- policy_value_w2_short e1: 317/320 optimal@1200, improved=0, regressed=3
- policy_value_w2_short e2: 317/320 optimal@1200, improved=0, regressed=3
- policy_value_w2_short e4: 317/320 optimal@1200, improved=0, regressed=3
- value_augmented_w1_short e1: 315/320 optimal@1200, improved=0, regressed=5
- value_augmented_w1_short e2: 316/320 optimal@1200, improved=0, regressed=4
- value_augmented_w1_short e4: 312/320 optimal@1200, improved=0, regressed=8

## 12. Sanity non-regression checks

Lightweight sanity checks were run on the first trace checkpoint. No catastrophic regression detected in corrected guard rows or standard initial state evaluation.

## 13. Training metrics

| trace_name | epochs | policy_loss | value_loss | total_loss | elapsed_s |
|---|---|---|---|---|---|
| policy_value_w1_short | 1 | 2.253709 | 0.289204 | - | 1.3 |
| policy_value_w1_short | 2 | 1.933847 | 0.279251 | - | 1.3 |
| policy_value_w1_short | 4 | 1.893542 | 0.239033 | - | 1.3 |
| policy_value_w2_short | 1 | 2.253709 | 0.289204 | - | 1.3 |
| policy_value_w2_short | 2 | 1.933847 | 0.279251 | - | 1.3 |
| policy_value_w2_short | 4 | 1.893542 | 0.239033 | - | 1.3 |
| value_augmented_w1_short | 1 | 1.769834 | 0.271059 | - | 1.3 |
| value_augmented_w1_short | 2 | 1.594194 | 0.230644 | - | 1.3 |
| value_augmented_w1_short | 4 | 1.489782 | 0.177622 | - | 1.3 |

## 14. Interpretation

The larger diagnostic tests whether the training signal observed in PR #75 generalizes to more diverse exact-tablebase positions. Baseline production optimal@1200 rate: 12/32. Results are compared against the baseline to determine if production candidates improve, controls remain stable, and holdouts do not regress.

## 15. Decision

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|
| larger_exact_tablebase_overfit_or_holdout_regression | prod_rate=0.64; control_regressed=True | larger_exact_tablebase_local_success; larger_exact_tablebase_weight_needed | add more preservation controls / reduce LR / soften policy targets before scaling |

## 16. Exactly one recommended next action

**add more preservation controls / reduce LR / soften policy targets before scaling**

### Acceptance criteria

- No arena was run.
- No model was promoted.
- `storage/ai/alphazero_lite/current` was not overwritten.
- Holdout rows were not used for training.
- Exhausted families were excluded.
- Exact tablebase labels were used with documented perspective.
- Final report recommends exactly one next branch.