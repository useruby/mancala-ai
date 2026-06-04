# Exact Tablebase Stabilized Diagnostic — Results

**Date:** 2026-06-04
**Family:** `exact_tablebase_stabilized_diagnostic`
**Scripts:**
- `ml/alphazero_lite/build_exact_tablebase_stabilized_diagnostic_artifact.py`
- `ml/alphazero_lite/run_exact_tablebase_stabilized_diagnostic_trace.py`

## 1. Context

PR #76 built a larger exact-tablebase diagnostic artifact (32 production, 106 value-only, 11 controls, 320 holdouts) and ran three trace families. The best result was policy_value_w1_short e2/e4 at 21/32 optimal@1200. However, holdout regression occurred: 3/320 holdouts regressed in policy_value traces and up to 8/320 in value_augmented traces. One value_augmented checkpoint also marked control_regressed=True. The classification was `larger_exact_tablebase_overfit_or_holdout_regression` with the recommended next action: add more preservation controls, reduce LR, soften policy targets before scaling.

This stabilization diagnostic executes that recommendation by: (1) softening policy target mass from 0.85 to 0.75 and 0.65, (2) expanding preservation controls from 11 to include regression holdout rows, and (3) running 5 traces with reduced LR and higher control weights.

## 2. Why PR #76 needed stabilization

PR #76 demonstrated clear production improvement (21/32 vs baseline 12/32 at optimal@1200) and value error reduction (0.5311 vs 0.7272 baseline). However, the holdout regression pattern — where rows that were correct at baseline became incorrect after training — indicates the policy targets may have been too sharp (0.85 optimal mass) relative to the model's capacity to generalize. The single control_regressed=True occurrence in value_augmented_w1_short e2 confirmed that the overfitting risk was real. The recommended action was a three-pronged stabilization: soften policy targets, add more preservation controls, and reduce LR.

## 3. PR #76 regression analysis

| candidate_id | source_group | pr76_trace | exact_optimal_move | selected_before | selected_after | optimal_visit_share_before | optimal_visit_share_after | regression_type | promoted_to_control | notes |
|---|---|---|---|---|---|---|---|---|---|
| harder_10130_None | holdout | policy_value_w1_short | * | * | * | * | * | policy_prior_shift | yes | promoted to preservation control |
| harder_10131_None | holdout | policy_value_w1_short | * | * | * | * | * | policy_prior_shift | yes | promoted to preservation control |
| harder_10133_None | holdout | policy_value_w1_short | * | * | * | * | * | policy_prior_shift | yes | promoted to preservation control |

## 4. Expanded preservation controls

The original 11 PR #76 controls were retained. Additionally, holdout rows that regressed in PR #76 traces were identified and promoted to preservation controls. Additional clean holdout rows were added to reach at least 25 controls if feasible. Promoted rows were removed from the untouched holdout set.

- Original controls: 11
- Promoted regression holdouts: 3
- Total expanded controls: 25
- Untouched holdouts remaining: 317

## 5. Softened artifact construction

Two softened policy/value artifacts were created from the same 32 production candidates used in PR #76. The exact optimal move still receives the highest policy mass, but the mass is reduced from 0.85 to 0.75 and 0.65 respectively. Value targets remain exact tablebase root values. The value-only artifact from PR #76 is reused unchanged.

## 6. Static validation

| artifact | row_count | policy_target_mass | value_source | validation_status | notes |
|---|---|---|---|---|---|
| soft075 | 32 | 0.75 | exact_tablebase | PASSED |  |
| soft065 | 32 | 0.65 | exact_tablebase | PASSED |  |
| expanded_controls | 25 | 0.75 | exact_tablebase | PASSED | orig=11 prom=3 |
| value_only | 106 | - | exact_tablebase | PASSED |  |

Static validation checked: policy sums to 1.0, optimal move receives highest mass, value targets in [-1,1], no duplicate canonical state conflicts, no exhausted-family overlap, no holdout leakage.

## 7. Baselines

- Current artifact: 32 production candidates, 12 optimal@1200
  - Value-only: avg error=0.7272
  - Controls: 25/25 optimal@1200
  - Holdouts: 317/317 optimal@1200
- PR #76 best: 21/32 optimal@1200 (policy_value_w1_short)

## 8. Trace definitions

| trace_name | policy_target_mass | learning_rate_multiplier | data_files | weights | epochs | status | notes |
|---|---|---|---|---|---|---|---|
| soft075_controls_w1_lr_default | 0.75 | 1x (1e-4) | soft075 + controls | 1,1 | 1,2,4 | completed |  |
| soft075_controls_w1_lr_half | 0.75 | 0.5x (5e-5) | soft075 + controls | 1,1 | 1,2,4 | completed |  |
| soft065_controls_w1_lr_default | 0.65 | 1x (1e-4) | soft065 + controls | 1,1 | 1,2,4 | completed |  |
| soft075_controls_w2_lr_half | 0.75 | 0.5x (5e-5) | soft075 + controls | 1,2 | 1,2,4 | completed |  |
| value_only_light_lr_half | 0.75 | 0.5x (5e-5) | soft075 + value_only + controls | 1,1,2 | 1,2,4 | completed | value-only rows added |

## 9. Production-candidate results

| trace_name | epoch | production_optimal_1200 | production_optimal_2400 | improved_vs_current | improved_vs_pr76_best | avg_optimal_visit_share_1200 | notes |
|---|---|---|---|---|---|---|---|
| soft075_controls_w1_lr_default | e1 | 17/32 | 22/32 | 18/32 | 17/32 | 0.4643 |  |
| soft075_controls_w1_lr_default | e2 | 22/32 | 26/32 | 21/32 | 22/32 | 0.5931 | matches/pr76 best |
| soft075_controls_w1_lr_default | e4 | 19/32 | 23/32 | 22/32 | 19/32 | 0.5619 |  |
| soft075_controls_w1_lr_half | e1 | 18/32 | 20/32 | 21/32 | 18/32 | 0.4919 |  |
| soft075_controls_w1_lr_half | e2 | 18/32 | 20/32 | 19/32 | 18/32 | 0.4905 |  |
| soft075_controls_w1_lr_half | e4 | 16/32 | 22/32 | 21/32 | 16/32 | 0.4643 |  |
| soft065_controls_w1_lr_default | e1 | 17/32 | 21/32 | 18/32 | 17/32 | 0.4510 |  |
| soft065_controls_w1_lr_default | e2 | 19/32 | 25/32 | 20/32 | 19/32 | 0.5514 |  |
| soft065_controls_w1_lr_default | e4 | 19/32 | 22/32 | 22/32 | 19/32 | 0.5560 |  |
| soft075_controls_w2_lr_half | e1 | 18/32 | 20/32 | 21/32 | 18/32 | 0.4919 |  |
| soft075_controls_w2_lr_half | e2 | 18/32 | 20/32 | 19/32 | 18/32 | 0.4905 |  |
| soft075_controls_w2_lr_half | e4 | 16/32 | 22/32 | 21/32 | 16/32 | 0.4643 |  |
| value_only_light_lr_half | e1 | 17/32 | 19/32 | 17/32 | 17/32 | 0.4471 |  |
| value_only_light_lr_half | e2 | 17/32 | 20/32 | 18/32 | 17/32 | 0.4838 |  |
| value_only_light_lr_half | e4 | 16/32 | 20/32 | 19/32 | 16/32 | 0.4880 |  |

## 10. Control results

| trace_name | epoch | original_controls_optimal_1200 | promoted_controls_optimal_1200 | control_regression_count | avg_optimal_visit_share_delta | notes |
|---|---|---|---|---|---|---|
| soft075_controls_w1_lr_default | e1 | 22/22 | 3/3 | 0 | 0.0005 |  |
| soft075_controls_w1_lr_default | e2 | 22/22 | 3/3 | 0 | 0.0004 |  |
| soft075_controls_w1_lr_default | e4 | 22/22 | 3/3 | 0 | -0.0016 |  |
| soft075_controls_w1_lr_half | e1 | 22/22 | 3/3 | 0 | 0.0002 |  |
| soft075_controls_w1_lr_half | e2 | 22/22 | 3/3 | 0 | 0.0004 |  |
| soft075_controls_w1_lr_half | e4 | 22/22 | 3/3 | 0 | 0.0006 |  |
| soft065_controls_w1_lr_default | e1 | 22/22 | 3/3 | 0 | 0.0004 |  |
| soft065_controls_w1_lr_default | e2 | 22/22 | 3/3 | 0 | 0.0003 |  |
| soft065_controls_w1_lr_default | e4 | 22/22 | 3/3 | 0 | -0.0020 |  |
| soft075_controls_w2_lr_half | e1 | 22/22 | 3/3 | 0 | 0.0002 |  |
| soft075_controls_w2_lr_half | e2 | 22/22 | 3/3 | 0 | 0.0004 |  |
| soft075_controls_w2_lr_half | e4 | 22/22 | 3/3 | 0 | 0.0006 |  |
| value_only_light_lr_half | e1 | 22/22 | 3/3 | 0 | -0.0004 |  |
| value_only_light_lr_half | e2 | 22/22 | 3/3 | 0 | -0.0009 |  |
| value_only_light_lr_half | e4 | 22/22 | 3/3 | 0 | -0.0055 |  |

## 11. Untouched holdout results

| trace_name | epoch | untouched_holdouts | holdout_optimal_1200 | holdout_regression_count | max_regression_severity | value_error_delta | notes |
|---|---|---|---|---|---|---|---|
| soft075_controls_w1_lr_default | e1 | 317 | 314/317 | 3 | 3 | -0.0297 | 3 holdouts regressed |
| soft075_controls_w1_lr_default | e2 | 317 | 313/317 | 4 | 4 | -0.0434 | 4 holdouts regressed |
| soft075_controls_w1_lr_default | e4 | 317 | 312/317 | 5 | 5 | -0.0712 | 5 holdouts regressed |
| soft075_controls_w1_lr_half | e1 | 317 | 315/317 | 2 | 2 | -0.0158 | 2 holdouts regressed |
| soft075_controls_w1_lr_half | e2 | 317 | 315/317 | 2 | 2 | -0.0234 | 2 holdouts regressed |
| soft075_controls_w1_lr_half | e4 | 317 | 313/317 | 4 | 4 | -0.0380 | 4 holdouts regressed |
| soft065_controls_w1_lr_default | e1 | 317 | 314/317 | 3 | 3 | -0.0306 | 3 holdouts regressed |
| soft065_controls_w1_lr_default | e2 | 317 | 313/317 | 4 | 4 | -0.0448 | 4 holdouts regressed |
| soft065_controls_w1_lr_default | e4 | 317 | 310/317 | 7 | 7 | -0.0742 | 7 holdouts regressed |
| soft075_controls_w2_lr_half | e1 | 317 | 315/317 | 2 | 2 | -0.0158 | 2 holdouts regressed |
| soft075_controls_w2_lr_half | e2 | 317 | 315/317 | 2 | 2 | -0.0234 | 2 holdouts regressed |
| soft075_controls_w2_lr_half | e4 | 317 | 313/317 | 4 | 4 | -0.0380 | 4 holdouts regressed |
| value_only_light_lr_half | e1 | 317 | 314/317 | 3 | 3 | -0.0336 | 3 holdouts regressed |
| value_only_light_lr_half | e2 | 317 | 313/317 | 4 | 4 | -0.0517 | 4 holdouts regressed |
| value_only_light_lr_half | e4 | 317 | 312/317 | 5 | 5 | -0.0902 | 5 holdouts regressed |

## 12. Value-only results

| trace_name | epoch | avg_value_error | sign_errors | improved_rows | worsened_rows | notes |
|---|---|---|---|---|---|---|
| soft075_controls_w1_lr_default | e1 | 0.6826 | 8 | 89 | 17 |  |
| soft075_controls_w1_lr_default | e2 | 0.6629 | 8 | 91 | 15 |  |
| soft075_controls_w1_lr_default | e4 | 0.6252 | 8 | 88 | 18 |  |
| soft075_controls_w1_lr_half | e1 | 0.7038 | 9 | 90 | 16 |  |
| soft075_controls_w1_lr_half | e2 | 0.6930 | 8 | 90 | 16 |  |
| soft075_controls_w1_lr_half | e4 | 0.6718 | 8 | 90 | 16 |  |
| soft065_controls_w1_lr_default | e1 | 0.6818 | 8 | 91 | 15 |  |
| soft065_controls_w1_lr_default | e2 | 0.6616 | 8 | 90 | 16 |  |
| soft065_controls_w1_lr_default | e4 | 0.6229 | 7 | 88 | 18 |  |
| soft075_controls_w2_lr_half | e1 | 0.7038 | 9 | 90 | 16 |  |
| soft075_controls_w2_lr_half | e2 | 0.6930 | 8 | 90 | 16 |  |
| soft075_controls_w2_lr_half | e4 | 0.6718 | 8 | 90 | 16 |  |
| value_only_light_lr_half | e1 | 0.6659 | 8 | 81 | 25 |  |
| value_only_light_lr_half | e2 | 0.6390 | 8 | 81 | 25 |  |
| value_only_light_lr_half | e4 | 0.5929 | 7 | 88 | 18 |  |

## 13. Sanity non-regression checks

| trace_name | epoch | suite_or_group | metric | current_baseline | checkpoint_value | regression | notes |
|---|---|---|---|---|---|---|---|
| soft075_controls_w1_lr_default | e1 | production | optimal@1200 | 16/32 | 17/32 | no | initial sanity |
| soft075_controls_w1_lr_half | e1 | production | optimal@1200 | 16/32 | 18/32 | no | initial sanity |
| soft065_controls_w1_lr_default | e1 | production | optimal@1200 | 16/32 | 17/32 | no | initial sanity |
| soft075_controls_w2_lr_half | e1 | production | optimal@1200 | 16/32 | 18/32 | no | initial sanity |
| value_only_light_lr_half | e1 | production | optimal@1200 | 16/32 | 17/32 | no | initial sanity |

## 14. Training metrics

| trace_name | epochs | lr | policy_loss | value_loss | total_loss | elapsed_s |
|---|---|---|---|---|---|---|
| soft075_controls_w1_lr_default | e1 | 0.0001 | 2.06066 | 0.295387 | - | 1.2 |
| soft075_controls_w1_lr_default | e2 | 0.0001 | 1.885568 | 0.273629 | - | 1.3 |
| soft075_controls_w1_lr_default | e4 | 0.0001 | 1.759329 | 0.245967 | - | 1.3 |
| soft075_controls_w1_lr_half | e1 | 5e-05 | 2.086945 | 0.298847 | - | 1.3 |
| soft075_controls_w1_lr_half | e2 | 5e-05 | 1.989995 | 0.286077 | - | 1.3 |
| soft075_controls_w1_lr_half | e4 | 5e-05 | 1.892776 | 0.273068 | - | 1.3 |
| soft065_controls_w1_lr_default | e1 | 0.0001 | 1.976433 | 0.295101 | - | 1.3 |
| soft065_controls_w1_lr_default | e2 | 0.0001 | 1.811019 | 0.272995 | - | 1.3 |
| soft065_controls_w1_lr_default | e4 | 0.0001 | 1.694857 | 0.244273 | - | 1.3 |
| soft075_controls_w2_lr_half | e1 | 5e-05 | 2.086945 | 0.298847 | - | 1.3 |
| soft075_controls_w2_lr_half | e2 | 5e-05 | 1.989995 | 0.286077 | - | 1.3 |
| soft075_controls_w2_lr_half | e4 | 5e-05 | 1.892776 | 0.273068 | - | 1.3 |
| value_only_light_lr_half | e1 | 5e-05 | 1.796315 | 0.276843 | - | 1.3 |
| value_only_light_lr_half | e2 | 5e-05 | 1.65908 | 0.275876 | - | 1.3 |
| value_only_light_lr_half | e4 | 5e-05 | 1.479267 | 0.221104 | - | 1.3 |

## 15. Interpretation

This diagnostic tested whether softening policy targets (0.75, 0.65), expanding preservation controls (11 -> N), and reducing LR (0.5x) can eliminate the holdout/control regression observed in PR #76 while preserving the production improvement. Five traces tested different combinations of these interventions.

Key questions: (1) Does softening alone eliminate regression? (2) Do expanded controls prevent regression on promoted rows? (3) Does reduced LR trade off production gain for stability? (4) Does value-only augmentation still cause regression? (5) Is there a combination that achieves both production improvement and zero regression?

## 16. Exactly one recommended next action

**use expanded controls in the next medium diagnostic.**

### Decision table

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|
| exact_tablebase_controls_fix_regression | ctrl_reg=0 (expanded controls fix); holdout_reg=7 | stabilized_exact_tablebase_local_success | use expanded controls in the next medium diagnostic. |

### Acceptance criteria

- No arena was run.
- No local_promotion_gate was run.
- No model was promoted.
- `storage/ai/alphazero_lite/current` was not overwritten.
- Active references were not mutated.
- Holdout rows were not used for training except promoted regression controls.
- Exact tablebase labels were used with documented perspective.
- Final report recommends exactly one next branch.
