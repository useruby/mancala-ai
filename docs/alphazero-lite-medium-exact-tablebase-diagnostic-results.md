# Medium Exact Tablebase Diagnostic — Results

**Date:** 2026-06-04
**Family:** `medium_exact_tablebase_diagnostic`
**Scripts:**
- `ml/alphazero_lite/build_medium_exact_tablebase_diagnostic_artifact.py`
- `ml/alphazero_lite/run_medium_exact_tablebase_diagnostic_trace.py`

## 1. Context

PR #77 ran a stabilized exact-tablebase diagnostic with softened policy targets (0.75), expanded controls (11 -> 25), and reduced LR. The best result was soft075_controls_w1_lr_default e2 at 22/32 optimal@1200 with control_regression=0 across all traces and max holdout regression of 7/317. Classification was `exact_tablebase_controls_fix_regression` with next action: "use expanded controls in the next medium diagnostic."

This medium diagnostic scales the artifact to ~80-120 production candidates and ~60+ preservation controls while maintaining strict holdout gates. It tests whether expanded controls and softened targets scale to larger row counts without regressing controls or damaging holdouts.

## 2. Why PR #77 justified medium diagnostic

PR #77 demonstrated that expanded controls (25 rows) completely eliminated control regression across all traces, and half-LR traces reduced holdout regression from 7/317 (soft065 default-LR) to 2/317 (soft075 half-LR). The classification `exact_tablebase_controls_fix_regression` recommended scaling these findings to a medium diagnostic with more production candidates, more controls, and strict holdout gates. PR #77 best production optimal@1200 was 22/32.

## 3. Medium row mining / reuse

The medium artifact was built by reusing all PR #76/77 clean split rows and generating fresh adversarial endgame candidates via random state enumeration in the tablebase range (2-14 seeds). PUCT baselines (budgets 64/384/1200, plus 2400 for persistent failures) and neural value rank scans were run on new candidates. Rows were classified into production/control/holdout/value-only buckets.

- Production candidates: 147
- Preservation controls: 91
- Value-only candidates: 544
- Untouched holdouts: 1351

## 4. Artifact construction

| artifact | row_count | policy_target_mass | value_source | validation_status | notes |
|---|---|---|---|---|---|
| policy_value | 147 | 0.75 | exact_tablebase | PASSED | |
| expanded_controls | 91 | 0.75 | exact_tablebase | PASSED | |
| value_only | 544 | - | exact_tablebase | PASSED | |

## 5. Static validation

Static validation checked: policy sums to 1.0, optimal move receives highest mass, value targets in [-1,1], no duplicate canonical state conflicts, no exhausted-family overlap, no holdout leakage.
Static validation PASSED.

## 6. Baselines

- Current artifact: 147 production candidates, 62 optimal@1200
  - Value-only: avg error=0.6938
  - Controls: 91/91 optimal@1200
  - Holdouts: 1344/1351 optimal@1200
- PR #77 best: 22/32 optimal@1200 (soft075_controls_w1_lr_default e2)

## 7. Trace definitions

| trace_name | policy_target_mass | learning_rate_multiplier | data_files | weights | epochs | status | notes |
|---|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | 0.75 | 1x (1e-4) | soft075 + controls | 1,1 | 1,2,4 | completed |  |
| medium_soft075_controls_w1_lr_half | 0.75 | 0.5x (5e-5) | soft075 + controls | 1,1 | 1,2,4 | completed |  |
| medium_soft075_controls_w2_lr_half | 0.75 | 0.5x (5e-5) | soft075 + controls | 1,2 | 1,2,4 | completed |  |
| medium_value_light_lr_half | 0.75 | 0.5x (5e-5) | soft075 + value_only + controls | 1,1,2 | 1,2,4 | completed | value-only rows added |

## 8. Production-candidate results

| trace_name | epoch | production_total | production_optimal_1200 | production_optimal_2400 | improved_vs_current | improved_vs_pr77_best | avg_optimal_visit_share_1200 | notes |
|---|---|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | e1 | 147 | 103/147 | 114/147 | 113/147 | 103/147 | 0.6558 |  |
| medium_soft075_controls_w1_lr_default | e2 | 147 | 104/147 | 114/147 | 111/147 | 104/147 | 0.6605 |  |
| medium_soft075_controls_w1_lr_default | e4 | 147 | 107/147 | 117/147 | 114/147 | 107/147 | 0.6702 |  |
| medium_soft075_controls_w1_lr_half | e1 | 147 | 97/147 | 108/147 | 109/147 | 97/147 | 0.6111 |  |
| medium_soft075_controls_w1_lr_half | e2 | 147 | 101/147 | 111/147 | 111/147 | 101/147 | 0.6329 |  |
| medium_soft075_controls_w1_lr_half | e4 | 147 | 104/147 | 114/147 | 112/147 | 104/147 | 0.6558 |  |
| medium_soft075_controls_w2_lr_half | e1 | 147 | 97/147 | 108/147 | 109/147 | 97/147 | 0.6111 |  |
| medium_soft075_controls_w2_lr_half | e2 | 147 | 101/147 | 111/147 | 111/147 | 101/147 | 0.6329 |  |
| medium_soft075_controls_w2_lr_half | e4 | 147 | 104/147 | 114/147 | 112/147 | 104/147 | 0.6558 |  |
| medium_value_light_lr_half | e1 | 147 | 99/147 | 110/147 | 104/147 | 99/147 | 0.6262 |  |
| medium_value_light_lr_half | e2 | 147 | 97/147 | 104/147 | 105/147 | 97/147 | 0.6079 |  |
| medium_value_light_lr_half | e4 | 147 | 94/147 | 100/147 | 101/147 | 94/147 | 0.5879 |  |

## 9. Control results

| trace_name | epoch | control_total | controls_optimal_1200 | controls_optimal_2400 | control_regression_count | avg_optimal_visit_share_delta | notes |
|---|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | e1 | 91 | 91/91 | 91/91 | 2 | -0.0128 | 2 regressed |
| medium_soft075_controls_w1_lr_default | e2 | 91 | 89/91 | 90/91 | 2 | -0.0278 | 2 regressed |
| medium_soft075_controls_w1_lr_default | e4 | 91 | 88/91 | 89/91 | 4 | -0.0408 | 4 regressed |
| medium_soft075_controls_w1_lr_half | e1 | 91 | 91/91 | 91/91 | 0 | -0.0006 |  |
| medium_soft075_controls_w1_lr_half | e2 | 91 | 90/91 | 91/91 | 2 | -0.0109 | 2 regressed |
| medium_soft075_controls_w1_lr_half | e4 | 91 | 91/91 | 91/91 | 2 | -0.0127 | 2 regressed |
| medium_soft075_controls_w2_lr_half | e1 | 91 | 91/91 | 91/91 | 0 | -0.0006 |  |
| medium_soft075_controls_w2_lr_half | e2 | 91 | 90/91 | 91/91 | 2 | -0.0109 | 2 regressed |
| medium_soft075_controls_w2_lr_half | e4 | 91 | 91/91 | 91/91 | 2 | -0.0127 | 2 regressed |
| medium_value_light_lr_half | e1 | 91 | 88/91 | 89/91 | 4 | -0.0397 | 4 regressed |
| medium_value_light_lr_half | e2 | 91 | 88/91 | 89/91 | 4 | -0.0362 | 4 regressed |
| medium_value_light_lr_half | e4 | 91 | 87/91 | 86/91 | 7 | -0.0589 | 7 regressed |

## 10. Untouched holdout results

| trace_name | epoch | untouched_holdout_total | holdout_optimal_1200 | holdout_regression_count | holdout_regression_rate | max_regression_severity | value_error_delta | notes |
|---|---|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | e1 | 1351 | 1329/1351 | 22 | 0.0163 | 22 | -0.0922 | 22 regressed |
| medium_soft075_controls_w1_lr_default | e2 | 1351 | 1317/1351 | 34 | 0.0252 | 34 | -0.1435 | 34 regressed |
| medium_soft075_controls_w1_lr_default | e4 | 1351 | 1302/1351 | 49 | 0.0363 | 49 | -0.2449 | 49 regressed |
| medium_soft075_controls_w1_lr_half | e1 | 1351 | 1337/1351 | 14 | 0.0104 | 14 | -0.0510 | 14 regressed |
| medium_soft075_controls_w1_lr_half | e2 | 1351 | 1334/1351 | 17 | 0.0126 | 17 | -0.0763 | 17 regressed |
| medium_soft075_controls_w1_lr_half | e4 | 1351 | 1326/1351 | 25 | 0.0185 | 25 | -0.1325 | 25 regressed |
| medium_soft075_controls_w2_lr_half | e1 | 1351 | 1337/1351 | 14 | 0.0104 | 14 | -0.0510 | 14 regressed |
| medium_soft075_controls_w2_lr_half | e2 | 1351 | 1334/1351 | 17 | 0.0126 | 17 | -0.0763 | 17 regressed |
| medium_soft075_controls_w2_lr_half | e4 | 1351 | 1326/1351 | 25 | 0.0185 | 25 | -0.1325 | 25 regressed |
| medium_value_light_lr_half | e1 | 1351 | 1314/1351 | 37 | 0.0274 | 37 | -0.1762 | 37 regressed |
| medium_value_light_lr_half | e2 | 1351 | 1306/1351 | 45 | 0.0333 | 45 | -0.2689 | 45 regressed |
| medium_value_light_lr_half | e4 | 1351 | 1281/1351 | 70 | 0.0518 | 70 | -0.3667 | 70 regressed |

## 11. Value-only results

| trace_name | epoch | avg_value_error | sign_errors | improved_rows | worsened_rows | notes |
|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | e1 | 0.5925 | 28 | 424 | 120 | |
| medium_soft075_controls_w1_lr_default | e2 | 0.5535 | 18 | 438 | 106 | |
| medium_soft075_controls_w1_lr_default | e4 | 0.4834 | 1 | 449 | 95 | |
| medium_soft075_controls_w1_lr_half | e1 | 0.6336 | 32 | 419 | 125 | |
| medium_soft075_controls_w1_lr_half | e2 | 0.6082 | 29 | 423 | 121 | |
| medium_soft075_controls_w1_lr_half | e4 | 0.5621 | 20 | 442 | 102 | |
| medium_soft075_controls_w2_lr_half | e1 | 0.6336 | 32 | 419 | 125 | |
| medium_soft075_controls_w2_lr_half | e2 | 0.6082 | 29 | 423 | 121 | |
| medium_soft075_controls_w2_lr_half | e4 | 0.5621 | 20 | 442 | 102 | |
| medium_value_light_lr_half | e1 | 0.5283 | 14 | 446 | 98 | |
| medium_value_light_lr_half | e2 | 0.4637 | 0 | 448 | 96 | |
| medium_value_light_lr_half | e4 | 0.3934 | 0 | 446 | 98 | |

## 12. Strict local gate results

| trace_name | epoch | production_gain_pass | control_gate_pass | holdout_gate_pass | sanity_gate_pass | locally_acceptable | notes |
|---|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | e1 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.628%  |
| medium_soft075_controls_w1_lr_default | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.517%  |
| medium_soft075_controls_w1_lr_default | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=3.627%  |
| medium_soft075_controls_w1_lr_half | e1 | PASS | PASS | FAIL | PASS | NO | hold_rate=1.036%  |
| medium_soft075_controls_w1_lr_half | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.258%  |
| medium_soft075_controls_w1_lr_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.850%  |
| medium_soft075_controls_w2_lr_half | e1 | PASS | PASS | FAIL | PASS | NO | hold_rate=1.036%  |
| medium_soft075_controls_w2_lr_half | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.258%  |
| medium_soft075_controls_w2_lr_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.850%  |
| medium_value_light_lr_half | e1 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.739%  |
| medium_value_light_lr_half | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=3.331%  |
| medium_value_light_lr_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=5.181%  |

## 13. Sanity non-regression checks

| trace_name | epoch | suite_or_group | metric | current_baseline | checkpoint_value | regression | notes |
|---|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | e1 | production | optimal@1200 | 62/147 | 103/147 | no | initial sanity |
| medium_soft075_controls_w1_lr_half | e1 | production | optimal@1200 | 62/147 | 97/147 | no | initial sanity |
| medium_soft075_controls_w2_lr_half | e1 | production | optimal@1200 | 62/147 | 97/147 | no | initial sanity |
| medium_value_light_lr_half | e1 | production | optimal@1200 | 62/147 | 99/147 | no | initial sanity |

## 14. Training metrics

| trace_name | epoch | policy_loss | value_loss | total_loss | production_cross_entropy | control_cross_entropy | value_only_loss | notes |
|---|---|---|---|---|---|---|---|---|
| medium_soft075_controls_w1_lr_default | e1 | - | - | - | - | - | - | |
| medium_soft075_controls_w1_lr_default | e2 | - | - | - | - | - | - | |
| medium_soft075_controls_w1_lr_default | e4 | - | - | - | - | - | - | |
| medium_soft075_controls_w1_lr_half | e1 | - | - | - | - | - | - | |
| medium_soft075_controls_w1_lr_half | e2 | - | - | - | - | - | - | |
| medium_soft075_controls_w1_lr_half | e4 | - | - | - | - | - | - | |
| medium_soft075_controls_w2_lr_half | e1 | - | - | - | - | - | - | |
| medium_soft075_controls_w2_lr_half | e2 | - | - | - | - | - | - | |
| medium_soft075_controls_w2_lr_half | e4 | - | - | - | - | - | - | |
| medium_value_light_lr_half | e1 | - | - | - | - | - | - | |
| medium_value_light_lr_half | e2 | - | - | - | - | - | - | |
| medium_value_light_lr_half | e4 | - | - | - | - | - | - | |

## 15. Interpretation

This medium diagnostic tested whether expanded controls and softened targets (0.75) scale to larger artifact sizes (80-120 production, 60+ controls) while maintaining the stability observed in PR #77. Four traces tested different combinations of control weight and learning rate.

Key questions: (1) Do expanded controls prevent regression at larger scale? (2) Does half-LR reduce holdout regression? (3) Does extra control weight (w2) improve stability? (4) Does value-only augmentation cause holdout regression in medium scale? (5) Is there a combination passing all strict gates?

## 16. Exactly one recommended next action

**add more preservation controls or soften targets further; the medium artifact scales production signal but holdout damage exceeds 0.5%% gate even at best checkpoint with zero control regression.**

### Decision table

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|
| medium_exact_tablebase_overfit_persists | best_prod=107/147 (trace=medium_soft075_controls_w1_lr_default e4); best_zero_ctrl_hold_reg_rate=1.036% > 0.5% threshold; min_hold_reg_rate=1.036% min across all checkpoints | medium_exact_tablebase_local_success; medium_exact_tablebase_controls_sufficient_but_gain_small | add more preservation controls or soften targets further; the medium artifact scales production signal but holdout damage exceeds 0.5%% gate even at best checkpoint with zero control regression. |

### Acceptance criteria

- No arena was run.
- No local_promotion_gate was run.
- No model was promoted.
- `storage/ai/alphazero_lite/current` was not overwritten.
- Active references were not mutated.
- Holdout rows were not used for training except promoted regression controls.
- Exact tablebase labels were used with documented perspective.
- Final report recommends exactly one next branch.
