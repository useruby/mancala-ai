# Medium Exact Tablebase Stabilization v2 — Results

**Date:** 2026-06-04
**Family:** `medium_exact_tablebase_stabilization_v2`
**Scripts:**
- `ml/alphazero_lite/run_medium_exact_tablebase_stabilization_v2.py`

## 1. Context

PR #78 ran a medium exact-tablebase diagnostic with softened policy targets (0.75), 91 expanded controls, and four traces. The best production result was 107/147 optimal@1200 (medium_soft075_controls_w1_lr_default e4). However, the best zero-control-regression checkpoints (medium_soft075_controls_w1_lr_half e1 and medium_soft075_controls_w2_lr_half e1) still had holdout regression rates of 1.036%, exceeding the 0.5% strict gate. The classification was `medium_exact_tablebase_overfit_persists` with next action: add more preservation controls or soften targets further.

This stabilization v2 diagnostic executes that recommendation by: (1) identifying specific regressed holdouts from PR #78 and promoting them to controls, (2) adding nearest-neighbor clean holdouts around production candidates, (3) softening policy target mass further to 0.65 and 0.55, (4) using targeted controls with quarter-LR as an additional option. Four traces test different combinations of soft target, control weight, and learning rate.

## 2. Why PR #78 needed stabilization v2

PR #78 demonstrated that expanded controls (91 rows) reduced but did not eliminate holdout regression. Even at the best zero-control-regression checkpoints, 14/1351 holdouts regressed (1.036%). The value-augmented trace made this worse (up to 70 regressions at e4). The classification `medium_exact_tablebase_overfit_persists` recommended either adding more preservation controls or softening targets further before any production-scale lane. This PR does both: it identifies the specific holdout rows that regressed and promotes them to controls, then softens targets to 0.65 and 0.55.

## 3. PR #78 regression analysis

| candidate_id | pr78_trace | exact_optimal_move | selected_before | selected_after | optimal_visit_share_before | optimal_visit_share_after | regression_type | promoted_to_control | notes |
|---|---|---|---|---|---|---|---|---|---|
| harder_10148_None | medium_soft075_controls_w1_lr_default_e1 | 3 | 3 | 1 | * | * | policy_prior_shift | no | |
| harder_10155_None | medium_soft075_controls_w1_lr_default_e1 | 2 | 2 | 0 | * | * | policy_prior_shift | yes | |
| harder_10238_None | medium_soft075_controls_w1_lr_default_e1 | 0 | 0 | 4 | * | * | policy_prior_shift | yes | |
| harder_10300_None | medium_soft075_controls_w1_lr_default_e1 | 1 | 1 | 4 | * | * | policy_prior_shift | yes | |
| harder_103_None | medium_soft075_controls_w1_lr_default_e1 | 2 | 2 | 3 | * | * | policy_prior_shift | yes | |
| harder_10367_None | medium_soft075_controls_w1_lr_default_e2 | 2 | 2 | 3 | * | * | policy_prior_shift | yes | |
| harder_10372_None | medium_soft075_controls_w1_lr_default_e2 | 1 | 1 | 4 | * | * | policy_prior_shift | yes | |
| harder_10222_None | medium_soft075_controls_w1_lr_default_e4 | 3 | 3 | 1 | * | * | policy_prior_shift | yes | |
| harder_10259_None | medium_soft075_controls_w1_lr_default_e4 | 2 | 2 | 0 | * | * | policy_prior_shift | no | |
| harder_10417_None | medium_soft075_controls_w1_lr_half_e1 | 1 | 1 | 4 | * | * | policy_prior_shift | yes | |
| harder_10376_None | medium_soft075_controls_w1_lr_half_e4 | 1 | 1 | 0 | * | * | policy_prior_shift | yes | |
| *62 unique* | *multiple* | * | * | * | * | * | * | * | |

## 4. Artifact construction

| artifact | row_count | policy_target_mass | value_source | validation_status | notes |
|---|---|---|---|---|---|
| soft065 | 147 | 0.65 | exact_tablebase | PASSED |  |
| soft055 | 147 | 0.55 | exact_tablebase | PASSED |  |
| targeted_controls | 206 | 0.75 | exact_tablebase | PASSED | orig=89 prom=59 nn=147 |
| value_only | 544 | - | exact_tablebase | PASSED | reused from PR #78 |

## 5. Static validation

Static validation PASSED.

- Production candidates: 147
- Targeted controls: 206
- Untouched holdouts: 1142
  - Original PR78 controls: 89
  - Promoted regression controls: 59
  - Nearest-neighbor controls: 147

## 6. Baselines

- Current artifact: 147 production candidates, 62 optimal@1200
  - Value-only: avg error=0.6938
  - Controls: 206/206 optimal@1200
  - Holdouts: 1135/1142 optimal@1200
  - pr78_best_production: 0/0 optimal@1200
  - pr78_best_stable: 0/0 optimal@1200

## 7. Trace definitions

| trace_name | policy_target_mass | learning_rate_multiplier | data_files | weights | epochs | status | notes |
|---|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | 0.65 | 0.5x (5e-5) | soft065 + controls | 1,1 | 1,2,4 | completed |  |
| soft065_targeted_controls_w2_lr_half | 0.65 | 0.5x (5e-5) | soft065 + controls | 1,2 | 1,2,4 | completed |  |
| soft055_targeted_controls_w1_lr_half | 0.55 | 0.5x (5e-5) | soft055 + controls | 1,1 | 1,2,4 | completed |  |
| soft065_targeted_controls_w1_lr_quarter | 0.65 | 0.25x (2.5e-5) | soft065 + controls | 1,1 | 1,2,4 | completed |  |

## 8. Production-candidate results

| trace_name | epoch | production_total | production_optimal_1200 | production_optimal_2400 | improved_vs_current | improved_vs_pr78_best_stable | avg_optimal_visit_share_1200 | notes |
|---|---|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | e1 | 147 | 99/147 | 109/147 | 109/147 | 99/147 | 0.6253 | |
| soft065_targeted_controls_w1_lr_half | e2 | 147 | 99/147 | 110/147 | 108/147 | 99/147 | 0.6339 | |
| soft065_targeted_controls_w1_lr_half | e4 | 147 | 98/147 | 110/147 | 106/147 | 98/147 | 0.6171 | |
| soft065_targeted_controls_w2_lr_half | e1 | 147 | 99/147 | 109/147 | 109/147 | 99/147 | 0.6253 | |
| soft065_targeted_controls_w2_lr_half | e2 | 147 | 99/147 | 110/147 | 108/147 | 99/147 | 0.6339 | |
| soft065_targeted_controls_w2_lr_half | e4 | 147 | 98/147 | 110/147 | 106/147 | 98/147 | 0.6171 | |
| soft055_targeted_controls_w1_lr_half | e1 | 147 | 100/147 | 110/147 | 109/147 | 100/147 | 0.6320 | |
| soft055_targeted_controls_w1_lr_half | e2 | 147 | 99/147 | 110/147 | 108/147 | 99/147 | 0.6271 | |
| soft055_targeted_controls_w1_lr_half | e4 | 147 | 95/147 | 105/147 | 104/147 | 95/147 | 0.5948 | |
| soft065_targeted_controls_w1_lr_quarter | e1 | 147 | 93/147 | 103/147 | 106/147 | 93/147 | 0.5677 | |
| soft065_targeted_controls_w1_lr_quarter | e2 | 147 | 95/147 | 107/147 | 104/147 | 95/147 | 0.5953 | |
| soft065_targeted_controls_w1_lr_quarter | e4 | 147 | 99/147 | 108/147 | 108/147 | 99/147 | 0.6252 | |
| cap128_soft065_w1_half | e1 | 147 | 104/147 | 109/147 | 111/147 | 104/147 | 0.6387 | |
| cap128_soft065_w1_half | e2 | 147 | 91/147 | 104/147 | 107/147 | 91/147 | 0.5802 | |
| cap128_soft065_w1_half | e4 | 147 | 86/147 | 102/147 | 99/147 | 86/147 | 0.5424 | |
| cap128_soft065_w2_half | e1 | 147 | 104/147 | 109/147 | 111/147 | 104/147 | 0.6387 | |
| cap128_soft065_w2_half | e2 | 147 | 91/147 | 104/147 | 107/147 | 91/147 | 0.5802 | |
| cap128_soft065_w2_half | e4 | 147 | 86/147 | 102/147 | 99/147 | 86/147 | 0.5424 | |
| cap128_soft055_w1_half | e1 | 147 | 103/147 | 106/147 | 113/147 | 103/147 | 0.6314 | |
| cap128_soft055_w1_half | e2 | 147 | 93/147 | 107/147 | 107/147 | 93/147 | 0.5811 | |
| cap128_soft055_w1_half | e4 | 147 | 85/147 | 100/147 | 96/147 | 85/147 | 0.5401 | |
| cap128_soft065_w1_quarter | e1 | 147 | 100/147 | 108/147 | 101/147 | 100/147 | 0.6262 | |
| cap128_soft065_w1_quarter | e2 | 147 | 98/147 | 106/147 | 111/147 | 98/147 | 0.6304 | |
| cap128_soft065_w1_quarter | e4 | 147 | 101/147 | 110/147 | 113/147 | 101/147 | 0.6369 | |
| reg_soft065_w1_half | e4 | 147 | 98/147 | 103/147 | 106/147 | 98/147 | 0.5992 | |
| reg_soft065_w1_half | e8 | 147 | 97/147 | 106/147 | 110/147 | 97/147 | 0.6045 | |
| reg_soft065_w2_half | e4 | 147 | 98/147 | 103/147 | 106/147 | 98/147 | 0.5992 | |
| reg_soft065_w2_half | e8 | 147 | 97/147 | 106/147 | 110/147 | 97/147 | 0.6045 | |
| reg_soft055_w1_half | e4 | 147 | 90/147 | 101/147 | 102/147 | 90/147 | 0.5643 | |
| reg_soft055_w1_half | e8 | 147 | 94/147 | 103/147 | 103/147 | 94/147 | 0.5834 | |
| reg_soft065_w1_quarter | e4 | 147 | 96/147 | 106/147 | 99/147 | 96/147 | 0.6002 | |
| reg_soft065_w1_quarter | e8 | 147 | 96/147 | 107/147 | 108/147 | 96/147 | 0.6027 | |

## 9. Targeted-control results

| trace_name | epoch | control_total | controls_optimal_1200 | controls_optimal_2400 | control_regression_count | original_controls_regression | promoted_controls_regression | nearest_neighbor_controls_regression | notes |
|---|---|---|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | e1 | 206 | 190/206 | 198/206 | 23 | 0 | 23 | 0 | 23 regressed |
| soft065_targeted_controls_w1_lr_half | e2 | 206 | 182/206 | 194/206 | 32 | 0 | 32 | 0 | 32 regressed |
| soft065_targeted_controls_w1_lr_half | e4 | 206 | 174/206 | 183/206 | 45 | 0 | 43 | 2 | 45 regressed |
| soft065_targeted_controls_w2_lr_half | e1 | 206 | 190/206 | 198/206 | 23 | 0 | 23 | 0 | 23 regressed |
| soft065_targeted_controls_w2_lr_half | e2 | 206 | 182/206 | 194/206 | 32 | 0 | 32 | 0 | 32 regressed |
| soft065_targeted_controls_w2_lr_half | e4 | 206 | 174/206 | 183/206 | 45 | 0 | 43 | 2 | 45 regressed |
| soft055_targeted_controls_w1_lr_half | e1 | 206 | 191/206 | 196/206 | 25 | 0 | 24 | 1 | 25 regressed |
| soft055_targeted_controls_w1_lr_half | e2 | 206 | 183/206 | 193/206 | 33 | 0 | 33 | 0 | 33 regressed |
| soft055_targeted_controls_w1_lr_half | e4 | 206 | 177/206 | 183/206 | 40 | 0 | 39 | 1 | 40 regressed |
| soft065_targeted_controls_w1_lr_quarter | e1 | 206 | 198/206 | 199/206 | 14 | 0 | 14 | 0 | 14 regressed |
| soft065_targeted_controls_w1_lr_quarter | e2 | 206 | 194/206 | 199/206 | 20 | 0 | 18 | 2 | 20 regressed |
| soft065_targeted_controls_w1_lr_quarter | e4 | 206 | 188/206 | 197/206 | 28 | 0 | 27 | 1 | 28 regressed |
| cap128_soft065_w1_half | e1 | 206 | 177/206 | 185/206 | 42 | 0 | 33 | 9 | 42 regressed |
| cap128_soft065_w1_half | e2 | 206 | 173/206 | 185/206 | 44 | 0 | 37 | 7 | 44 regressed |
| cap128_soft065_w1_half | e4 | 206 | 170/206 | 183/206 | 48 | 0 | 41 | 7 | 48 regressed |
| cap128_soft065_w2_half | e1 | 206 | 177/206 | 185/206 | 42 | 0 | 33 | 9 | 42 regressed |
| cap128_soft065_w2_half | e2 | 206 | 173/206 | 185/206 | 44 | 0 | 37 | 7 | 44 regressed |
| cap128_soft065_w2_half | e4 | 206 | 170/206 | 183/206 | 48 | 0 | 41 | 7 | 48 regressed |
| cap128_soft055_w1_half | e1 | 206 | 176/206 | 183/206 | 43 | 0 | 34 | 9 | 43 regressed |
| cap128_soft055_w1_half | e2 | 206 | 174/206 | 182/206 | 42 | 0 | 35 | 7 | 42 regressed |
| cap128_soft055_w1_half | e4 | 206 | 167/206 | 181/206 | 53 | 0 | 43 | 10 | 53 regressed |
| cap128_soft065_w1_quarter | e1 | 206 | 186/206 | 190/206 | 31 | 0 | 27 | 4 | 31 regressed |
| cap128_soft065_w1_quarter | e2 | 206 | 184/206 | 189/206 | 37 | 0 | 29 | 8 | 37 regressed |
| cap128_soft065_w1_quarter | e4 | 206 | 177/206 | 186/206 | 40 | 0 | 33 | 7 | 40 regressed |
| reg_soft065_w1_half | e4 | 206 | 171/206 | 180/206 | 45 | 0 | 41 | 4 | 45 regressed |
| reg_soft065_w1_half | e8 | 206 | 167/206 | 178/206 | 50 | 0 | 43 | 7 | 50 regressed |
| reg_soft065_w2_half | e4 | 206 | 171/206 | 180/206 | 45 | 0 | 41 | 4 | 45 regressed |
| reg_soft065_w2_half | e8 | 206 | 167/206 | 178/206 | 50 | 0 | 43 | 7 | 50 regressed |
| reg_soft055_w1_half | e4 | 206 | 169/206 | 180/206 | 44 | 0 | 42 | 2 | 44 regressed |
| reg_soft055_w1_half | e8 | 206 | 169/206 | 182/206 | 46 | 0 | 42 | 4 | 46 regressed |
| reg_soft065_w1_quarter | e4 | 206 | 181/206 | 188/206 | 39 | 0 | 38 | 1 | 39 regressed |
| reg_soft065_w1_quarter | e8 | 206 | 170/206 | 182/206 | 45 | 0 | 42 | 3 | 45 regressed |

## 10. Untouched holdout results

| trace_name | epoch | untouched_holdout_total | holdout_optimal_1200 | holdout_regression_count | holdout_regression_rate | max_regression_severity | value_error_delta | notes |
|---|---|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | e1 | 1142 | 1141/1142 | 1 | 0.0009 | 1 | -0.0798 | 1 regressed |
| soft065_targeted_controls_w1_lr_half | e2 | 1142 | 1140/1142 | 2 | 0.0018 | 2 | -0.1259 | 2 regressed |
| soft065_targeted_controls_w1_lr_half | e4 | 1142 | 1139/1142 | 3 | 0.0026 | 3 | -0.2139 | 3 regressed |
| soft065_targeted_controls_w2_lr_half | e1 | 1142 | 1141/1142 | 1 | 0.0009 | 1 | -0.0798 | 1 regressed |
| soft065_targeted_controls_w2_lr_half | e2 | 1142 | 1140/1142 | 2 | 0.0018 | 2 | -0.1259 | 2 regressed |
| soft065_targeted_controls_w2_lr_half | e4 | 1142 | 1139/1142 | 3 | 0.0026 | 3 | -0.2139 | 3 regressed |
| soft055_targeted_controls_w1_lr_half | e1 | 1142 | 1141/1142 | 1 | 0.0009 | 1 | -0.0827 | 1 regressed |
| soft055_targeted_controls_w1_lr_half | e2 | 1142 | 1140/1142 | 2 | 0.0018 | 2 | -0.1316 | 2 regressed |
| soft055_targeted_controls_w1_lr_half | e4 | 1142 | 1137/1142 | 5 | 0.0044 | 5 | -0.2216 | 5 regressed |
| soft065_targeted_controls_w1_lr_quarter | e1 | 1142 | 1141/1142 | 1 | 0.0009 | 1 | -0.0404 | 1 regressed |
| soft065_targeted_controls_w1_lr_quarter | e2 | 1142 | 1139/1142 | 3 | 0.0026 | 3 | -0.0616 | 3 regressed |
| soft065_targeted_controls_w1_lr_quarter | e4 | 1142 | 1140/1142 | 2 | 0.0018 | 2 | -0.1097 | 2 regressed |
| cap128_soft065_w1_half | e1 | 1142 | 1124/1142 | 18 | 0.0158 | 18 | -0.1612 | 18 regressed |
| cap128_soft065_w1_half | e2 | 1142 | 1119/1142 | 23 | 0.0201 | 23 | -0.2401 | 23 regressed |
| cap128_soft065_w1_half | e4 | 1142 | 1102/1142 | 40 | 0.0350 | 40 | -0.3393 | 40 regressed |
| cap128_soft065_w2_half | e1 | 1142 | 1124/1142 | 18 | 0.0158 | 18 | -0.1612 | 18 regressed |
| cap128_soft065_w2_half | e2 | 1142 | 1119/1142 | 23 | 0.0201 | 23 | -0.2401 | 23 regressed |
| cap128_soft065_w2_half | e4 | 1142 | 1102/1142 | 40 | 0.0350 | 40 | -0.3393 | 40 regressed |
| cap128_soft055_w1_half | e1 | 1142 | 1125/1142 | 17 | 0.0149 | 17 | -0.1655 | 17 regressed |
| cap128_soft055_w1_half | e2 | 1142 | 1116/1142 | 26 | 0.0228 | 26 | -0.2460 | 26 regressed |
| cap128_soft055_w1_half | e4 | 1142 | 1103/1142 | 39 | 0.0342 | 39 | -0.3454 | 39 regressed |
| cap128_soft065_w1_quarter | e1 | 1142 | 1113/1142 | 29 | 0.0254 | 29 | -0.0787 | 29 regressed |
| cap128_soft065_w1_quarter | e2 | 1142 | 1117/1142 | 25 | 0.0219 | 25 | -0.1225 | 25 regressed |
| cap128_soft065_w1_quarter | e4 | 1142 | 1122/1142 | 20 | 0.0175 | 20 | -0.2058 | 20 regressed |
| reg_soft065_w1_half | e4 | 1142 | 1132/1142 | 10 | 0.0088 | 10 | -0.3104 | 10 regressed |
| reg_soft065_w1_half | e8 | 1142 | 1118/1142 | 24 | 0.0210 | 24 | -0.4224 | 24 regressed |
| reg_soft065_w2_half | e4 | 1142 | 1132/1142 | 10 | 0.0088 | 10 | -0.3104 | 10 regressed |
| reg_soft065_w2_half | e8 | 1142 | 1118/1142 | 24 | 0.0210 | 24 | -0.4224 | 24 regressed |
| reg_soft055_w1_half | e4 | 1142 | 1126/1142 | 16 | 0.0140 | 16 | -0.3188 | 16 regressed |
| reg_soft055_w1_half | e8 | 1142 | 1114/1142 | 28 | 0.0245 | 28 | -0.4248 | 28 regressed |
| reg_soft065_w1_quarter | e4 | 1142 | 1140/1142 | 2 | 0.0018 | 2 | -0.1822 | 2 regressed |
| reg_soft065_w1_quarter | e8 | 1142 | 1130/1142 | 12 | 0.0105 | 12 | -0.3104 | 12 regressed |

## 11. Value-only results

| trace_name | epoch | avg_value_error | sign_errors | improved_rows | worsened_rows | notes |
|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | e1 | 0.6060 | 27 | 424 | 120 | |
| soft065_targeted_controls_w1_lr_half | e2 | 0.5698 | 20 | 444 | 100 | |
| soft065_targeted_controls_w1_lr_half | e4 | 0.5086 | 3 | 449 | 95 | |
| soft065_targeted_controls_w2_lr_half | e1 | 0.6060 | 27 | 424 | 120 | |
| soft065_targeted_controls_w2_lr_half | e2 | 0.5698 | 20 | 444 | 100 | |
| soft065_targeted_controls_w2_lr_half | e4 | 0.5086 | 3 | 449 | 95 | |
| soft055_targeted_controls_w1_lr_half | e1 | 0.6040 | 27 | 428 | 116 | |
| soft055_targeted_controls_w1_lr_half | e2 | 0.5662 | 19 | 448 | 96 | |
| soft055_targeted_controls_w1_lr_half | e4 | 0.5038 | 2 | 448 | 96 | |
| soft065_targeted_controls_w1_lr_quarter | e1 | 0.6455 | 37 | 411 | 133 | |
| soft065_targeted_controls_w1_lr_quarter | e2 | 0.6238 | 31 | 428 | 116 | |
| soft065_targeted_controls_w1_lr_quarter | e4 | 0.5832 | 22 | 445 | 99 | |
| cap128_soft065_w1_half | e1 | 0.5531 | 14 | 420 | 124 | |
| cap128_soft065_w1_half | e2 | 0.4949 | 3 | 438 | 106 | |
| cap128_soft065_w1_half | e4 | 0.4185 | 0 | 445 | 99 | |
| cap128_soft065_w2_half | e1 | 0.5531 | 14 | 420 | 124 | |
| cap128_soft065_w2_half | e2 | 0.4949 | 3 | 438 | 106 | |
| cap128_soft065_w2_half | e4 | 0.4185 | 0 | 445 | 99 | |
| cap128_soft055_w1_half | e1 | 0.5495 | 12 | 421 | 123 | |
| cap128_soft055_w1_half | e2 | 0.4901 | 3 | 438 | 106 | |
| cap128_soft055_w1_half | e4 | 0.4136 | 0 | 446 | 98 | |
| cap128_soft065_w1_quarter | e1 | 0.6139 | 33 | 348 | 196 | |
| cap128_soft065_w1_quarter | e2 | 0.5808 | 24 | 400 | 144 | |
| cap128_soft065_w1_quarter | e4 | 0.5204 | 6 | 429 | 115 | |
| reg_soft065_w1_half | e4 | 0.4444 | 0 | 448 | 96 | |
| reg_soft065_w1_half | e8 | 0.3645 | 0 | 447 | 97 | |
| reg_soft065_w2_half | e4 | 0.4444 | 0 | 448 | 96 | |
| reg_soft065_w2_half | e8 | 0.3645 | 0 | 447 | 97 | |
| reg_soft055_w1_half | e4 | 0.4386 | 0 | 447 | 97 | |
| reg_soft055_w1_half | e8 | 0.3628 | 0 | 447 | 97 | |
| reg_soft065_w1_quarter | e4 | 0.5307 | 6 | 451 | 93 | |
| reg_soft065_w1_quarter | e8 | 0.4452 | 0 | 447 | 97 | |

## 12. Strict local gate results

| trace_name | epoch | production_gain_pass | control_gate_pass | holdout_gate_pass | sanity_gate_pass | locally_acceptable | notes |
|---|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | e1 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w1_lr_half | e2 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w1_lr_half | e4 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w2_lr_half | e1 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w2_lr_half | e2 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w2_lr_half | e4 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft055_targeted_controls_w1_lr_half | e1 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft055_targeted_controls_w1_lr_half | e2 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft055_targeted_controls_w1_lr_half | e4 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w1_lr_quarter | e1 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w1_lr_quarter | e2 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| soft065_targeted_controls_w1_lr_quarter | e4 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| cap128_soft065_w1_half | e1 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.576%  |
| cap128_soft065_w1_half | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.014%  |
| cap128_soft065_w1_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=3.503%  |
| cap128_soft065_w2_half | e1 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.576%  |
| cap128_soft065_w2_half | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.014%  |
| cap128_soft065_w2_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=3.503%  |
| cap128_soft055_w1_half | e1 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.489%  |
| cap128_soft055_w1_half | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.277%  |
| cap128_soft055_w1_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=3.415%  |
| cap128_soft065_w1_quarter | e1 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.539%  |
| cap128_soft065_w1_quarter | e2 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.189%  |
| cap128_soft065_w1_quarter | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.751%  |
| reg_soft065_w1_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=0.876%  |
| reg_soft065_w1_half | e8 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.102%  |
| reg_soft065_w2_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=0.876%  |
| reg_soft065_w2_half | e8 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.102%  |
| reg_soft055_w1_half | e4 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.401%  |
| reg_soft055_w1_half | e8 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=2.452%  |
| reg_soft065_w1_quarter | e4 | PASS | FAIL | PASS | PASS | NO | ctrl_fail  |
| reg_soft065_w1_quarter | e8 | PASS | FAIL | FAIL | PASS | NO | ctrl_fail hold_rate=1.051%  |

## 13. Sanity non-regression checks

| trace_name | epoch | suite_or_group | metric | current_baseline | checkpoint_value | regression | notes |
|---|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | e1 | production | optimal@1200 | 62/147 | 99/147 | no | initial sanity |
| soft065_targeted_controls_w2_lr_half | e1 | production | optimal@1200 | 62/147 | 99/147 | no | initial sanity |
| soft055_targeted_controls_w1_lr_half | e1 | production | optimal@1200 | 62/147 | 100/147 | no | initial sanity |
| soft065_targeted_controls_w1_lr_quarter | e1 | production | optimal@1200 | 62/147 | 93/147 | no | initial sanity |
| cap128_soft065_w1_half | e1 | production | optimal@1200 | 62/147 | 104/147 | no | initial sanity |
| cap128_soft065_w2_half | e1 | production | optimal@1200 | 62/147 | 104/147 | no | initial sanity |
| cap128_soft055_w1_half | e1 | production | optimal@1200 | 62/147 | 103/147 | no | initial sanity |
| cap128_soft065_w1_quarter | e1 | production | optimal@1200 | 62/147 | 100/147 | no | initial sanity |
| reg_soft065_w1_half | e4 | production | optimal@1200 | 62/147 | 98/147 | no | initial sanity |
| reg_soft065_w2_half | e4 | production | optimal@1200 | 62/147 | 98/147 | no | initial sanity |
| reg_soft055_w1_half | e4 | production | optimal@1200 | 62/147 | 90/147 | no | initial sanity |
| reg_soft065_w1_quarter | e4 | production | optimal@1200 | 62/147 | 96/147 | no | initial sanity |

## 14. Training metrics

| trace_name | epoch | policy_loss | value_loss | total_loss | production_cross_entropy | control_cross_entropy | value_only_loss | notes |
|---|---|---|---|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_half | e1 | - | - | - | - | - | - | |
| soft065_targeted_controls_w1_lr_half | e2 | - | - | - | - | - | - | |
| soft065_targeted_controls_w1_lr_half | e4 | - | - | - | - | - | - | |
| soft065_targeted_controls_w2_lr_half | e1 | - | - | - | - | - | - | |
| soft065_targeted_controls_w2_lr_half | e2 | - | - | - | - | - | - | |
| soft065_targeted_controls_w2_lr_half | e4 | - | - | - | - | - | - | |
| soft055_targeted_controls_w1_lr_half | e1 | - | - | - | - | - | - | |
| soft055_targeted_controls_w1_lr_half | e2 | - | - | - | - | - | - | |
| soft055_targeted_controls_w1_lr_half | e4 | - | - | - | - | - | - | |
| soft065_targeted_controls_w1_lr_quarter | e1 | - | - | - | - | - | - | |
| soft065_targeted_controls_w1_lr_quarter | e2 | - | - | - | - | - | - | |
| soft065_targeted_controls_w1_lr_quarter | e4 | - | - | - | - | - | - | |
| cap128_soft065_w1_half | e1 | - | - | - | - | - | - | |
| cap128_soft065_w1_half | e2 | - | - | - | - | - | - | |
| cap128_soft065_w1_half | e4 | - | - | - | - | - | - | |
| cap128_soft065_w2_half | e1 | - | - | - | - | - | - | |
| cap128_soft065_w2_half | e2 | - | - | - | - | - | - | |
| cap128_soft065_w2_half | e4 | - | - | - | - | - | - | |
| cap128_soft055_w1_half | e1 | - | - | - | - | - | - | |
| cap128_soft055_w1_half | e2 | - | - | - | - | - | - | |
| cap128_soft055_w1_half | e4 | - | - | - | - | - | - | |
| cap128_soft065_w1_quarter | e1 | - | - | - | - | - | - | |
| cap128_soft065_w1_quarter | e2 | - | - | - | - | - | - | |
| cap128_soft065_w1_quarter | e4 | - | - | - | - | - | - | |
| reg_soft065_w1_half | e4 | - | - | - | - | - | - | |
| reg_soft065_w1_half | e8 | - | - | - | - | - | - | |
| reg_soft065_w2_half | e4 | - | - | - | - | - | - | |
| reg_soft065_w2_half | e8 | - | - | - | - | - | - | |
| reg_soft055_w1_half | e4 | - | - | - | - | - | - | |
| reg_soft055_w1_half | e8 | - | - | - | - | - | - | |
| reg_soft065_w1_quarter | e4 | - | - | - | - | - | - | |
| reg_soft065_w1_quarter | e8 | - | - | - | - | - | - | |

## 15. Interpretation

This stabilization v2 diagnostic tested whether further softening (0.65, 0.55) combined with targeted preservation controls (including promoted regression holdouts and nearest-neighbor clean rows) can reduce holdout regression below the 0.5% strict gate while maintaining meaningful production improvement over the current artifact.

Key questions: (1) Does further softening (0.55) reduce holdout regression? (2) Do targeted controls (promoted regression rows + nearest neighbors) prevent control regression? (3) Does quarter-LR add stability? (4) Is there a combination passing all strict gates?

## 16. Exactly one recommended next action

**inspect target format/value perspective/training path.**

### Decision table

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|
| exact_tablebase_no_local_signal | best_prod=104/147; min_hold_reg_rate=0.088% |  | inspect target format/value perspective/training path. |

### Acceptance criteria

- No arena was run.
- No local_promotion_gate was run.
- No model was promoted.
- `storage/ai/alphazero_lite/current` was not overwritten.
- Active references were not mutated.
- Untouched holdouts were not used for training.
- Promoted regression controls were removed from untouched holdout evaluation.
- Exhausted families were excluded.
- Exact tablebase labels were used with documented perspective.
- Final report recommends exactly one next branch.
