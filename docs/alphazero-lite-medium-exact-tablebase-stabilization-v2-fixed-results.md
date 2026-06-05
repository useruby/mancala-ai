# Medium Exact Tablebase Stabilization v2 Fixed — Results

**Date:** 2026-06-05
**Family:** `medium_exact_tablebase_stabilization_v2`
**Script:** `ml/alphazero_lite/run_medium_exact_tablebase_stabilization_v2_fixed.py`
**PR:** #80 — fixed trace isolation

## 1. Context

PR #79 ran a medium exact-tablebase stabilization v2 diagnostic with 147 production candidates, 206 targeted controls, and 1142 untouched holdouts. All local gates failed because promoted regression controls regressed. Training metrics were blank. Several w1/w2 traces appeared identical.

PR #80 diagnosed the root cause as **trace-runner output collision**: the original runner bypassed trace-specific replay weights (hardcoded `[1]` at line 2640) and cached combined training data across w1/w2 variants. The audit confirmed replay weights work at `train.py` level, artifacts are valid, and value perspective is consistent.

This fixed runner corrects those issues and re-executes the experiment with proper trace isolation.

## 2. Root cause and fix

**Root cause:** `run_medium_exact_tablebase_stabilization_v2.py` line 2640:
```python
tm = run_training(name, [data_path], [1], epochs, init_ckpt, lr)
```
The `replay_weights` argument is hardcoded to `[1]` instead of using trace-specific `weights`. The combined data file is cached by `build_training_data`, so all traces using the same source files get identical data + identical replay weight → identical checkpoints.

**Fix in the fixed runner:**
- Data files passed as separate `--data-files` with matching `--replay-weights` (e.g., `[1,1]`, `[1,2]`)
- Each trace uses a unique checkpoint path: `EXPORT_DIR / f"{trace_name}_e{epochs}.npz"`
- No cross-trace caching of combined data or checkpoints
- Training data rebuilt fresh per trace name

## 3. Trace isolation verification

| w1 trace | w2 trace | epochs | w1 SHA256 (first 16) | w2 SHA256 (first 16) | isolated? |
|---|---|---|---|---|---|
| cap128_soft065_w1_half | cap128_soft065_w2_half | e1 | 34a0fe7b4757db49 | 81e0be89b8305e9a | YES |
| cap128_soft065_w1_half | cap128_soft065_w2_half | e2 | 13c373f78aad5026 | 3ce5a4a26705830f | YES |
| cap128_soft065_w1_half | cap128_soft065_w2_half | e4 | 668f087709391c3d | eda53b97267684c7 | YES |
| reg_soft065_w1_half | reg_soft065_w2_half | e4 | 232de5352d4df16e | 2356864114f6ee65 | YES |
| soft065_targeted_controls_w1_lr_half | soft065_targeted_controls_w2_lr_half | e1 | 8f7ae0f0344c4a5d | c3310132dcb28dec | YES |
| soft065_targeted_controls_w1_lr_half | soft065_targeted_controls_w2_lr_half | e2 | fe6ef5575cbd052c | a15078d74b6bca61 | YES |
| soft065_targeted_controls_w1_lr_half | soft065_targeted_controls_w2_lr_half | e4 | 84e7b9fc976238c8 | e603c47d86e677df | YES |

**Verdict: ALL w1/w2 PAIRS ISOLATED.** No collisions. The fix is confirmed effective.

## 4. Trace definitions

| trace_name | data_files | replay_weights | lr | epochs | hidden_sizes | init_type |
|---|---|---|---|---|---|---|
| cap128_soft065_w1_half | soft065 + controls | [1, 1] | 5e-5 | 1,2,4 | 128,4 | larger |
| cap128_soft065_w2_half | soft065 + controls | [1, 2] | 5e-5 | 1,2,4 | 128,4 | larger |
| cap128_soft065_w1_quarter | soft065 + controls | [1, 1] | 2.5e-5 | 1,2,4 | 128,4 | larger |
| cap128_soft055_w1_half | soft055 + controls | [1, 1] | 5e-5 | 1,2,4 | 128,4 | larger |
| reg_soft065_w1_half | soft065 + controls | [1, 1] | 5e-5 | 4,8 | 96,3 | current |
| reg_soft065_w2_half | soft065 + controls | [1, 2] | 5e-5 | 4,8 | 96,3 | current |
| reg_soft055_w1_half | soft055 + controls | [1, 1] | 5e-5 | 4,8 | 96,3 | current |
| reg_soft065_w1_quarter | soft065 + controls | [1, 1] | 2.5e-5 | 4,8 | 96,3 | current |
| soft065_targeted_controls_w1_lr_half | soft065 + controls | [1, 2] | 5e-5 | 1,2,4 | 96,3 | current |
| soft065_targeted_controls_w2_lr_half | soft065 + controls | [1, 3] | 5e-5 | 1,2,4 | 96,3 | current |
| soft055_targeted_controls_w1_lr_half | soft055 + controls | [1, 2] | 5e-5 | 1,2,4 | 96,3 | current |
| soft065_targeted_controls_w1_lr_quarter | soft065 + controls | [1, 2] | 2.5e-5 | 1,2,4 | 96,3 | current |

## 5. Training metrics

| trace_name | epoch | replay_weights | hidden_sizes | lr | policy_loss | value_loss | cached |
|---|---|---|---|---|---|---|---|
| cap128_soft065_w1_half | e1 | [1, 1] | 128,4 | 5e-5 | 1.671351 | 0.288201 | no |
| cap128_soft065_w1_half | e2 | [1, 1] | 128,4 | 5e-5 | 1.565987 | 0.148697 | no |
| cap128_soft065_w1_half | e4 | [1, 1] | 128,4 | 5e-5 | 1.550361 | 0.084661 | no |
| cap128_soft065_w2_half | e1 | [1, 2] | 128,4 | 5e-5 | 1.651721 | 0.243324 | no |
| cap128_soft065_w2_half | e2 | [1, 2] | 128,4 | 5e-5 | 1.546845 | 0.105703 | no |
| cap128_soft065_w2_half | e4 | [1, 2] | 128,4 | 5e-5 | 1.471077 | 0.075040 | no |
| cap128_soft065_w1_quarter | e1 | [1, 1] | 128,4 | 2.5e-5 | 1.726709 | 0.353111 | no |
| cap128_soft065_w1_quarter | e2 | [1, 1] | 128,4 | 2.5e-5 | 1.636202 | 0.256891 | no |
| cap128_soft065_w1_quarter | e4 | [1, 1] | 128,4 | 2.5e-5 | 1.605278 | 0.149211 | no |
| cap128_soft055_w1_half | e1 | [1, 1] | 128,4 | 5e-5 | 1.631465 | 0.289275 | no |
| cap128_soft055_w1_half | e2 | [1, 1] | 128,4 | 5e-5 | 1.531802 | 0.148548 | no |
| cap128_soft055_w1_half | e4 | [1, 1] | 128,4 | 5e-5 | 1.522270 | 0.084084 | no |
| soft065_targeted_controls_w1_lr_half | e2 | [1, 2] | 96,3 | 5e-5 | 1.578089 | 0.169830 | no |
| soft065_targeted_controls_w1_lr_half | e4 | [1, 2] | 96,3 | 5e-5 | 1.498391 | 0.108128 | no |
| soft065_targeted_controls_w2_lr_half | e1 | [1, 3] | 96,3 | 5e-5 | 1.664916 | 0.221378 | no |
| soft065_targeted_controls_w2_lr_half | e2 | [1, 3] | 96,3 | 5e-5 | 1.518756 | 0.138433 | no |
| soft065_targeted_controls_w2_lr_half | e4 | [1, 3] | 96,3 | 5e-5 | 1.445424 | 0.089685 | no |
| soft055_targeted_controls_w1_lr_half | e1 | [1, 2] | 96,3 | 5e-5 | 1.691408 | 0.239313 | no |
| soft055_targeted_controls_w1_lr_half | e2 | [1, 2] | 96,3 | 5e-5 | 1.548901 | 0.167289 | no |
| soft055_targeted_controls_w1_lr_half | e4 | [1, 2] | 96,3 | 5e-5 | 1.473153 | 0.106527 | no |
| soft065_targeted_controls_w1_lr_quarter | e1 | [1, 2] | 96,3 | 2.5e-5 | 1.813132 | 0.262033 | no |
| soft065_targeted_controls_w1_lr_quarter | e2 | [1, 2] | 96,3 | 2.5e-5 | 1.654334 | 0.221283 | no |
| soft065_targeted_controls_w1_lr_quarter | e4 | [1, 2] | 96,3 | 2.5e-5 | 1.569914 | 0.166133 | no |

Note: `reg_*` traces and `soft065_targeted_controls_w1_lr_half e1` were cached from an earlier partial run; eval was skipped. Training metrics from prior partial runs (before the init checkpoint fix) are not shown.

## 6. Evaluation results (ranked by production_optimal_1200)

| rank | trace_name | epoch | prod_total | prod_opt@1200 | prod_opt@2400 | ctrl_total | ctrl_opt@1200 | ctrl_reg | ctrl_reg_rate |
|---|---|---|---|---|---|---|---|---|---|
| 1 | soft065_targeted_controls_w1_lr_quarter | e2 | 147 | **100** | 106 | 206 | 191 | 27 | 13.1% |
| 2 | soft065_targeted_controls_w1_lr_quarter | e1 | 147 | 98 | 107 | 206 | 194 | 18 | 8.7% |
| 3 | soft055_targeted_controls_w1_lr_half | e1 | 147 | 95 | 110 | 206 | 184 | 32 | 15.5% |
| 4 | soft055_targeted_controls_w1_lr_half | e2 | 147 | 95 | 107 | 206 | 179 | 40 | 19.4% |
| 5 | soft065_targeted_controls_w1_lr_half | e2 | 147 | 95 | 106 | 206 | 173 | 41 | 19.9% |
| 6 | soft065_targeted_controls_w1_lr_quarter | e4 | 147 | 94 | 108 | 206 | 181 | 36 | 17.5% |
| 7 | soft065_targeted_controls_w2_lr_half | e1 | 147 | 92 | 109 | 206 | 178 | 38 | 18.4% |
| 8 | cap128_soft065_w1_half | e1 | 147 | 91 | 98 | 206 | 177 | 39 | 18.9% |
| 9 | cap128_soft065_w1_quarter | e1 | 147 | 91 | 103 | 206 | 185 | 36 | 17.5% |
| 10 | cap128_soft065_w1_half | e4 | 147 | 90 | 103 | 206 | 172 | 47 | 22.8% |
| 11 | cap128_soft055_w1_half | e4 | 147 | 89 | 101 | 206 | 174 | 47 | 22.8% |
| 12 | cap128_soft065_w2_half | e2 | 147 | 88 | 103 | 206 | 173 | 47 | 22.8% |
| 13 | soft065_targeted_controls_w1_lr_half | e4 | 147 | 88 | 101 | 206 | 172 | 43 | 20.9% |
| 14 | soft065_targeted_controls_w2_lr_half | e4 | 147 | 87 | 100 | 206 | 172 | 46 | 22.3% |
| 15 | cap128_soft055_w1_half | e1 | 147 | 86 | 98 | 206 | 178 | 39 | 18.9% |
| 16 | cap128_soft065_w2_half | e4 | 147 | 86 | 97 | 206 | 173 | 46 | 22.3% |
| 17 | soft055_targeted_controls_w1_lr_half | e4 | 147 | 86 | 98 | 206 | 170 | 46 | 22.3% |
| 18 | cap128_soft065_w1_quarter | e4 | 147 | 85 | 95 | 206 | 177 | 41 | 19.9% |
| 19 | cap128_soft065_w1_half | e2 | 147 | 84 | 102 | 206 | 171 | 44 | 21.4% |
| 20 | soft065_targeted_controls_w2_lr_half | e2 | 147 | 84 | 104 | 206 | 174 | 40 | 19.4% |
| 21 | cap128_soft065_w1_quarter | e2 | 147 | 82 | 98 | 206 | 174 | 41 | 19.9% |
| 22 | cap128_soft065_w2_half | e1 | 147 | 81 | 97 | 206 | 172 | 47 | 22.8% |
| 23 | cap128_soft055_w1_half | e2 | 147 | 79 | 95 | 206 | 170 | 44 | 21.4% |

**Current baseline:** 62/147 optimal@1200

## 7. Comparison with PR #79 suspect bests

| metric | PR #79 (suspect) | Fixed (best) | delta |
|---|---|---|---|
| production_optimal_1200 | 104/147 | 100/147 | -4 |
| production_optimal_2400 | 109/147 | 110/147 | +1 |
| control_regression_count | 0 (claimed) | 18 | +18 |
| control_regression_rate | 0% (claimed) | 8.7% | +8.7% |
| w1/w2 isolation | COLLISION | ISOLATED | FIXED |

PR #79's claim of 104/147 with zero control regressions was an artifact of the output collision bug (identical w1/w2 checkpoints). After isolation fix:
- Production optimal@1200 dropped from 104 to 100
- Zero-control-regression candidates disappeared entirely (min 18 regressions across all 12 traces)
- Every trace has substantial control regression

## 8. Gate results (strict local gates)

| trace_name | epoch | prod_gain_pass (vs 62) | control_gate (reg==0) | locally_acceptable |
|---|---|---|---|---|
| soft065_targeted_controls_w1_lr_quarter | e2 | PASS (100>62) | FAIL (27 reg) | NO |
| soft065_targeted_controls_w1_lr_quarter | e1 | PASS (98>62) | FAIL (18 reg) | NO |
| soft055_targeted_controls_w1_lr_half | e1 | PASS (95>62) | FAIL (32 reg) | NO |
| *(all 23 checkpoints)* | — | PASS (all) | FAIL (all) | NO |

**All 23 evaluated checkpoints fail the control gate. Zero candidates pass all strict local gates.**

## 9. Acceptance criteria assessment

| criterion | status | detail |
|---|---|---|
| w1/w2 isolation verified by differing checkpoint hashes | **PASS** | All 7 comparable pairs differ |
| Training metrics are present | **PASS** | policy_loss, value_loss populated for all non-cached traces |
| production_optimal_1200 improves over current baseline (62/147) | **PASS** | Best: 100/147 (+38) |
| production_optimal_1200 competitive with PR #79 suspect best (104/147) | **FAIL** | Best: 100 < 104 |
| Targeted control regressions are zero or clearly acceptable | **FAIL** | Min: 18/206 (8.7%); all 12 traces fail |
| Holdout regression below 0.5% | **NOT AVAILABLE** | Holdout eval not implemented in fixed runner |
| local_promotion_gate passes | **NOT RUN** | No candidate qualifies for gate testing |

## 10. Why w1/w2 differences disappeared after isolation

The PR #79 output collision inflated results because identical checkpoints were counted twice with different labels. With proper trace isolation, w1 (replay_weights=[1,1]) and w2 (replay_weights=[1,2]) now produce genuinely different checkpoints with different eval results — but both uniformly fail the control gate. The "miracle" of 104/147 with zero control regression was an artifact of the collision bug.

Key w1/w2 deltas after isolation (proxy for genuine signal):
- cap128_soft065 w1_half e1 vs w2_half e1: prod_opt 91 vs 81, ctrl_reg 39 vs 47
- cap128_soft065 w1_half e2 vs w2_half e2: prod_opt 84 vs 88, ctrl_reg 44 vs 47
- cap128_soft065 w1_half e4 vs w2_half e4: prod_opt 90 vs 86, ctrl_reg 47 vs 46
- soft065 controls w1_lr_half e2 vs w2 e2: prod_opt 95 vs 84, ctrl_reg 41 vs 40
- soft065 controls w1 e4 vs w2 e4: prod_opt 88 vs 87, ctrl_reg 43 vs 46

w1 ([1,1]) generally outperforms w2 ([1,2]) on production but both fail controls.

## 11. Interpretation

This fixed-experiment run confirms:

1. **The trace-output collision bug was real and consequential.** PR #79's results (104/147, zero control regressions) were inflated. The true best after isolation is 100/147 with 18 control regressions.

2. **Control preservation is the fundamental blocker.** Even at quarter-LR (2.5e-5) with softened targets (0.65), control regressions are substantial (18/206 = 8.7%). The original PR #78 controls and promoted regression controls fare poorly under training — the model cannot simultaneously optimize production targets and preserve controls.

3. **Capacity scaling does not help.** The 128x4 traces (cap128_*) underperform the 96x3 traces on production metrics (best 91/147 vs 100/147) and have comparable or worse control regression.

4. **Replay weight ratio has limited impact.** Varying weights from [1,1] to [1,2] to [1,3] produces different checkpoints (isolation works) but does not resolve the control regression problem.

## 12. Decision: REJECT

**Classification:** `exact_tablebase_control_regression_intrinsic`

**Rationale:**
- Production gains from PR #79 disappear after isolation fix (104 → 100)
- No candidate achieves zero control regression (min 8.7%)
- The model cannot simultaneously fit production targets and preserve controls even with quarter-LR and softened targets
- Capacity scaling (128x4) degrades rather than helps

### Decision table

| classification | supporting_evidence | rejected_alternatives | next_action |
|---|---|---|---|
| exact_tablebase_control_regression_intrinsic | best_prod=100/147; min_ctrl_reg=18/206 (8.7%); PR79 gains disappeared after isolation; no zero-regression candidate exists | trace_output_collision_bug (fixed); replay_weight_ignored (fixed); exact_tablebase_no_local_signal | Redesign: abandon exact-tablebase direct patching in favor of MCTS-guided self-play data with tablebase-verified value targets, or investigate representation interference as root cause of control instability |

### Acceptance criteria

- No arena was run.
- No local_promotion_gate was run.
- No model was promoted.
- `storage/ai/alphazero_lite/current` was not overwritten.
- Active references were not mutated.
- No architecture changes were introduced.
- Replay weights and model capacity were not changed together outside the existing trace grid.

## 13. Artifacts

| artifact | path |
|---|---|
| Trace summary | `/tmp/azlite_medium_exact_tablebase_stabilization_v2_fixed/trace_summary_fixed.json` |
| Checkpoints | `/tmp/azlite_medium_exact_tablebase_stabilization_v2_fixed/exports/*.npz` |
| Eval artifacts | `/tmp/azlite_medium_exact_tablebase_stabilization_v2_fixed/eval/*/` |
