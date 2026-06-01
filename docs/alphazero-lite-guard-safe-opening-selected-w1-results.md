# AlphaZero-lite Guard-Safe Opening Selected W1 Results

## 1. Context

- Task scope: exactly one controlled AlphaZero-lite training lane using `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` at replay weight `1`.
- Current baseline artifact stayed fixed at `storage/ai/alphazero_lite/current`.
- No replay-weight sweep, no auto-promotion, no overwrite of `storage/ai/alphazero_lite/current`.
- Top-k checkpoints were screened before any candidate could be trusted.
- Arena, MCTS1200, benchmark, and local promotion gate were blocked unless a candidate passed the corrected guard kill gate first.

## 2. Why This Lane Was Allowed

- PR #35 classified the selected artifact as `compatible_with_inherited_policy_drift`.
- PR #35 validated the selected artifact at `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` with `26` rows, corrected guard rows present, `duplicate_conflicts = 0`, and `stale_reference_conflicts = 0`.
- PR #35 also reported no post-init corrected guard search regression in the low-epoch drift trace, so one small guarded full lane was the next allowed experiment.
- The lane still required the existing corrected guard search gate before any arena or promotion decision.

## 3. Config Summary

| run_id | config_path | selected_artifact | replay_weight | seed | model_type | input_encoding | versions_dir |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `exp-v3-guard-safe-opening-selected-w1` | `ml/alphazero_lite/configs/exp_v3_guard_safe_opening_selected_w1.json` | `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` | 1 | 42 | `residual_v3` | `kalah_v3` | `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions` |

- Effective iteration dir: `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1`.
- Execution note: the lane used `sharpened` policy/value target modes so training could consume the selected replay artifact, whose rows are tagged with `policy_target_mode=value_target_mode=sharpened`.

## 4. Selected Artifact Validation

| artifact_path | row_count | guard_rows_present | duplicate_conflicts | stale_reference_conflicts | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` | 26 | true | 0 | 0 | `ok` | reused the PR #35 validated selected artifact |

## 5. Training Result

- Pipeline result: `completed`.
- Manifest: `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/run_manifest.json`.
- Device: `cpu`.
- Final logged training metrics from `train.log`:
  policy_loss `1.228448`, value_loss `0.060752`, best_val_loss `1.253882`.
- Exported checkpoints:
  `checkpoint.npz`, `checkpoint.top1.npz`, `checkpoint.top2.npz`, `checkpoint.top3.npz`.

## 6. Corrected Guard Kill-Gate Results

- Gate settings: corrected tracked references from `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`, with the existing train-only fallback fixture used only to supply missing `capture_available-006` metadata, matching prior guarded-reference tooling.
- Budgets: `384`, `1200`.
- Search settings: existing `build_eval_search_options()` baseline, no root-prior transforms and no extra search intervention.
- Gate outcome: every exported candidate failed the kill gate because rows `capture_available-002` and `capture_available-003` selected move `1` instead of corrected reference move `2` at budget `384`.

| candidate | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `checkpoint.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2318 | 0.3950 | false | selected_move_not_reference |
| `checkpoint.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3255 | 0.4067 | false | selected_move_not_reference |
| `checkpoint.npz` | `capture_available-006` | 2 | 2 | 2 | 0.3385 | 0.5225 | true | pass_reference_selected |
| `checkpoint.npz` | `capture_available-007` | 2 | 2 | 2 | 0.3776 | 0.4475 | true | pass_reference_selected |
| `checkpoint.npz` | `capture_available-008` | 1 | 1 | 1 | 0.6250 | 0.7350 | true | pass_reference_selected |
| `checkpoint.top1.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2318 | 0.3950 | false | selected_move_not_reference |
| `checkpoint.top1.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3255 | 0.4067 | false | selected_move_not_reference |
| `checkpoint.top1.npz` | `capture_available-006` | 2 | 2 | 2 | 0.3385 | 0.5225 | true | pass_reference_selected |
| `checkpoint.top1.npz` | `capture_available-007` | 2 | 2 | 2 | 0.3776 | 0.4475 | true | pass_reference_selected |
| `checkpoint.top1.npz` | `capture_available-008` | 1 | 1 | 1 | 0.6250 | 0.7350 | true | pass_reference_selected |
| `checkpoint.top2.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2865 | 0.4333 | false | selected_move_not_reference |
| `checkpoint.top2.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3411 | 0.4575 | false | selected_move_not_reference |
| `checkpoint.top2.npz` | `capture_available-006` | 2 | 2 | 2 | 0.4062 | 0.5108 | true | pass_reference_selected |
| `checkpoint.top2.npz` | `capture_available-007` | 2 | 2 | 2 | 0.3958 | 0.5100 | true | pass_reference_selected |
| `checkpoint.top2.npz` | `capture_available-008` | 1 | 1 | 1 | 0.6250 | 0.7217 | true | pass_reference_selected |
| `checkpoint.top3.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2396 | 0.4100 | false | selected_move_not_reference |
| `checkpoint.top3.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3229 | 0.4242 | false | selected_move_not_reference |
| `checkpoint.top3.npz` | `capture_available-006` | 2 | 2 | 2 | 0.3177 | 0.4842 | true | pass_reference_selected |
| `checkpoint.top3.npz` | `capture_available-007` | 2 | 2 | 2 | 0.4349 | 0.5033 | true | pass_reference_selected |
| `checkpoint.top3.npz` | `capture_available-008` | 1 | 1 | 1 | 0.6380 | 0.7242 | true | pass_reference_selected |

## 7. Top-K Evaluation

| checkpoint | top_k_score | guard_gate_pass | arena_score_if_available | unstable_decision | selected_for_promotion_gate | notes |
| --- | --- | --- | --- | --- | --- | --- |
| `checkpoint.npz` | - | false | - | - | false | failed corrected guard kill gate on 002/003 at 384; arena skipped |
| `checkpoint.top1.npz` | 1.253882 | false | - | - | false | failed corrected guard kill gate on 002/003 at 384; arena skipped |
| `checkpoint.top2.npz` | 1.268750 | false | - | - | false | failed corrected guard kill gate on 002/003 at 384; arena skipped |
| `checkpoint.top3.npz` | 1.285996 | false | - | - | false | failed corrected guard kill gate on 002/003 at 384; arena skipped |

- Because no top-k candidate passed the corrected guard kill gate, no arena report, no MCTS1200 report, and no benchmark/promotion evidence were generated for this lane.

## 8. Local Promotion Gate

| candidate | arena_score | arena_ci_low | arena_ci_high | unstable_decision | mcts1200_result | benchmark_pass | decision | report_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| - | - | - | - | - | - | - | `not_run_guard_gate_failed` | - |

## 9. Interpretation

- Classification: `full_lane_guard_regression`.
- This lane did produce checkpoints, but every exported top-k candidate regressed the corrected guard kill gate at budget `384` on `capture_available-002` and `capture_available-003`.
- The regressions were guard-search failures, not just validation-loss noise, so the lane was rejected before any arena or promotion evidence could be collected.
- No production artifact was promoted.
- No auto-promotion path was entered.
- `checkpoint.top2` and `checkpoint.top3` changed visit shares slightly, but none cleared the required all-five-row `384` and `1200` pass condition.

## 10. Exactly One Recommended Next Action

Recommendation: **reproduce with a low-cost full-pipeline trace and identify whether self-play/background replay mix caused the regression.**
