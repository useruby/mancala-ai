# AlphaZero-lite Denoised Opening-Min384 Selected-W1 Guard-W2 Results

## 1. Context

- PR #40 classified the diagnostic tradeoff run as `guard_anchor_sufficient`.
- The only epoch-4 trace that passed all five corrected guard rows was `denoised_opening_min384_plus_selected_w1_guard_controls_w2`.
- This follow-up ran exactly one production-scale lane with the same replay mix, corrected guard kill gate before arena, no auto-promotion, and explicit `storage/ai/alphazero_lite/current` paths.

## 2. Why PR #40 Justified Exactly This Lane

- PR #40 showed selected replay at weight 1 plus guard controls at weight 2 was the only trace that recovered `capture_available-002`, `003`, `006`, `007`, and `008` together.
- The same trace was stable by `384` sims for `002` and `003`, and by `256` sims for `006`, `007`, and `008`.
- No replay-weight sweep was repeated here. No selected-only rerun was performed. No default self-play behavior was changed.

## 3. Config Summary

| run_id | config_path | versions_dir | self_play_games | base_simulations | opening_min_simulations | policy_target_noise_mode | selected_artifact_weight | guard_controls_weight | seed | model_type | input_encoding |
| --- | --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- | --- |
| `exp-v3-denoised-opening-min384-selected-w1-guard-w2` | `ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2.json` | `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions` | 1600 | 192 | 384 | `denoised` | 1 | 2 | 42 | `residual_v3` | `kalah_v3` |

## 4. Input Validation

- Config JSON validated with `.venv/bin/python -m json.tool ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2.json`.
- Selected artifact exists at `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` with `26` rows.
- Guard-controls artifact exists at `/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl` with `5` rows.
- Corrected reference artifact exists at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- `capture_available-006` is supplied through the existing fallback merge path from `ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json`, which is exactly what `corrected_guard_kill_gate.py` uses.
- Current artifact exists at `storage/ai/alphazero_lite/current`.

## 5. Training Result

- Pipeline command: `.venv/bin/python ml/alphazero_lite/pipeline.py --config ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2.json`
- Iteration dir: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter1`
- Self-play output: `63959` rows.
- Training completed and wrote `checkpoint.npz`, `checkpoint.top1.npz`, `checkpoint.top2.npz`, and `checkpoint.top3.npz`.
- Final logged training metrics: `policy_loss=1.120452`, `value_loss=0.056106`, `best_val_loss=1.145113`.
- Validation loss was recorded only as training metadata, not used as promotion evidence.

## 6. Top-k Corrected Guard Gate Results

| checkpoint | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | row_pass | gate_pass | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| `checkpoint.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2266 | 0.4500 | false | false | `selected_move_not_reference` |
| `checkpoint.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3464 | 0.4608 | false | false | `selected_move_not_reference` |
| `checkpoint.npz` | `capture_available-006` | 2 | 2 | 2 | 0.4219 | 0.5717 | true | false | `pass_reference_selected` |
| `checkpoint.npz` | `capture_available-007` | 2 | 1 | 2 | 0.4193 | 0.5250 | false | false | `selected_move_not_reference` |
| `checkpoint.npz` | `capture_available-008` | 1 | 1 | 1 | 0.7812 | 0.8125 | true | false | `pass_reference_selected` |
| `checkpoint.top1.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2266 | 0.4500 | false | false | `selected_move_not_reference` |
| `checkpoint.top1.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3464 | 0.4608 | false | false | `selected_move_not_reference` |
| `checkpoint.top1.npz` | `capture_available-006` | 2 | 2 | 2 | 0.4219 | 0.5717 | true | false | `pass_reference_selected` |
| `checkpoint.top1.npz` | `capture_available-007` | 2 | 1 | 2 | 0.4193 | 0.5250 | false | false | `selected_move_not_reference` |
| `checkpoint.top1.npz` | `capture_available-008` | 1 | 1 | 1 | 0.7812 | 0.8125 | true | false | `pass_reference_selected` |
| `checkpoint.top2.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2812 | 0.4642 | false | false | `selected_move_not_reference` |
| `checkpoint.top2.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3776 | 0.4708 | false | false | `selected_move_not_reference` |
| `checkpoint.top2.npz` | `capture_available-006` | 2 | 2 | 2 | 0.4505 | 0.6300 | true | false | `pass_reference_selected` |
| `checkpoint.top2.npz` | `capture_available-007` | 2 | 2 | 2 | 0.4479 | 0.5575 | true | false | `pass_reference_selected` |
| `checkpoint.top2.npz` | `capture_available-008` | 1 | 1 | 1 | 0.7578 | 0.8025 | true | false | `pass_reference_selected` |
| `checkpoint.top3.npz` | `capture_available-002` | 2 | 1 | 2 | 0.2604 | 0.3875 | false | false | `selected_move_not_reference` |
| `checkpoint.top3.npz` | `capture_available-003` | 2 | 1 | 2 | 0.3646 | 0.4533 | false | false | `selected_move_not_reference` |
| `checkpoint.top3.npz` | `capture_available-006` | 2 | 2 | 2 | 0.3958 | 0.5958 | true | false | `pass_reference_selected` |
| `checkpoint.top3.npz` | `capture_available-007` | 2 | 2 | 2 | 0.4583 | 0.5267 | true | false | `pass_reference_selected` |
| `checkpoint.top3.npz` | `capture_available-008` | 1 | 1 | 1 | 0.7500 | 0.7858 | true | false | `pass_reference_selected` |

## 7. Top-k Arena Results

| checkpoint | guard_gate_pass | arena_skipped_due_to_guard | arena_score | wins | losses | draws | unstable_decision | selected_for_local_gate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `checkpoint.npz` | false | true | - | - | - | - | - | false | corrected guard gate failed before arena |
| `checkpoint.top1.npz` | false | true | - | - | - | - | - | false | corrected guard gate failed before arena |
| `checkpoint.top2.npz` | false | true | - | - | - | - | - | false | corrected guard gate failed before arena |
| `checkpoint.top3.npz` | false | true | - | - | - | - | - | false | corrected guard gate failed before arena |

## 8. Local Promotion Gate Results, If Run

- Not run. No top-k candidate passed the corrected guard gate, so `script/ai/local_promotion_gate` was intentionally skipped.

## 9. Larger Confirmation, If Run

- Not run. The preconditions for larger confirmation were not met.

## 10. Interpretation

- Classification: `production_lane_guard_regression`.
- Result: all four exported candidates failed the corrected guard gate, and arena was skipped for every candidate.
- Common failure pattern: every candidate still selected move `1` instead of corrected reference move `2` on both `capture_available-002` and `capture_available-003` at the `384`-simulation budget.
- Secondary detail: `checkpoint.top2.npz` and `checkpoint.top3.npz` recovered `capture_available-007`, but that was not enough because `002` and `003` still failed.
- No model was promoted automatically. `storage/ai/alphazero_lite/current` was not overwritten.

## 11. Exactly One Recommended Next Action

Recommendation: **compare the PR #40 diagnostic trace against this full pipeline lane with deterministic data-order and checkpoint-hash inspection to explain why the diagnostic trace passed while the production-scale lane regressed.**
