# AlphaZero-lite Guard-Safe Opening Full-Pipeline Interaction Trace Results

## 1. Context

- PR #36 trained the selected replay lane but every exported candidate failed the corrected guard kill gate on `capture_available-002` and `capture_available-003` at `384` simulations.
- PR #35 already showed the selected artifact was statically guard-safe and did not regress under low-epoch artifact-only drift tracing.
- This run stayed diagnostic-only: no arena, no MCTS1200 eval lane, no promotion, and no overwrite of `storage/ai/alphazero_lite/current`.

## 2. Why PR #36 Stopped The Selected Lane

- PR #36 stopped because rows `capture_available-002` and `capture_available-003` selected move `1` instead of corrected reference move `2` at `384` simulations, even though both rows recovered to move `2` at `1200` simulations.
- That made the failure look like a full-pipeline interaction or search-sensitivity issue rather than a simple static replay-artifact conflict.

## 3. Data/Input Verification

| input | path | exists | rows_or_size | status | notes |
| --- | --- | --- | --- | --- | --- |
| pr36_self_play | `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl` | true | 62954 | `reused_existing` | reused PR #36 self_play.jsonl |
| selected_artifact | `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` | true | 26 | `ok` | ok |
| guard_controls_artifact | `/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl` | true | 5 | `ok` | available for Trace D |
| corrected_references | `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json` | true | 322711 | `ok` | ok |
| current_artifact | `storage/ai/alphazero_lite/current` | true | 2 | `ok` | initializer artifact present; materialized checkpoint reused under /tmp |
| current_init_checkpoint | `/tmp/azlite_guard_safe_opening_full_pipeline_trace/current_init_checkpoint.npz` | true | 298302 | `ok` | materialized from current weights.json |

## 4. Current Baseline Guard Result

| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | row_pass | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current_baseline | 0 | capture_available-002 | 2 | 2 | 2 | 0.4036 | 0.5308 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| current_baseline | 0 | capture_available-003 | 2 | 2 | 2 | 0.4740 | 0.6742 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| current_baseline | 0 | capture_available-006 | 2 | 2 | 2 | 0.5547 | 0.6600 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| current_baseline | 0 | capture_available-007 | 2 | 2 | 2 | 0.5859 | 0.7400 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| current_baseline | 0 | capture_available-008 | 1 | 1 | 1 | 0.7682 | 0.8733 | 0.0000 | 0.0000 | true | true | pass_reference_selected |

## 5. Trace Definitions

| trace_name | data_files | replay_weights | lr | epochs | init_checkpoint | purpose | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| self_play_only | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl"]` | `[1]` | 0.001 | `[1, 2, 4]` | `/tmp/azlite_guard_safe_opening_full_pipeline_trace/current_init_checkpoint.npz` | test whether generated self-play/background data alone causes guard regression. | `completed` |
| selected_artifact_only_pipeline_hparams | `["/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1]` | 0.001 | `[1, 2, 4]` | `/tmp/azlite_guard_safe_opening_full_pipeline_trace/current_init_checkpoint.npz` | compare PR #35 artifact-only conclusion under the exact PR #36 training settings. | `completed` |
| self_play_plus_selected_artifact_w1 | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1, 1]` | 0.001 | `[1, 2, 4]` | `/tmp/azlite_guard_safe_opening_full_pipeline_trace/current_init_checkpoint.npz` | reproduce the PR #36 full-lane regression and identify when it appears. | `completed` |
| self_play_plus_guard_controls | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl", "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"]` | `[1, 1, 2]` | 0.001 | `[1, 2, 4]` | `/tmp/azlite_guard_safe_opening_full_pipeline_trace/current_init_checkpoint.npz` | test whether stronger guard anchoring prevents the 384-sim 002/003 drift. | `completed` |
| self_play_plus_selected_artifact_low_lr | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1, 1]` | 0.0001 | `[1, 2, 4]` | `/tmp/azlite_guard_safe_opening_full_pipeline_trace/current_init_checkpoint.npz` | test update-size sensitivity in the full-data setting. | `completed` |

## 6. Guard Drift Results

| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | row_pass | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| self_play_only | 1 | capture_available-002 | 2 | 1 | 1 | 0.1589 | 0.3050 | 0.0053 | -0.0104 | false | false | selected_move_not_reference |
| self_play_only | 1 | capture_available-003 | 2 | 1 | 1 | 0.3438 | 0.3350 | -0.0443 | -0.0215 | false | false | selected_move_not_reference |
| self_play_only | 1 | capture_available-006 | 2 | 1 | 2 | 0.2891 | 0.4508 | -0.0373 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only | 1 | capture_available-007 | 2 | 2 | 2 | 0.3932 | 0.4283 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 1 | capture_available-008 | 1 | 1 | 1 | 0.6172 | 0.7075 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 2 | capture_available-002 | 2 | 2 | 2 | 0.3411 | 0.4492 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 2 | capture_available-003 | 2 | 1 | 2 | 0.3542 | 0.4958 | -0.0307 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only | 2 | capture_available-006 | 2 | 2 | 2 | 0.4609 | 0.5358 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 2 | capture_available-007 | 2 | 2 | 2 | 0.4557 | 0.5383 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 2 | capture_available-008 | 1 | 1 | 1 | 0.6589 | 0.7550 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 4 | capture_available-002 | 2 | 1 | 2 | 0.2708 | 0.4033 | -0.0147 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only | 4 | capture_available-003 | 2 | 1 | 2 | 0.3464 | 0.4233 | -0.0417 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only | 4 | capture_available-006 | 2 | 2 | 2 | 0.3542 | 0.4583 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 4 | capture_available-007 | 2 | 2 | 2 | 0.3854 | 0.4308 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only | 4 | capture_available-008 | 1 | 1 | 1 | 0.6406 | 0.7300 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 1 | capture_available-002 | 2 | 1 | 2 | 0.2422 | 0.3292 | -0.0079 | 0.0000 | false | false | selected_move_not_reference |
| selected_artifact_only_pipeline_hparams | 1 | capture_available-003 | 2 | 1 | 1 | 0.1693 | 0.3383 | -0.0127 | -0.0204 | false | false | selected_move_not_reference |
| selected_artifact_only_pipeline_hparams | 1 | capture_available-006 | 2 | 2 | 2 | 0.4219 | 0.3983 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 1 | capture_available-007 | 2 | 2 | 2 | 0.3698 | 0.4333 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 1 | capture_available-008 | 1 | 1 | 1 | 0.6146 | 0.7417 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 2 | capture_available-002 | 2 | 2 | 2 | 0.4505 | 0.3892 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 2 | capture_available-003 | 2 | 2 | 2 | 0.3802 | 0.5975 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 2 | capture_available-006 | 2 | 2 | 2 | 0.3073 | 0.6300 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 2 | capture_available-007 | 2 | 2 | 2 | 0.5234 | 0.5475 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 2 | capture_available-008 | 1 | 1 | 1 | 0.4922 | 0.6592 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 4 | capture_available-002 | 2 | 2 | 2 | 0.4792 | 0.5658 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 4 | capture_available-003 | 2 | 2 | 2 | 0.6354 | 0.6575 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 4 | capture_available-006 | 2 | 2 | 2 | 0.6406 | 0.5842 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 4 | capture_available-007 | 2 | 2 | 2 | 0.7552 | 0.7575 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| selected_artifact_only_pipeline_hparams | 4 | capture_available-008 | 1 | 1 | 1 | 0.4740 | 0.4908 | 0.0000 | 0.0000 | true | true | pass_reference_selected |
| self_play_plus_selected_artifact_w1 | 1 | capture_available-002 | 2 | 1 | 2 | 0.1458 | 0.3525 | 0.0087 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 1 | capture_available-003 | 2 | 1 | 1 | 0.2839 | 0.3758 | -0.0392 | -0.0286 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 1 | capture_available-006 | 2 | 2 | 2 | 0.3229 | 0.4750 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_w1 | 1 | capture_available-007 | 2 | 1 | 1 | 0.2917 | 0.3550 | -0.0386 | -0.0175 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 1 | capture_available-008 | 1 | 1 | 1 | 0.6458 | 0.7583 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_w1 | 2 | capture_available-002 | 2 | 1 | 2 | 0.2396 | 0.4267 | -0.0079 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 2 | capture_available-003 | 2 | 1 | 2 | 0.3464 | 0.4325 | -0.0388 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 2 | capture_available-006 | 2 | 2 | 2 | 0.3359 | 0.4858 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_w1 | 2 | capture_available-007 | 2 | 2 | 2 | 0.4401 | 0.5017 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_w1 | 2 | capture_available-008 | 1 | 1 | 1 | 0.6328 | 0.7233 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_w1 | 4 | capture_available-002 | 2 | 1 | 2 | 0.2344 | 0.3858 | -0.0134 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 4 | capture_available-003 | 2 | 1 | 2 | 0.3255 | 0.4233 | -0.0425 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 4 | capture_available-006 | 2 | 2 | 2 | 0.3281 | 0.5250 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_w1 | 4 | capture_available-007 | 2 | 1 | 2 | 0.3698 | 0.4625 | -0.0382 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_w1 | 4 | capture_available-008 | 1 | 1 | 1 | 0.6328 | 0.7283 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 1 | capture_available-002 | 2 | 1 | 2 | 0.1458 | 0.4083 | 0.0417 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_guard_controls | 1 | capture_available-003 | 2 | 2 | 2 | 0.4349 | 0.4467 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 1 | capture_available-006 | 2 | 0 | 2 | 0.2734 | 0.4542 | -0.0341 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_guard_controls | 1 | capture_available-007 | 2 | 2 | 2 | 0.4505 | 0.5075 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 1 | capture_available-008 | 1 | 1 | 1 | 0.5833 | 0.6967 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 2 | capture_available-002 | 2 | 1 | 1 | 0.1536 | 0.3275 | 0.0144 | -0.0160 | false | false | selected_move_not_reference |
| self_play_plus_guard_controls | 2 | capture_available-003 | 2 | 1 | 1 | 0.3411 | 0.3792 | -0.0394 | -0.0223 | false | false | selected_move_not_reference |
| self_play_plus_guard_controls | 2 | capture_available-006 | 2 | 2 | 2 | 0.2734 | 0.4533 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 2 | capture_available-007 | 2 | 2 | 2 | 0.3958 | 0.5125 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 2 | capture_available-008 | 1 | 1 | 1 | 0.5859 | 0.7008 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 4 | capture_available-002 | 2 | 1 | 2 | 0.2474 | 0.4317 | -0.0170 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_guard_controls | 4 | capture_available-003 | 2 | 1 | 2 | 0.3594 | 0.4508 | -0.0444 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_guard_controls | 4 | capture_available-006 | 2 | 2 | 2 | 0.3672 | 0.5417 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 4 | capture_available-007 | 2 | 2 | 2 | 0.4375 | 0.5300 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_guard_controls | 4 | capture_available-008 | 1 | 1 | 1 | 0.6510 | 0.7358 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 1 | capture_available-002 | 2 | 0 | 2 | 0.2578 | 0.3817 | -0.0512 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_low_lr | 1 | capture_available-003 | 2 | 1 | 2 | 0.3906 | 0.5375 | -0.0489 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_low_lr | 1 | capture_available-006 | 2 | 2 | 2 | 0.3490 | 0.5583 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 1 | capture_available-007 | 2 | 2 | 2 | 0.4479 | 0.5650 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 1 | capture_available-008 | 1 | 1 | 1 | 0.6693 | 0.7667 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 2 | capture_available-002 | 2 | 0 | 2 | 0.2240 | 0.3458 | -0.0407 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_low_lr | 2 | capture_available-003 | 2 | 1 | 2 | 0.3594 | 0.5000 | -0.0456 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_low_lr | 2 | capture_available-006 | 2 | 2 | 2 | 0.3203 | 0.4875 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 2 | capture_available-007 | 2 | 2 | 2 | 0.4167 | 0.4983 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 2 | capture_available-008 | 1 | 1 | 1 | 0.6641 | 0.7617 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 4 | capture_available-002 | 2 | 0 | 2 | 0.2135 | 0.3692 | -0.0334 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_low_lr | 4 | capture_available-003 | 2 | 1 | 2 | 0.3828 | 0.4600 | -0.0489 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_selected_artifact_low_lr | 4 | capture_available-006 | 2 | 2 | 2 | 0.3203 | 0.5167 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 4 | capture_available-007 | 2 | 2 | 2 | 0.4193 | 0.4817 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_selected_artifact_low_lr | 4 | capture_available-008 | 1 | 1 | 1 | 0.6953 | 0.7733 | 0.0000 | 0.0000 | true | false | pass_reference_selected |

## 7. Training Metric Trace

| trace_name | epoch | policy_loss | value_loss | total_loss | val_loss | guard_cross_entropy | selected_artifact_cross_entropy | self_play_cross_entropy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| self_play_only | 1 | 1.2781 | 0.0646 | 1.2975 | 1.3087 | - | - | 1.2781 | full_dataset_snapshot |
| self_play_only | 2 | 1.2526 | 0.0652 | 1.2722 | 1.2966 | - | - | 1.2526 | full_dataset_snapshot |
| self_play_only | 4 | 1.2048 | 0.0622 | 1.2235 | 1.2513 | - | - | 1.2048 | full_dataset_snapshot |
| selected_artifact_only_pipeline_hparams | 1 | 2.0777 | 0.0431 | 2.0906 | 2.2476 | 1.7774 | 2.0777 | - | full_dataset_snapshot |
| selected_artifact_only_pipeline_hparams | 2 | 1.7327 | 0.0288 | 1.7414 | 2.1215 | 1.2664 | 1.7327 | - | full_dataset_snapshot |
| selected_artifact_only_pipeline_hparams | 4 | 1.6112 | 0.0247 | 1.6186 | 2.0181 | 1.1568 | 1.6112 | - | full_dataset_snapshot |
| self_play_plus_selected_artifact_w1 | 1 | 1.2951 | 0.0668 | 1.3151 | 1.3401 | 2.1596 | 1.9558 | 1.2948 | full_dataset_snapshot |
| self_play_plus_selected_artifact_w1 | 2 | 1.2396 | 0.0634 | 1.2586 | 1.2842 | 1.9015 | 1.8598 | 1.2394 | full_dataset_snapshot |
| self_play_plus_selected_artifact_w1 | 4 | 1.2012 | 0.0618 | 1.2197 | 1.2539 | 1.9534 | 1.8502 | 1.2009 | full_dataset_snapshot |
| self_play_plus_guard_controls | 1 | 1.2716 | 0.0649 | 1.2911 | 1.3105 | 1.7535 | 1.9293 | 1.2713 | full_dataset_snapshot |
| self_play_plus_guard_controls | 2 | 1.2642 | 0.0643 | 1.2835 | 1.3099 | 1.8934 | 1.9338 | 1.2639 | full_dataset_snapshot |
| self_play_plus_guard_controls | 4 | 1.2022 | 0.0620 | 1.2208 | 1.2562 | 1.8497 | 1.8387 | 1.2019 | full_dataset_snapshot |
| self_play_plus_selected_artifact_low_lr | 1 | 1.2930 | 0.0637 | 1.3121 | 1.3252 | 1.9543 | 1.9481 | 1.2927 | full_dataset_snapshot |
| self_play_plus_selected_artifact_low_lr | 2 | 1.2801 | 0.0635 | 1.2991 | 1.3142 | 2.0251 | 1.9291 | 1.2798 | full_dataset_snapshot |
| self_play_plus_selected_artifact_low_lr | 4 | 1.2725 | 0.0632 | 1.2915 | 1.3085 | 2.0100 | 1.9419 | 1.2723 | full_dataset_snapshot |

## 8. Cause Classification

| classification | supporting_evidence | rejected_alternatives | next_action |
| --- | --- | --- | --- |
| `self_play_background_causes_guard_regression` | self_play_only reproduced the PR #36-style 384-sim 002/003 regression without any selected replay artifact. Follow-up self-play inspection found a shared predecessor state whose targets under-support the corrected move-2 branch and frequently steer sibling lines toward move 1. | Selected artifact was not required to trigger the failure. | build a small corrective artifact anchored on `opening_plies_1_8-017` and descendant `capture_available-002` / `003` / `007` rows, then rerun the 384/1200 guard eval. |

## 9. Self-Play Follow-Up

- Exact PR #36 self-play matches existed for `capture_available-002` (2 rows), `capture_available-003` (7 rows), and `opening_plies_1_8-068` (16 rows).
- All matched rows used `teacher_source: puct`, so the bad labels came from search targets, not opening-cache reuse.
- `capture_available-002` was directly mis-targeted in the dataset: its averaged policy preferred move `1` (`0.4973`) over corrected move `2` (`0.1849`).
- `capture_available-003` was less damaged but still noisy: its averaged policy kept move `2` on top (`0.3898`) while move `1` stayed close (`0.3638`).
- Both failing guard rows descend from the same predecessor, `opening_plies_1_8-017`, whose corrected reference move is `2`.
- That predecessor appeared 32 times in PR #36 self-play, but only 3 of those rows had move `2` as the top policy target; 22 instead preferred sibling moves `4` or `5`.
- The successor branch reinforced the same local drift: `capture_available-002` and `capture_available-007` both averaged to move `1`, while only `capture_available-003` still averaged to move `2`.
- This local target pattern is sufficient to explain why 384-sim search drifts to move `1` on `capture_available-002` / `003` even though 1200 sims can still recover the corrected move.

## 10. Exactly One Recommended Next Action

Recommendation: **build a small corrective artifact anchored on `opening_plies_1_8-017` and descendant `capture_available-002` / `003` / `007` rows, then rerun the 384/1200 guard eval.**
