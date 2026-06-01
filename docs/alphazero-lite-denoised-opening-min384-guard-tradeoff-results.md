# AlphaZero-lite Denoised Opening-Min384 Guard Tradeoff Results

## 1. Context

- PR #39 added denoised self-play targets, optional root telemetry, and opening-min simulation controls without changing default production behavior.
- PR #39 ended as `denoised_targets_partial_003_needs_more_sims`: no arena, no promotion, and no full corrected guard pass.
- The remaining question is narrower than generic denoising: under denoised opening-min-384 training, self-play-only helps `capture_available-002` while selected replay helps `capture_available-003`.

## 2. Why PR #39 Was Not Enough

- `denoised_opening_min384_self_play_only` at epoch 4 previously passed `002/006/007/008` but still failed `003` at `384` sims.
- `denoised_opening_min384_plus_selected_w1` at epoch 4 previously passed `003/006/007/008` but regressed `002` at `384` sims.
- No prior trace achieved a full corrected guard pass, so the remaining issue is a `002` vs `003` tradeoff under the `384`-sim gate.

## 3. Input and Target-Pressure Inspection

| input | path | exists | rows_or_size | status | notes |
| --- | --- | --- | --- | --- | --- |
| denoised_opening_min384_self_play | `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl` | true | 63959 | `reused_existing` | reused PR39 denoised opening-min-384 self_play.jsonl |
| selected_artifact | `/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl` | true | 26 | `ok` | guard-safe static artifact |
| guard_controls_artifact | `/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl` | true | 5 | `ok` | diagnostic-only guard artifact |
| corrected_references | `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json` | true | 322711 | `ok` | corrected incumbent forensic references |
| current_artifact | `storage/ai/alphazero_lite/current` | true | 2 | `ok` | initializer only; never overwritten |

Near-duplicate scan support: self_play.jsonl exposes exact canonical matches and search telemetry, but no row_id/opening_family labels for broader near-duplicate grouping.

| row_id | source | hit_count | corrected_reference_move | average_reference_mass | move_1_mass | move_2_mass | top_target_distribution | in_selected_artifact | in_guard_controls | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | self_play_exact + selected_artifact | 16 | 2 | 0.2365 | 0.1675 | 0.2365 | `{2:16}` | true | false | selected_artifact_top=2 move1=0.0717 move2=0.2683, selected_artifact_adds_opening_anchor_row |
| capture_available-002 | self_play_exact + selected_artifact + guard_controls | 4 | 2 | 0.5461 | 0.2086 | 0.5461 | `{2:4}` | true | true | selected_artifact_top=2 move1=0.0127 move2=0.9049, guard_controls_artifact_top=2 move1=0.0127 move2=0.9049, self_play_and_selected_both_anchor_002_exactly |
| capture_available-003 | selected_artifact + guard_controls | 0 | 2 | - | - | - | `-` | true | true | no_exact_self_play_hits, selected_artifact_top=2 move1=0.0472 move2=0.9035, guard_controls_artifact_top=2 move1=0.0472 move2=0.9035, selected_artifact_supplies_exact_003_anchor_absent_from_self_play |
| capture_available-006 | selected_artifact + guard_controls | 0 | 2 | - | - | - | `-` | true | true | no_exact_self_play_hits, selected_artifact_top=2 move1=0.0992 move2=0.5050, guard_controls_artifact_top=2 move1=0.0992 move2=0.5050 |
| capture_available-007 | self_play_exact + selected_artifact + guard_controls | 3 | 2 | 0.4893 | 0.3209 | 0.4893 | `{2:3}` | true | true | selected_artifact_top=2 move1=0.0507 move2=0.7240, guard_controls_artifact_top=2 move1=0.0507 move2=0.7240 |
| capture_available-008 | self_play_exact + selected_artifact + guard_controls | 3 | 1 | 0.8072 | 0.8072 | 0.0396 | `{1:3}` | true | true | selected_artifact_top=1 move1=0.7339 move2=0.1595, guard_controls_artifact_top=1 move1=0.7339 move2=0.1595 |

## 4. Trace Definitions

| trace_name | data_files | replay_weights | epochs | status | notes |
| --- | --- | --- | --- | --- | --- |
| denoised_opening_min384_self_play_only | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl"]` | `[1]` | `[1, 2, 4]` | `completed` | ok |
| denoised_opening_min384_plus_selected_w1 | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1, 1]` | `[1, 2, 4]` | `completed` | ok |
| denoised_opening_min384_plus_guard_controls_w1 | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"]` | `[1, 1]` | `[1, 2, 4]` | `completed` | ok |
| denoised_opening_min384_plus_guard_controls_w2 | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"]` | `[1, 2]` | `[1, 2, 4]` | `completed` | ok |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl", "/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl"]` | `[1, 1, 2]` | `[1, 2, 4]` | `completed` | ok |
| denoised_opening_min384_plus_selected_half_if_supported | `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl", "/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl"]` | `[1, 0.5]` | `[1, 2, 4]` | `skipped` | fractional replay weights not supported cleanly by load_jsonl_replay |

## 5. Compact Decision Table

Final epoch summary for the five corrected guard rows. Cells are `selected_move_384 / selected_move_1200`.

| trace_name | capture_available-002 | capture_available-003 | capture_available-006 | capture_available-007 | capture_available-008 | gate_status |
| --- | --- | --- | --- | --- | --- | --- |
| denoised_opening_min384_self_play_only | `0 / 2` | `1 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | fail |
| denoised_opening_min384_plus_selected_w1 | `0 / 2` | `1 / 1` | `2 / 2` | `1 / 2` | `1 / 1` | fail |
| denoised_opening_min384_plus_guard_controls_w1 | `1 / 2` | `1 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | fail |
| denoised_opening_min384_plus_guard_controls_w2 | `0 / 2` | `1 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | fail |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | `2 / 2` | `2 / 2` | `2 / 2` | `2 / 2` | `1 / 1` | pass |

384-sim stability shorthand for the three rows that drove the tradeoff:

| trace_name | capture_available-002 first_correct_budget | capture_available-003 first_correct_budget | capture_available-007 first_correct_budget | notes |
| --- | --- | --- | --- | --- |
| denoised_opening_min384_self_play_only | `1200` | `1200` | `1200` | very brittle |
| denoised_opening_min384_plus_selected_w1 | `768` | `-` | `1200` | selected replay alone does not recover `003` |
| denoised_opening_min384_plus_guard_controls_w1 | `1200` | `1200` | `512` | partial guard anchor |
| denoised_opening_min384_plus_guard_controls_w2 | `512` | `768` | `768` | stronger guard anchor, still insufficient at `384` |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | `384` | `384` | `256` | only trace stable enough for the corrected gate |

## 6. Corrected Guard Results

| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | row_pass | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| denoised_opening_min384_self_play_only | 1 | capture_available-002 | 2 | 1 | 2 | 0.2057 | 0.3667 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 1 | capture_available-003 | 2 | 1 | 2 | 0.3359 | 0.4517 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 1 | capture_available-006 | 2 | 2 | 2 | 0.3776 | 0.5550 | true | false | pass_reference_selected |
| denoised_opening_min384_self_play_only | 1 | capture_available-007 | 2 | 1 | 2 | 0.3724 | 0.4642 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 1 | capture_available-008 | 1 | 1 | 1 | 0.7734 | 0.7983 | true | false | pass_reference_selected |
| denoised_opening_min384_self_play_only | 2 | capture_available-002 | 2 | 0 | 2 | 0.2161 | 0.4425 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 2 | capture_available-003 | 2 | 1 | 1 | 0.2891 | 0.3092 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 2 | capture_available-006 | 2 | 2 | 2 | 0.3984 | 0.5667 | true | false | pass_reference_selected |
| denoised_opening_min384_self_play_only | 2 | capture_available-007 | 2 | 1 | 2 | 0.2917 | 0.5358 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 2 | capture_available-008 | 1 | 1 | 1 | 0.7344 | 0.8067 | true | false | pass_reference_selected |
| denoised_opening_min384_self_play_only | 4 | capture_available-002 | 2 | 0 | 2 | 0.2344 | 0.3708 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 4 | capture_available-003 | 2 | 1 | 2 | 0.2188 | 0.4167 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 4 | capture_available-006 | 2 | 2 | 2 | 0.4349 | 0.6617 | true | false | pass_reference_selected |
| denoised_opening_min384_self_play_only | 4 | capture_available-007 | 2 | 1 | 2 | 0.2812 | 0.4642 | false | false | selected_move_not_reference |
| denoised_opening_min384_self_play_only | 4 | capture_available-008 | 1 | 1 | 1 | 0.6797 | 0.7675 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1 | 1 | capture_available-002 | 2 | 0 | 2 | 0.1875 | 0.3317 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 1 | capture_available-003 | 2 | 1 | 1 | 0.3411 | 0.3275 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 1 | capture_available-006 | 2 | 2 | 2 | 0.3542 | 0.5525 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1 | 1 | capture_available-007 | 2 | 2 | 2 | 0.4453 | 0.5392 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1 | 1 | capture_available-008 | 1 | 1 | 1 | 0.7734 | 0.8100 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1 | 2 | capture_available-002 | 2 | 0 | 2 | 0.2240 | 0.3350 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 2 | capture_available-003 | 2 | 1 | 1 | 0.2734 | 0.3517 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 2 | capture_available-006 | 2 | 2 | 2 | 0.4167 | 0.5800 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1 | 2 | capture_available-007 | 2 | 1 | 2 | 0.3255 | 0.4908 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 2 | capture_available-008 | 1 | 1 | 1 | 0.6406 | 0.7400 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-002 | 2 | 0 | 2 | 0.2812 | 0.4783 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-003 | 2 | 1 | 1 | 0.3229 | 0.4117 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-006 | 2 | 2 | 2 | 0.4115 | 0.6233 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-007 | 2 | 1 | 2 | 0.3281 | 0.5242 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-008 | 1 | 1 | 1 | 0.6484 | 0.7667 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w1 | 1 | capture_available-002 | 2 | 1 | 1 | 0.2891 | 0.3008 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 1 | capture_available-003 | 2 | 1 | 1 | 0.3594 | 0.4375 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 1 | capture_available-006 | 2 | 2 | 2 | 0.3672 | 0.4917 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w1 | 1 | capture_available-007 | 2 | 1 | 2 | 0.3542 | 0.5033 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 1 | capture_available-008 | 1 | 1 | 1 | 0.7839 | 0.8250 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w1 | 2 | capture_available-002 | 2 | 1 | 2 | 0.2995 | 0.4317 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 2 | capture_available-003 | 2 | 1 | 2 | 0.3724 | 0.4792 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 2 | capture_available-006 | 2 | 2 | 2 | 0.4375 | 0.5717 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w1 | 2 | capture_available-007 | 2 | 1 | 2 | 0.3828 | 0.5767 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 2 | capture_available-008 | 1 | 1 | 1 | 0.7396 | 0.7792 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-002 | 2 | 1 | 2 | 0.2422 | 0.4517 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-003 | 2 | 1 | 2 | 0.3229 | 0.4233 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-006 | 2 | 2 | 2 | 0.3568 | 0.5542 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-007 | 2 | 1 | 2 | 0.3880 | 0.4817 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-008 | 1 | 1 | 1 | 0.7031 | 0.7758 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w2 | 1 | capture_available-002 | 2 | 1 | 1 | 0.2031 | 0.3242 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 1 | capture_available-003 | 2 | 1 | 1 | 0.3125 | 0.4492 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 1 | capture_available-006 | 2 | 1 | 2 | 0.2969 | 0.4933 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 1 | capture_available-007 | 2 | 1 | 2 | 0.3854 | 0.5083 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 1 | capture_available-008 | 1 | 1 | 1 | 0.8333 | 0.8575 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w2 | 2 | capture_available-002 | 2 | 1 | 2 | 0.2370 | 0.4200 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 2 | capture_available-003 | 2 | 1 | 1 | 0.3594 | 0.4217 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 2 | capture_available-006 | 2 | 2 | 2 | 0.4141 | 0.6042 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w2 | 2 | capture_available-007 | 2 | 1 | 2 | 0.3490 | 0.5458 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 2 | capture_available-008 | 1 | 1 | 1 | 0.7578 | 0.8267 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-002 | 2 | 0 | 2 | 0.3047 | 0.5125 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-003 | 2 | 1 | 2 | 0.3125 | 0.4525 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-006 | 2 | 2 | 2 | 0.4792 | 0.6733 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-007 | 2 | 1 | 2 | 0.2995 | 0.5283 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-008 | 1 | 1 | 1 | 0.6328 | 0.7667 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 1 | capture_available-002 | 2 | 1 | 2 | 0.2630 | 0.3700 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 1 | capture_available-003 | 2 | 1 | 1 | 0.3203 | 0.3967 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 1 | capture_available-006 | 2 | 2 | 2 | 0.3333 | 0.5083 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 1 | capture_available-007 | 2 | 2 | 2 | 0.4297 | 0.5342 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 1 | capture_available-008 | 1 | 1 | 1 | 0.7448 | 0.7933 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 2 | capture_available-002 | 2 | 2 | 2 | 0.3776 | 0.5100 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 2 | capture_available-003 | 2 | 1 | 2 | 0.4167 | 0.4517 | false | false | selected_move_not_reference |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 2 | capture_available-006 | 2 | 2 | 2 | 0.4974 | 0.5842 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 2 | capture_available-007 | 2 | 2 | 2 | 0.4531 | 0.6200 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 2 | capture_available-008 | 1 | 1 | 1 | 0.7318 | 0.8008 | true | false | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-002 | 2 | 2 | 2 | 0.3828 | 0.5075 | true | true | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-003 | 2 | 2 | 2 | 0.4089 | 0.5700 | true | true | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-006 | 2 | 2 | 2 | 0.4531 | 0.5875 | true | true | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-007 | 2 | 2 | 2 | 0.4531 | 0.5983 | true | true | pass_reference_selected |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-008 | 1 | 1 | 1 | 0.7370 | 0.7825 | true | true | pass_reference_selected |

## 7. 384-Sim Stability Probe

| trace_name | epoch | row_id | selected_256 | selected_384 | selected_512 | selected_768 | selected_1200 | first_budget_reference_selected | diagnosis | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| denoised_opening_min384_self_play_only | 4 | capture_available-002 | 0 | 0 | 0 | 1 | 2 | 1200 | requires_1200 | ok |
| denoised_opening_min384_self_play_only | 4 | capture_available-003 | 1 | 1 | 1 | 1 | 2 | 1200 | requires_1200 | ok |
| denoised_opening_min384_self_play_only | 4 | capture_available-006 | 2 | 2 | 2 | 2 | 2 | 256 | stable_from_256 | ok |
| denoised_opening_min384_self_play_only | 4 | capture_available-007 | 1 | 1 | 1 | 1 | 2 | 1200 | requires_1200 | ok |
| denoised_opening_min384_self_play_only | 4 | capture_available-008 | 1 | 1 | 1 | 1 | 1 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-002 | 0 | 0 | 1 | 2 | 2 | 768 | fails_at_384_and_512_recovers_by_768 | ok |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-003 | 1 | 1 | 1 | 1 | 1 | - | reference_never_selected | ok |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-006 | 2 | 2 | 2 | 2 | 2 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-007 | 1 | 1 | 1 | 1 | 2 | 1200 | requires_1200 | ok |
| denoised_opening_min384_plus_selected_w1 | 4 | capture_available-008 | 1 | 1 | 1 | 1 | 1 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-002 | 1 | 1 | 1 | 1 | 2 | 1200 | requires_1200 | ok |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-003 | 1 | 1 | 1 | 1 | 2 | 1200 | requires_1200 | ok |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-006 | 2 | 2 | 2 | 2 | 2 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-007 | 1 | 1 | 2 | 2 | 2 | 512 | fails_only_at_384_recovers_by_512 | 512_or_higher_stable |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | capture_available-008 | 1 | 1 | 1 | 1 | 1 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-002 | 0 | 0 | 2 | 2 | 2 | 512 | fails_only_at_384_recovers_by_512 | 512_or_higher_stable |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-003 | 1 | 1 | 1 | 2 | 2 | 768 | fails_at_384_and_512_recovers_by_768 | ok |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-006 | 2 | 2 | 2 | 2 | 2 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-007 | 1 | 1 | 1 | 2 | 2 | 768 | fails_at_384_and_512_recovers_by_768 | ok |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | capture_available-008 | 1 | 1 | 1 | 1 | 1 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-002 | 1 | 2 | 2 | 2 | 2 | 384 | recovers_by_384 | ok |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-003 | 1 | 2 | 2 | 2 | 2 | 384 | recovers_by_384 | ok |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-006 | 2 | 2 | 2 | 2 | 2 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-007 | 2 | 2 | 2 | 2 | 2 | 256 | stable_from_256 | ok |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | capture_available-008 | 1 | 1 | 1 | 1 | 1 | 256 | stable_from_256 | ok |

## 8. Training Metrics

| trace_name | epoch | policy_loss | value_loss | total_loss | self_play_cross_entropy | selected_artifact_cross_entropy | guard_controls_cross_entropy | guard_cross_entropy | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| denoised_opening_min384_self_play_only | 1 | 1.1903 | 0.0641 | 1.2095 | 1.1903 | - | - | 1.5403 | full_dataset_snapshot |
| denoised_opening_min384_self_play_only | 2 | 1.1585 | 0.0663 | 1.1784 | 1.1585 | - | - | 1.5315 | full_dataset_snapshot |
| denoised_opening_min384_self_play_only | 4 | 1.1002 | 0.0635 | 1.1192 | 1.1002 | - | - | 1.6703 | full_dataset_snapshot |
| denoised_opening_min384_plus_selected_w1 | 1 | 1.1977 | 0.0674 | 1.2179 | 1.1974 | 1.9608 | - | 1.6218 | full_dataset_snapshot |
| denoised_opening_min384_plus_selected_w1 | 2 | 1.1362 | 0.0639 | 1.1554 | 1.1359 | 1.9743 | - | 1.7773 | full_dataset_snapshot |
| denoised_opening_min384_plus_selected_w1 | 4 | 1.0852 | 0.0622 | 1.1039 | 1.0849 | 1.8383 | - | 1.7015 | full_dataset_snapshot |
| denoised_opening_min384_plus_guard_controls_w1 | 1 | 1.1955 | 0.0649 | 1.2150 | 1.1955 | - | 1.9943 | 1.6593 | full_dataset_snapshot |
| denoised_opening_min384_plus_guard_controls_w1 | 2 | 1.1399 | 0.0633 | 1.1589 | 1.1399 | - | 1.9108 | 1.6184 | full_dataset_snapshot |
| denoised_opening_min384_plus_guard_controls_w1 | 4 | 1.0832 | 0.0617 | 1.1017 | 1.0831 | - | 1.9282 | 1.6087 | full_dataset_snapshot |
| denoised_opening_min384_plus_guard_controls_w2 | 1 | 1.1725 | 0.0640 | 1.1917 | 1.1724 | - | 1.9486 | 1.6263 | full_dataset_snapshot |
| denoised_opening_min384_plus_guard_controls_w2 | 2 | 1.1617 | 0.0647 | 1.1811 | 1.1616 | - | 1.9409 | 1.6111 | full_dataset_snapshot |
| denoised_opening_min384_plus_guard_controls_w2 | 4 | 1.1092 | 0.0632 | 1.1281 | 1.1091 | - | 2.0067 | 1.7331 | full_dataset_snapshot |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 1 | 1.1857 | 0.0645 | 1.2051 | 1.1854 | 1.9588 | 1.8682 | 1.6616 | full_dataset_snapshot |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 2 | 1.1443 | 0.0630 | 1.1632 | 1.1439 | 1.9632 | 1.7815 | 1.6061 | full_dataset_snapshot |
| denoised_opening_min384_plus_selected_w1_guard_controls_w2 | 4 | 1.0924 | 0.0621 | 1.1111 | 1.0921 | 1.9138 | 1.7071 | 1.5399 | full_dataset_snapshot |

## 9. Interpretation

- Classification: `guard_anchor_sufficient`.
- Evidence: selected w1 plus guard controls w2 passes all five rows at epoch 4; pure guard-controls traces alone still miss 002 and/or 003.
- Selected artifact summary: `guard-safe static artifact` with `21` opening rows.
- Guard controls summary: `diagnostic-only guard artifact` with `5` exact guard rows.

## 10. Exactly One Recommended Next Action

Recommendation: **run one controlled production-scale denoised opening-min-384 lane with guard controls and corrected guard kill gate before arena.**
