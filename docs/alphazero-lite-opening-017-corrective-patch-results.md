# AlphaZero-lite Opening 017 Corrective Patch Results

## 1. Context

- PR #37 classified the merged failure chain as `self_play_background_causes_guard_regression`.
- This follow-up builds a minimal train-only corrective artifact around `opening_plies_1_8-017` and descendant guard rows, then reruns low-cost guarded training traces from `storage/ai/alphazero_lite/current`.
- This run did not use arena, did not run any production MCTS1200 lane, did not promote, and did not overwrite `storage/ai/alphazero_lite/current`.

## 2. Why PR #37 Shifted Blame To Self-Play Targets

- `self_play_only` reproduced the PR #36-style corrected guard regression without the selected replay artifact.
- Follow-up inspection showed `opening_plies_1_8-017` often under-targeted corrected move `2`, while descendants `capture_available-002` and `capture_available-007` drifted toward move `1`.
- That made a narrow corrective target patch the next lowest-cost diagnostic.

## 3. Failure-Chain Extraction

| row_id | self_play_count | corrected_reference_move | averaged_self_play_top_move | corrected_reference_mass | top_move_mass | teacher_source | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | 32 | 2 | 4 | 0.1673 | 0.2522 | puct | predecessor_target_drift |
| capture_available-002 | 2 | 2 | 1 | 0.1849 | 0.4973 | puct | descendant_shift_to_move_1 |
| capture_available-003 | 7 | 2 | 2 | 0.3898 | 0.3898 | puct | reference_move_still_top_but_noisy |
| capture_available-007 | 5 | 2 | 1 | 0.3316 | 0.4357 | puct | descendant_shift_to_move_1 |
| capture_available-006 | 1 | 2 | 3 | 0.1175 | 0.4043 | puct | descendant_target_noise |
| capture_available-008 | 8 | 1 | 1 | 0.4863 | 0.4863 | puct | reference_move_supported |

## 4. Corrective Artifact Construction

| row_id | reason | corrected_reference_move | target_reference_mass | legal_moves | value_source | train_only | status |
| --- | --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | predecessor_target_correction | 2 | 0.8500 | `[0, 1, 2, 3, 4, 5]` | generated | true | ok |
| capture_available-002 | descendant_guard_correction | 2 | 0.8500 | `[0, 1, 2, 3, 4]` | audited_override | true | ok |
| capture_available-003 | descendant_guard_correction | 2 | 0.8500 | `[0, 1, 2, 3, 4]` | audited_override | true | ok |
| capture_available-007 | descendant_guard_correction | 2 | 0.8500 | `[0, 1, 2, 3, 4]` | audited_override | true | ok |
| capture_available-006 | guard_preservation_control | 2 | 0.8500 | `[0, 1, 2, 3, 4]` | unknown | true | ok |
| capture_available-008 | guard_preservation_control | 1 | 0.8500 | `[0, 1, 2, 3, 4]` | audited_override | true | ok |

## 5. Static Validation

- Status: `ok`.
- All required rows present: `true`.
- Duplicate conflicts: `0`.
- Stale reference conflicts: `0`.
- Notes: `ok`.

## 6. Diagnostic Patch Traces

- `self_play_only_reproduction` used data `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl"]` with replay weights `[1]` and epochs `[1, 2, 4]`.
- `self_play_plus_corrective_patch_w1` used data `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl", "/tmp/azlite_opening_017_corrective_patch/opening_017_corrective_patch_artifact.jsonl"]` with replay weights `[1, 1]` and epochs `[1, 2, 4]`.
- `self_play_plus_corrective_patch_w2` used data `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl", "/tmp/azlite_opening_017_corrective_patch/opening_017_corrective_patch_artifact.jsonl"]` with replay weights `[1, 2]` and epochs `[1, 2, 4]`.
- `self_play_plus_corrective_patch_w4` used data `["/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl", "/tmp/azlite_opening_017_corrective_patch/opening_017_corrective_patch_artifact.jsonl"]` with replay weights `[1, 4]` and epochs `[1, 2, 4]`.

## 7. Corrected Guard Kill-Gate Results

| trace_name | epoch | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | row_pass | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| self_play_only_reproduction | 1 | capture_available-002 | 2 | 1 | 1 | 0.1589 | 0.3050 | 0.0054 | -0.0104 | false | false | selected_move_not_reference |
| self_play_only_reproduction | 1 | capture_available-003 | 2 | 1 | 1 | 0.3438 | 0.3350 | -0.0443 | -0.0215 | false | false | selected_move_not_reference |
| self_play_only_reproduction | 1 | capture_available-006 | 2 | 1 | 2 | 0.2891 | 0.4508 | -0.0373 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only_reproduction | 1 | capture_available-007 | 2 | 2 | 2 | 0.3932 | 0.4283 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 1 | capture_available-008 | 1 | 1 | 1 | 0.6172 | 0.7075 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 2 | capture_available-002 | 2 | 2 | 2 | 0.3411 | 0.4492 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 2 | capture_available-003 | 2 | 1 | 2 | 0.3542 | 0.4958 | -0.0307 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only_reproduction | 2 | capture_available-006 | 2 | 2 | 2 | 0.4609 | 0.5358 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 2 | capture_available-007 | 2 | 2 | 2 | 0.4557 | 0.5383 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 2 | capture_available-008 | 1 | 1 | 1 | 0.6589 | 0.7550 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 4 | capture_available-002 | 2 | 1 | 2 | 0.2708 | 0.4033 | -0.0147 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only_reproduction | 4 | capture_available-003 | 2 | 1 | 2 | 0.3464 | 0.4233 | -0.0417 | 0.0000 | false | false | selected_move_not_reference |
| self_play_only_reproduction | 4 | capture_available-006 | 2 | 2 | 2 | 0.3542 | 0.4583 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 4 | capture_available-007 | 2 | 2 | 2 | 0.3854 | 0.4308 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_only_reproduction | 4 | capture_available-008 | 1 | 1 | 1 | 0.6406 | 0.7300 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-002 | 2 | 1 | 2 | 0.2396 | 0.3708 | -0.0018 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-003 | 2 | 1 | 2 | 0.2995 | 0.4333 | -0.0316 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-006 | 2 | 2 | 2 | 0.2995 | 0.4467 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-007 | 2 | 2 | 2 | 0.3802 | 0.4550 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-008 | 1 | 1 | 1 | 0.6042 | 0.7050 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-002 | 2 | 0 | 2 | 0.2135 | 0.3767 | -0.0303 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-003 | 2 | 1 | 1 | 0.3255 | 0.3858 | -0.0452 | -0.0296 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-006 | 2 | 2 | 2 | 0.3307 | 0.5125 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-007 | 2 | 1 | 2 | 0.3255 | 0.4158 | -0.0344 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-008 | 1 | 1 | 1 | 0.6016 | 0.7133 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-002 | 2 | 1 | 2 | 0.2630 | 0.3992 | -0.0143 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-003 | 2 | 1 | 2 | 0.3177 | 0.4100 | -0.0398 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-006 | 2 | 2 | 2 | 0.3672 | 0.5050 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-007 | 2 | 2 | 2 | 0.4036 | 0.4625 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-008 | 1 | 1 | 1 | 0.6641 | 0.7417 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-002 | 2 | 0 | 2 | 0.2005 | 0.3325 | 0.0067 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-003 | 2 | 1 | 1 | 0.3464 | 0.2750 | -0.0401 | -0.0078 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-006 | 2 | 2 | 2 | 0.3047 | 0.4450 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-007 | 2 | 2 | 2 | 0.4062 | 0.4050 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-008 | 1 | 1 | 1 | 0.6042 | 0.6817 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-002 | 2 | 1 | 2 | 0.2760 | 0.4092 | 0.0003 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-003 | 2 | 1 | 1 | 0.3307 | 0.3325 | -0.0304 | -0.0137 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-006 | 2 | 2 | 2 | 0.3229 | 0.5092 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-007 | 2 | 2 | 2 | 0.4010 | 0.5367 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-008 | 1 | 1 | 1 | 0.5911 | 0.6958 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-002 | 2 | 1 | 2 | 0.3021 | 0.4350 | -0.0135 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-003 | 2 | 1 | 2 | 0.3333 | 0.4200 | -0.0385 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-006 | 2 | 2 | 2 | 0.3698 | 0.5192 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-007 | 2 | 1 | 2 | 0.3542 | 0.4958 | -0.0265 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-008 | 1 | 1 | 1 | 0.6120 | 0.7342 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-002 | 2 | 1 | 2 | 0.2188 | 0.4233 | -0.0110 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-003 | 2 | 1 | 1 | 0.3802 | 0.3592 | -0.0585 | -0.0279 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-006 | 2 | 2 | 2 | 0.3281 | 0.4983 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-007 | 2 | 2 | 2 | 0.3880 | 0.4350 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-008 | 1 | 1 | 1 | 0.6823 | 0.7575 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-002 | 2 | 1 | 2 | 0.2526 | 0.4383 | -0.0058 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-003 | 2 | 1 | 1 | 0.3203 | 0.3958 | -0.0365 | -0.0300 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-006 | 2 | 2 | 2 | 0.3385 | 0.4767 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-007 | 2 | 2 | 2 | 0.3932 | 0.4983 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-008 | 1 | 1 | 1 | 0.6224 | 0.7250 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-002 | 2 | 1 | 2 | 0.2630 | 0.4050 | -0.0097 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-003 | 2 | 1 | 2 | 0.3151 | 0.4133 | -0.0338 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-006 | 2 | 2 | 2 | 0.3464 | 0.4975 | 0.0000 | 0.0000 | true | false | pass_reference_selected |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-007 | 2 | 1 | 2 | 0.3542 | 0.5000 | -0.0334 | 0.0000 | false | false | selected_move_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-008 | 1 | 1 | 1 | 0.6068 | 0.7058 | 0.0000 | 0.0000 | true | false | pass_reference_selected |

## 8. Policy-Target Movement

| trace_name | epoch | row_id | corrected_reference_move | policy_top_move | reference_policy_probability | reference_policy_rank | old_self_play_top_move | old_top_policy_probability | reference_minus_old_top_margin | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| self_play_only_reproduction | 1 | opening_plies_1_8-017 | 2 | 1 | 0.1836 | 2 | 4 | 0.0883 | 0.0953 | policy_top_not_reference |
| self_play_only_reproduction | 1 | capture_available-002 | 2 | 0 | 0.1045 | 3 | 1 | 0.3058 | -0.2013 | policy_top_not_reference |
| self_play_only_reproduction | 1 | capture_available-003 | 2 | 1 | 0.1021 | 4 | 2 | 0.1021 | 0.0000 | policy_top_not_reference |
| self_play_only_reproduction | 1 | capture_available-007 | 2 | 1 | 0.1534 | 4 | 1 | 0.4081 | -0.2547 | policy_top_not_reference |
| self_play_only_reproduction | 1 | capture_available-006 | 2 | 1 | 0.1186 | 4 | 3 | 0.0611 | 0.0575 | policy_top_not_reference |
| self_play_only_reproduction | 1 | capture_available-008 | 1 | 1 | 0.4467 | 1 | 1 | 0.4467 | 0.0000 | reference_policy_top |
| self_play_only_reproduction | 2 | opening_plies_1_8-017 | 2 | 2 | 0.2706 | 1 | 4 | 0.1119 | 0.1587 | reference_policy_top |
| self_play_only_reproduction | 2 | capture_available-002 | 2 | 0 | 0.1434 | 3 | 1 | 0.3470 | -0.2036 | policy_top_not_reference |
| self_play_only_reproduction | 2 | capture_available-003 | 2 | 1 | 0.1410 | 4 | 2 | 0.1410 | 0.0000 | policy_top_not_reference |
| self_play_only_reproduction | 2 | capture_available-007 | 2 | 1 | 0.1899 | 2 | 1 | 0.4033 | -0.2134 | policy_top_not_reference |
| self_play_only_reproduction | 2 | capture_available-006 | 2 | 1 | 0.1610 | 3 | 3 | 0.0820 | 0.0790 | policy_top_not_reference |
| self_play_only_reproduction | 2 | capture_available-008 | 1 | 1 | 0.4551 | 1 | 1 | 0.4551 | 0.0000 | reference_policy_top |
| self_play_only_reproduction | 4 | opening_plies_1_8-017 | 2 | 1 | 0.2718 | 2 | 4 | 0.0951 | 0.1767 | policy_top_not_reference |
| self_play_only_reproduction | 4 | capture_available-002 | 2 | 0 | 0.1486 | 3 | 1 | 0.3572 | -0.2086 | policy_top_not_reference |
| self_play_only_reproduction | 4 | capture_available-003 | 2 | 1 | 0.1199 | 4 | 2 | 0.1199 | 0.0000 | policy_top_not_reference |
| self_play_only_reproduction | 4 | capture_available-007 | 2 | 1 | 0.1782 | 2 | 1 | 0.4597 | -0.2815 | policy_top_not_reference |
| self_play_only_reproduction | 4 | capture_available-006 | 2 | 1 | 0.1652 | 3 | 3 | 0.0796 | 0.0856 | policy_top_not_reference |
| self_play_only_reproduction | 4 | capture_available-008 | 1 | 1 | 0.4813 | 1 | 1 | 0.4813 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w1 | 1 | opening_plies_1_8-017 | 2 | 1 | 0.2088 | 2 | 4 | 0.1103 | 0.0985 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-002 | 2 | 0 | 0.1219 | 3 | 1 | 0.2302 | -0.1083 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-003 | 2 | 1 | 0.1020 | 4 | 2 | 0.1020 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-007 | 2 | 1 | 0.1510 | 4 | 1 | 0.3956 | -0.2446 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-006 | 2 | 0 | 0.1271 | 4 | 3 | 0.1012 | 0.0259 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 1 | capture_available-008 | 1 | 1 | 0.3959 | 1 | 1 | 0.3959 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w1 | 2 | opening_plies_1_8-017 | 2 | 1 | 0.2347 | 2 | 4 | 0.0855 | 0.1492 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-002 | 2 | 0 | 0.1134 | 3 | 1 | 0.2435 | -0.1301 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-003 | 2 | 1 | 0.0909 | 4 | 2 | 0.0909 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-007 | 2 | 1 | 0.1363 | 4 | 1 | 0.4492 | -0.3129 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-006 | 2 | 0 | 0.1246 | 3 | 3 | 0.0837 | 0.0409 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 2 | capture_available-008 | 1 | 1 | 0.4395 | 1 | 1 | 0.4395 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w1 | 4 | opening_plies_1_8-017 | 2 | 1 | 0.2254 | 2 | 4 | 0.0782 | 0.1472 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-002 | 2 | 1 | 0.1402 | 3 | 1 | 0.3747 | -0.2345 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-003 | 2 | 1 | 0.1029 | 4 | 2 | 0.1029 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-007 | 2 | 1 | 0.1569 | 2 | 1 | 0.5076 | -0.3507 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-006 | 2 | 1 | 0.1546 | 3 | 3 | 0.0707 | 0.0839 | policy_top_not_reference |
| self_play_plus_corrective_patch_w1 | 4 | capture_available-008 | 1 | 1 | 0.4926 | 1 | 1 | 0.4926 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w2 | 1 | opening_plies_1_8-017 | 2 | 2 | 0.3134 | 1 | 4 | 0.1187 | 0.1947 | reference_policy_top |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-002 | 2 | 0 | 0.1929 | 3 | 1 | 0.2666 | -0.0737 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-003 | 2 | 1 | 0.1416 | 2 | 2 | 0.1416 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-007 | 2 | 1 | 0.2129 | 2 | 1 | 0.4293 | -0.2164 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-006 | 2 | 1 | 0.1988 | 3 | 3 | 0.1341 | 0.0647 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 1 | capture_available-008 | 1 | 1 | 0.4974 | 1 | 1 | 0.4974 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w2 | 2 | opening_plies_1_8-017 | 2 | 2 | 0.3151 | 1 | 4 | 0.1069 | 0.2082 | reference_policy_top |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-002 | 2 | 0 | 0.2080 | 3 | 1 | 0.2582 | -0.0502 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-003 | 2 | 1 | 0.1740 | 2 | 2 | 0.1740 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-007 | 2 | 1 | 0.2602 | 2 | 1 | 0.3735 | -0.1133 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-006 | 2 | 0 | 0.2221 | 3 | 3 | 0.1066 | 0.1155 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 2 | capture_available-008 | 1 | 1 | 0.4494 | 1 | 1 | 0.4494 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w2 | 4 | opening_plies_1_8-017 | 2 | 1 | 0.2390 | 2 | 4 | 0.1100 | 0.1290 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-002 | 2 | 0 | 0.1694 | 3 | 1 | 0.3232 | -0.1538 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-003 | 2 | 1 | 0.1269 | 4 | 2 | 0.1269 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-007 | 2 | 1 | 0.1849 | 2 | 1 | 0.4248 | -0.2399 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-006 | 2 | 1 | 0.1811 | 3 | 3 | 0.0778 | 0.1033 | policy_top_not_reference |
| self_play_plus_corrective_patch_w2 | 4 | capture_available-008 | 1 | 1 | 0.4675 | 1 | 1 | 0.4675 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w4 | 1 | opening_plies_1_8-017 | 2 | 1 | 0.1702 | 3 | 4 | 0.0978 | 0.0724 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-002 | 2 | 1 | 0.1164 | 4 | 1 | 0.3886 | -0.2722 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-003 | 2 | 1 | 0.0942 | 4 | 2 | 0.0942 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-007 | 2 | 1 | 0.1413 | 3 | 1 | 0.5210 | -0.3797 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-006 | 2 | 1 | 0.1156 | 4 | 3 | 0.0797 | 0.0359 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 1 | capture_available-008 | 1 | 1 | 0.5941 | 1 | 1 | 0.5941 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w4 | 2 | opening_plies_1_8-017 | 2 | 1 | 0.2820 | 2 | 4 | 0.0913 | 0.1907 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-002 | 2 | 0 | 0.1626 | 3 | 1 | 0.3274 | -0.1648 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-003 | 2 | 1 | 0.1331 | 2 | 2 | 0.1331 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-007 | 2 | 1 | 0.2173 | 2 | 1 | 0.4510 | -0.2337 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-006 | 2 | 1 | 0.1521 | 3 | 3 | 0.0761 | 0.0760 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 2 | capture_available-008 | 1 | 1 | 0.4926 | 1 | 1 | 0.4926 | 0.0000 | reference_policy_top |
| self_play_plus_corrective_patch_w4 | 4 | opening_plies_1_8-017 | 2 | 1 | 0.2266 | 2 | 4 | 0.0966 | 0.1300 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-002 | 2 | 0 | 0.1484 | 3 | 1 | 0.3149 | -0.1665 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-003 | 2 | 1 | 0.1311 | 3 | 2 | 0.1311 | 0.0000 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-007 | 2 | 1 | 0.1973 | 2 | 1 | 0.4101 | -0.2128 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-006 | 2 | 1 | 0.1529 | 3 | 3 | 0.0825 | 0.0704 | policy_top_not_reference |
| self_play_plus_corrective_patch_w4 | 4 | capture_available-008 | 1 | 1 | 0.4585 | 1 | 1 | 0.4585 | 0.0000 | reference_policy_top |

## 9. Training Metrics

| trace_name | epoch | policy_loss | value_loss | total_loss | self_play_cross_entropy | corrective_patch_cross_entropy | guard_cross_entropy | val_loss | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| self_play_only_reproduction | 1 | 1.2781 | 0.0646 | 1.2975 | 1.2781 | - | - | 1.3087 | full_dataset_snapshot |
| self_play_only_reproduction | 2 | 1.2526 | 0.0652 | 1.2722 | 1.2526 | - | - | 1.2966 | full_dataset_snapshot |
| self_play_only_reproduction | 4 | 1.2048 | 0.0622 | 1.2235 | 1.2048 | - | - | 1.2513 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w1 | 1 | 1.2968 | 0.0661 | 1.3166 | 1.2968 | 1.9044 | 1.9599 | 1.3317 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w1 | 2 | 1.2607 | 0.0651 | 1.2802 | 1.2606 | 1.8776 | 1.9466 | 1.3030 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w1 | 4 | 1.2049 | 0.0621 | 1.2236 | 1.2049 | 1.7803 | 1.8210 | 1.2526 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w2 | 1 | 1.2954 | 0.0675 | 1.3157 | 1.2954 | 1.5509 | 1.6023 | 1.3301 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w2 | 2 | 1.2463 | 0.0637 | 1.2654 | 1.2463 | 1.4886 | 1.5250 | 1.2899 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w2 | 4 | 1.2074 | 0.0620 | 1.2260 | 1.2074 | 1.6855 | 1.7173 | 1.2587 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w4 | 1 | 1.2932 | 0.0654 | 1.3128 | 1.2931 | 1.9065 | 1.9273 | 1.3323 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w4 | 2 | 1.2642 | 0.0637 | 1.2833 | 1.2642 | 1.6496 | 1.7014 | 1.3082 | full_dataset_snapshot |
| self_play_plus_corrective_patch_w4 | 4 | 1.2038 | 0.0623 | 1.2225 | 1.2038 | 1.7093 | 1.7372 | 1.2538 | full_dataset_snapshot |

## 10. Interpretation

- Classification: `self_play_target_generation_broken`.
- Evidence: No corrective replay weight passed the corrected guard kill gate at epoch 4.
- Trace D executed: `true`.

## 11. Exactly One Recommended Next Action

Recommendation: **audit PUCT target generation for opening_plies_1_8-017 and descendants, not replay anchoring.**
