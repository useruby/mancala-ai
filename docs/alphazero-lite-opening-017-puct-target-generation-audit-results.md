# AlphaZero-lite Opening 017 PUCT Target Generation Audit Results

## 1. Context

- PR #38 concluded the opening-017 failure chain is `self_play_target_generation_broken` and recommended auditing PUCT target generation instead of adding more replay weight.
- This audit stayed diagnostic-only: no training, no arena, no promotion, no new corrective replay artifact, and no replay-weight sweep.
- The audit reused corrected references from `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`, the current artifact under `storage/ai/alphazero_lite/current`, and PR #36 self-play from `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_versions/exp-v3-guard-safe-opening-selected-w1-iter1/self_play.jsonl`.

## 2. Why Replay Anchoring Failed

- PR #38 already showed the corrective artifact statically matched the intended rows, so the failure was not a stale-reference or duplicate-conflict issue.
- The corrected patch still failed the corrected guard kill gate at epoch 4 for every tested replay weight, which means the underlying self-play targets remained strong enough to reintroduce the bad branch.
- That shifts the audit from replay composition to how self-play PUCT constructed the original targets for `opening_plies_1_8-017` and its descendants.

## 3. Audited Rows and Corrected References

| row_id | corrected_reference_move | pr36_self_play_count | pr36_top_target_move | pr36_corrected_reference_mass | legal_moves | notes |
| --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | 2 | 32 | 4 | 0.1673 | `[0, 1, 2, 3, 4, 5]` | pr36_rows_do_not_store_teacher_root_summary |
| capture_available-002 | 2 | 2 | 1 | 0.1849 | `[0, 1, 2, 3, 4]` | pr36_rows_do_not_store_teacher_root_summary |
| capture_available-003 | 2 | 7 | 2 | 0.3898 | `[0, 1, 2, 3, 4]` | pr36_rows_do_not_store_teacher_root_summary |
| capture_available-006 | 2 | 1 | 3 | 0.1175 | `[0, 1, 2, 3, 4]` | pr36_rows_do_not_store_teacher_root_summary |
| capture_available-007 | 2 | 5 | 1 | 0.3316 | `[0, 1, 2, 3, 4]` | pr36_rows_do_not_store_teacher_root_summary |
| capture_available-008 | 1 | 8 | 1 | 0.4863 | `[0, 1, 2, 3, 4]` | pr36_rows_do_not_store_teacher_root_summary |

## 4. PR #36 Self-Play Target Reproduction

| row_id | seed | simulations | dirichlet_epsilon | policy_target_mode | top_target_move | corrected_reference_mass_raw | corrected_reference_mass_sharpened | search_value | reproduces_pr36_bad_target | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | 41 | 192 | 0.3000 | sharpened | 1 | 0.3226 | 0.3903 | -0.0442 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| opening_plies_1_8-017 | 42 | 192 | 0.3000 | sharpened | 1 | 0.2147 | 0.1820 | -0.0415 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| opening_plies_1_8-017 | 43 | 192 | 0.3000 | sharpened | 1 | 0.1991 | 0.1712 | -0.0285 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| opening_plies_1_8-017 | 101 | 192 | 0.3000 | sharpened | 1 | 0.2621 | 0.2679 | -0.0358 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| opening_plies_1_8-017 | 202 | 192 | 0.3000 | sharpened | 1 | 0.2140 | 0.1799 | -0.0307 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| opening_plies_1_8-017 | 303 | 192 | 0.3000 | sharpened | 1 | 0.3184 | 0.3827 | -0.0420 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-002 | 41 | 192 | 0.3000 | sharpened | 2 | 0.4366 | 0.6905 | 0.0208 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-002 | 42 | 192 | 0.3000 | sharpened | 3 | 0.1362 | 0.0820 | -0.0085 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-002 | 43 | 192 | 0.3000 | sharpened | 1 | 0.1509 | 0.1029 | -0.0132 | true | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-002 | 101 | 192 | 0.3000 | sharpened | 2 | 0.4324 | 0.6757 | 0.0206 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-002 | 202 | 192 | 0.3000 | sharpened | 1 | 0.2604 | 0.3039 | 0.0071 | true | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-002 | 303 | 192 | 0.3000 | sharpened | 2 | 0.4696 | 0.7425 | 0.0230 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-003 | 41 | 192 | 0.3000 | sharpened | 2 | 0.4496 | 0.5895 | 0.0737 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-003 | 42 | 192 | 0.3000 | sharpened | 1 | 0.3159 | 0.3553 | 0.0485 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-003 | 43 | 192 | 0.3000 | sharpened | 1 | 0.3043 | 0.3052 | 0.0557 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-003 | 101 | 192 | 0.3000 | sharpened | 2 | 0.4691 | 0.6129 | 0.0701 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-003 | 202 | 192 | 0.3000 | sharpened | 1 | 0.3077 | 0.3123 | 0.0565 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-003 | 303 | 192 | 0.3000 | sharpened | 2 | 0.4946 | 0.6432 | 0.0732 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-006 | 41 | 192 | 0.3000 | sharpened | 2 | 0.5114 | 0.8124 | -0.0039 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-006 | 42 | 192 | 0.3000 | sharpened | 3 | 0.2378 | 0.2363 | -0.0293 | true | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-006 | 43 | 192 | 0.3000 | sharpened | 2 | 0.3108 | 0.4347 | -0.0294 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-006 | 101 | 192 | 0.3000 | sharpened | 2 | 0.5212 | 0.8209 | -0.0036 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-006 | 202 | 192 | 0.3000 | sharpened | 2 | 0.3708 | 0.5744 | -0.0198 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-006 | 303 | 192 | 0.3000 | sharpened | 2 | 0.5499 | 0.8542 | 0.0035 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-007 | 41 | 192 | 0.3000 | sharpened | 2 | 0.4878 | 0.6523 | 0.0690 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-007 | 42 | 192 | 0.3000 | sharpened | 1 | 0.3305 | 0.3787 | 0.0438 | true | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-007 | 43 | 192 | 0.3000 | sharpened | 1 | 0.3612 | 0.4051 | 0.0442 | true | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-007 | 101 | 192 | 0.3000 | sharpened | 2 | 0.4773 | 0.6429 | 0.0688 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-007 | 202 | 192 | 0.3000 | sharpened | 1 | 0.3553 | 0.3956 | 0.0493 | true | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-007 | 303 | 192 | 0.3000 | sharpened | 2 | 0.5181 | 0.6905 | 0.0719 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-008 | 41 | 192 | 0.3000 | sharpened | 1 | 0.6389 | 0.8904 | 0.0129 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-008 | 42 | 192 | 0.3000 | sharpened | 1 | 0.6476 | 0.9076 | 0.0087 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-008 | 43 | 192 | 0.3000 | sharpened | 1 | 0.6669 | 0.9273 | 0.0124 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-008 | 101 | 192 | 0.3000 | sharpened | 1 | 0.6384 | 0.8933 | 0.0151 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-008 | 202 | 192 | 0.3000 | sharpened | 1 | 0.6851 | 0.9469 | 0.0145 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |
| capture_available-008 | 303 | 192 | 0.3000 | sharpened | 1 | 0.6455 | 0.8864 | 0.0132 | false | tree_reuse_not_applicable_for_isolated_root, tree_reuse_not_applicable_for_isolated_root |

## 5. Target-Generation Ablation Matrix

| row_id | variant | simulations | seeds | dirichlet_epsilon | policy_target_mode | corrected_reference_top_rate | average_corrected_reference_mass | majority_top_move | target_entropy | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opening_plies_1_8-017 | baseline_self_play_target | 192 | `41,42,43,101,202,303` | 0.3000 | sharpened | 0.0000 | 0.2623 | 1 | 1.6901 | selected_target_misses_corrected_reference |
| opening_plies_1_8-017 | no_dirichlet | 192 | `41,42,43,101,202,303` | 0.0000 | sharpened | 0.0000 | 0.2284 | 1 | 1.4302 | selected_target_misses_corrected_reference |
| opening_plies_1_8-017 | unsharpened_policy_target | 192 | `41,42,43,101,202,303` | 0.3000 | default | 0.0000 | 0.2552 | 1 | 2.1752 | selected_target_misses_corrected_reference |
| opening_plies_1_8-017 | no_dirichlet_unsharpened | 192 | `41,42,43,101,202,303` | 0.0000 | default | 0.0000 | 0.2539 | 1 | 2.0885 | selected_target_misses_corrected_reference |
| opening_plies_1_8-017 | higher_sims_384 | 384 | `41,42,43,101,202,303` | 0.0000 | default | 0.0000 | 0.1778 | 1 | 2.0447 | selected_target_misses_corrected_reference |
| opening_plies_1_8-017 | higher_sims_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 0.0000 | 0.1090 | 1 | 1.8831 | selected_target_misses_corrected_reference |
| opening_plies_1_8-017 | eval_search_control | 384 | `41,42,43,101,202,303` | 0.0000 | default | 0.0000 | 0.1778 | 1 | 2.0447 | selected_target_misses_corrected_reference |
| opening_plies_1_8-017 | classic_teacher_control_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 0.1667 | 0.1900 | 4 | 2.3479 | selected_target_unstable_across_seeds |
| opening_plies_1_8-017 | classic_teacher_control_2400 | 2400 | `41,42,43,101,202,303` | 0.0000 | default | 0.5000 | 0.3159 | 2 | 1.8092 | selected_target_unstable_across_seeds |
| capture_available-002 | baseline_self_play_target | 192 | `41,42,43,101,202,303` | 0.3000 | sharpened | 0.5000 | 0.4329 | 2 | 1.7098 | selected_target_unstable_across_seeds |
| capture_available-002 | no_dirichlet | 192 | `41,42,43,101,202,303` | 0.0000 | sharpened | 1.0000 | 0.3802 | 2 | 1.9404 | selected_target_matches_corrected_reference |
| capture_available-002 | unsharpened_policy_target | 192 | `41,42,43,101,202,303` | 0.3000 | default | 0.5000 | 0.3144 | 2 | 2.1488 | selected_target_unstable_across_seeds |
| capture_available-002 | no_dirichlet_unsharpened | 192 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.2982 | 2 | 2.1906 | selected_target_matches_corrected_reference |
| capture_available-002 | higher_sims_384 | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.3514 | 2 | 2.1570 | selected_target_matches_corrected_reference |
| capture_available-002 | higher_sims_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4728 | 2 | 1.9911 | selected_target_matches_corrected_reference |
| capture_available-002 | eval_search_control | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.3846 | 2 | 2.0960 | selected_target_matches_corrected_reference |
| capture_available-002 | classic_teacher_control_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.3732 | 2 | 2.1711 | selected_target_matches_corrected_reference |
| capture_available-002 | classic_teacher_control_2400 | 2400 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4583 | 2 | 2.0128 | selected_target_matches_corrected_reference |
| capture_available-003 | baseline_self_play_target | 192 | `41,42,43,101,202,303` | 0.3000 | sharpened | 0.5000 | 0.4697 | 1 | 1.3281 | selected_target_unstable_across_seeds |
| capture_available-003 | no_dirichlet | 192 | `41,42,43,101,202,303` | 0.0000 | sharpened | 0.0000 | 0.2789 | 1 | 1.2451 | selected_target_misses_corrected_reference |
| capture_available-003 | unsharpened_policy_target | 192 | `41,42,43,101,202,303` | 0.3000 | default | 0.5000 | 0.3902 | 1 | 1.8376 | selected_target_unstable_across_seeds |
| capture_available-003 | no_dirichlet_unsharpened | 192 | `41,42,43,101,202,303` | 0.0000 | default | 0.0000 | 0.3056 | 1 | 1.8242 | selected_target_misses_corrected_reference |
| capture_available-003 | higher_sims_384 | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4121 | 2 | 1.7414 | selected_target_matches_corrected_reference |
| capture_available-003 | higher_sims_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.6239 | 2 | 1.4132 | selected_target_matches_corrected_reference |
| capture_available-003 | eval_search_control | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4552 | 2 | 1.7572 | selected_target_matches_corrected_reference |
| capture_available-003 | classic_teacher_control_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4038 | 2 | 2.1201 | selected_target_matches_corrected_reference |
| capture_available-003 | classic_teacher_control_2400 | 2400 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4524 | 2 | 1.9926 | selected_target_matches_corrected_reference |
| capture_available-006 | baseline_self_play_target | 192 | `41,42,43,101,202,303` | 0.3000 | sharpened | 0.8333 | 0.6221 | 2 | 1.4361 | selected_target_unstable_across_seeds |
| capture_available-006 | no_dirichlet | 192 | `41,42,43,101,202,303` | 0.0000 | sharpened | 1.0000 | 0.6250 | 2 | 1.6456 | selected_target_matches_corrected_reference |
| capture_available-006 | unsharpened_policy_target | 192 | `41,42,43,101,202,303` | 0.3000 | default | 0.8333 | 0.4170 | 2 | 2.0724 | selected_target_unstable_across_seeds |
| capture_available-006 | no_dirichlet_unsharpened | 192 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.3990 | 2 | 2.1473 | selected_target_matches_corrected_reference |
| capture_available-006 | higher_sims_384 | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4833 | 2 | 2.0148 | selected_target_matches_corrected_reference |
| capture_available-006 | higher_sims_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.5965 | 2 | 1.7583 | selected_target_matches_corrected_reference |
| capture_available-006 | eval_search_control | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.5196 | 2 | 1.9252 | selected_target_matches_corrected_reference |
| capture_available-006 | classic_teacher_control_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4687 | 2 | 1.9782 | selected_target_matches_corrected_reference |
| capture_available-006 | classic_teacher_control_2400 | 2400 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.5822 | 2 | 1.7214 | selected_target_matches_corrected_reference |
| capture_available-007 | baseline_self_play_target | 192 | `41,42,43,101,202,303` | 0.3000 | sharpened | 0.5000 | 0.5275 | 1 | 1.2572 | selected_target_unstable_across_seeds |
| capture_available-007 | no_dirichlet | 192 | `41,42,43,101,202,303` | 0.0000 | sharpened | 1.0000 | 0.4995 | 2 | 1.2196 | selected_target_matches_corrected_reference |
| capture_available-007 | unsharpened_policy_target | 192 | `41,42,43,101,202,303` | 0.3000 | default | 0.5000 | 0.4217 | 1 | 1.8064 | selected_target_unstable_across_seeds |
| capture_available-007 | no_dirichlet_unsharpened | 192 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.4186 | 2 | 1.7594 | selected_target_matches_corrected_reference |
| capture_available-007 | higher_sims_384 | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.5277 | 2 | 1.6172 | selected_target_matches_corrected_reference |
| capture_available-007 | higher_sims_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.6958 | 2 | 1.2931 | selected_target_matches_corrected_reference |
| capture_available-007 | eval_search_control | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.5570 | 2 | 1.6229 | selected_target_matches_corrected_reference |
| capture_available-007 | classic_teacher_control_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.3395 | 2 | 2.2103 | selected_target_matches_corrected_reference |
| capture_available-007 | classic_teacher_control_2400 | 2400 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.3419 | 2 | 2.1669 | selected_target_matches_corrected_reference |
| capture_available-008 | baseline_self_play_target | 192 | `41,42,43,101,202,303` | 0.3000 | sharpened | 1.0000 | 0.9087 | 1 | 0.5373 | selected_target_matches_corrected_reference |
| capture_available-008 | no_dirichlet | 192 | `41,42,43,101,202,303` | 0.0000 | sharpened | 1.0000 | 0.9573 | 1 | 0.3166 | selected_target_matches_corrected_reference |
| capture_available-008 | unsharpened_policy_target | 192 | `41,42,43,101,202,303` | 0.3000 | default | 1.0000 | 0.6537 | 1 | 1.5289 | selected_target_matches_corrected_reference |
| capture_available-008 | no_dirichlet_unsharpened | 192 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.7231 | 1 | 1.3651 | selected_target_matches_corrected_reference |
| capture_available-008 | higher_sims_384 | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.7415 | 1 | 1.3086 | selected_target_matches_corrected_reference |
| capture_available-008 | higher_sims_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.8438 | 1 | 0.9188 | selected_target_matches_corrected_reference |
| capture_available-008 | eval_search_control | 384 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.7249 | 1 | 1.3771 | selected_target_matches_corrected_reference |
| capture_available-008 | classic_teacher_control_1200 | 1200 | `41,42,43,101,202,303` | 0.0000 | default | 0.6667 | 0.2820 | 1 | 2.2546 | selected_target_unstable_across_seeds |
| capture_available-008 | classic_teacher_control_2400 | 2400 | `41,42,43,101,202,303` | 0.0000 | default | 1.0000 | 0.3985 | 1 | 2.1335 | selected_target_matches_corrected_reference |

## 6. Isolated-Root vs Full-Trajectory Comparison

- `opening_plies_1_8-017`: `inconsistent_or_unreproduced`. Notes: pr36_top_target_move=4, pr36_corrected_reference_mass=0.1673, baseline_majority_top_move=1, baseline_avg_corrected_reference_mass=0.2623.
- `capture_available-002`: `bad_only_in_full_trajectory`. Notes: pr36_rows_store_no_teacher_root_summary, root_visit_counts_not_stored_in_self_play_row, policy_target_built_before_temperature_sampling, canonical_state_matches_reference_exactly, pr36_top_target_move=1, pr36_corrected_reference_mass=0.1849, baseline_majority_top_move=2, baseline_avg_corrected_reference_mass=0.4329.
- `capture_available-003`: `bad_due_to_low_sims`. Notes: pr36_top_target_move=2, pr36_corrected_reference_mass=0.3898, baseline_majority_top_move=1, baseline_avg_corrected_reference_mass=0.4697.
- `capture_available-006`: `bad_only_in_full_trajectory`. Notes: pr36_rows_store_no_teacher_root_summary, root_visit_counts_not_stored_in_self_play_row, policy_target_built_before_temperature_sampling, canonical_state_matches_reference_exactly, pr36_top_target_move=3, pr36_corrected_reference_mass=0.1175, baseline_majority_top_move=2, baseline_avg_corrected_reference_mass=0.6221.
- `capture_available-007`: `isolated_reproduces_bad_target`. Notes: pr36_top_target_move=1, pr36_corrected_reference_mass=0.3316, baseline_majority_top_move=1, baseline_avg_corrected_reference_mass=0.5275.
- `capture_available-008`: `inconsistent_or_unreproduced`. Notes: pr36_control_row_or_reference_supported, pr36_top_target_move=1, pr36_corrected_reference_mass=0.4863, baseline_majority_top_move=1, baseline_avg_corrected_reference_mass=0.9087.

## 7. Diagnostic Target-Generation Fix Candidates

| fix_candidate | rows_evaluated | corrected_reference_top_rate | average_corrected_reference_mass | regressions | implementation_risk | recommended | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| no_noise_targets | 6 | 0.6667 | 0.4949 | capture_available-003: loses corrected top move; capture_available-003: corrected mass drops | medium | true | Uses the no-Dirichlet isolated-root target while leaving exploration-only noise available for action sampling. |
| unsharpened_opening_targets | 6 | 0.5000 | 0.4087 | capture_available-003: loses corrected top move | low | false | Uses default visit-policy targets for opening rows instead of squared sharpening. |
| min_sims_for_opening_targets | 6 | 0.8333 | 0.4490 | - | medium | false | Diagnostic summary uses `higher_sims_384` as the stronger opening-only minimum-simulation candidate. |
| teacher_fallback_for_corrected_forensic_hits | 6 | 1.0000 | 0.6733 | - | medium | false | Uses the corrected forensic teacher policy when self-play hits an exact audited forensic state. |

## 8. Interpretation

- Classification: `dirichlet_noise_leaks_into_targets`.
- Evidence: Removing Dirichlet noise restores the corrected reference as the majority target on 3/5 problem rows.
- PR #36 self-play row canonicalization against corrected references: `true`.
- PR #36 row-level root telemetry available for audited rows: `false`.

## 9. Exactly One Recommended Next Action

Recommendation: **modify self-play target generation so training targets are built from de-noised or separate no-noise search while exploration noise remains only for action sampling.**
