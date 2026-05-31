# AlphaZero-lite Corrected-Reference Guarded Search-Prior Control Results

## 1. Context

- PR #32 established that corrected-reference replay cleaned up labels but did not improve arena strength.
- The active blocker remained the corrected `capture_available-002/003` gate, especially on replay/opening replay branches.
- Default corrected reference artifact: `/home/alex/Mancala/ai/ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.

## 2. Why replay is paused

- Overall classification: `replay_induced_guard_regression`.
- Classification rationale: Current already satisfies the corrected 002/003 gate at baseline, corrected hard-state replay candidates do not improve that gate, and the corrected opening replay branch regresses it.
- No training, broad arena, promotion, or new replay dataset construction was run in this control branch.

## 3. Corrected-reference baseline behavior

| artifact | row_id | budget | search_setting | corrected_reference_move | selected_move | selected_is_corrected_reference | reference_policy_probability | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | capture_available-002 | 64 | baseline_eval_search | 2 | 0 | false | 0.1768 | 0.2344 | 0.3281 | -0.0570 | false | u_favors_selected |
| current | capture_available-002 | 64 | no_subtree_reuse | 2 | 0 | false | 0.1768 | 0.2344 | 0.3281 | -0.0570 | false | u_favors_selected |
| current | capture_available-002 | 64 | parent_q_fpu | 2 | 0 | false | 0.1768 | 0.2344 | 0.3281 | -0.0570 | false | u_favors_selected |
| current | capture_available-002 | 64 | normalized_values | 2 | 2 | true | 0.1768 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 64 | deterministic_root_policy | 2 | 0 | false | 0.1768 | 0.2344 | 0.3281 | -0.0570 | false | u_favors_selected |
| current | capture_available-002 | 64 | low_cpuct | 2 | 2 | true | 0.1768 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 64 | high_cpuct | 2 | 0 | false | 0.1768 | 0.2188 | 0.3438 | -0.0703 | false | u_favors_selected |
| current | capture_available-002 | 64 | no_tactical_root_bias | 2 | 0 | false | 0.1299 | 0.2031 | 0.3281 | -0.0749 | false | u_favors_selected |
| current | capture_available-002 | 64 | full_search_control | 2 | 2 | true | 0.1768 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 128 | baseline_eval_search | 2 | 1 | false | 0.1768 | 0.2266 | 0.2969 | -0.0376 | false | u_favors_selected |
| current | capture_available-002 | 128 | no_subtree_reuse | 2 | 1 | false | 0.1768 | 0.2266 | 0.2969 | -0.0376 | false | u_favors_selected |
| current | capture_available-002 | 128 | parent_q_fpu | 2 | 2 | true | 0.1768 | 0.2734 | 0.2734 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 128 | normalized_values | 2 | 2 | true | 0.1768 | 0.7109 | 0.7109 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 128 | deterministic_root_policy | 2 | 1 | false | 0.1768 | 0.2266 | 0.2969 | -0.0376 | false | u_favors_selected |
| current | capture_available-002 | 128 | low_cpuct | 2 | 2 | true | 0.1768 | 0.3906 | 0.3906 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 128 | high_cpuct | 2 | 1 | false | 0.1768 | 0.1953 | 0.2969 | -0.0247 | false | u_favors_selected |
| current | capture_available-002 | 128 | no_tactical_root_bias | 2 | 1 | false | 0.1299 | 0.1641 | 0.2969 | -0.0262 | false | u_favors_selected |
| current | capture_available-002 | 128 | full_search_control | 2 | 2 | true | 0.1768 | 0.7109 | 0.7109 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | baseline_eval_search | 2 | 2 | true | 0.1768 | 0.4036 | 0.4036 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1768 | 0.4036 | 0.4036 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | parent_q_fpu | 2 | 2 | true | 0.1768 | 0.4323 | 0.4323 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | normalized_values | 2 | 2 | true | 0.1768 | 0.8516 | 0.8516 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1768 | 0.4036 | 0.4036 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | low_cpuct | 2 | 2 | true | 0.1768 | 0.6302 | 0.6302 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | high_cpuct | 2 | 2 | true | 0.1768 | 0.3255 | 0.3255 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1299 | 0.3672 | 0.3672 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 384 | full_search_control | 2 | 2 | true | 0.1768 | 0.8542 | 0.8542 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1768 | 0.5308 | 0.5308 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1768 | 0.5308 | 0.5308 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1768 | 0.5542 | 0.5542 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | normalized_values | 2 | 2 | true | 0.1768 | 0.9400 | 0.9400 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1768 | 0.5308 | 0.5308 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | low_cpuct | 2 | 2 | true | 0.1768 | 0.7883 | 0.7883 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | high_cpuct | 2 | 2 | true | 0.1768 | 0.4317 | 0.4317 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1299 | 0.5025 | 0.5025 | 0.0000 | true | pass_reference_selected |
| current | capture_available-002 | 1200 | full_search_control | 2 | 2 | true | 0.1768 | 0.9308 | 0.9308 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 64 | baseline_eval_search | 2 | 2 | true | 0.1577 | 0.4219 | 0.4219 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 64 | no_subtree_reuse | 2 | 2 | true | 0.1577 | 0.4219 | 0.4219 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 64 | parent_q_fpu | 2 | 1 | false | 0.1577 | 0.3750 | 0.4531 | -0.1145 | false | u_favors_selected |
| current | capture_available-003 | 64 | normalized_values | 2 | 1 | false | 0.1577 | 0.4688 | 0.5156 | -0.0789 | false | u_favors_selected |
| current | capture_available-003 | 64 | deterministic_root_policy | 2 | 2 | true | 0.1577 | 0.4219 | 0.4219 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 64 | low_cpuct | 2 | 2 | true | 0.1577 | 0.6250 | 0.6250 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 64 | high_cpuct | 2 | 1 | false | 0.1577 | 0.2812 | 0.5156 | -0.1134 | false | u_favors_selected |
| current | capture_available-003 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1050 | 0.2500 | 0.5469 | -0.1452 | false | u_favors_selected |
| current | capture_available-003 | 64 | full_search_control | 2 | 2 | true | 0.1577 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 128 | baseline_eval_search | 2 | 1 | false | 0.1577 | 0.3672 | 0.4609 | -0.0875 | false | u_favors_selected |
| current | capture_available-003 | 128 | no_subtree_reuse | 2 | 1 | false | 0.1577 | 0.3672 | 0.4609 | -0.0875 | false | u_favors_selected |
| current | capture_available-003 | 128 | parent_q_fpu | 2 | 1 | false | 0.1577 | 0.3750 | 0.4453 | -0.0883 | false | u_favors_selected |
| current | capture_available-003 | 128 | normalized_values | 2 | 2 | true | 0.1577 | 0.7031 | 0.7031 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 128 | deterministic_root_policy | 2 | 1 | false | 0.1577 | 0.3672 | 0.4609 | -0.0875 | false | u_favors_selected |
| current | capture_available-003 | 128 | low_cpuct | 2 | 2 | true | 0.1577 | 0.5078 | 0.5078 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 128 | high_cpuct | 2 | 1 | false | 0.1577 | 0.3047 | 0.4844 | -0.0937 | false | u_favors_selected |
| current | capture_available-003 | 128 | no_tactical_root_bias | 2 | 1 | false | 0.1050 | 0.3203 | 0.5000 | -0.0918 | false | u_favors_selected |
| current | capture_available-003 | 128 | full_search_control | 2 | 2 | true | 0.1577 | 0.7891 | 0.7891 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | baseline_eval_search | 2 | 2 | true | 0.1577 | 0.4740 | 0.4740 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1577 | 0.4740 | 0.4740 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | parent_q_fpu | 2 | 2 | true | 0.1577 | 0.4896 | 0.4896 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | normalized_values | 2 | 2 | true | 0.1577 | 0.8229 | 0.8229 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1577 | 0.4740 | 0.4740 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | low_cpuct | 2 | 2 | true | 0.1577 | 0.6536 | 0.6536 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | high_cpuct | 2 | 1 | false | 0.1577 | 0.3776 | 0.4401 | -0.0758 | false | u_favors_selected |
| current | capture_available-003 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1050 | 0.4245 | 0.4245 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 384 | full_search_control | 2 | 2 | true | 0.1577 | 0.7865 | 0.7865 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1577 | 0.6742 | 0.6742 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1577 | 0.6742 | 0.6742 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1577 | 0.7050 | 0.7050 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | normalized_values | 2 | 2 | true | 0.1577 | 0.9350 | 0.9350 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1577 | 0.6742 | 0.6742 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | low_cpuct | 2 | 2 | true | 0.1577 | 0.7867 | 0.7867 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | high_cpuct | 2 | 2 | true | 0.1577 | 0.5350 | 0.5350 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1050 | 0.6542 | 0.6542 | 0.0000 | true | pass_reference_selected |
| current | capture_available-003 | 1200 | full_search_control | 2 | 2 | true | 0.1577 | 0.9300 | 0.9300 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 64 | baseline_eval_search | 2 | 0 | false | 0.1614 | 0.1406 | 0.2500 | -0.0207 | false | u_favors_selected |
| current | capture_available-006 | 64 | no_subtree_reuse | 2 | 0 | false | 0.1614 | 0.1406 | 0.2500 | -0.0207 | false | u_favors_selected |
| current | capture_available-006 | 64 | parent_q_fpu | 2 | 4 | false | 0.1614 | 0.1406 | 0.2344 | 0.0924 | false | q_favors_selected |
| current | capture_available-006 | 64 | normalized_values | 2 | 2 | true | 0.1614 | 0.7031 | 0.7031 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 64 | deterministic_root_policy | 2 | 0 | false | 0.1614 | 0.1406 | 0.2500 | -0.0207 | false | u_favors_selected |
| current | capture_available-006 | 64 | low_cpuct | 2 | 4 | false | 0.1614 | 0.1406 | 0.2656 | 0.0675 | false | q_favors_selected |
| current | capture_available-006 | 64 | high_cpuct | 2 | 1 | false | 0.1614 | 0.1406 | 0.2812 | -0.0645 | false | u_favors_selected |
| current | capture_available-006 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1098 | 0.1094 | 0.2812 | -0.0782 | false | u_favors_selected |
| current | capture_available-006 | 64 | full_search_control | 2 | 2 | true | 0.1614 | 0.7031 | 0.7031 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | baseline_eval_search | 2 | 2 | true | 0.1614 | 0.3359 | 0.3359 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1614 | 0.3359 | 0.3359 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | parent_q_fpu | 2 | 2 | true | 0.1614 | 0.3359 | 0.3359 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | normalized_values | 2 | 2 | true | 0.1614 | 0.8516 | 0.8516 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1614 | 0.3359 | 0.3359 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | low_cpuct | 2 | 2 | true | 0.1614 | 0.5078 | 0.5078 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | high_cpuct | 2 | 2 | true | 0.1614 | 0.2891 | 0.2891 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1098 | 0.2969 | 0.2969 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 128 | full_search_control | 2 | 2 | true | 0.1614 | 0.8516 | 0.8516 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | baseline_eval_search | 2 | 2 | true | 0.1614 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1614 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | parent_q_fpu | 2 | 2 | true | 0.1614 | 0.5365 | 0.5365 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | normalized_values | 2 | 2 | true | 0.1614 | 0.8958 | 0.8958 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1614 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | low_cpuct | 2 | 2 | true | 0.1614 | 0.6667 | 0.6667 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | high_cpuct | 2 | 2 | true | 0.1614 | 0.4010 | 0.4010 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1098 | 0.5156 | 0.5156 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 384 | full_search_control | 2 | 2 | true | 0.1614 | 0.8542 | 0.8542 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1614 | 0.6600 | 0.6600 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1614 | 0.6600 | 0.6600 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1614 | 0.6667 | 0.6667 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | normalized_values | 2 | 2 | true | 0.1614 | 0.9483 | 0.9483 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1614 | 0.6600 | 0.6600 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | low_cpuct | 2 | 2 | true | 0.1614 | 0.8150 | 0.8150 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | high_cpuct | 2 | 2 | true | 0.1614 | 0.5692 | 0.5692 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1098 | 0.6375 | 0.6375 | 0.0000 | true | pass_reference_selected |
| current | capture_available-006 | 1200 | full_search_control | 2 | 2 | true | 0.1614 | 0.9300 | 0.9300 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 64 | baseline_eval_search | 2 | 2 | true | 0.2762 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 64 | no_subtree_reuse | 2 | 2 | true | 0.2762 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 64 | parent_q_fpu | 2 | 1 | false | 0.2762 | 0.3594 | 0.5000 | -0.0186 | false | u_favors_selected |
| current | capture_available-007 | 64 | normalized_values | 2 | 2 | true | 0.2762 | 0.8125 | 0.8125 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 64 | deterministic_root_policy | 2 | 2 | true | 0.2762 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 64 | low_cpuct | 2 | 2 | true | 0.2762 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 64 | high_cpuct | 2 | 1 | false | 0.2762 | 0.3750 | 0.4531 | -0.0496 | false | u_favors_selected |
| current | capture_available-007 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.2590 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 64 | full_search_control | 2 | 2 | true | 0.2762 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 128 | baseline_eval_search | 2 | 1 | false | 0.2762 | 0.3984 | 0.4297 | -0.0281 | false | u_favors_selected |
| current | capture_available-007 | 128 | no_subtree_reuse | 2 | 1 | false | 0.2762 | 0.3984 | 0.4297 | -0.0281 | false | u_favors_selected |
| current | capture_available-007 | 128 | parent_q_fpu | 2 | 1 | false | 0.2762 | 0.3672 | 0.4609 | -0.0232 | false | u_favors_selected |
| current | capture_available-007 | 128 | normalized_values | 2 | 2 | true | 0.2762 | 0.7266 | 0.7266 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 128 | deterministic_root_policy | 2 | 1 | false | 0.2762 | 0.3984 | 0.4297 | -0.0281 | false | u_favors_selected |
| current | capture_available-007 | 128 | low_cpuct | 2 | 2 | true | 0.2762 | 0.5234 | 0.5234 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 128 | high_cpuct | 2 | 1 | false | 0.2762 | 0.3438 | 0.4453 | -0.0226 | false | u_favors_selected |
| current | capture_available-007 | 128 | no_tactical_root_bias | 2 | 1 | false | 0.2590 | 0.3906 | 0.4531 | -0.0364 | false | u_favors_selected |
| current | capture_available-007 | 128 | full_search_control | 2 | 2 | true | 0.2762 | 0.7031 | 0.7031 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | baseline_eval_search | 2 | 2 | true | 0.2762 | 0.5859 | 0.5859 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2762 | 0.5859 | 0.5859 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | parent_q_fpu | 2 | 2 | true | 0.2762 | 0.5729 | 0.5729 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | normalized_values | 2 | 2 | true | 0.2762 | 0.8620 | 0.8620 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2762 | 0.5859 | 0.5859 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | low_cpuct | 2 | 2 | true | 0.2762 | 0.6745 | 0.6745 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | high_cpuct | 2 | 2 | true | 0.2762 | 0.5182 | 0.5182 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.2590 | 0.5521 | 0.5521 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 384 | full_search_control | 2 | 2 | true | 0.2762 | 0.8568 | 0.8568 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2762 | 0.7400 | 0.7400 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2762 | 0.7400 | 0.7400 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2762 | 0.7475 | 0.7475 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | normalized_values | 2 | 2 | true | 0.2762 | 0.9508 | 0.9508 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2762 | 0.7400 | 0.7400 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | low_cpuct | 2 | 2 | true | 0.2762 | 0.8383 | 0.8383 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | high_cpuct | 2 | 2 | true | 0.2762 | 0.6158 | 0.6158 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.2590 | 0.7308 | 0.7308 | 0.0000 | true | pass_reference_selected |
| current | capture_available-007 | 1200 | full_search_control | 2 | 2 | true | 0.2762 | 0.9475 | 0.9475 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | baseline_eval_search | 1 | 1 | true | 0.6295 | 0.7812 | 0.7812 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | no_subtree_reuse | 1 | 1 | true | 0.6295 | 0.7812 | 0.7812 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | parent_q_fpu | 1 | 1 | true | 0.6295 | 0.7969 | 0.7969 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | normalized_values | 1 | 1 | true | 0.6295 | 0.8750 | 0.8750 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | deterministic_root_policy | 1 | 1 | true | 0.6295 | 0.7812 | 0.7812 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | low_cpuct | 1 | 1 | true | 0.6295 | 0.8125 | 0.8125 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | high_cpuct | 1 | 1 | true | 0.6295 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | no_tactical_root_bias | 1 | 1 | true | 0.6554 | 0.7656 | 0.7656 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 64 | full_search_control | 1 | 1 | true | 0.6295 | 0.8438 | 0.8438 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | baseline_eval_search | 1 | 1 | true | 0.6295 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | no_subtree_reuse | 1 | 1 | true | 0.6295 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | parent_q_fpu | 1 | 1 | true | 0.6295 | 0.7500 | 0.7500 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | normalized_values | 1 | 1 | true | 0.6295 | 0.8672 | 0.8672 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | deterministic_root_policy | 1 | 1 | true | 0.6295 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | low_cpuct | 1 | 1 | true | 0.6295 | 0.8047 | 0.8047 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | high_cpuct | 1 | 1 | true | 0.6295 | 0.7266 | 0.7266 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | no_tactical_root_bias | 1 | 1 | true | 0.6554 | 0.7891 | 0.7891 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 128 | full_search_control | 1 | 1 | true | 0.6295 | 0.8672 | 0.8672 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | baseline_eval_search | 1 | 1 | true | 0.6295 | 0.7682 | 0.7682 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | no_subtree_reuse | 1 | 1 | true | 0.6295 | 0.7682 | 0.7682 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | parent_q_fpu | 1 | 1 | true | 0.6295 | 0.7839 | 0.7839 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | normalized_values | 1 | 1 | true | 0.6295 | 0.8776 | 0.8776 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | deterministic_root_policy | 1 | 1 | true | 0.6295 | 0.7682 | 0.7682 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | low_cpuct | 1 | 1 | true | 0.6295 | 0.8151 | 0.8151 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | high_cpuct | 1 | 1 | true | 0.6295 | 0.7526 | 0.7526 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | no_tactical_root_bias | 1 | 1 | true | 0.6554 | 0.7839 | 0.7839 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 384 | full_search_control | 1 | 1 | true | 0.6295 | 0.8698 | 0.8698 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | baseline_eval_search | 1 | 1 | true | 0.6295 | 0.8733 | 0.8733 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | no_subtree_reuse | 1 | 1 | true | 0.6295 | 0.8733 | 0.8733 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | parent_q_fpu | 1 | 1 | true | 0.6295 | 0.8750 | 0.8750 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | normalized_values | 1 | 1 | true | 0.6295 | 0.9600 | 0.9600 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | deterministic_root_policy | 1 | 1 | true | 0.6295 | 0.8733 | 0.8733 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | low_cpuct | 1 | 1 | true | 0.6295 | 0.9067 | 0.9067 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | high_cpuct | 1 | 1 | true | 0.6295 | 0.8150 | 0.8150 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | no_tactical_root_bias | 1 | 1 | true | 0.6554 | 0.8792 | 0.8792 | 0.0000 | true | pass_reference_selected |
| current | capture_available-008 | 1200 | full_search_control | 1 | 1 | true | 0.6295 | 0.9575 | 0.9575 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | baseline_eval_search | 2 | 2 | true | 0.1984 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | no_subtree_reuse | 2 | 2 | true | 0.1984 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | parent_q_fpu | 2 | 2 | true | 0.1984 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | normalized_values | 2 | 2 | true | 0.1984 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | deterministic_root_policy | 2 | 2 | true | 0.1984 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | low_cpuct | 2 | 2 | true | 0.1984 | 0.3594 | 0.3594 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | high_cpuct | 2 | 2 | true | 0.1984 | 0.2344 | 0.2344 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 64 | no_tactical_root_bias | 2 | 0 | false | 0.1579 | 0.1719 | 0.2344 | -0.1582 | false | u_favors_selected |
| corrected_replay_w1 | capture_available-002 | 64 | full_search_control | 2 | 1 | false | 0.1984 | 0.0469 | 0.8594 | 0.1194 | false | q_favors_selected |
| corrected_replay_w1 | capture_available-002 | 128 | baseline_eval_search | 2 | 2 | true | 0.1984 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1984 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | parent_q_fpu | 2 | 2 | true | 0.1984 | 0.5469 | 0.5469 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | normalized_values | 2 | 2 | true | 0.1984 | 0.8203 | 0.8203 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1984 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | low_cpuct | 2 | 2 | true | 0.1984 | 0.6797 | 0.6797 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | high_cpuct | 2 | 2 | true | 0.1984 | 0.4766 | 0.4766 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1579 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 128 | full_search_control | 2 | 1 | false | 0.1984 | 0.0234 | 0.9219 | 0.0999 | false | q_favors_selected |
| corrected_replay_w1 | capture_available-002 | 384 | baseline_eval_search | 2 | 2 | true | 0.1984 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1984 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | parent_q_fpu | 2 | 2 | true | 0.1984 | 0.6432 | 0.6432 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | normalized_values | 2 | 2 | true | 0.1984 | 0.9167 | 0.9167 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1984 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | low_cpuct | 2 | 2 | true | 0.1984 | 0.8620 | 0.8620 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | high_cpuct | 2 | 2 | true | 0.1984 | 0.5911 | 0.5911 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1579 | 0.7161 | 0.7161 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 384 | full_search_control | 2 | 1 | false | 0.1984 | 0.0234 | 0.5938 | 0.0344 | false | q_favors_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1984 | 0.8592 | 0.8592 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1984 | 0.8592 | 0.8592 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1984 | 0.8550 | 0.8550 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | normalized_values | 2 | 2 | true | 0.1984 | 0.9617 | 0.9617 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1984 | 0.8592 | 0.8592 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | low_cpuct | 2 | 2 | true | 0.1984 | 0.9458 | 0.9458 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | high_cpuct | 2 | 2 | true | 0.1984 | 0.7842 | 0.7842 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1579 | 0.8567 | 0.8567 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-002 | 1200 | full_search_control | 2 | 2 | true | 0.1984 | 0.5025 | 0.5025 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | baseline_eval_search | 2 | 2 | true | 0.1775 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | no_subtree_reuse | 2 | 2 | true | 0.1775 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | parent_q_fpu | 2 | 2 | true | 0.1775 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | normalized_values | 2 | 2 | true | 0.1775 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | deterministic_root_policy | 2 | 2 | true | 0.1775 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | low_cpuct | 2 | 2 | true | 0.1775 | 0.7500 | 0.7500 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | high_cpuct | 2 | 1 | false | 0.1775 | 0.3750 | 0.4062 | -0.0895 | false | u_favors_selected |
| corrected_replay_w1 | capture_available-003 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1307 | 0.3750 | 0.3750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 64 | full_search_control | 2 | 2 | true | 0.1775 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | baseline_eval_search | 2 | 2 | true | 0.1775 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1775 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | parent_q_fpu | 2 | 2 | true | 0.1775 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | normalized_values | 2 | 2 | true | 0.1775 | 0.7656 | 0.7656 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1775 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | low_cpuct | 2 | 2 | true | 0.1775 | 0.7969 | 0.7969 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | high_cpuct | 2 | 2 | true | 0.1775 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1307 | 0.6016 | 0.6016 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 128 | full_search_control | 2 | 2 | true | 0.1775 | 0.8281 | 0.8281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | baseline_eval_search | 2 | 2 | true | 0.1775 | 0.8099 | 0.8099 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1775 | 0.8099 | 0.8099 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | parent_q_fpu | 2 | 2 | true | 0.1775 | 0.7917 | 0.7917 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | normalized_values | 2 | 2 | true | 0.1775 | 0.9062 | 0.9062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1775 | 0.8099 | 0.8099 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | low_cpuct | 2 | 2 | true | 0.1775 | 0.8828 | 0.8828 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | high_cpuct | 2 | 2 | true | 0.1775 | 0.7083 | 0.7083 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1307 | 0.8073 | 0.8073 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 384 | full_search_control | 2 | 2 | true | 0.1775 | 0.9062 | 0.9062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1775 | 0.9092 | 0.9092 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1775 | 0.9092 | 0.9092 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1775 | 0.8958 | 0.8958 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | normalized_values | 2 | 2 | true | 0.1775 | 0.9550 | 0.9550 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1775 | 0.9092 | 0.9092 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | low_cpuct | 2 | 2 | true | 0.1775 | 0.9400 | 0.9400 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | high_cpuct | 2 | 2 | true | 0.1775 | 0.8358 | 0.8358 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1307 | 0.9008 | 0.9008 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-003 | 1200 | full_search_control | 2 | 2 | true | 0.1775 | 0.9442 | 0.9442 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | baseline_eval_search | 2 | 2 | true | 0.2148 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | no_subtree_reuse | 2 | 2 | true | 0.2148 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | parent_q_fpu | 2 | 2 | true | 0.2148 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | normalized_values | 2 | 4 | false | 0.2148 | 0.0312 | 0.4219 | 0.0721 | false | q_favors_selected |
| corrected_replay_w1 | capture_available-006 | 64 | deterministic_root_policy | 2 | 2 | true | 0.2148 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | low_cpuct | 2 | 2 | true | 0.2148 | 0.5312 | 0.5312 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | high_cpuct | 2 | 2 | true | 0.2148 | 0.4375 | 0.4375 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1792 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 64 | full_search_control | 2 | 4 | false | 0.2148 | 0.0312 | 0.3750 | 0.1058 | false | q_favors_selected |
| corrected_replay_w1 | capture_available-006 | 128 | baseline_eval_search | 2 | 2 | true | 0.2148 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 128 | no_subtree_reuse | 2 | 2 | true | 0.2148 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 128 | parent_q_fpu | 2 | 2 | true | 0.2148 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 128 | normalized_values | 2 | 4 | false | 0.2148 | 0.1562 | 0.3984 | -0.2962 | false | selected_move_not_reference |
| corrected_replay_w1 | capture_available-006 | 128 | deterministic_root_policy | 2 | 2 | true | 0.2148 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 128 | low_cpuct | 2 | 2 | true | 0.2148 | 0.6641 | 0.6641 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 128 | high_cpuct | 2 | 2 | true | 0.2148 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1792 | 0.5312 | 0.5312 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 128 | full_search_control | 2 | 4 | false | 0.2148 | 0.1719 | 0.3281 | -0.1736 | false | selected_move_not_reference |
| corrected_replay_w1 | capture_available-006 | 384 | baseline_eval_search | 2 | 2 | true | 0.2148 | 0.7214 | 0.7214 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2148 | 0.7214 | 0.7214 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | parent_q_fpu | 2 | 2 | true | 0.2148 | 0.7031 | 0.7031 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | normalized_values | 2 | 2 | true | 0.2148 | 0.7109 | 0.7109 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2148 | 0.7214 | 0.7214 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | low_cpuct | 2 | 2 | true | 0.2148 | 0.8359 | 0.8359 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | high_cpuct | 2 | 2 | true | 0.2148 | 0.6510 | 0.6510 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1792 | 0.6927 | 0.6927 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 384 | full_search_control | 2 | 2 | true | 0.2148 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2148 | 0.8883 | 0.8883 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2148 | 0.8883 | 0.8883 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2148 | 0.8950 | 0.8950 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | normalized_values | 2 | 2 | true | 0.2148 | 0.9025 | 0.9025 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2148 | 0.8883 | 0.8883 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | low_cpuct | 2 | 2 | true | 0.2148 | 0.9442 | 0.9442 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | high_cpuct | 2 | 2 | true | 0.2148 | 0.8283 | 0.8283 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1792 | 0.8808 | 0.8808 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-006 | 1200 | full_search_control | 2 | 2 | true | 0.2148 | 0.9050 | 0.9050 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | baseline_eval_search | 2 | 2 | true | 0.2159 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | no_subtree_reuse | 2 | 2 | true | 0.2159 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | parent_q_fpu | 2 | 2 | true | 0.2159 | 0.4531 | 0.4531 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | normalized_values | 2 | 2 | true | 0.2159 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | deterministic_root_policy | 2 | 2 | true | 0.2159 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | low_cpuct | 2 | 2 | true | 0.2159 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | high_cpuct | 2 | 2 | true | 0.2159 | 0.4219 | 0.4219 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1806 | 0.3906 | 0.3906 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 64 | full_search_control | 2 | 2 | true | 0.2159 | 0.8125 | 0.8125 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | baseline_eval_search | 2 | 2 | true | 0.2159 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | no_subtree_reuse | 2 | 2 | true | 0.2159 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | parent_q_fpu | 2 | 2 | true | 0.2159 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | normalized_values | 2 | 2 | true | 0.2159 | 0.7109 | 0.7109 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | deterministic_root_policy | 2 | 2 | true | 0.2159 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | low_cpuct | 2 | 2 | true | 0.2159 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | high_cpuct | 2 | 2 | true | 0.2159 | 0.5078 | 0.5078 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1806 | 0.5469 | 0.5469 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 128 | full_search_control | 2 | 2 | true | 0.2159 | 0.8516 | 0.8516 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | baseline_eval_search | 2 | 2 | true | 0.2159 | 0.6589 | 0.6589 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2159 | 0.6589 | 0.6589 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | parent_q_fpu | 2 | 2 | true | 0.2159 | 0.6901 | 0.6901 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | normalized_values | 2 | 2 | true | 0.2159 | 0.8672 | 0.8672 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2159 | 0.6589 | 0.6589 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | low_cpuct | 2 | 2 | true | 0.2159 | 0.8464 | 0.8464 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | high_cpuct | 2 | 2 | true | 0.2159 | 0.5651 | 0.5651 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1806 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 384 | full_search_control | 2 | 2 | true | 0.2159 | 0.8880 | 0.8880 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2159 | 0.8692 | 0.8692 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2159 | 0.8692 | 0.8692 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2159 | 0.8600 | 0.8600 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | normalized_values | 2 | 2 | true | 0.2159 | 0.9475 | 0.9475 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2159 | 0.8692 | 0.8692 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | low_cpuct | 2 | 2 | true | 0.2159 | 0.9408 | 0.9408 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | high_cpuct | 2 | 2 | true | 0.2159 | 0.8000 | 0.8000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1806 | 0.8633 | 0.8633 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-007 | 1200 | full_search_control | 2 | 2 | true | 0.2159 | 0.9467 | 0.9467 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | baseline_eval_search | 1 | 1 | true | 0.3669 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | no_subtree_reuse | 1 | 1 | true | 0.3669 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | parent_q_fpu | 1 | 1 | true | 0.3669 | 0.5469 | 0.5469 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | normalized_values | 1 | 1 | true | 0.3669 | 0.8906 | 0.8906 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | deterministic_root_policy | 1 | 1 | true | 0.3669 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | low_cpuct | 1 | 1 | true | 0.3669 | 0.8438 | 0.8438 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | high_cpuct | 1 | 1 | true | 0.3669 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | no_tactical_root_bias | 1 | 1 | true | 0.3403 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 64 | full_search_control | 1 | 1 | true | 0.3669 | 0.8906 | 0.8906 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | baseline_eval_search | 1 | 1 | true | 0.3669 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | no_subtree_reuse | 1 | 1 | true | 0.3669 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | parent_q_fpu | 1 | 1 | true | 0.3669 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | normalized_values | 1 | 1 | true | 0.3669 | 0.9141 | 0.9141 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | deterministic_root_policy | 1 | 1 | true | 0.3669 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | low_cpuct | 1 | 1 | true | 0.3669 | 0.8359 | 0.8359 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | high_cpuct | 1 | 1 | true | 0.3669 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | no_tactical_root_bias | 1 | 1 | true | 0.3403 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 128 | full_search_control | 1 | 1 | true | 0.3669 | 0.9141 | 0.9141 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | baseline_eval_search | 1 | 1 | true | 0.3669 | 0.7865 | 0.7865 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | no_subtree_reuse | 1 | 1 | true | 0.3669 | 0.7865 | 0.7865 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | parent_q_fpu | 1 | 1 | true | 0.3669 | 0.8151 | 0.8151 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | normalized_values | 1 | 1 | true | 0.3669 | 0.9323 | 0.9323 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | deterministic_root_policy | 1 | 1 | true | 0.3669 | 0.7865 | 0.7865 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | low_cpuct | 1 | 1 | true | 0.3669 | 0.9010 | 0.9010 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | high_cpuct | 1 | 1 | true | 0.3669 | 0.7396 | 0.7396 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | no_tactical_root_bias | 1 | 1 | true | 0.3403 | 0.7682 | 0.7682 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 384 | full_search_control | 1 | 1 | true | 0.3669 | 0.9427 | 0.9427 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | baseline_eval_search | 1 | 1 | true | 0.3669 | 0.9025 | 0.9025 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | no_subtree_reuse | 1 | 1 | true | 0.3669 | 0.9025 | 0.9025 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | parent_q_fpu | 1 | 1 | true | 0.3669 | 0.8942 | 0.8942 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | normalized_values | 1 | 1 | true | 0.3669 | 0.9642 | 0.9642 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | deterministic_root_policy | 1 | 1 | true | 0.3669 | 0.9025 | 0.9025 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | low_cpuct | 1 | 1 | true | 0.3669 | 0.9392 | 0.9392 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | high_cpuct | 1 | 1 | true | 0.3669 | 0.8650 | 0.8650 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | no_tactical_root_bias | 1 | 1 | true | 0.3403 | 0.8975 | 0.8975 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w1 | capture_available-008 | 1200 | full_search_control | 1 | 1 | true | 0.3669 | 0.9642 | 0.9642 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 64 | baseline_eval_search | 2 | 1 | false | 0.1963 | 0.2344 | 0.3438 | -0.0051 | false | policy_prior_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1963 | 0.2344 | 0.3438 | -0.0051 | false | policy_prior_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | parent_q_fpu | 2 | 1 | false | 0.1963 | 0.1406 | 0.3125 | 0.0758 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | normalized_values | 2 | 1 | false | 0.1963 | 0.0312 | 0.7812 | 0.1305 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1963 | 0.2344 | 0.3438 | -0.0051 | false | policy_prior_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | low_cpuct | 2 | 1 | false | 0.1963 | 0.0938 | 0.4688 | 0.0527 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | high_cpuct | 2 | 1 | false | 0.1963 | 0.2031 | 0.2656 | -0.0083 | false | u_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1552 | 0.0625 | 0.3438 | 0.1957 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 64 | full_search_control | 2 | 1 | false | 0.1963 | 0.0312 | 0.8281 | 0.1291 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 128 | baseline_eval_search | 2 | 2 | true | 0.1963 | 0.4297 | 0.4297 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1963 | 0.4297 | 0.4297 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 128 | parent_q_fpu | 2 | 2 | true | 0.1963 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 128 | normalized_values | 2 | 1 | false | 0.1963 | 0.0234 | 0.8750 | 0.0965 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1963 | 0.4297 | 0.4297 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 128 | low_cpuct | 2 | 2 | true | 0.1963 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 128 | high_cpuct | 2 | 2 | true | 0.1963 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1552 | 0.3672 | 0.3672 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 128 | full_search_control | 2 | 1 | false | 0.1963 | 0.0156 | 0.8984 | 0.1240 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 384 | baseline_eval_search | 2 | 2 | true | 0.1963 | 0.6979 | 0.6979 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1963 | 0.6979 | 0.6979 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 384 | parent_q_fpu | 2 | 2 | true | 0.1963 | 0.7604 | 0.7604 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 384 | normalized_values | 2 | 1 | false | 0.1963 | 0.0104 | 0.5625 | 0.0989 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1963 | 0.6979 | 0.6979 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 384 | low_cpuct | 2 | 2 | true | 0.1963 | 0.8099 | 0.8099 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 384 | high_cpuct | 2 | 2 | true | 0.1963 | 0.6276 | 0.6276 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1552 | 0.7005 | 0.7005 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 384 | full_search_control | 2 | 1 | false | 0.1963 | 0.0104 | 0.6667 | 0.1235 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1963 | 0.8692 | 0.8692 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1963 | 0.8692 | 0.8692 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1963 | 0.8475 | 0.8475 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | normalized_values | 2 | 2 | true | 0.1963 | 0.6483 | 0.6483 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1963 | 0.8692 | 0.8692 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | low_cpuct | 2 | 2 | true | 0.1963 | 0.9342 | 0.9342 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | high_cpuct | 2 | 2 | true | 0.1963 | 0.7875 | 0.7875 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1552 | 0.8658 | 0.8658 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-002 | 1200 | full_search_control | 2 | 2 | true | 0.1963 | 0.5733 | 0.5733 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | baseline_eval_search | 2 | 2 | true | 0.1707 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | no_subtree_reuse | 2 | 2 | true | 0.1707 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | parent_q_fpu | 2 | 2 | true | 0.1707 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | normalized_values | 2 | 2 | true | 0.1707 | 0.5312 | 0.5312 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | deterministic_root_policy | 2 | 2 | true | 0.1707 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | low_cpuct | 2 | 2 | true | 0.1707 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | high_cpuct | 2 | 1 | false | 0.1707 | 0.3438 | 0.3594 | -0.0791 | false | u_favors_selected |
| corrected_replay_w2 | capture_available-003 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1219 | 0.3750 | 0.3750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 64 | full_search_control | 2 | 2 | true | 0.1707 | 0.7031 | 0.7031 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | baseline_eval_search | 2 | 2 | true | 0.1707 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1707 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | parent_q_fpu | 2 | 2 | true | 0.1707 | 0.5078 | 0.5078 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | normalized_values | 2 | 2 | true | 0.1707 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1707 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | low_cpuct | 2 | 2 | true | 0.1707 | 0.6953 | 0.6953 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | high_cpuct | 2 | 2 | true | 0.1707 | 0.5156 | 0.5156 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1219 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 128 | full_search_control | 2 | 2 | true | 0.1707 | 0.8047 | 0.8047 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | baseline_eval_search | 2 | 2 | true | 0.1707 | 0.8047 | 0.8047 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1707 | 0.8047 | 0.8047 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | parent_q_fpu | 2 | 2 | true | 0.1707 | 0.7943 | 0.7943 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | normalized_values | 2 | 2 | true | 0.1707 | 0.8880 | 0.8880 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1707 | 0.8047 | 0.8047 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | low_cpuct | 2 | 2 | true | 0.1707 | 0.8698 | 0.8698 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | high_cpuct | 2 | 2 | true | 0.1707 | 0.6354 | 0.6354 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1219 | 0.7943 | 0.7943 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 384 | full_search_control | 2 | 2 | true | 0.1707 | 0.8776 | 0.8776 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1707 | 0.9092 | 0.9092 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1707 | 0.9092 | 0.9092 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1707 | 0.8783 | 0.8783 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | normalized_values | 2 | 2 | true | 0.1707 | 0.9567 | 0.9567 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1707 | 0.9092 | 0.9092 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | low_cpuct | 2 | 2 | true | 0.1707 | 0.9417 | 0.9417 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | high_cpuct | 2 | 2 | true | 0.1707 | 0.8350 | 0.8350 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1219 | 0.9050 | 0.9050 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-003 | 1200 | full_search_control | 2 | 2 | true | 0.1707 | 0.9483 | 0.9483 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 64 | baseline_eval_search | 2 | 2 | true | 0.2091 | 0.3281 | 0.3281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 64 | no_subtree_reuse | 2 | 2 | true | 0.2091 | 0.3281 | 0.3281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 64 | parent_q_fpu | 2 | 1 | false | 0.2091 | 0.2031 | 0.2656 | -0.1066 | false | u_favors_selected |
| corrected_replay_w2 | capture_available-006 | 64 | normalized_values | 2 | 3 | false | 0.2091 | 0.0312 | 0.3594 | 0.1760 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-006 | 64 | deterministic_root_policy | 2 | 2 | true | 0.2091 | 0.3281 | 0.3281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 64 | low_cpuct | 2 | 2 | true | 0.2091 | 0.3281 | 0.3281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 64 | high_cpuct | 2 | 2 | true | 0.2091 | 0.3906 | 0.3906 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1719 | 0.2656 | 0.2656 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 64 | full_search_control | 2 | 4 | false | 0.2091 | 0.0312 | 0.3750 | 0.1263 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-006 | 128 | baseline_eval_search | 2 | 2 | true | 0.2091 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 128 | no_subtree_reuse | 2 | 2 | true | 0.2091 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 128 | parent_q_fpu | 2 | 2 | true | 0.2091 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 128 | normalized_values | 2 | 3 | false | 0.2091 | 0.2812 | 0.3828 | -0.1407 | false | selected_move_not_reference |
| corrected_replay_w2 | capture_available-006 | 128 | deterministic_root_policy | 2 | 2 | true | 0.2091 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 128 | low_cpuct | 2 | 2 | true | 0.2091 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 128 | high_cpuct | 2 | 2 | true | 0.2091 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1719 | 0.5234 | 0.5234 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 128 | full_search_control | 2 | 3 | false | 0.2091 | 0.0156 | 0.6328 | 0.1448 | false | q_favors_selected |
| corrected_replay_w2 | capture_available-006 | 384 | baseline_eval_search | 2 | 2 | true | 0.2091 | 0.6979 | 0.6979 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2091 | 0.6979 | 0.6979 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | parent_q_fpu | 2 | 2 | true | 0.2091 | 0.6901 | 0.6901 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | normalized_values | 2 | 2 | true | 0.2091 | 0.7500 | 0.7500 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2091 | 0.6979 | 0.6979 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | low_cpuct | 2 | 2 | true | 0.2091 | 0.8307 | 0.8307 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | high_cpuct | 2 | 2 | true | 0.2091 | 0.5729 | 0.5729 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1719 | 0.6693 | 0.6693 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 384 | full_search_control | 2 | 2 | true | 0.2091 | 0.4766 | 0.4766 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2091 | 0.8575 | 0.8575 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2091 | 0.8575 | 0.8575 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2091 | 0.8592 | 0.8592 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | normalized_values | 2 | 2 | true | 0.2091 | 0.9142 | 0.9142 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2091 | 0.8575 | 0.8575 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | low_cpuct | 2 | 2 | true | 0.2091 | 0.9458 | 0.9458 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | high_cpuct | 2 | 2 | true | 0.2091 | 0.8000 | 0.8000 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1719 | 0.8442 | 0.8442 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-006 | 1200 | full_search_control | 2 | 2 | true | 0.2091 | 0.8267 | 0.8267 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | baseline_eval_search | 2 | 2 | true | 0.2097 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | no_subtree_reuse | 2 | 2 | true | 0.2097 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | parent_q_fpu | 2 | 2 | true | 0.2097 | 0.3906 | 0.3906 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | normalized_values | 2 | 2 | true | 0.2097 | 0.6406 | 0.6406 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | deterministic_root_policy | 2 | 2 | true | 0.2097 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | low_cpuct | 2 | 2 | true | 0.2097 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | high_cpuct | 2 | 2 | true | 0.2097 | 0.3750 | 0.3750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1726 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 64 | full_search_control | 2 | 2 | true | 0.2097 | 0.7812 | 0.7812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | baseline_eval_search | 2 | 2 | true | 0.2097 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | no_subtree_reuse | 2 | 2 | true | 0.2097 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | parent_q_fpu | 2 | 2 | true | 0.2097 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | normalized_values | 2 | 2 | true | 0.2097 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | deterministic_root_policy | 2 | 2 | true | 0.2097 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | low_cpuct | 2 | 2 | true | 0.2097 | 0.6953 | 0.6953 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | high_cpuct | 2 | 2 | true | 0.2097 | 0.4922 | 0.4922 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1726 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 128 | full_search_control | 2 | 2 | true | 0.2097 | 0.7969 | 0.7969 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | baseline_eval_search | 2 | 2 | true | 0.2097 | 0.7057 | 0.7057 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2097 | 0.7057 | 0.7057 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | parent_q_fpu | 2 | 2 | true | 0.2097 | 0.6146 | 0.6146 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | normalized_values | 2 | 2 | true | 0.2097 | 0.8750 | 0.8750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2097 | 0.7057 | 0.7057 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | low_cpuct | 2 | 2 | true | 0.2097 | 0.8698 | 0.8698 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | high_cpuct | 2 | 2 | true | 0.2097 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1726 | 0.7057 | 0.7057 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 384 | full_search_control | 2 | 2 | true | 0.2097 | 0.8828 | 0.8828 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2097 | 0.8575 | 0.8575 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2097 | 0.8575 | 0.8575 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2097 | 0.8525 | 0.8525 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | normalized_values | 2 | 2 | true | 0.2097 | 0.9475 | 0.9475 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2097 | 0.8575 | 0.8575 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | low_cpuct | 2 | 2 | true | 0.2097 | 0.9375 | 0.9375 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | high_cpuct | 2 | 2 | true | 0.2097 | 0.7933 | 0.7933 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1726 | 0.8592 | 0.8592 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-007 | 1200 | full_search_control | 2 | 2 | true | 0.2097 | 0.9483 | 0.9483 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | baseline_eval_search | 1 | 1 | true | 0.3226 | 0.6250 | 0.6250 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | no_subtree_reuse | 1 | 1 | true | 0.3226 | 0.6250 | 0.6250 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | parent_q_fpu | 1 | 1 | true | 0.3226 | 0.6250 | 0.6250 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | normalized_values | 1 | 1 | true | 0.3226 | 0.8750 | 0.8750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | deterministic_root_policy | 1 | 1 | true | 0.3226 | 0.6250 | 0.6250 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | low_cpuct | 1 | 1 | true | 0.3226 | 0.8281 | 0.8281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | high_cpuct | 1 | 1 | true | 0.3226 | 0.5312 | 0.5312 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | no_tactical_root_bias | 1 | 1 | true | 0.2871 | 0.5312 | 0.5312 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 64 | full_search_control | 1 | 1 | true | 0.3226 | 0.8594 | 0.8594 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | baseline_eval_search | 1 | 1 | true | 0.3226 | 0.7266 | 0.7266 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | no_subtree_reuse | 1 | 1 | true | 0.3226 | 0.7266 | 0.7266 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | parent_q_fpu | 1 | 1 | true | 0.3226 | 0.7109 | 0.7109 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | normalized_values | 1 | 1 | true | 0.3226 | 0.9219 | 0.9219 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | deterministic_root_policy | 1 | 1 | true | 0.3226 | 0.7266 | 0.7266 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | low_cpuct | 1 | 1 | true | 0.3226 | 0.8281 | 0.8281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | high_cpuct | 1 | 1 | true | 0.3226 | 0.6484 | 0.6484 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | no_tactical_root_bias | 1 | 1 | true | 0.2871 | 0.6953 | 0.6953 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 128 | full_search_control | 1 | 1 | true | 0.3226 | 0.9141 | 0.9141 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | baseline_eval_search | 1 | 1 | true | 0.3226 | 0.7760 | 0.7760 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | no_subtree_reuse | 1 | 1 | true | 0.3226 | 0.7760 | 0.7760 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | parent_q_fpu | 1 | 1 | true | 0.3226 | 0.7760 | 0.7760 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | normalized_values | 1 | 1 | true | 0.3226 | 0.9505 | 0.9505 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | deterministic_root_policy | 1 | 1 | true | 0.3226 | 0.7760 | 0.7760 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | low_cpuct | 1 | 1 | true | 0.3226 | 0.8802 | 0.8802 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | high_cpuct | 1 | 1 | true | 0.3226 | 0.7240 | 0.7240 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | no_tactical_root_bias | 1 | 1 | true | 0.2871 | 0.7604 | 0.7604 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 384 | full_search_control | 1 | 1 | true | 0.3226 | 0.9375 | 0.9375 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | baseline_eval_search | 1 | 1 | true | 0.3226 | 0.8950 | 0.8950 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | no_subtree_reuse | 1 | 1 | true | 0.3226 | 0.8950 | 0.8950 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | parent_q_fpu | 1 | 1 | true | 0.3226 | 0.8892 | 0.8892 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | normalized_values | 1 | 1 | true | 0.3226 | 0.9633 | 0.9633 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | deterministic_root_policy | 1 | 1 | true | 0.3226 | 0.8950 | 0.8950 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | low_cpuct | 1 | 1 | true | 0.3226 | 0.9275 | 0.9275 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | high_cpuct | 1 | 1 | true | 0.3226 | 0.8533 | 0.8533 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | no_tactical_root_bias | 1 | 1 | true | 0.2871 | 0.8892 | 0.8892 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w2 | capture_available-008 | 1200 | full_search_control | 1 | 1 | true | 0.3226 | 0.9642 | 0.9642 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 64 | baseline_eval_search | 2 | 1 | false | 0.1874 | 0.2031 | 0.2969 | -0.0222 | false | u_favors_selected |
| corrected_replay_w4 | capture_available-002 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1874 | 0.2031 | 0.2969 | -0.0222 | false | u_favors_selected |
| corrected_replay_w4 | capture_available-002 | 64 | parent_q_fpu | 2 | 0 | false | 0.1874 | 0.2344 | 0.2500 | -0.1417 | false | u_favors_selected |
| corrected_replay_w4 | capture_available-002 | 64 | normalized_values | 2 | 1 | false | 0.1874 | 0.0312 | 0.7969 | 0.1624 | false | q_favors_selected |
| corrected_replay_w4 | capture_available-002 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1874 | 0.2031 | 0.2969 | -0.0222 | false | u_favors_selected |
| corrected_replay_w4 | capture_available-002 | 64 | low_cpuct | 2 | 1 | false | 0.1874 | 0.1719 | 0.3438 | 0.0017 | false | q_favors_selected |
| corrected_replay_w4 | capture_available-002 | 64 | high_cpuct | 2 | 1 | false | 0.1874 | 0.2031 | 0.3125 | 0.0023 | false | selected_move_beats_reference_on_q_and_u |
| corrected_replay_w4 | capture_available-002 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1436 | 0.1406 | 0.2969 | -0.0073 | false | policy_prior_favors_selected |
| corrected_replay_w4 | capture_available-002 | 64 | full_search_control | 2 | 3 | false | 0.1874 | 0.0312 | 0.5312 | 0.1733 | false | q_favors_selected |
| corrected_replay_w4 | capture_available-002 | 128 | baseline_eval_search | 2 | 2 | true | 0.1874 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1874 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 128 | parent_q_fpu | 2 | 2 | true | 0.1874 | 0.5312 | 0.5312 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 128 | normalized_values | 2 | 1 | false | 0.1874 | 0.0156 | 0.8047 | 0.1417 | false | q_favors_selected |
| corrected_replay_w4 | capture_available-002 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1874 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 128 | low_cpuct | 2 | 2 | true | 0.1874 | 0.5859 | 0.5859 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 128 | high_cpuct | 2 | 2 | true | 0.1874 | 0.4297 | 0.4297 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1436 | 0.5156 | 0.5156 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 128 | full_search_control | 2 | 1 | false | 0.1874 | 0.0156 | 0.4453 | 0.1350 | false | q_favors_selected |
| corrected_replay_w4 | capture_available-002 | 384 | baseline_eval_search | 2 | 2 | true | 0.1874 | 0.7083 | 0.7083 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1874 | 0.7083 | 0.7083 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 384 | parent_q_fpu | 2 | 2 | true | 0.1874 | 0.6797 | 0.6797 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 384 | normalized_values | 2 | 1 | false | 0.1874 | 0.4375 | 0.4818 | -0.1378 | false | u_favors_selected |
| corrected_replay_w4 | capture_available-002 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1874 | 0.7083 | 0.7083 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 384 | low_cpuct | 2 | 2 | true | 0.1874 | 0.8568 | 0.8568 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 384 | high_cpuct | 2 | 2 | true | 0.1874 | 0.5964 | 0.5964 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1436 | 0.7005 | 0.7005 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 384 | full_search_control | 2 | 1 | false | 0.1874 | 0.2057 | 0.5990 | -0.2115 | false | policy_prior_favors_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1874 | 0.8633 | 0.8633 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1874 | 0.8633 | 0.8633 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1874 | 0.8567 | 0.8567 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | normalized_values | 2 | 2 | true | 0.1874 | 0.8150 | 0.8150 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1874 | 0.8633 | 0.8633 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | low_cpuct | 2 | 2 | true | 0.1874 | 0.9425 | 0.9425 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | high_cpuct | 2 | 2 | true | 0.1874 | 0.7975 | 0.7975 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1436 | 0.8567 | 0.8567 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-002 | 1200 | full_search_control | 2 | 2 | true | 0.1874 | 0.7417 | 0.7417 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | baseline_eval_search | 2 | 2 | true | 0.1787 | 0.6406 | 0.6406 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | no_subtree_reuse | 2 | 2 | true | 0.1787 | 0.6406 | 0.6406 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | parent_q_fpu | 2 | 2 | true | 0.1787 | 0.6406 | 0.6406 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | normalized_values | 2 | 2 | true | 0.1787 | 0.5469 | 0.5469 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | deterministic_root_policy | 2 | 2 | true | 0.1787 | 0.6406 | 0.6406 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | low_cpuct | 2 | 2 | true | 0.1787 | 0.6250 | 0.6250 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | high_cpuct | 2 | 2 | true | 0.1787 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1323 | 0.6250 | 0.6250 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 64 | full_search_control | 2 | 2 | true | 0.1787 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | baseline_eval_search | 2 | 2 | true | 0.1787 | 0.7734 | 0.7734 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1787 | 0.7734 | 0.7734 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | parent_q_fpu | 2 | 2 | true | 0.1787 | 0.7578 | 0.7578 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | normalized_values | 2 | 2 | true | 0.1787 | 0.7656 | 0.7656 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1787 | 0.7734 | 0.7734 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | low_cpuct | 2 | 2 | true | 0.1787 | 0.8047 | 0.8047 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | high_cpuct | 2 | 2 | true | 0.1787 | 0.6016 | 0.6016 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1323 | 0.6953 | 0.6953 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 128 | full_search_control | 2 | 2 | true | 0.1787 | 0.8203 | 0.8203 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | baseline_eval_search | 2 | 2 | true | 0.1787 | 0.8151 | 0.8151 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1787 | 0.8151 | 0.8151 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | parent_q_fpu | 2 | 2 | true | 0.1787 | 0.8047 | 0.8047 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | normalized_values | 2 | 2 | true | 0.1787 | 0.9036 | 0.9036 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1787 | 0.8151 | 0.8151 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | low_cpuct | 2 | 2 | true | 0.1787 | 0.9010 | 0.9010 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | high_cpuct | 2 | 2 | true | 0.1787 | 0.6771 | 0.6771 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1323 | 0.8021 | 0.8021 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 384 | full_search_control | 2 | 2 | true | 0.1787 | 0.9062 | 0.9062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1787 | 0.9133 | 0.9133 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1787 | 0.9133 | 0.9133 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1787 | 0.8975 | 0.8975 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | normalized_values | 2 | 2 | true | 0.1787 | 0.9592 | 0.9592 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1787 | 0.9133 | 0.9133 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | low_cpuct | 2 | 2 | true | 0.1787 | 0.9492 | 0.9492 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | high_cpuct | 2 | 2 | true | 0.1787 | 0.8550 | 0.8550 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1323 | 0.9050 | 0.9050 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-003 | 1200 | full_search_control | 2 | 2 | true | 0.1787 | 0.9500 | 0.9500 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | baseline_eval_search | 2 | 2 | true | 0.2004 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | no_subtree_reuse | 2 | 2 | true | 0.2004 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | parent_q_fpu | 2 | 2 | true | 0.2004 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | normalized_values | 2 | 4 | false | 0.2004 | 0.0312 | 0.4375 | 0.1577 | false | q_favors_selected |
| corrected_replay_w4 | capture_available-006 | 64 | deterministic_root_policy | 2 | 2 | true | 0.2004 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | low_cpuct | 2 | 2 | true | 0.2004 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | high_cpuct | 2 | 2 | true | 0.2004 | 0.3438 | 0.3438 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1605 | 0.4531 | 0.4531 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 64 | full_search_control | 2 | 4 | false | 0.2004 | 0.0312 | 0.4219 | 0.1587 | false | q_favors_selected |
| corrected_replay_w4 | capture_available-006 | 128 | baseline_eval_search | 2 | 2 | true | 0.2004 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | no_subtree_reuse | 2 | 2 | true | 0.2004 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | parent_q_fpu | 2 | 2 | true | 0.2004 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | normalized_values | 2 | 2 | true | 0.2004 | 0.3750 | 0.3750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | deterministic_root_policy | 2 | 2 | true | 0.2004 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | low_cpuct | 2 | 2 | true | 0.2004 | 0.6328 | 0.6328 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | high_cpuct | 2 | 2 | true | 0.2004 | 0.4922 | 0.4922 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1605 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 128 | full_search_control | 2 | 4 | false | 0.2004 | 0.2969 | 0.4219 | -0.2566 | false | selected_move_not_reference |
| corrected_replay_w4 | capture_available-006 | 384 | baseline_eval_search | 2 | 2 | true | 0.2004 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2004 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | parent_q_fpu | 2 | 2 | true | 0.2004 | 0.7240 | 0.7240 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | normalized_values | 2 | 2 | true | 0.2004 | 0.7812 | 0.7812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2004 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | low_cpuct | 2 | 2 | true | 0.2004 | 0.8203 | 0.8203 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | high_cpuct | 2 | 2 | true | 0.2004 | 0.6458 | 0.6458 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1605 | 0.7214 | 0.7214 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 384 | full_search_control | 2 | 2 | true | 0.2004 | 0.7552 | 0.7552 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2004 | 0.9017 | 0.9017 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2004 | 0.9017 | 0.9017 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2004 | 0.9083 | 0.9083 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | normalized_values | 2 | 2 | true | 0.2004 | 0.9250 | 0.9250 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2004 | 0.9017 | 0.9017 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | low_cpuct | 2 | 2 | true | 0.2004 | 0.9425 | 0.9425 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | high_cpuct | 2 | 2 | true | 0.2004 | 0.8433 | 0.8433 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1605 | 0.8892 | 0.8892 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-006 | 1200 | full_search_control | 2 | 2 | true | 0.2004 | 0.9167 | 0.9167 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | baseline_eval_search | 2 | 2 | true | 0.2090 | 0.4531 | 0.4531 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | no_subtree_reuse | 2 | 2 | true | 0.2090 | 0.4531 | 0.4531 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | parent_q_fpu | 2 | 2 | true | 0.2090 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | normalized_values | 2 | 2 | true | 0.2090 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | deterministic_root_policy | 2 | 2 | true | 0.2090 | 0.4531 | 0.4531 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | low_cpuct | 2 | 2 | true | 0.2090 | 0.5938 | 0.5938 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | high_cpuct | 2 | 2 | true | 0.2090 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | no_tactical_root_bias | 2 | 2 | true | 0.1717 | 0.4375 | 0.4375 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 64 | full_search_control | 2 | 2 | true | 0.2090 | 0.8125 | 0.8125 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | baseline_eval_search | 2 | 2 | true | 0.2090 | 0.6641 | 0.6641 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | no_subtree_reuse | 2 | 2 | true | 0.2090 | 0.6641 | 0.6641 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | parent_q_fpu | 2 | 2 | true | 0.2090 | 0.4922 | 0.4922 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | normalized_values | 2 | 2 | true | 0.2090 | 0.8281 | 0.8281 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | deterministic_root_policy | 2 | 2 | true | 0.2090 | 0.6641 | 0.6641 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | low_cpuct | 2 | 2 | true | 0.2090 | 0.7578 | 0.7578 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | high_cpuct | 2 | 1 | false | 0.2090 | 0.3906 | 0.4375 | -0.0381 | false | u_favors_selected |
| corrected_replay_w4 | capture_available-007 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1717 | 0.6094 | 0.6094 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 128 | full_search_control | 2 | 2 | true | 0.2090 | 0.8750 | 0.8750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | baseline_eval_search | 2 | 2 | true | 0.2090 | 0.6927 | 0.6927 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2090 | 0.6927 | 0.6927 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | parent_q_fpu | 2 | 2 | true | 0.2090 | 0.6953 | 0.6953 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | normalized_values | 2 | 2 | true | 0.2090 | 0.8932 | 0.8932 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2090 | 0.6927 | 0.6927 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | low_cpuct | 2 | 2 | true | 0.2090 | 0.8880 | 0.8880 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | high_cpuct | 2 | 2 | true | 0.2090 | 0.6094 | 0.6094 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1717 | 0.6849 | 0.6849 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 384 | full_search_control | 2 | 2 | true | 0.2090 | 0.9062 | 0.9062 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2090 | 0.8725 | 0.8725 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2090 | 0.8725 | 0.8725 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2090 | 0.8633 | 0.8633 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | normalized_values | 2 | 2 | true | 0.2090 | 0.9483 | 0.9483 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2090 | 0.8725 | 0.8725 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | low_cpuct | 2 | 2 | true | 0.2090 | 0.9408 | 0.9408 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | high_cpuct | 2 | 2 | true | 0.2090 | 0.8142 | 0.8142 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1717 | 0.8667 | 0.8667 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-007 | 1200 | full_search_control | 2 | 2 | true | 0.2090 | 0.9383 | 0.9383 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | baseline_eval_search | 1 | 1 | true | 0.3992 | 0.7500 | 0.7500 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | no_subtree_reuse | 1 | 1 | true | 0.3992 | 0.7500 | 0.7500 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | parent_q_fpu | 1 | 1 | true | 0.3992 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | normalized_values | 1 | 1 | true | 0.3992 | 0.8906 | 0.8906 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | deterministic_root_policy | 1 | 1 | true | 0.3992 | 0.7500 | 0.7500 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | low_cpuct | 1 | 1 | true | 0.3992 | 0.8438 | 0.8438 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | high_cpuct | 1 | 1 | true | 0.3992 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | no_tactical_root_bias | 1 | 1 | true | 0.3791 | 0.7500 | 0.7500 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 64 | full_search_control | 1 | 1 | true | 0.3992 | 0.8750 | 0.8750 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | baseline_eval_search | 1 | 1 | true | 0.3992 | 0.7734 | 0.7734 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | no_subtree_reuse | 1 | 1 | true | 0.3992 | 0.7734 | 0.7734 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | parent_q_fpu | 1 | 1 | true | 0.3992 | 0.7812 | 0.7812 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | normalized_values | 1 | 1 | true | 0.3992 | 0.9297 | 0.9297 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | deterministic_root_policy | 1 | 1 | true | 0.3992 | 0.7734 | 0.7734 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | low_cpuct | 1 | 1 | true | 0.3992 | 0.8672 | 0.8672 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | high_cpuct | 1 | 1 | true | 0.3992 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | no_tactical_root_bias | 1 | 1 | true | 0.3791 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 128 | full_search_control | 1 | 1 | true | 0.3992 | 0.9375 | 0.9375 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | baseline_eval_search | 1 | 1 | true | 0.3992 | 0.8490 | 0.8490 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | no_subtree_reuse | 1 | 1 | true | 0.3992 | 0.8490 | 0.8490 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | parent_q_fpu | 1 | 1 | true | 0.3992 | 0.8516 | 0.8516 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | normalized_values | 1 | 1 | true | 0.3992 | 0.9531 | 0.9531 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | deterministic_root_policy | 1 | 1 | true | 0.3992 | 0.8490 | 0.8490 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | low_cpuct | 1 | 1 | true | 0.3992 | 0.8984 | 0.8984 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | high_cpuct | 1 | 1 | true | 0.3992 | 0.7786 | 0.7786 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | no_tactical_root_bias | 1 | 1 | true | 0.3791 | 0.8333 | 0.8333 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 384 | full_search_control | 1 | 1 | true | 0.3992 | 0.9661 | 0.9661 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | baseline_eval_search | 1 | 1 | true | 0.3992 | 0.9008 | 0.9008 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | no_subtree_reuse | 1 | 1 | true | 0.3992 | 0.9008 | 0.9008 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | parent_q_fpu | 1 | 1 | true | 0.3992 | 0.9042 | 0.9042 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | normalized_values | 1 | 1 | true | 0.3992 | 0.9650 | 0.9650 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | deterministic_root_policy | 1 | 1 | true | 0.3992 | 0.9008 | 0.9008 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | low_cpuct | 1 | 1 | true | 0.3992 | 0.9442 | 0.9442 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | high_cpuct | 1 | 1 | true | 0.3992 | 0.8783 | 0.8783 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | no_tactical_root_bias | 1 | 1 | true | 0.3791 | 0.8983 | 0.8983 | 0.0000 | true | pass_reference_selected |
| corrected_replay_w4 | capture_available-008 | 1200 | full_search_control | 1 | 1 | true | 0.3992 | 0.9700 | 0.9700 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | baseline_eval_search | 2 | 0 | false | 0.1450 | 0.1719 | 0.4219 | -0.0330 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | no_subtree_reuse | 2 | 0 | false | 0.1450 | 0.1719 | 0.4219 | -0.0330 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | parent_q_fpu | 2 | 0 | false | 0.1450 | 0.1719 | 0.4062 | -0.0355 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | normalized_values | 2 | 2 | true | 0.1450 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | deterministic_root_policy | 2 | 0 | false | 0.1450 | 0.1719 | 0.4219 | -0.0330 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | low_cpuct | 2 | 1 | false | 0.1450 | 0.1719 | 0.3750 | -0.0020 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | high_cpuct | 2 | 0 | false | 0.1450 | 0.1562 | 0.4375 | -0.0169 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | no_tactical_root_bias | 2 | 0 | false | 0.0885 | 0.0938 | 0.4688 | -0.0244 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 64 | full_search_control | 2 | 1 | false | 0.1450 | 0.1094 | 0.7656 | 0.0244 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | baseline_eval_search | 2 | 0 | false | 0.1450 | 0.1875 | 0.4297 | -0.0258 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | no_subtree_reuse | 2 | 0 | false | 0.1450 | 0.1875 | 0.4297 | -0.0258 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | parent_q_fpu | 2 | 0 | false | 0.1450 | 0.1875 | 0.4297 | -0.0246 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | normalized_values | 2 | 1 | false | 0.1450 | 0.3828 | 0.5391 | 0.0020 | false | selected_move_beats_reference_on_q_and_u |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | deterministic_root_policy | 2 | 0 | false | 0.1450 | 0.1875 | 0.4297 | -0.0258 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | low_cpuct | 2 | 0 | false | 0.1450 | 0.2188 | 0.3984 | -0.0316 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | high_cpuct | 2 | 0 | false | 0.1450 | 0.1719 | 0.4297 | -0.0313 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | no_tactical_root_bias | 2 | 0 | false | 0.0885 | 0.0938 | 0.4766 | -0.0082 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 128 | full_search_control | 2 | 1 | false | 0.1450 | 0.3438 | 0.5781 | -0.0055 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | baseline_eval_search | 2 | 1 | false | 0.1450 | 0.1797 | 0.4036 | 0.0015 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | no_subtree_reuse | 2 | 1 | false | 0.1450 | 0.1797 | 0.4036 | 0.0015 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | parent_q_fpu | 2 | 1 | false | 0.1450 | 0.1823 | 0.4062 | 0.0004 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | normalized_values | 2 | 1 | false | 0.1450 | 0.4375 | 0.4766 | -0.0067 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | deterministic_root_policy | 2 | 1 | false | 0.1450 | 0.1797 | 0.4036 | 0.0015 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | low_cpuct | 2 | 1 | false | 0.1450 | 0.1771 | 0.4583 | -0.0061 | false | policy_prior_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | high_cpuct | 2 | 1 | false | 0.1450 | 0.1745 | 0.3828 | 0.0017 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | no_tactical_root_bias | 2 | 1 | false | 0.0885 | 0.1146 | 0.4193 | 0.0019 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 384 | full_search_control | 2 | 2 | true | 0.1450 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | baseline_eval_search | 2 | 1 | false | 0.1450 | 0.3075 | 0.4017 | -0.0115 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | no_subtree_reuse | 2 | 1 | false | 0.1450 | 0.3075 | 0.4017 | -0.0115 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | parent_q_fpu | 2 | 1 | false | 0.1450 | 0.3183 | 0.3958 | -0.0120 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | normalized_values | 2 | 2 | true | 0.1450 | 0.8192 | 0.8192 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | deterministic_root_policy | 2 | 1 | false | 0.1450 | 0.3075 | 0.4017 | -0.0115 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | low_cpuct | 2 | 2 | true | 0.1450 | 0.4492 | 0.4492 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | high_cpuct | 2 | 1 | false | 0.1450 | 0.2575 | 0.3808 | -0.0130 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | no_tactical_root_bias | 2 | 1 | false | 0.0885 | 0.2258 | 0.4258 | -0.0131 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | 1200 | full_search_control | 2 | 2 | true | 0.1450 | 0.8783 | 0.8783 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | baseline_eval_search | 2 | 1 | false | 0.1514 | 0.2031 | 0.4844 | -0.0157 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1514 | 0.2031 | 0.4844 | -0.0157 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | parent_q_fpu | 2 | 1 | false | 0.1514 | 0.2031 | 0.4844 | -0.0212 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | normalized_values | 2 | 2 | true | 0.1514 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1514 | 0.2031 | 0.4844 | -0.0157 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | low_cpuct | 2 | 1 | false | 0.1514 | 0.2344 | 0.5156 | -0.0126 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | high_cpuct | 2 | 1 | false | 0.1514 | 0.1875 | 0.4531 | -0.0253 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.0968 | 0.1406 | 0.5000 | -0.0339 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 64 | full_search_control | 2 | 2 | true | 0.1514 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | baseline_eval_search | 2 | 1 | false | 0.1514 | 0.2188 | 0.4531 | -0.0303 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | no_subtree_reuse | 2 | 1 | false | 0.1514 | 0.2188 | 0.4531 | -0.0303 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | parent_q_fpu | 2 | 1 | false | 0.1514 | 0.2109 | 0.4688 | -0.0219 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | normalized_values | 2 | 2 | true | 0.1514 | 0.7812 | 0.7812 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | deterministic_root_policy | 2 | 1 | false | 0.1514 | 0.2188 | 0.4531 | -0.0303 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | low_cpuct | 2 | 1 | false | 0.1514 | 0.2734 | 0.5000 | -0.0077 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | high_cpuct | 2 | 1 | false | 0.1514 | 0.1953 | 0.4453 | -0.0273 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | no_tactical_root_bias | 2 | 1 | false | 0.0968 | 0.1484 | 0.4766 | -0.0258 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 128 | full_search_control | 2 | 2 | true | 0.1514 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | baseline_eval_search | 2 | 1 | false | 0.1514 | 0.2526 | 0.4453 | -0.0222 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | no_subtree_reuse | 2 | 1 | false | 0.1514 | 0.2526 | 0.4453 | -0.0222 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | parent_q_fpu | 2 | 1 | false | 0.1514 | 0.2578 | 0.4505 | -0.0221 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | normalized_values | 2 | 2 | true | 0.1514 | 0.8984 | 0.8984 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | deterministic_root_policy | 2 | 1 | false | 0.1514 | 0.2526 | 0.4453 | -0.0222 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | low_cpuct | 2 | 1 | false | 0.1514 | 0.3594 | 0.4167 | -0.0225 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | high_cpuct | 2 | 1 | false | 0.1514 | 0.2188 | 0.4453 | -0.0215 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | no_tactical_root_bias | 2 | 1 | false | 0.0968 | 0.1693 | 0.4922 | -0.0210 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 384 | full_search_control | 2 | 2 | true | 0.1514 | 0.8672 | 0.8672 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | baseline_eval_search | 2 | 1 | false | 0.1514 | 0.3517 | 0.4117 | -0.0212 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | no_subtree_reuse | 2 | 1 | false | 0.1514 | 0.3517 | 0.4117 | -0.0212 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | parent_q_fpu | 2 | 1 | false | 0.1514 | 0.3892 | 0.3967 | -0.0248 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | normalized_values | 2 | 2 | true | 0.1514 | 0.9508 | 0.9508 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | deterministic_root_policy | 2 | 1 | false | 0.1514 | 0.3517 | 0.4117 | -0.0212 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | low_cpuct | 2 | 2 | true | 0.1514 | 0.5467 | 0.5467 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | high_cpuct | 2 | 1 | false | 0.1514 | 0.2808 | 0.4417 | -0.0209 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | no_tactical_root_bias | 2 | 1 | false | 0.0968 | 0.2808 | 0.4583 | -0.0228 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | 1200 | full_search_control | 2 | 2 | true | 0.1514 | 0.9492 | 0.9492 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | baseline_eval_search | 2 | 1 | false | 0.1645 | 0.2188 | 0.3125 | -0.0572 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1645 | 0.2188 | 0.3125 | -0.0572 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | parent_q_fpu | 2 | 1 | false | 0.1645 | 0.2344 | 0.3125 | -0.0497 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | normalized_values | 2 | 2 | true | 0.1645 | 0.4688 | 0.4688 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1645 | 0.2188 | 0.3125 | -0.0572 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | low_cpuct | 2 | 1 | false | 0.1645 | 0.2656 | 0.3125 | -0.0463 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | high_cpuct | 2 | 1 | false | 0.1645 | 0.2031 | 0.3281 | -0.0528 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1139 | 0.1562 | 0.3281 | -0.0595 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 64 | full_search_control | 2 | 2 | true | 0.1645 | 0.8125 | 0.8125 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | baseline_eval_search | 2 | 0 | false | 0.1645 | 0.2891 | 0.2969 | -0.0755 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | no_subtree_reuse | 2 | 0 | false | 0.1645 | 0.2891 | 0.2969 | -0.0755 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | parent_q_fpu | 2 | 2 | true | 0.1645 | 0.2891 | 0.2891 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | normalized_values | 2 | 2 | true | 0.1645 | 0.7344 | 0.7344 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | deterministic_root_policy | 2 | 0 | false | 0.1645 | 0.2891 | 0.2969 | -0.0755 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | low_cpuct | 2 | 2 | true | 0.1645 | 0.4062 | 0.4062 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | high_cpuct | 2 | 0 | false | 0.1645 | 0.2422 | 0.3203 | -0.0740 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | no_tactical_root_bias | 2 | 0 | false | 0.1139 | 0.1797 | 0.3203 | -0.0637 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 128 | full_search_control | 2 | 2 | true | 0.1645 | 0.8984 | 0.8984 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | baseline_eval_search | 2 | 2 | true | 0.1645 | 0.3151 | 0.3151 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1645 | 0.3151 | 0.3151 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | parent_q_fpu | 2 | 2 | true | 0.1645 | 0.3099 | 0.3099 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | normalized_values | 2 | 2 | true | 0.1645 | 0.8958 | 0.8958 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1645 | 0.3151 | 0.3151 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | low_cpuct | 2 | 2 | true | 0.1645 | 0.4818 | 0.4818 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | high_cpuct | 2 | 0 | false | 0.1645 | 0.2370 | 0.3151 | -0.0460 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | no_tactical_root_bias | 2 | 1 | false | 0.1139 | 0.2396 | 0.2891 | -0.0447 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 384 | full_search_control | 2 | 2 | true | 0.1645 | 0.9193 | 0.9193 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1645 | 0.4625 | 0.4625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1645 | 0.4625 | 0.4625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1645 | 0.4250 | 0.4250 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | normalized_values | 2 | 2 | true | 0.1645 | 0.9500 | 0.9500 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1645 | 0.4625 | 0.4625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | low_cpuct | 2 | 2 | true | 0.1645 | 0.6375 | 0.6375 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | high_cpuct | 2 | 2 | true | 0.1645 | 0.3492 | 0.3492 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1139 | 0.3733 | 0.3733 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-006 | 1200 | full_search_control | 2 | 2 | true | 0.1645 | 0.9567 | 0.9567 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | baseline_eval_search | 2 | 1 | false | 0.1856 | 0.2500 | 0.4219 | -0.0246 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1856 | 0.2500 | 0.4219 | -0.0246 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | parent_q_fpu | 2 | 1 | false | 0.1856 | 0.2500 | 0.4219 | -0.0194 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | normalized_values | 2 | 2 | true | 0.1856 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1856 | 0.2500 | 0.4219 | -0.0246 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | low_cpuct | 2 | 1 | false | 0.1856 | 0.2812 | 0.4219 | -0.0209 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | high_cpuct | 2 | 1 | false | 0.1856 | 0.2344 | 0.4062 | -0.0218 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1413 | 0.2031 | 0.4375 | -0.0292 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 64 | full_search_control | 2 | 2 | true | 0.1856 | 0.7031 | 0.7031 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | baseline_eval_search | 2 | 1 | false | 0.1856 | 0.2500 | 0.4688 | -0.0073 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | no_subtree_reuse | 2 | 1 | false | 0.1856 | 0.2500 | 0.4688 | -0.0073 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | parent_q_fpu | 2 | 1 | false | 0.1856 | 0.2500 | 0.4688 | -0.0084 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | normalized_values | 2 | 2 | true | 0.1856 | 0.6797 | 0.6797 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | deterministic_root_policy | 2 | 1 | false | 0.1856 | 0.2500 | 0.4688 | -0.0073 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | low_cpuct | 2 | 1 | false | 0.1856 | 0.2969 | 0.4922 | -0.0085 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | high_cpuct | 2 | 1 | false | 0.1856 | 0.2344 | 0.4219 | -0.0159 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | no_tactical_root_bias | 2 | 1 | false | 0.1413 | 0.1719 | 0.5000 | 0.0005 | false | q_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 128 | full_search_control | 2 | 2 | true | 0.1856 | 0.5547 | 0.5547 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | baseline_eval_search | 2 | 1 | false | 0.1856 | 0.2760 | 0.4089 | -0.0173 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | no_subtree_reuse | 2 | 1 | false | 0.1856 | 0.2760 | 0.4089 | -0.0173 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | parent_q_fpu | 2 | 1 | false | 0.1856 | 0.2734 | 0.4141 | -0.0150 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | normalized_values | 2 | 2 | true | 0.1856 | 0.8542 | 0.8542 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | deterministic_root_policy | 2 | 1 | false | 0.1856 | 0.2760 | 0.4089 | -0.0173 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | low_cpuct | 2 | 2 | true | 0.1856 | 0.4219 | 0.4219 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | high_cpuct | 2 | 1 | false | 0.1856 | 0.2240 | 0.4193 | -0.0075 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | no_tactical_root_bias | 2 | 1 | false | 0.1413 | 0.2109 | 0.4453 | -0.0119 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 384 | full_search_control | 2 | 2 | true | 0.1856 | 0.7734 | 0.7734 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1856 | 0.4167 | 0.4167 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1856 | 0.4167 | 0.4167 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1856 | 0.4242 | 0.4242 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | normalized_values | 2 | 2 | true | 0.1856 | 0.9392 | 0.9392 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1856 | 0.4167 | 0.4167 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | low_cpuct | 2 | 2 | true | 0.1856 | 0.6217 | 0.6217 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | high_cpuct | 2 | 1 | false | 0.1856 | 0.3333 | 0.4233 | -0.0166 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | no_tactical_root_bias | 2 | 1 | false | 0.1413 | 0.3575 | 0.4150 | -0.0218 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-007 | 1200 | full_search_control | 2 | 2 | true | 0.1856 | 0.9117 | 0.9117 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | baseline_eval_search | 1 | 1 | true | 0.4093 | 0.5156 | 0.5156 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | no_subtree_reuse | 1 | 1 | true | 0.4093 | 0.5156 | 0.5156 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | parent_q_fpu | 1 | 1 | true | 0.4093 | 0.5156 | 0.5156 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | normalized_values | 1 | 1 | true | 0.4093 | 0.9375 | 0.9375 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | deterministic_root_policy | 1 | 1 | true | 0.4093 | 0.5156 | 0.5156 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | low_cpuct | 1 | 1 | true | 0.4093 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | high_cpuct | 1 | 1 | true | 0.4093 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | no_tactical_root_bias | 1 | 1 | true | 0.3912 | 0.4844 | 0.4844 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 64 | full_search_control | 1 | 1 | true | 0.4093 | 0.8281 | 0.8281 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | baseline_eval_search | 1 | 1 | true | 0.4093 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | no_subtree_reuse | 1 | 1 | true | 0.4093 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | parent_q_fpu | 1 | 1 | true | 0.4093 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | normalized_values | 1 | 1 | true | 0.4093 | 0.9062 | 0.9062 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | deterministic_root_policy | 1 | 1 | true | 0.4093 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | low_cpuct | 1 | 1 | true | 0.4093 | 0.6016 | 0.6016 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | high_cpuct | 1 | 1 | true | 0.4093 | 0.4922 | 0.4922 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | no_tactical_root_bias | 1 | 1 | true | 0.3912 | 0.5078 | 0.5078 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 128 | full_search_control | 1 | 1 | true | 0.4093 | 0.9062 | 0.9062 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | baseline_eval_search | 1 | 1 | true | 0.4093 | 0.6094 | 0.6094 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | no_subtree_reuse | 1 | 1 | true | 0.4093 | 0.6094 | 0.6094 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | parent_q_fpu | 1 | 1 | true | 0.4093 | 0.6146 | 0.6146 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | normalized_values | 1 | 1 | true | 0.4093 | 0.9453 | 0.9453 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | deterministic_root_policy | 1 | 1 | true | 0.4093 | 0.6094 | 0.6094 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | low_cpuct | 1 | 1 | true | 0.4093 | 0.6901 | 0.6901 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | high_cpuct | 1 | 1 | true | 0.4093 | 0.5573 | 0.5573 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | no_tactical_root_bias | 1 | 1 | true | 0.3912 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 384 | full_search_control | 1 | 1 | true | 0.4093 | 0.9479 | 0.9479 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | baseline_eval_search | 1 | 1 | true | 0.4093 | 0.7033 | 0.7033 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | no_subtree_reuse | 1 | 1 | true | 0.4093 | 0.7033 | 0.7033 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | parent_q_fpu | 1 | 1 | true | 0.4093 | 0.7200 | 0.7200 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | normalized_values | 1 | 1 | true | 0.4093 | 0.9625 | 0.9625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | deterministic_root_policy | 1 | 1 | true | 0.4093 | 0.7033 | 0.7033 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | low_cpuct | 1 | 1 | true | 0.4093 | 0.7992 | 0.7992 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | high_cpuct | 1 | 1 | true | 0.4093 | 0.6567 | 0.6567 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | no_tactical_root_bias | 1 | 1 | true | 0.3912 | 0.6817 | 0.6817 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w1 | capture_available-008 | 1200 | full_search_control | 1 | 1 | true | 0.4093 | 0.9717 | 0.9717 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | baseline_eval_search | 2 | 1 | false | 0.1737 | 0.2344 | 0.3438 | -0.0356 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1737 | 0.2344 | 0.3438 | -0.0356 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | parent_q_fpu | 2 | 1 | false | 0.1737 | 0.2344 | 0.3438 | -0.0362 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | normalized_values | 2 | 2 | true | 0.1737 | 0.5000 | 0.5000 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1737 | 0.2344 | 0.3438 | -0.0356 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | low_cpuct | 2 | 1 | false | 0.1737 | 0.3125 | 0.3281 | -0.0410 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | high_cpuct | 2 | 1 | false | 0.1737 | 0.2031 | 0.3594 | -0.0282 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1258 | 0.1406 | 0.3750 | 0.0016 | false | selected_move_beats_reference_on_q_and_u |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 64 | full_search_control | 2 | 2 | true | 0.1737 | 0.5312 | 0.5312 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | baseline_eval_search | 2 | 0 | false | 0.1737 | 0.2578 | 0.3203 | -0.0558 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | no_subtree_reuse | 2 | 0 | false | 0.1737 | 0.2578 | 0.3203 | -0.0558 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | parent_q_fpu | 2 | 0 | false | 0.1737 | 0.2578 | 0.3203 | -0.0551 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | normalized_values | 2 | 2 | true | 0.1737 | 0.7266 | 0.7266 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | deterministic_root_policy | 2 | 0 | false | 0.1737 | 0.2578 | 0.3203 | -0.0558 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | low_cpuct | 2 | 2 | true | 0.1737 | 0.3047 | 0.3047 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | high_cpuct | 2 | 0 | false | 0.1737 | 0.2266 | 0.3438 | -0.0535 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | no_tactical_root_bias | 2 | 0 | false | 0.1258 | 0.1953 | 0.3516 | -0.0536 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 128 | full_search_control | 2 | 2 | true | 0.1737 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | baseline_eval_search | 2 | 1 | false | 0.1737 | 0.2578 | 0.3750 | -0.0159 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | no_subtree_reuse | 2 | 1 | false | 0.1737 | 0.2578 | 0.3750 | -0.0159 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | parent_q_fpu | 2 | 1 | false | 0.1737 | 0.2448 | 0.3828 | -0.0136 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | normalized_values | 2 | 2 | true | 0.1737 | 0.8438 | 0.8438 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | deterministic_root_policy | 2 | 1 | false | 0.1737 | 0.2578 | 0.3750 | -0.0159 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | low_cpuct | 2 | 2 | true | 0.1737 | 0.3490 | 0.3490 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | high_cpuct | 2 | 1 | false | 0.1737 | 0.2318 | 0.3724 | -0.0150 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | no_tactical_root_bias | 2 | 1 | false | 0.1258 | 0.2083 | 0.3802 | -0.0175 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 384 | full_search_control | 2 | 2 | true | 0.1737 | 0.8307 | 0.8307 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1737 | 0.4075 | 0.4075 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1737 | 0.4075 | 0.4075 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1737 | 0.4083 | 0.4083 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | normalized_values | 2 | 2 | true | 0.1737 | 0.9408 | 0.9408 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1737 | 0.4075 | 0.4075 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | low_cpuct | 2 | 2 | true | 0.1737 | 0.5267 | 0.5267 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | high_cpuct | 2 | 1 | false | 0.1737 | 0.3158 | 0.3608 | -0.0209 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1258 | 0.3617 | 0.3617 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | 1200 | full_search_control | 2 | 2 | true | 0.1737 | 0.9425 | 0.9425 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | baseline_eval_search | 2 | 1 | false | 0.1680 | 0.2031 | 0.5312 | -0.0135 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1680 | 0.2031 | 0.5312 | -0.0135 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | parent_q_fpu | 2 | 1 | false | 0.1680 | 0.2188 | 0.5156 | -0.0250 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | normalized_values | 2 | 2 | true | 0.1680 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1680 | 0.2031 | 0.5312 | -0.0135 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | low_cpuct | 2 | 1 | false | 0.1680 | 0.2344 | 0.5312 | -0.0159 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | high_cpuct | 2 | 1 | false | 0.1680 | 0.2031 | 0.5156 | -0.0148 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1184 | 0.1562 | 0.5469 | -0.0160 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 64 | full_search_control | 2 | 2 | true | 0.1680 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | baseline_eval_search | 2 | 1 | false | 0.1680 | 0.2578 | 0.4609 | -0.0387 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | no_subtree_reuse | 2 | 1 | false | 0.1680 | 0.2578 | 0.4609 | -0.0387 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | parent_q_fpu | 2 | 1 | false | 0.1680 | 0.2188 | 0.5000 | -0.0121 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | normalized_values | 2 | 2 | true | 0.1680 | 0.7422 | 0.7422 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | deterministic_root_policy | 2 | 1 | false | 0.1680 | 0.2578 | 0.4609 | -0.0387 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | low_cpuct | 2 | 1 | false | 0.1680 | 0.2969 | 0.4609 | -0.0185 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | high_cpuct | 2 | 1 | false | 0.1680 | 0.2266 | 0.4688 | -0.0395 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | no_tactical_root_bias | 2 | 1 | false | 0.1184 | 0.1797 | 0.5234 | -0.0283 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 128 | full_search_control | 2 | 2 | true | 0.1680 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | baseline_eval_search | 2 | 1 | false | 0.1680 | 0.3125 | 0.4323 | -0.0338 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | no_subtree_reuse | 2 | 1 | false | 0.1680 | 0.3125 | 0.4323 | -0.0338 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | parent_q_fpu | 2 | 1 | false | 0.1680 | 0.3099 | 0.4323 | -0.0334 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | normalized_values | 2 | 2 | true | 0.1680 | 0.8724 | 0.8724 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | deterministic_root_policy | 2 | 1 | false | 0.1680 | 0.3125 | 0.4323 | -0.0338 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | low_cpuct | 2 | 2 | true | 0.1680 | 0.4297 | 0.4297 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | high_cpuct | 2 | 1 | false | 0.1680 | 0.2760 | 0.4349 | -0.0397 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | no_tactical_root_bias | 2 | 1 | false | 0.1184 | 0.2760 | 0.4688 | -0.0392 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 384 | full_search_control | 2 | 2 | true | 0.1680 | 0.8568 | 0.8568 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1680 | 0.4050 | 0.4050 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1680 | 0.4050 | 0.4050 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1680 | 0.4142 | 0.4142 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | normalized_values | 2 | 2 | true | 0.1680 | 0.9233 | 0.9233 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1680 | 0.4050 | 0.4050 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | low_cpuct | 2 | 2 | true | 0.1680 | 0.5542 | 0.5542 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | high_cpuct | 2 | 1 | false | 0.1680 | 0.3033 | 0.4475 | -0.0234 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | no_tactical_root_bias | 2 | 1 | false | 0.1184 | 0.3225 | 0.4633 | -0.0267 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | 1200 | full_search_control | 2 | 2 | true | 0.1680 | 0.9133 | 0.9133 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | baseline_eval_search | 2 | 1 | false | 0.1835 | 0.2500 | 0.3125 | -0.0679 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | no_subtree_reuse | 2 | 1 | false | 0.1835 | 0.2500 | 0.3125 | -0.0679 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | parent_q_fpu | 2 | 1 | false | 0.1835 | 0.2500 | 0.3125 | -0.0703 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | normalized_values | 2 | 2 | true | 0.1835 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | deterministic_root_policy | 2 | 1 | false | 0.1835 | 0.2500 | 0.3125 | -0.0679 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | low_cpuct | 2 | 2 | true | 0.1835 | 0.2969 | 0.2969 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | high_cpuct | 2 | 1 | false | 0.1835 | 0.2344 | 0.3281 | -0.0674 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1385 | 0.2031 | 0.3125 | -0.0688 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 64 | full_search_control | 2 | 2 | true | 0.1835 | 0.6719 | 0.6719 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | baseline_eval_search | 2 | 2 | true | 0.1835 | 0.3438 | 0.3438 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | no_subtree_reuse | 2 | 2 | true | 0.1835 | 0.3438 | 0.3438 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | parent_q_fpu | 2 | 2 | true | 0.1835 | 0.3359 | 0.3359 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | normalized_values | 2 | 2 | true | 0.1835 | 0.8359 | 0.8359 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | deterministic_root_policy | 2 | 2 | true | 0.1835 | 0.3438 | 0.3438 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | low_cpuct | 2 | 2 | true | 0.1835 | 0.4453 | 0.4453 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | high_cpuct | 2 | 2 | true | 0.1835 | 0.2812 | 0.2812 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | no_tactical_root_bias | 2 | 2 | true | 0.1385 | 0.2891 | 0.2891 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 128 | full_search_control | 2 | 2 | true | 0.1835 | 0.8359 | 0.8359 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | baseline_eval_search | 2 | 2 | true | 0.1835 | 0.3828 | 0.3828 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | no_subtree_reuse | 2 | 2 | true | 0.1835 | 0.3828 | 0.3828 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | parent_q_fpu | 2 | 2 | true | 0.1835 | 0.3594 | 0.3594 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | normalized_values | 2 | 2 | true | 0.1835 | 0.8802 | 0.8802 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | deterministic_root_policy | 2 | 2 | true | 0.1835 | 0.3828 | 0.3828 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | low_cpuct | 2 | 2 | true | 0.1835 | 0.5052 | 0.5052 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | high_cpuct | 2 | 2 | true | 0.1835 | 0.3099 | 0.3099 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | no_tactical_root_bias | 2 | 2 | true | 0.1385 | 0.3099 | 0.3099 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 384 | full_search_control | 2 | 2 | true | 0.1835 | 0.8932 | 0.8932 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | baseline_eval_search | 2 | 2 | true | 0.1835 | 0.5408 | 0.5408 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.1835 | 0.5408 | 0.5408 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | parent_q_fpu | 2 | 2 | true | 0.1835 | 0.5383 | 0.5383 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | normalized_values | 2 | 2 | true | 0.1835 | 0.9417 | 0.9417 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.1835 | 0.5408 | 0.5408 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | low_cpuct | 2 | 2 | true | 0.1835 | 0.6975 | 0.6975 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | high_cpuct | 2 | 2 | true | 0.1835 | 0.3950 | 0.3950 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1385 | 0.4983 | 0.4983 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-006 | 1200 | full_search_control | 2 | 2 | true | 0.1835 | 0.9467 | 0.9467 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | baseline_eval_search | 2 | 1 | false | 0.2142 | 0.2812 | 0.4375 | -0.0176 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | no_subtree_reuse | 2 | 1 | false | 0.2142 | 0.2812 | 0.4375 | -0.0176 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | parent_q_fpu | 2 | 1 | false | 0.2142 | 0.2812 | 0.4375 | -0.0190 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | normalized_values | 2 | 2 | true | 0.2142 | 0.6562 | 0.6562 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | deterministic_root_policy | 2 | 1 | false | 0.2142 | 0.2812 | 0.4375 | -0.0176 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | low_cpuct | 2 | 1 | false | 0.2142 | 0.3281 | 0.4219 | -0.0308 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | high_cpuct | 2 | 1 | false | 0.2142 | 0.2656 | 0.4375 | -0.0204 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | no_tactical_root_bias | 2 | 1 | false | 0.1784 | 0.2344 | 0.4375 | -0.0275 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 64 | full_search_control | 2 | 2 | true | 0.2142 | 0.6875 | 0.6875 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | baseline_eval_search | 2 | 1 | false | 0.2142 | 0.2891 | 0.4453 | -0.0175 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | no_subtree_reuse | 2 | 1 | false | 0.2142 | 0.2891 | 0.4453 | -0.0175 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | parent_q_fpu | 2 | 1 | false | 0.2142 | 0.2969 | 0.4453 | -0.0158 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | normalized_values | 2 | 2 | true | 0.2142 | 0.7656 | 0.7656 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | deterministic_root_policy | 2 | 1 | false | 0.2142 | 0.2891 | 0.4453 | -0.0175 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | low_cpuct | 2 | 1 | false | 0.2142 | 0.3594 | 0.4688 | -0.0117 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | high_cpuct | 2 | 1 | false | 0.2142 | 0.2812 | 0.4375 | -0.0164 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | no_tactical_root_bias | 2 | 1 | false | 0.1784 | 0.2578 | 0.4609 | -0.0257 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 128 | full_search_control | 2 | 2 | true | 0.2142 | 0.7578 | 0.7578 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | baseline_eval_search | 2 | 2 | true | 0.2142 | 0.3854 | 0.3854 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | no_subtree_reuse | 2 | 2 | true | 0.2142 | 0.3854 | 0.3854 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | parent_q_fpu | 2 | 2 | true | 0.2142 | 0.3802 | 0.3802 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | normalized_values | 2 | 2 | true | 0.2142 | 0.8880 | 0.8880 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | deterministic_root_policy | 2 | 2 | true | 0.2142 | 0.3854 | 0.3854 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | low_cpuct | 2 | 2 | true | 0.2142 | 0.5234 | 0.5234 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | high_cpuct | 2 | 1 | false | 0.2142 | 0.3177 | 0.3750 | -0.0342 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | no_tactical_root_bias | 2 | 1 | false | 0.1784 | 0.3568 | 0.3828 | -0.0387 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 384 | full_search_control | 2 | 2 | true | 0.2142 | 0.8229 | 0.8229 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | baseline_eval_search | 2 | 2 | true | 0.2142 | 0.4708 | 0.4708 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | no_subtree_reuse | 2 | 2 | true | 0.2142 | 0.4708 | 0.4708 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | parent_q_fpu | 2 | 2 | true | 0.2142 | 0.4733 | 0.4733 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | normalized_values | 2 | 2 | true | 0.2142 | 0.9475 | 0.9475 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | deterministic_root_policy | 2 | 2 | true | 0.2142 | 0.4708 | 0.4708 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | low_cpuct | 2 | 2 | true | 0.2142 | 0.6833 | 0.6833 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | high_cpuct | 2 | 1 | false | 0.2142 | 0.3717 | 0.4017 | -0.0201 | false | u_favors_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | no_tactical_root_bias | 2 | 2 | true | 0.1784 | 0.4392 | 0.4392 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-007 | 1200 | full_search_control | 2 | 2 | true | 0.2142 | 0.9333 | 0.9333 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | baseline_eval_search | 1 | 1 | true | 0.4650 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | no_subtree_reuse | 1 | 1 | true | 0.4650 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | parent_q_fpu | 1 | 1 | true | 0.4650 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | normalized_values | 1 | 1 | true | 0.4650 | 0.8594 | 0.8594 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | deterministic_root_policy | 1 | 1 | true | 0.4650 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | low_cpuct | 1 | 1 | true | 0.4650 | 0.6406 | 0.6406 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | high_cpuct | 1 | 1 | true | 0.4650 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | no_tactical_root_bias | 1 | 1 | true | 0.4580 | 0.5781 | 0.5781 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 64 | full_search_control | 1 | 1 | true | 0.4650 | 0.8750 | 0.8750 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | baseline_eval_search | 1 | 1 | true | 0.4650 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | no_subtree_reuse | 1 | 1 | true | 0.4650 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | parent_q_fpu | 1 | 1 | true | 0.4650 | 0.5625 | 0.5625 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | normalized_values | 1 | 1 | true | 0.4650 | 0.9219 | 0.9219 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | deterministic_root_policy | 1 | 1 | true | 0.4650 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | low_cpuct | 1 | 1 | true | 0.4650 | 0.6484 | 0.6484 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | high_cpuct | 1 | 1 | true | 0.4650 | 0.5391 | 0.5391 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | no_tactical_root_bias | 1 | 1 | true | 0.4580 | 0.5703 | 0.5703 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 128 | full_search_control | 1 | 1 | true | 0.4650 | 0.9219 | 0.9219 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | baseline_eval_search | 1 | 1 | true | 0.4650 | 0.6224 | 0.6224 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | no_subtree_reuse | 1 | 1 | true | 0.4650 | 0.6224 | 0.6224 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | parent_q_fpu | 1 | 1 | true | 0.4650 | 0.6354 | 0.6354 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | normalized_values | 1 | 1 | true | 0.4650 | 0.9349 | 0.9349 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | deterministic_root_policy | 1 | 1 | true | 0.4650 | 0.6224 | 0.6224 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | low_cpuct | 1 | 1 | true | 0.4650 | 0.7188 | 0.7188 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | high_cpuct | 1 | 1 | true | 0.4650 | 0.5859 | 0.5859 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | no_tactical_root_bias | 1 | 1 | true | 0.4580 | 0.6172 | 0.6172 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 384 | full_search_control | 1 | 1 | true | 0.4650 | 0.8594 | 0.8594 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | baseline_eval_search | 1 | 1 | true | 0.4650 | 0.7225 | 0.7225 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | no_subtree_reuse | 1 | 1 | true | 0.4650 | 0.7225 | 0.7225 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | parent_q_fpu | 1 | 1 | true | 0.4650 | 0.7342 | 0.7342 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | normalized_values | 1 | 1 | true | 0.4650 | 0.9692 | 0.9692 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | deterministic_root_policy | 1 | 1 | true | 0.4650 | 0.7225 | 0.7225 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | low_cpuct | 1 | 1 | true | 0.4650 | 0.8067 | 0.8067 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | high_cpuct | 1 | 1 | true | 0.4650 | 0.6708 | 0.6708 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | no_tactical_root_bias | 1 | 1 | true | 0.4580 | 0.7100 | 0.7100 | 0.0000 | true | pass_reference_selected |
| opening_extra_turn_overbias_corrected_w2 | capture_available-008 | 1200 | full_search_control | 1 | 1 | true | 0.4650 | 0.9458 | 0.9458 | 0.0000 | true | pass_reference_selected |

## 4. Search-setting matrix

- `baseline_eval_search`: current `build_eval_search_options()` defaults.
- `no_subtree_reuse`: explicit `reuse_subtree=false`.
- `parent_q_fpu`: `fpu_mode=parent_q`.
- `normalized_values`: `normalize_values=true`.
- `deterministic_root_policy`: explicit deterministic root tie-break.
- `low_cpuct`: `c_puct=0.75`.
- `high_cpuct`: `c_puct=1.75`.
- `no_tactical_root_bias`: `tactical_root_bias=0.0`.
- `full_search_control`: `parent_q` FPU, subtree reuse, normalized values, deterministic root, tactical bias `0.1`.

## 5. Current vs replay-candidate guard comparison

| row_id | current_selected | current_pass | candidate | candidate_selected | candidate_pass | classification | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | 2 | true | corrected_replay_w1 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-003 | 2 | true | corrected_replay_w1 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-006 | 2 | true | corrected_replay_w1 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-007 | 2 | true | corrected_replay_w1 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-008 | 1 | true | corrected_replay_w1 | 1 | true | `current_pass_candidate_pass` | 1200 selected=1 pass=true |
| capture_available-002 | 2 | true | corrected_replay_w2 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-003 | 2 | true | corrected_replay_w2 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-006 | 2 | true | corrected_replay_w2 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-007 | 2 | true | corrected_replay_w2 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-008 | 1 | true | corrected_replay_w2 | 1 | true | `current_pass_candidate_pass` | 1200 selected=1 pass=true |
| capture_available-002 | 2 | true | corrected_replay_w4 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-003 | 2 | true | corrected_replay_w4 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-006 | 2 | true | corrected_replay_w4 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-007 | 2 | true | corrected_replay_w4 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-008 | 1 | true | corrected_replay_w4 | 1 | true | `current_pass_candidate_pass` | 1200 selected=1 pass=true |
| capture_available-002 | 2 | true | opening_extra_turn_overbias_corrected_w1 | 1 | false | `current_pass_candidate_fail` | 1200 selected=1 pass=false |
| capture_available-003 | 2 | true | opening_extra_turn_overbias_corrected_w1 | 1 | false | `current_pass_candidate_fail` | 1200 selected=1 pass=false |
| capture_available-006 | 2 | true | opening_extra_turn_overbias_corrected_w1 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-007 | 2 | true | opening_extra_turn_overbias_corrected_w1 | 1 | false | `current_pass_candidate_fail` | 1200 selected=2 pass=true |
| capture_available-008 | 1 | true | opening_extra_turn_overbias_corrected_w1 | 1 | true | `current_pass_candidate_pass` | 1200 selected=1 pass=true |
| capture_available-002 | 2 | true | opening_extra_turn_overbias_corrected_w2 | 1 | false | `current_pass_candidate_fail` | 1200 selected=2 pass=true |
| capture_available-003 | 2 | true | opening_extra_turn_overbias_corrected_w2 | 1 | false | `current_pass_candidate_fail` | 1200 selected=2 pass=true |
| capture_available-006 | 2 | true | opening_extra_turn_overbias_corrected_w2 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-007 | 2 | true | opening_extra_turn_overbias_corrected_w2 | 2 | true | `current_pass_candidate_pass` | 1200 selected=2 pass=true |
| capture_available-008 | 1 | true | opening_extra_turn_overbias_corrected_w2 | 1 | true | `current_pass_candidate_pass` | 1200 selected=1 pass=true |

## 6. Selection-pressure trace

| artifact | row_id | search_setting | budget | corrected_reference_move | selected_move | p_reference | p_selected | q_reference | q_selected | u_reference | u_selected | n_reference | n_selected | score_margin_selected_minus_reference | diagnosis |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | baseline_eval_search | 384 | 2 | 1 | 0.1450 | 0.3145 | 0.0063 | 0.0078 | 0.0507 | 0.0494 | 69 | 155 | 0.0001 | q_supports_selected_move |
| opening_extra_turn_overbias_corrected_w1 | capture_available-002 | baseline_eval_search | 1200 | 2 | 1 | 0.1450 | 0.3145 | 0.0091 | -0.0024 | 0.0170 | 0.0282 | 369 | 482 | -0.0003 | failure_disappears_under_low_cpuct |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | baseline_eval_search | 384 | 2 | 1 | 0.1514 | 0.4209 | 0.0629 | 0.0407 | 0.0378 | 0.0599 | 97 | 171 | -0.0000 | u_or_prior_pressure_explains_selection |
| opening_extra_turn_overbias_corrected_w1 | capture_available-003 | baseline_eval_search | 1200 | 2 | 1 | 0.1514 | 0.4209 | 0.0469 | 0.0257 | 0.0155 | 0.0368 | 422 | 494 | 0.0001 | failure_disappears_under_low_cpuct |
| opening_extra_turn_overbias_corrected_w2 | capture_available-002 | baseline_eval_search | 384 | 2 | 1 | 0.1737 | 0.3406 | 0.0306 | 0.0147 | 0.0425 | 0.0575 | 99 | 144 | -0.0009 | failure_disappears_under_low_cpuct |
| opening_extra_turn_overbias_corrected_w2 | capture_available-003 | baseline_eval_search | 384 | 2 | 1 | 0.1680 | 0.4626 | 0.0766 | 0.0428 | 0.0340 | 0.0678 | 120 | 166 | 0.0000 | failure_disappears_under_low_cpuct |

## 7. Guard-specific diagnostic interventions

- Diagnostic interventions were only run on rows that still failed corrected reference at `384` or `1200` simulations.
- `equal_initialize_q` was approximated with `fpu_mode=zero`, which equalizes root FPU initialization across unvisited children without changing production code.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `384` intervention `equalize_priors`: selected `1`, pass `false`, reason `selected_move_not_reference`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `384` intervention `zero_tactical_root_bias`: selected `1`, pass `false`, reason `q_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `384` intervention `clamp_selected_prior_to_reference`: selected `0`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `384` intervention `equal_initialize_q`: selected `1`, pass `false`, reason `q_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `384` intervention `uniform_legal_prior`: selected `1`, pass `false`, reason `q_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `1200` intervention `equalize_priors`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `1200` intervention `zero_tactical_root_bias`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `1200` intervention `clamp_selected_prior_to_reference`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `1200` intervention `equal_initialize_q`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-002` budget `1200` intervention `uniform_legal_prior`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `384` intervention `equalize_priors`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `384` intervention `zero_tactical_root_bias`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `384` intervention `clamp_selected_prior_to_reference`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `384` intervention `equal_initialize_q`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `384` intervention `uniform_legal_prior`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `1200` intervention `equalize_priors`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `1200` intervention `zero_tactical_root_bias`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `1200` intervention `clamp_selected_prior_to_reference`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `1200` intervention `equal_initialize_q`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w1` `capture_available-003` budget `1200` intervention `uniform_legal_prior`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-002` budget `384` intervention `equalize_priors`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-002` budget `384` intervention `zero_tactical_root_bias`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-002` budget `384` intervention `clamp_selected_prior_to_reference`: selected `0`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-002` budget `384` intervention `equal_initialize_q`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-002` budget `384` intervention `uniform_legal_prior`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-003` budget `384` intervention `equalize_priors`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-003` budget `384` intervention `zero_tactical_root_bias`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-003` budget `384` intervention `clamp_selected_prior_to_reference`: selected `2`, pass `true`, reason `pass_reference_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-003` budget `384` intervention `equal_initialize_q`: selected `1`, pass `false`, reason `u_favors_selected`.
- `opening_extra_turn_overbias_corrected_w2` `capture_available-003` budget `384` intervention `uniform_legal_prior`: selected `2`, pass `true`, reason `pass_reference_selected`.

## 8. Opening-subfamily compatibility check

| subfamily | rows_sampled | baseline_pass_rate | best_guard_setting_pass_rate | regressions | notes |
| --- | --- | --- | --- | --- | --- |
| opening_extra_turn_overbias | 3 | 0.0000 | 0.3333 | `[]` | helps |
| opening_edge_move_5_preference | 3 | 0.0000 | 0.3333 | `[]` | helps |
| opening_missed_extra_turn_continuation | 3 | 0.0000 | 0.0000 | `[]` | neutral |

## 9. Interpretation

- Best 002/003 search setting by replay-row pass count: `low_cpuct`.
- Corrected-guard decision: `replay_induced_guard_regression`.
- The observed regression is specific to `opening_extra_turn_overbias_corrected_w1/w2`; corrected hard-state replay `w1/w2/w4` still pass the `1200`-budget guard rows.
- Recommended next action is driven by the explicit decision rules and the current-vs-candidate comparison.

## 10. Exactly one recommended next action

Recommendation: **abandon corrected-reference replay branches and rebuild candidate datasets without rows that perturb corrected guards.**.
