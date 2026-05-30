# AlphaZero-lite Root-Prior Transform Rollout Validation Results

## Context

This run evaluates whether the narrow root-prior transform is a robust PUCT search-time improvement or only a local diagnostic patch.

- transform: `seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5`
- current artifact: `storage/ai/alphazero_lite/current`
- candidate artifact: `aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- output summary: `<out-root>/root_prior_transform_rollout_validation_summary.json`
- arena supports side-specific transforms: `true`

## Matrix Definition

- Matrix A comparisons: challenger/current baseline, challenger transform vs current baseline, current transform vs current baseline, challenger transform vs current transform.
- Matrix B budgets: equal low 256/256, equal high 640/640, superhuman-style 640/256.
- fair-budget design improvement: paired random opening prefixes with `4` plies on equal-budget rows.
- seeds: `11, 23, 37`

## Local Guard Results

| artifact | transform | row_id | simulations | selected_move | reference_move | selected_is_reference | reference_visit_share | pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | None | capture_available-002 | 384 | 2 | 4 | false | 0.0026 | false | selected_non_reference |
| current | None | capture_available-002 | 1200 | 2 | 4 | false | 0.0008 | false | selected_non_reference |
| current | None | capture_available-003 | 384 | 1 | 1 | true | 0.9766 | true | ok |
| current | None | capture_available-003 | 1200 | 1 | 1 | true | 0.9883 | true | ok |
| current | None | capture_available-005 | 384 | 1 | 4 | false | 0.0078 | true | reported_only_not_hard_fail,selected_non_reference |
| current | None | capture_available-005 | 1200 | 1 | 4 | false | 0.0025 | true | reported_only_not_hard_fail,selected_non_reference |
| current | None | capture_available-006 | 384 | 2 | 2 | true | 0.8542 | true | ok |
| current | None | capture_available-006 | 1200 | 2 | 2 | true | 0.93 | true | ok |
| current | None | capture_available-007 | 384 | 1 | 1 | true | 0.9688 | true | ok |
| current | None | capture_available-007 | 1200 | 1 | 1 | true | 0.9725 | true | ok |
| current | None | capture_available-008 | 384 | 1 | 1 | true | 0.8932 | true | ok |
| current | None | capture_available-008 | 1200 | 1 | 1 | true | 0.9642 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-002 | 384 | 4 | 4 | true | 0.5781 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-002 | 1200 | 4 | 4 | true | 0.8625 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-003 | 384 | 1 | 1 | true | 0.9766 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-003 | 1200 | 1 | 1 | true | 0.9883 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-005 | 384 | 1 | 4 | false | 0.0078 | true | reported_only_not_hard_fail,selected_non_reference |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-005 | 1200 | 1 | 4 | false | 0.0025 | true | reported_only_not_hard_fail,selected_non_reference |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-006 | 384 | 2 | 2 | true | 0.8542 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-006 | 1200 | 2 | 2 | true | 0.93 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-007 | 384 | 1 | 1 | true | 0.9688 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-007 | 1200 | 1 | 1 | true | 0.9725 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-008 | 384 | 1 | 1 | true | 0.8932 | true | ok |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-008 | 1200 | 1 | 1 | true | 0.9642 | true | ok |
| guarded-w2 | None | capture_available-002 | 384 | 2 | 4 | false | 0.1458 | false | selected_non_reference |
| guarded-w2 | None | capture_available-002 | 1200 | 2 | 4 | false | 0.0467 | false | selected_non_reference |
| guarded-w2 | None | capture_available-003 | 384 | 1 | 1 | true | 0.9401 | true | ok |
| guarded-w2 | None | capture_available-003 | 1200 | 1 | 1 | true | 0.9433 | true | ok |
| guarded-w2 | None | capture_available-005 | 384 | 1 | 4 | false | 0.0312 | true | reported_only_not_hard_fail,selected_non_reference |
| guarded-w2 | None | capture_available-005 | 1200 | 1 | 4 | false | 0.01 | true | reported_only_not_hard_fail,selected_non_reference |
| guarded-w2 | None | capture_available-006 | 384 | 2 | 2 | true | 0.8854 | true | ok |
| guarded-w2 | None | capture_available-006 | 1200 | 2 | 2 | true | 0.9533 | true | ok |
| guarded-w2 | None | capture_available-007 | 384 | 1 | 1 | true | 0.8984 | true | ok |
| guarded-w2 | None | capture_available-007 | 1200 | 1 | 1 | true | 0.9608 | true | ok |
| guarded-w2 | None | capture_available-008 | 384 | 1 | 1 | true | 0.7135 | true | ok |
| guarded-w2 | None | capture_available-008 | 1200 | 1 | 1 | true | 0.9033 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-002 | 384 | 4 | 4 | true | 0.8385 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-002 | 1200 | 4 | 4 | true | 0.9383 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-003 | 384 | 1 | 1 | true | 0.9401 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-003 | 1200 | 1 | 1 | true | 0.9433 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-005 | 384 | 1 | 4 | false | 0.0312 | true | reported_only_not_hard_fail,selected_non_reference |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-005 | 1200 | 1 | 4 | false | 0.01 | true | reported_only_not_hard_fail,selected_non_reference |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-006 | 384 | 2 | 2 | true | 0.8854 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-006 | 1200 | 2 | 2 | true | 0.9533 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-007 | 384 | 1 | 1 | true | 0.8984 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-007 | 1200 | 1 | 1 | true | 0.9608 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-008 | 384 | 1 | 1 | true | 0.7135 | true | ok |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | capture_available-008 | 1200 | 1 | 1 | true | 0.9033 | true | ok |

## Arena Matrix Results

| comparison_id | challenger_artifact | current_artifact | challenger_transform | current_transform | challenger_sims | current_sims | seed | games | score | wins | losses | draws | ci_low | ci_high | unstable_decision | report_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 256 | 256 | 11 | 120 | 0.5208333333333334 | 58 | 53 | 9 | 0.4321983184552195 | 0.6081758369174503 | True | <out-root>/arena_A1_candidate_vs_current_baseline_equal_low_seed11.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 256 | 256 | 23 | 120 | 0.525 | 59 | 53 | 8 | 0.4362683592400492 | 0.6121806272071545 | True | <out-root>/arena_A1_candidate_vs_current_baseline_equal_low_seed23.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 256 | 256 | 37 | 120 | 0.5333333333333333 | 60 | 52 | 8 | 0.44442629105796966 | 0.6201723575383018 | True | <out-root>/arena_A1_candidate_vs_current_baseline_equal_low_seed37.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 640 | 640 | 11 | 120 | 0.5041666666666667 | 53 | 52 | 15 | 0.4159775159037542 | 0.5920973151707797 | True | <out-root>/arena_A1_candidate_vs_current_baseline_equal_high_seed11.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 640 | 640 | 23 | 120 | 0.5083333333333333 | 56 | 54 | 10 | 0.420023820848017 | 0.5961258413010508 | True | <out-root>/arena_A1_candidate_vs_current_baseline_equal_high_seed23.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 640 | 640 | 37 | 120 | 0.5 | 55 | 55 | 10 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A1_candidate_vs_current_baseline_equal_high_seed37.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 640 | 256 | 11 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A1_candidate_vs_current_baseline_superhuman_style_seed11.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 640 | 256 | 23 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A1_candidate_vs_current_baseline_superhuman_style_seed23.json |
| A1_candidate_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | None | None | 640 | 256 | 37 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A1_candidate_vs_current_baseline_superhuman_style_seed37.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 256 | 256 | 11 | 120 | 0.5208333333333334 | 58 | 53 | 9 | 0.4321983184552195 | 0.6081758369174503 | True | <out-root>/arena_A2_candidate_transform_vs_current_baseline_equal_low_seed11.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 256 | 256 | 23 | 120 | 0.525 | 59 | 53 | 8 | 0.4362683592400492 | 0.6121806272071545 | True | <out-root>/arena_A2_candidate_transform_vs_current_baseline_equal_low_seed23.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 256 | 256 | 37 | 120 | 0.5333333333333333 | 60 | 52 | 8 | 0.44442629105796966 | 0.6201723575383018 | True | <out-root>/arena_A2_candidate_transform_vs_current_baseline_equal_low_seed37.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 640 | 11 | 120 | 0.5041666666666667 | 53 | 52 | 15 | 0.4159775159037542 | 0.5920973151707797 | True | <out-root>/arena_A2_candidate_transform_vs_current_baseline_equal_high_seed11.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 640 | 23 | 120 | 0.5083333333333333 | 56 | 54 | 10 | 0.420023820848017 | 0.5961258413010508 | True | <out-root>/arena_A2_candidate_transform_vs_current_baseline_equal_high_seed23.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 640 | 37 | 120 | 0.5 | 55 | 55 | 10 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A2_candidate_transform_vs_current_baseline_equal_high_seed37.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 256 | 11 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A2_candidate_transform_vs_current_baseline_superhuman_style_seed11.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 256 | 23 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A2_candidate_transform_vs_current_baseline_superhuman_style_seed23.json |
| A2_candidate_transform_vs_current_baseline | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 256 | 37 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A2_candidate_transform_vs_current_baseline_superhuman_style_seed37.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 256 | 256 | 11 | 120 | 0.5 | 52 | 52 | 16 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A3_current_transform_vs_current_baseline_equal_low_seed11.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 256 | 256 | 23 | 120 | 0.5 | 52 | 52 | 16 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A3_current_transform_vs_current_baseline_equal_low_seed23.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 256 | 256 | 37 | 120 | 0.5 | 51 | 51 | 18 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A3_current_transform_vs_current_baseline_equal_low_seed37.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 640 | 11 | 120 | 0.5 | 57 | 57 | 6 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A3_current_transform_vs_current_baseline_equal_high_seed11.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 640 | 23 | 120 | 0.5 | 55 | 55 | 10 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A3_current_transform_vs_current_baseline_equal_high_seed23.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 640 | 37 | 120 | 0.5 | 56 | 56 | 8 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A3_current_transform_vs_current_baseline_equal_high_seed37.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 256 | 11 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A3_current_transform_vs_current_baseline_superhuman_style_seed11.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 256 | 23 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A3_current_transform_vs_current_baseline_superhuman_style_seed23.json |
| A3_current_transform_vs_current_baseline | current | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | None | 640 | 256 | 37 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A3_current_transform_vs_current_baseline_superhuman_style_seed37.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 256 | 256 | 11 | 120 | 0.5208333333333334 | 58 | 53 | 9 | 0.4321983184552195 | 0.6081758369174503 | True | <out-root>/arena_A4_candidate_transform_vs_current_transform_equal_low_seed11.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 256 | 256 | 23 | 120 | 0.525 | 59 | 53 | 8 | 0.4362683592400492 | 0.6121806272071545 | True | <out-root>/arena_A4_candidate_transform_vs_current_transform_equal_low_seed23.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 256 | 256 | 37 | 120 | 0.5333333333333333 | 60 | 52 | 8 | 0.44442629105796966 | 0.6201723575383018 | True | <out-root>/arena_A4_candidate_transform_vs_current_transform_equal_low_seed37.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 640 | 640 | 11 | 120 | 0.5041666666666667 | 53 | 52 | 15 | 0.4159775159037542 | 0.5920973151707797 | True | <out-root>/arena_A4_candidate_transform_vs_current_transform_equal_high_seed11.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 640 | 640 | 23 | 120 | 0.5083333333333333 | 56 | 54 | 10 | 0.420023820848017 | 0.5961258413010508 | True | <out-root>/arena_A4_candidate_transform_vs_current_transform_equal_high_seed23.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 640 | 640 | 37 | 120 | 0.5 | 55 | 55 | 10 | 0.411937137430229 | 0.588062862569771 | True | <out-root>/arena_A4_candidate_transform_vs_current_transform_equal_high_seed37.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 640 | 256 | 11 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A4_candidate_transform_vs_current_transform_superhuman_style_seed11.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 640 | 256 | 23 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A4_candidate_transform_vs_current_transform_superhuman_style_seed23.json |
| A4_candidate_transform_vs_current_transform | aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1 | current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 640 | 256 | 37 | 120 | 1.0 | 120 | 0 | 0 | 0.9689797289440706 | 1.0 | False | <out-root>/arena_A4_candidate_transform_vs_current_transform_superhuman_style_seed37.json |

## PUCT Hard-State Validation Results

| artifact | transform | simulations | positions | average_regret | blunder_rate | value_calibration_mae | activation_count | activation_rate | average_mass_shift | report_path |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| current | None | 384 | 224 | 0.0711 | 0.5893 | 0.4513 | 0 | 0.0 | None | <out-root>/hard_state_current_original_prior_384.json |
| current | None | 1200 | 224 | 0.0874 | 0.5446 | 0.432 | 0 | 0.0 | None | <out-root>/hard_state_current_original_prior_1200.json |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 224 | 0.0711 | 0.5893 | 0.4513 | 0 | 0.0 | None | <out-root>/hard_state_current_seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5_384.json |
| current | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 224 | 0.0874 | 0.5446 | 0.432 | 0 | 0.0 | None | <out-root>/hard_state_current_seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5_1200.json |
| guarded-w2 | None | 384 | 224 | 0.0694 | 0.5714 | 0.456 | 0 | 0.0 | None | <out-root>/hard_state_guarded-w2_original_prior_384.json |
| guarded-w2 | None | 1200 | 224 | 0.0907 | 0.4955 | 0.4405 | 0 | 0.0 | None | <out-root>/hard_state_guarded-w2_original_prior_1200.json |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 224 | 0.0694 | 0.5714 | 0.456 | 0 | 0.0 | None | <out-root>/hard_state_guarded-w2_seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5_384.json |
| guarded-w2 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 224 | 0.0907 | 0.4955 | 0.4405 | 0 | 0.0 | None | <out-root>/hard_state_guarded-w2_seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5_1200.json |

## Transform Activation Telemetry

- `current::original_prior::384`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`
- `current::original_prior::1200`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`
- `current::seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5::384`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`
- `current::seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5::1200`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`
- `guarded-w2::original_prior::384`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`
- `guarded-w2::original_prior::1200`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`
- `guarded-w2::seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5::384`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`
- `guarded-w2::seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5::1200`: `{'root_states_evaluated': 224, 'activation_count': 0, 'activation_rate': 0.0, 'average_mass_shift': None, 'legal_move_context_counts': {'seed4_extra_turn_count': 0, 'no_extra_turn_capture_count': 0, 'no_extra_turn_noncapture_seed5_count': 0, 'legal_move_count': 0}, 'top_changed_move_feature_patterns': []}`

## Interpretation

- classification: `budget_advantage_not_transform_evidence`
- interpretation: budget advantage not transform evidence

## Recommended Next Action

Recommendation: **do not promote; rerun fair-budget or improve evaluation design.**
