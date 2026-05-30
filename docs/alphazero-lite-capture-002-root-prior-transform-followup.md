# AlphaZero-lite Capture 002 Root-Prior Transform Follow-up

## Context

This follow-up narrows extra-turn damping to a tighter legal-move feature pattern derived from the first ablation.

- source branch: `more precise move-feature conditioning; do not deploy damping`
- artifact: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- summary JSON: `/tmp/azlite_capture_002_root_prior_transform_followup/root_prior_transform_followup_summary.json`

## Results

| row_id | transform | simulations | selected_move | reference_move | selected_is_reference | reference_visit_share | selected_minus_reference_q | original_reference_prior | transformed_reference_prior | gate_pass | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-002 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 384 | 4 | 4 | true | 0.4948 | 0.0 | 0.0668 | 0.0793 | false | 002@1200_revert_2 |
| capture_available-002 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 1200 | 2 | 4 | false | 0.1583 | 0.0448 | 0.0668 | 0.0793 | false | 002@1200_revert_2 |
| capture_available-002 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 384 | 4 | 4 | true | 0.8333 | 0.0 | 0.0668 | 0.0875 | false | 002@1200_revert_2 |
| capture_available-002 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 1200 | 2 | 4 | false | 0.28 | 0.0469 | 0.0668 | 0.0875 | false | 002@1200_revert_2 |
| capture_available-002 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 384 | 4 | 4 | true | 0.8385 | 0.0 | 0.0668 | 0.0933 | true | followup_local_ok |
| capture_available-002 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 1200 | 4 | 4 | true | 0.9383 | 0.0 | 0.0668 | 0.0933 | true | followup_local_ok |
| capture_available-002 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 4 | 4 | true | 0.8385 | 0.0 | 0.0668 | 0.0933 | true | followup_local_ok |
| capture_available-002 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 4 | 4 | true | 0.9383 | 0.0 | 0.0668 | 0.0933 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.9401 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9433 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.9401 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9433 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.9401 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9433 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.9401 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-003 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9433 | 0.0 | 0.3886 | 0.3886 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 384 | 1 | 4 | false | 0.0312 | 0.0423 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 1200 | 1 | 4 | false | 0.01 | 0.1093 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 384 | 1 | 4 | false | 0.0312 | 0.0423 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 1200 | 1 | 4 | false | 0.01 | 0.1093 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 384 | 1 | 4 | false | 0.0312 | 0.0423 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 1200 | 1 | 4 | false | 0.01 | 0.1093 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 1 | 4 | false | 0.0312 | 0.0423 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-005 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 1 | 4 | false | 0.01 | 0.1093 | 0.0287 | 0.0287 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 384 | 2 | 2 | true | 0.8854 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 1200 | 2 | 2 | true | 0.9533 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 384 | 2 | 2 | true | 0.8854 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 1200 | 2 | 2 | true | 0.9533 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 384 | 2 | 2 | true | 0.8854 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 1200 | 2 | 2 | true | 0.9533 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 2 | 2 | true | 0.8854 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-006 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 2 | 2 | true | 0.9533 | 0.0 | 0.1796 | 0.1796 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.8984 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9608 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.8984 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9608 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.8984 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9608 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.8984 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-007 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9608 | 0.0 | 0.3884 | 0.3884 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.7135 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_050_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9033 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.7135 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_025_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9033 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.7135 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_010_when_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9033 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 384 | 1 | 1 | true | 0.7135 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |
| capture_available-008 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 | 1200 | 1 | 1 | true | 0.9033 | 0.0 | 0.4939 | 0.4939 | true | followup_local_ok |

## Recommendation

Recommendation: **broader arena/MCTS1200 validation for the best narrow feature-conditioned transform**.
