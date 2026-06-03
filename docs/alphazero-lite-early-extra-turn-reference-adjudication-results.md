# AlphaZero-lite Early Extra Turn Reference Adjudication Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay/value-calibration artifacts were created.
- Active corrected references were not mutated.
- Active corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows path: `/tmp/azlite_corrected_non_opening_failure_mining_v7/selected_non_opening_family_rows_v7.jsonl`.

## 2. Why PR #67 blocked value training

- PR #67 did not confirm that `early_extra_turn` is cleanly trainable.
- Family-level classification: `reference_family_uncertain`.
- Mechanism counts: `{"blocked_next_action": "adjudicate early_extra_turn references before training.", "corrected_reference_suspicious_count": 5, "family_classification": "reference_family_uncertain", "inconclusive_count": 0, "puct_child_search_mismatch_count": 0, "root_selection_pressure_count": 0, "value_head_miscalibration_count": 1}`.
- Exact PR #67 recommendation: **adjudicate early_extra_turn references before training.**

## 3. Row validation

| row_id | pr67_classification | active_reference_move | legal | reference_unstable | canonical_state_match | current_selected_384 | current_selected_1200 | current_selected_2400 | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-016 | corrected_reference_suspicious | 2 | true | false | true | 5 | 5 | 5 | ok | validated for adjudication |
| early_extra_turn-012 | corrected_reference_suspicious | 3 | true | false | true | 5 | 5 | 5 | ok | validated for adjudication |
| early_extra_turn-013 | corrected_reference_suspicious | 3 | true | false | true | 1 | 5 | 5 | ok | validated for adjudication |
| early_extra_turn-014 | corrected_reference_suspicious | 1 | true | false | true | 5 | 5 | 5 | ok | validated for adjudication |
| early_extra_turn-018 | corrected_reference_suspicious | 2 | true | false | true | 5 | 5 | 5 | ok | validated for adjudication |
| early_extra_turn-025 | value_head_miscalibration | 0 | true | false | true | 4 | 5 | 4 | ok | reported as value-head miscalibration context only |
| early_extra_turn-017 | holdout_context | 2 | true | false | true | 5 | 5 | 5 | ok | reported as holdout context only |
| early_extra_turn-007 | preservation_control | 5 | true | false | true | 5 | 5 | 5 | ok | reported as preservation control only |
| early_extra_turn-011 | preservation_control | 5 | true | false | true | 5 | 5 | 5 | ok | reported as preservation control only |
| early_extra_turn-002 | preservation_control | 4 | true | false | true | 4 | 4 | 4 | ok | reported as preservation control only |
| early_extra_turn-009 | preservation_control | 4 | true | false | true | 4 | 4 | 4 | ok | reported as preservation control only |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Repo convention: `follow state_to_root_perspective_value: child values stay positive for the same player-to-move and are sign-flipped when the turn hands off`.

## 4. Root ClassicMCTS adjudication

| row_id | budget | seeds | active_reference_move | observed_top_moves | majority_move | majority_fraction | reference_selected_fraction | top1_margin_mean | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-016 | 1200 | 7 | 2 | 2:4, 4:3 | 2 | 0.5714 | 0.5714 | 0.3407 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-016 | 2400 | 7 | 2 | 2:4, 4:3 | 2 | 0.5714 | 0.5714 | 0.5625 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-016 | 5000 | 7 | 2 | 2:4, 4:3 | 2 | 0.5714 | 0.5714 | 0.7538 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-016 | 10000 | 7 | 2 | 2:4, 4:3 | 2 | 0.5714 | 0.5714 | 0.9384 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-016 | 30000 | 7 | 2 | 2:4, 4:3 | 2 | 0.5714 | 0.5714 | 1.1515 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-012 | 1200 | 7 | 3 | 3:2, 5:5 | 5 | 0.7143 | 0.2857 | 0.0451 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-012 | 2400 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0537 | supports_active_reference | high-budget majority stays on the active reference |
| early_extra_turn-012 | 5000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.1564 | supports_active_reference | high-budget majority stays on the active reference |
| early_extra_turn-012 | 10000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.2125 | supports_active_reference | high-budget majority stays on the active reference |
| early_extra_turn-012 | 30000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.2683 | supports_active_reference | high-budget majority stays on the active reference |
| early_extra_turn-013 | 1200 | 7 | 3 | 1:1, 4:6 | 4 | 0.8571 | 0.0000 | 0.2278 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-013 | 2400 | 7 | 3 | 1:2, 4:5 | 4 | 0.7143 | 0.0000 | 0.4282 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-013 | 5000 | 7 | 3 | 1:2, 4:5 | 4 | 0.7143 | 0.0000 | 0.6401 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-013 | 10000 | 7 | 3 | 1:2, 4:5 | 4 | 0.7143 | 0.0000 | 0.8451 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-013 | 30000 | 7 | 3 | 1:2, 4:5 | 4 | 0.7143 | 0.0000 | 1.0849 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-014 | 1200 | 7 | 1 | 0:1, 1:4, 3:2 | 1 | 0.5714 | 0.5714 | 0.2553 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-014 | 2400 | 7 | 1 | 1:4, 3:3 | 1 | 0.5714 | 0.5714 | 0.4480 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-014 | 5000 | 7 | 1 | 1:4, 3:3 | 1 | 0.5714 | 0.5714 | 0.6893 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-014 | 10000 | 7 | 1 | 1:4, 3:3 | 1 | 0.5714 | 0.5714 | 0.9056 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-014 | 30000 | 7 | 1 | 1:4, 3:3 | 1 | 0.5714 | 0.5714 | 1.1192 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-018 | 1200 | 7 | 2 | 2:4, 5:3 | 2 | 0.5714 | 0.5714 | 0.3747 | unstable_or_mixed | seed majorities remain mixed at this budget |
| early_extra_turn-018 | 2400 | 7 | 2 | 2:2, 5:5 | 5 | 0.7143 | 0.2857 | 0.3650 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-018 | 5000 | 7 | 2 | 2:2, 5:5 | 5 | 0.7143 | 0.2857 | 0.4793 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-018 | 10000 | 7 | 2 | 2:2, 5:5 | 5 | 0.7143 | 0.2857 | 0.6132 | supports_flip_candidate | high-budget majority prefers another move |
| early_extra_turn-018 | 30000 | 7 | 2 | 2:2, 5:5 | 5 | 0.7143 | 0.2857 | 0.7803 | supports_flip_candidate | high-budget majority prefers another move |

## 5. Child-afterstate adjudication

| row_id | active_reference_move | comparison_move | budget | active_reference_child_value_root_mean | comparison_child_value_root_mean | active_minus_comparison_value | child_selected_moves | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-016 | 2 | 5 | 1200 | -0.5226 | 0.2233 | -0.7459 | ref[1:1, 2:3, 3:1] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-016 | 2 | 5 | 2400 | -0.7090 | 0.3604 | -1.0694 | ref[1:2, 2:3] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-016 | 2 | 5 | 5000 | -0.8390 | 0.4522 | -1.2912 | ref[1:2, 2:3] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-012 | 3 | 5 | 1200 | -0.2220 | 0.7137 | -0.9357 | ref[5:5] cmp[3:1, 4:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-012 | 3 | 5 | 2400 | -0.3294 | 0.7504 | -1.0798 | ref[2:1, 5:4] cmp[4:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-012 | 3 | 5 | 5000 | -0.5204 | 0.7897 | -1.3101 | ref[1:2, 2:3] cmp[3:2, 4:3] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-013 | 3 | 4 | 1200 | -0.5151 | -0.5578 | 0.0427 | ref[1:5] cmp[4:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-013 | 3 | 5 | 1200 | -0.5151 | 0.1504 | -0.6655 | ref[1:5] cmp[1:4, 3:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-013 | 3 | 4 | 2400 | -0.6663 | -0.5755 | -0.0908 | ref[1:5] cmp[4:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-013 | 3 | 5 | 2400 | -0.6663 | 0.2875 | -0.9538 | ref[1:5] cmp[1:4, 3:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-013 | 3 | 4 | 5000 | -0.7908 | -0.7259 | -0.0649 | ref[1:5] cmp[2:2, 4:3] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-013 | 3 | 5 | 5000 | -0.7908 | 0.4135 | -1.2043 | ref[1:5] cmp[1:4, 3:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-014 | 1 | 5 | 1200 | -0.5252 | 0.1597 | -0.6849 | ref[2:4, 3:1] cmp[3:4, 4:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-014 | 1 | 5 | 2400 | -0.6921 | 0.3087 | -1.0008 | ref[2:5] cmp[3:4, 4:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-014 | 1 | 5 | 5000 | -0.7998 | 0.4488 | -1.2486 | ref[2:5] cmp[3:4, 4:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-018 | 2 | 5 | 1200 | -0.6097 | 0.4527 | -1.0624 | ref[2:5] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-018 | 2 | 5 | 2400 | -0.6150 | 0.4747 | -1.0897 | ref[2:5] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| early_extra_turn-018 | 2 | 5 | 5000 | -0.6130 | 0.5313 | -1.1443 | ref[2:5] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_value_root | tablebase_preferred_move | agrees_with_classic_majority | agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-016 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-016 | child_after_move_2 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-016 | child_after_move_5 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-012 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-012 | child_after_move_3 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-012 | child_after_move_5 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-013 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-013 | child_after_move_3 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-013 | child_after_move_4 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-013 | child_after_move_5 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-014 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-014 | child_after_move_1 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-014 | child_after_move_5 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-018 | root | 42 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-018 | child_after_move_2 | 41 | false | - | - | - | - | not solvable under the repo threshold |
| early_extra_turn-018 | child_after_move_5 | 41 | false | - | - | - | - | not solvable under the repo threshold |

## 7. PUCT/artifact teacher comparison

| row_id | budget | active_reference_move | puct_selected_move | puct_reference_visit_share | puct_selected_visit_share | puct_agrees_with_classic_majority | puct_agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-016 | 384 | 2 | 5 | 0.0234 | 0.9661 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-016 | 1200 | 2 | 5 | 0.0108 | 0.9833 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-016 | 2400 | 2 | 5 | 0.0254 | 0.9621 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-016 | 5000 | 2 | 5 | 0.0126 | 0.9808 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-012 | 384 | 3 | 5 | 0.0026 | 0.9922 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-012 | 1200 | 3 | 5 | 0.0008 | 0.9967 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-012 | 2400 | 3 | 5 | 0.0004 | 0.9979 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-012 | 5000 | 3 | 5 | 0.0004 | 0.9986 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-013 | 384 | 3 | 1 | 0.0104 | 0.8490 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-013 | 1200 | 3 | 5 | 0.0033 | 0.7033 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-013 | 2400 | 3 | 5 | 0.0017 | 0.8517 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-013 | 5000 | 3 | 5 | 0.0012 | 0.9274 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-014 | 384 | 1 | 5 | 0.0885 | 0.8828 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-014 | 1200 | 1 | 5 | 0.0333 | 0.9525 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-014 | 2400 | 1 | 5 | 0.0167 | 0.9762 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-014 | 5000 | 1 | 5 | 0.0080 | 0.9882 | false | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-018 | 384 | 2 | 5 | 0.0078 | 0.9818 | true | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-018 | 1200 | 2 | 5 | 0.0108 | 0.9808 | true | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-018 | 2400 | 2 | 5 | 0.0187 | 0.9746 | true | false | deterministic artifact PUCT with PR #67 settings |
| early_extra_turn-018 | 5000 | 2 | 5 | 0.0092 | 0.9870 | true | false | deterministic artifact PUCT with PR #67 settings |

## 8. Row decisions

| row_id | active_reference_move | adjudicated_decision | proposed_reference_move | proposed_unstable | evidence_summary | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-012 | 3 | still_inconclusive | - | false | highest_root_majority=3 fraction=1.0000; highest_puct_selected=5; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| early_extra_turn-013 | 3 | reference_should_flip | 4 | false | highest_root_majority=4 fraction=0.7143; highest_puct_selected=5; child_support=supports_comparison_move; tablebase_root_move=- | requires reviewed reference patch before training use | high-budget ClassicMCTS root and child-afterstate evidence consistently prefer another move |
| early_extra_turn-014 | 1 | reference_unstable | - | true | highest_root_majority=1 fraction=0.5714; highest_puct_selected=5; child_support=mixed; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| early_extra_turn-016 | 2 | reference_unstable | - | true | highest_root_majority=2 fraction=0.5714; highest_puct_selected=5; child_support=mixed; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| early_extra_turn-018 | 2 | reference_should_flip | 5 | false | highest_root_majority=5 fraction=0.7143; highest_puct_selected=5; child_support=supports_comparison_move; tablebase_root_move=- | requires reviewed reference patch before training use | high-budget ClassicMCTS root and child-afterstate evidence consistently prefer another move |

## 9. Proposed non-mutating patch artifact

- Proposed review artifact: `/tmp/azlite_early_extra_turn_reference_adjudication/early_extra_turn_reference_review_patch_v1.json`.
- `do_not_auto_apply` is set for every proposed row.
- The active fixture remained unchanged.

## 10. Projected clean targetability

| bucket | row_count | rows | train_target_eligible_count | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| confirmed_active_reference_targets | 0 | - | 0 | still limited to rows with confirmed active references only | usable only if the family still clears the final branch decision |
| reference_flip_candidates | 2 | early_extra_turn-013, early_extra_turn-018 | 0 | fixture changes need explicit review before they are safe to use | review the proposed patch artifact before any training |
| unstable_or_excluded | 3 | early_extra_turn-012, early_extra_turn-014, early_extra_turn-016 | 0 | unstable or mixed labels would contaminate any hard target set | exclude from hard gates and training targets |
| puct_teacher_divergence_rows | 0 | - | 0 | teacher-policy ambiguity remains unresolved for these rows | decide teacher policy before any training use |
| tablebase_tie_or_arbitrary_policy_rows | 0 | - | 0 | tablebase-tied positions have arbitrary policy labels | exclude from hard policy gates; keep as diagnostic context only |
| confirmed_value_head_miscalibration_context | 1 | early_extra_turn-025 | 0 | miscalibration may require separate value-calibration pass | reserve for later value-calibration if references settle |
| confirmed_preservation_controls | 4 | early_extra_turn-002, early_extra_turn-007, early_extra_turn-009, early_extra_turn-011 | 0 | controls must stay unchanged during any follow-up | preserve unchanged as regression checks |
| holdout_context_rows | 1 | early_extra_turn-017 | 0 | holdouts should remain context-only during follow-up | report only; do not train on them in this branch |

- Target-candidate rows still eligible under active references: `0`.
- Target-candidate rows that would require reference patching: `2`.
- Target-candidate rows that should stay excluded from training: `3`.
- `early_extra_turn` remains a good training target after adjudication: `false`.
- `early_extra_turn-025` is safe enough to seed a later value-calibration bucket: `false`.
- Projected family decision: `reference_suite_too_noisy_for_early_extra_turn`.

## 11. Exactly one recommended next action

Recommendation: **exclude early_extra_turn and select the next non-opening family.**
