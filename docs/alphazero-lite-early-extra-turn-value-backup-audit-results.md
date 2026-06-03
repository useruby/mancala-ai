# AlphaZero-lite Sparse Endgame Value/Backup Audit Results

## 1. Context

- This audit evaluates child-afterstate values and backup behavior for the `early_extra_turn` corrected non-opening failure family.
- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references were not mutated.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows: `/tmp/azlite_corrected_non_opening_failure_mining_v7/selected_non_opening_family_rows_v7.jsonl`.

## 2. Why early_extra_turn was selected

- PR #67 (v7 family mining) selected `early_extra_turn` as the next corrected non-opening failure family after sparse_endgame was excluded due to forced/tied dominance.
- Family stats: `{"avg_reference_visit_share_1200": 0.6359, "avg_reference_visit_share_384": 0.6, "avg_selected_minus_reference_q_margin_1200": 0.1097, "avg_selected_minus_reference_q_margin_384": 0.084, "classification": "value_or_backup_issue", "conflicting_target_count": 0, "dominant_failure_mode": "value_q", "duplicate_canonical_state_count": 0, "fail_rows": 8, "failure_rate": 0.3333, "family": "early_extra_turn", "high_severity_count": 0, "medium_severity_count": 8, "notes": "persistent failures remain Q/value-dominant", "pass_rows": 16, "persistent_1200_failures": 7, "rank_score": 211.7816, "recovered_at_1200": 0, "stable_corrected_reference_count": 24, "total_rows": 24}`.

## 3. Row validation

| row_id | role | corrected_reference_move | legal | reference_unstable | remaining_seed_count | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | target_candidate | 0 | true | false | 42 | ok | validated, representative target row present |
| early_extra_turn-016 | target_candidate | 2 | true | false | 42 | ok | validated, representative target row present |
| early_extra_turn-012 | target_candidate | 3 | true | false | 42 | ok | validated, representative target row present |
| early_extra_turn-013 | target_candidate | 3 | true | false | 42 | ok | validated, representative target row present |
| early_extra_turn-014 | target_candidate | 1 | true | false | 42 | ok | validated, representative target row present |
| early_extra_turn-018 | target_candidate | 2 | true | false | 42 | ok | validated, representative target row present |
| early_extra_turn-017 | holdout_candidate | 2 | true | false | 42 | ok | validated |
| early_extra_turn-007 | preservation_control | 5 | true | false | 42 | ok | validated, representative control row present |
| early_extra_turn-011 | preservation_control | 5 | true | false | 42 | ok | validated, representative control row present |
| early_extra_turn-002 | preservation_control | 4 | true | false | 42 | ok | validated, representative control row present |
| early_extra_turn-009 | preservation_control | 4 | true | false | 42 | ok | validated, representative control row present |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Implementation rule: `PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player`.

## 4. Root PUCT baseline

| row_id | role | budget | corrected_reference_move | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | reference_q | selected_q | selected_minus_reference_q_margin | reference_policy_probability | selected_policy_probability | remaining_seed_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-002 | preservation_control | 384 | 4 | 4 | true | 0.9375 | 0.9375 | 0.3484 | 0.3484 | 0.0000 | 0.6838 | 0.6838 | 42 | preservation_control, selected reference |
| early_extra_turn-002 | preservation_control | 1200 | 4 | 4 | true | 0.9767 | 0.9767 | 0.4042 | 0.4042 | 0.0000 | 0.6838 | 0.6838 | 42 | preservation_control, selected reference |
| early_extra_turn-002 | preservation_control | 2400 | 4 | 4 | true | 0.9842 | 0.9842 | 0.4340 | 0.4340 | 0.0000 | 0.6838 | 0.6838 | 42 | preservation_control, selected reference |
| early_extra_turn-007 | preservation_control | 384 | 5 | 5 | true | 0.9974 | 0.9974 | 0.1553 | 0.1553 | 0.0000 | 0.9918 | 0.9918 | 42 | preservation_control, selected reference |
| early_extra_turn-007 | preservation_control | 1200 | 5 | 5 | true | 0.9975 | 0.9975 | 0.1653 | 0.1653 | 0.0000 | 0.9918 | 0.9918 | 42 | preservation_control, selected reference |
| early_extra_turn-007 | preservation_control | 2400 | 5 | 5 | true | 0.9988 | 0.9988 | 0.1853 | 0.1853 | 0.0000 | 0.9918 | 0.9918 | 42 | preservation_control, selected reference |
| early_extra_turn-009 | preservation_control | 384 | 4 | 4 | true | 0.9297 | 0.9297 | 0.1936 | 0.1936 | 0.0000 | 0.6746 | 0.6746 | 42 | preservation_control, selected reference |
| early_extra_turn-009 | preservation_control | 1200 | 4 | 4 | true | 0.9717 | 0.9717 | 0.1977 | 0.1977 | 0.0000 | 0.6746 | 0.6746 | 42 | preservation_control, selected reference |
| early_extra_turn-009 | preservation_control | 2400 | 4 | 4 | true | 0.9758 | 0.9758 | 0.2022 | 0.2022 | 0.0000 | 0.6746 | 0.6746 | 42 | preservation_control, selected reference |
| early_extra_turn-011 | preservation_control | 384 | 5 | 5 | true | 0.9505 | 0.9505 | 0.1267 | 0.1267 | 0.0000 | 0.9899 | 0.9899 | 42 | preservation_control, selected reference |
| early_extra_turn-011 | preservation_control | 1200 | 5 | 5 | true | 0.9817 | 0.9817 | 0.1392 | 0.1392 | 0.0000 | 0.9899 | 0.9899 | 42 | preservation_control, selected reference |
| early_extra_turn-011 | preservation_control | 2400 | 5 | 5 | true | 0.9904 | 0.9904 | 0.1375 | 0.1375 | 0.0000 | 0.9899 | 0.9899 | 42 | preservation_control, selected reference |
| early_extra_turn-012 | target_candidate | 384 | 3 | 5 | false | 0.0026 | 0.9922 | 0.0683 | 0.1865 | 0.1182 | 0.0086 | 0.9672 | 42 | target_candidate, selected away from reference |
| early_extra_turn-012 | target_candidate | 1200 | 3 | 5 | false | 0.0008 | 0.9967 | 0.0683 | 0.1638 | 0.0955 | 0.0086 | 0.9672 | 42 | target_candidate, selected away from reference |
| early_extra_turn-012 | target_candidate | 2400 | 3 | 5 | false | 0.0004 | 0.9979 | 0.0683 | 0.1691 | 0.1007 | 0.0086 | 0.9672 | 42 | target_candidate, selected away from reference |
| early_extra_turn-013 | target_candidate | 384 | 3 | 1 | false | 0.0104 | 0.8490 | -0.1325 | -0.1060 | 0.0265 | 0.0281 | 0.1162 | 42 | target_candidate, selected away from reference |
| early_extra_turn-013 | target_candidate | 1200 | 3 | 5 | false | 0.0033 | 0.7033 | -0.1325 | -0.0411 | 0.0914 | 0.0281 | 0.7815 | 42 | target_candidate, selected away from reference |
| early_extra_turn-013 | target_candidate | 2400 | 3 | 5 | false | 0.0017 | 0.8517 | -0.1325 | -0.0814 | 0.0511 | 0.0281 | 0.7815 | 42 | target_candidate, selected away from reference |
| early_extra_turn-014 | target_candidate | 384 | 1 | 5 | false | 0.0885 | 0.8828 | -0.1943 | -0.1830 | 0.0112 | 0.1534 | 0.6831 | 42 | target_candidate, selected away from reference |
| early_extra_turn-014 | target_candidate | 1200 | 1 | 5 | false | 0.0333 | 0.9525 | -0.2117 | -0.1385 | 0.0732 | 0.1534 | 0.6831 | 42 | target_candidate, selected away from reference |
| early_extra_turn-014 | target_candidate | 2400 | 1 | 5 | false | 0.0167 | 0.9762 | -0.2117 | -0.1295 | 0.0822 | 0.1534 | 0.6831 | 42 | target_candidate, selected away from reference |
| early_extra_turn-016 | target_candidate | 384 | 2 | 5 | false | 0.0234 | 0.9661 | -0.2296 | -0.1349 | 0.0947 | 0.0831 | 0.7990 | 42 | target_candidate, selected away from reference |
| early_extra_turn-016 | target_candidate | 1200 | 2 | 5 | false | 0.0108 | 0.9833 | -0.2346 | -0.1188 | 0.1158 | 0.0831 | 0.7990 | 42 | target_candidate, selected away from reference |
| early_extra_turn-016 | target_candidate | 2400 | 2 | 5 | false | 0.0254 | 0.9621 | -0.2024 | -0.1795 | 0.0229 | 0.0831 | 0.7990 | 42 | target_candidate, selected away from reference |
| early_extra_turn-017 | holdout_candidate | 384 | 2 | 5 | false | 0.0521 | 0.8646 | -0.1957 | -0.1921 | 0.0036 | 0.0798 | 0.7920 | 42 | holdout_candidate, selected away from reference |
| early_extra_turn-017 | holdout_candidate | 1200 | 2 | 5 | false | 0.0208 | 0.9500 | -0.2230 | -0.1919 | 0.0311 | 0.0798 | 0.7920 | 42 | holdout_candidate, selected away from reference |
| early_extra_turn-017 | holdout_candidate | 2400 | 2 | 5 | false | 0.0104 | 0.9750 | -0.2230 | -0.1532 | 0.0697 | 0.0798 | 0.7920 | 42 | holdout_candidate, selected away from reference |
| early_extra_turn-018 | target_candidate | 384 | 2 | 5 | false | 0.0078 | 0.9818 | -0.3052 | -0.1823 | 0.1229 | 0.0672 | 0.8202 | 42 | target_candidate, selected away from reference |
| early_extra_turn-018 | target_candidate | 1200 | 2 | 5 | false | 0.0108 | 0.9808 | -0.2292 | -0.1566 | 0.0726 | 0.0672 | 0.8202 | 42 | target_candidate, selected away from reference |
| early_extra_turn-018 | target_candidate | 2400 | 2 | 5 | false | 0.0187 | 0.9746 | -0.2146 | -0.1941 | 0.0204 | 0.0672 | 0.8202 | 42 | target_candidate, selected away from reference |
| early_extra_turn-025 | target_candidate | 384 | 0 | 4 | false | 0.0026 | 0.6068 | -0.1715 | 0.1236 | 0.2951 | 0.0376 | 0.5332 | 42 | target_candidate, selected away from reference |
| early_extra_turn-025 | target_candidate | 1200 | 0 | 5 | false | 0.0008 | 0.5017 | -0.1715 | 0.1166 | 0.2880 | 0.0376 | 0.0698 | 42 | target_candidate, selected away from reference |
| early_extra_turn-025 | target_candidate | 2400 | 0 | 4 | false | 0.0037 | 0.6583 | 0.0352 | 0.0976 | 0.0624 | 0.0376 | 0.5332 | 42 | target_candidate, selected away from reference |

- ran 2400 root budget: projected ~1.6s across 6 target rows

## 5. Move consequence and endgame audit

| row_id | move | is_corrected_reference | is_selected | gives_extra_turn | produces_capture | capture_count | immediate_store_delta | side_to_move_after | game_over_after_move | remaining_seed_count_after | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | 0 | true | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-025 | 1 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-025 | 2 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-025 | 3 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-025 | 4 | false | false | true | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-025 | 5 | false | true | false | false | 0 | 1 | 1 | false | 41 | selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| early_extra_turn-016 | 0 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-016 | 1 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-016 | 2 | true | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-016 | 3 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-016 | 4 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-016 | 5 | false | true | true | false | 0 | 1 | 1 | false | 41 | selected move gains extra turn; selected move keeps the move |
| early_extra_turn-012 | 0 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-012 | 1 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-012 | 2 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-012 | 3 | true | false | false | false | 0 | 1 | 1 | false | 41 |  |
| early_extra_turn-012 | 4 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-012 | 5 | false | true | true | false | 0 | 1 | 0 | false | 41 | selected move gains extra turn; selected move keeps the move |
| early_extra_turn-013 | 0 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-013 | 1 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-013 | 2 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-013 | 3 | true | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-013 | 4 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-013 | 5 | false | true | true | false | 0 | 1 | 1 | false | 41 | selected move gains extra turn; selected move keeps the move |
| early_extra_turn-014 | 0 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-014 | 1 | true | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-014 | 2 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-014 | 3 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-014 | 4 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-014 | 5 | false | true | true | false | 0 | 1 | 1 | false | 41 | selected move gains extra turn; selected move keeps the move |
| early_extra_turn-018 | 0 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-018 | 1 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-018 | 2 | true | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-018 | 3 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-018 | 4 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-018 | 5 | false | true | true | false | 0 | 1 | 1 | false | 41 | selected move gains extra turn; selected move keeps the move |
| early_extra_turn-017 | 0 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-017 | 1 | false | false | false | false | 0 | 0 | 0 | false | 42 |  |
| early_extra_turn-017 | 2 | true | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-017 | 3 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-017 | 4 | false | false | false | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-017 | 5 | false | true | true | false | 0 | 1 | 1 | false | 41 | selected move gains extra turn; selected move keeps the move |
| early_extra_turn-007 | 0 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-007 | 1 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-007 | 2 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-007 | 3 | false | false | false | false | 0 | 1 | 1 | false | 41 |  |
| early_extra_turn-007 | 4 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-007 | 5 | true | true | true | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-011 | 0 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-011 | 1 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-011 | 2 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-011 | 3 | false | false | false | false | 0 | 1 | 1 | false | 41 |  |
| early_extra_turn-011 | 4 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-011 | 5 | true | true | true | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-002 | 0 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-002 | 1 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-002 | 2 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-002 | 3 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-002 | 4 | true | true | true | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-002 | 5 | false | false | false | false | 0 | 1 | 1 | false | 41 |  |
| early_extra_turn-009 | 0 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-009 | 1 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-009 | 2 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-009 | 3 | false | false | false | false | 0 | 0 | 1 | false | 42 |  |
| early_extra_turn-009 | 4 | true | true | true | false | 0 | 1 | 0 | false | 41 |  |
| early_extra_turn-009 | 5 | false | false | false | false | 0 | 1 | 1 | false | 41 |  |

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_preferred_move | tablebase_value_root | agrees_with_active_reference | agrees_with_puct_selected | agrees_with_classic_teacher | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | root | 42 | false | - | - | false | false | - | not solvable under the 16-seed threshold |
| early_extra_turn-025 | child_after_corrected_reference | 42 | false | - | - | - | - | false | not solvable under the 16-seed threshold |
| early_extra_turn-025 | child_after_current_selected | 41 | false | - | - | - | - | - | not solvable under the 16-seed threshold |
| early_extra_turn-016 | root | 42 | false | - | - | false | false | - | not solvable under the 16-seed threshold |
| early_extra_turn-016 | child_after_corrected_reference | 41 | false | - | - | - | - | false | not solvable under the 16-seed threshold |
| early_extra_turn-016 | child_after_current_selected | 41 | false | - | - | - | - | - | not solvable under the 16-seed threshold |
| early_extra_turn-012 | root | 42 | false | - | - | false | false | - | not solvable under the 16-seed threshold |
| early_extra_turn-012 | child_after_corrected_reference | 41 | false | - | - | - | - | false | not solvable under the 16-seed threshold |
| early_extra_turn-012 | child_after_current_selected | 41 | false | - | - | - | - | - | not solvable under the 16-seed threshold |
| early_extra_turn-013 | root | 42 | false | - | - | false | false | - | not solvable under the 16-seed threshold |
| early_extra_turn-013 | child_after_corrected_reference | 41 | false | - | - | - | - | false | not solvable under the 16-seed threshold |
| early_extra_turn-013 | child_after_current_selected | 41 | false | - | - | - | - | - | not solvable under the 16-seed threshold |
| early_extra_turn-014 | root | 42 | false | - | - | false | false | - | not solvable under the 16-seed threshold |
| early_extra_turn-014 | child_after_corrected_reference | 41 | false | - | - | - | - | false | not solvable under the 16-seed threshold |
| early_extra_turn-014 | child_after_current_selected | 41 | false | - | - | - | - | - | not solvable under the 16-seed threshold |
| early_extra_turn-018 | root | 42 | false | - | - | false | false | - | not solvable under the 16-seed threshold |
| early_extra_turn-018 | child_after_corrected_reference | 41 | false | - | - | - | - | false | not solvable under the 16-seed threshold |
| early_extra_turn-018 | child_after_current_selected | 41 | false | - | - | - | - | - | not solvable under the 16-seed threshold |

## 7. Neural child-afterstate value audit

| row_id | corrected_reference_move | selected_move | child | raw_value | root_perspective_value | tablebase_value_root_if_available | child_ref_minus_child_selected | neural_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | 0 | 5 | corrected_reference_child | 0.1715 | -0.1715 | - | -0.3849 | false | sign flip to root perspective |
| early_extra_turn-025 | 0 | 5 | selected_child | -0.2135 | 0.2135 | - | -0.3849 | false | sign flip to root perspective |
| early_extra_turn-016 | 2 | 5 | corrected_reference_child | 0.2319 | -0.2319 | - | -0.1183 | false | sign flip to root perspective |
| early_extra_turn-016 | 2 | 5 | selected_child | -0.1136 | -0.1136 | - | -0.1183 | false | identity conversion |
| early_extra_turn-012 | 3 | 5 | corrected_reference_child | -0.0683 | 0.0683 | - | -0.2273 | false | sign flip to root perspective |
| early_extra_turn-012 | 3 | 5 | selected_child | 0.2956 | 0.2956 | - | -0.2273 | false | identity conversion |
| early_extra_turn-013 | 3 | 5 | corrected_reference_child | 0.0096 | -0.0096 | - | 0.1634 | true | sign flip to root perspective |
| early_extra_turn-013 | 3 | 5 | selected_child | -0.1731 | -0.1731 | - | 0.1634 | true | identity conversion |
| early_extra_turn-014 | 1 | 5 | corrected_reference_child | 0.5305 | -0.5305 | - | -0.4382 | false | sign flip to root perspective |
| early_extra_turn-014 | 1 | 5 | selected_child | -0.0923 | -0.0923 | - | -0.4382 | false | identity conversion |
| early_extra_turn-018 | 2 | 5 | corrected_reference_child | 0.2652 | -0.2652 | - | -0.0431 | false | sign flip to root perspective |
| early_extra_turn-018 | 2 | 5 | selected_child | -0.2221 | -0.2221 | - | -0.0431 | false | identity conversion |

## 8. ClassicMCTS child-afterstate audit

| row_id | budget | seeds | child_ref_value_root_mean | child_selected_value_root_mean | child_ref_minus_child_selected | teacher_prefers_reference | stable | tablebase_agrees_if_available | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | 1200 | [11, 23, 37, 42, 101] | -0.5436 | -0.6506 | 0.1070 | true | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-025 | 2400 | [11, 23, 37, 42, 101] | -0.5586 | -0.6904 | 0.1318 | true | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-025 | 5000 | [11, 23, 37, 42, 101] | -0.6954 | -0.7278 | 0.0324 | true | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-016 | 1200 | [11, 23, 37, 42, 101] | -0.5226 | 0.2233 | -0.7460 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-016 | 2400 | [11, 23, 37, 42, 101] | -0.7090 | 0.3604 | -1.0694 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-016 | 5000 | [11, 23, 37, 42, 101] | -0.8390 | 0.4522 | -1.2912 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-012 | 1200 | [11, 23, 37, 42, 101] | -0.2220 | 0.7137 | -0.9357 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-012 | 2400 | [11, 23, 37, 42, 101] | -0.3294 | 0.7504 | -1.0798 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-012 | 5000 | [11, 23, 37, 42, 101] | -0.5204 | 0.7897 | -1.3101 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-013 | 1200 | [11, 23, 37, 42, 101] | -0.5151 | 0.1504 | -0.6655 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-013 | 2400 | [11, 23, 37, 42, 101] | -0.6663 | 0.2875 | -0.9539 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-013 | 5000 | [11, 23, 37, 42, 101] | -0.7908 | 0.4135 | -1.2044 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-014 | 1200 | [11, 23, 37, 42, 101] | -0.5252 | 0.1597 | -0.6850 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-014 | 2400 | [11, 23, 37, 42, 101] | -0.6921 | 0.3087 | -1.0008 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-014 | 5000 | [11, 23, 37, 42, 101] | -0.7998 | 0.4488 | -1.2486 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-018 | 1200 | [11, 23, 37, 42, 101] | -0.6097 | 0.4527 | -1.0624 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-018 | 2400 | [11, 23, 37, 42, 101] | -0.6150 | 0.4747 | -1.0897 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |
| early_extra_turn-018 | 5000 | [11, 23, 37, 42, 101] | -0.6130 | 0.5313 | -1.1443 | false | true | - | ClassicMCTS child-afterstate teacher aggregate |

## 9. PUCT child-afterstate audit

| row_id | budget | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | puct_prefers_reference | tablebase_agreement_if_available | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | 384 | 0.0353 | 0.1383 | -0.1030 | false | - | disagrees with teacher; agrees with neural |
| early_extra_turn-025 | 1200 | 0.0367 | 0.0924 | -0.0557 | false | - | disagrees with teacher; agrees with neural |
| early_extra_turn-016 | 384 | -0.3557 | -0.1388 | -0.2168 | false | - | agrees with teacher; agrees with neural |
| early_extra_turn-016 | 1200 | -0.2810 | -0.1210 | -0.1600 | false | - | agrees with teacher; agrees with neural |
| early_extra_turn-012 | 384 | 0.0009 | 0.1863 | -0.1854 | false | - | agrees with teacher; agrees with neural |
| early_extra_turn-012 | 1200 | 0.0559 | 0.1627 | -0.1068 | false | - | agrees with teacher; agrees with neural |
| early_extra_turn-013 | 384 | -0.1000 | -0.0562 | -0.0437 | false | - | agrees with teacher; disagrees with neural |
| early_extra_turn-013 | 1200 | -0.1111 | -0.0550 | -0.0561 | false | - | agrees with teacher; disagrees with neural |
| early_extra_turn-014 | 384 | -0.2397 | -0.1802 | -0.0596 | false | - | agrees with teacher; agrees with neural |
| early_extra_turn-014 | 1200 | -0.2573 | -0.1382 | -0.1191 | false | - | agrees with teacher; agrees with neural |
| early_extra_turn-018 | 384 | -0.2510 | -0.1812 | -0.0698 | false | - | agrees with teacher; agrees with neural |
| early_extra_turn-018 | 1200 | -0.2070 | -0.1546 | -0.0524 | false | - | agrees with teacher; agrees with neural |

## 10. Root counterfactual diagnostics

| row_id | intervention | budget | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | flipped | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | original | 384 | 4 | false | 0.0026 | 0.6068 | 0.2951 | false | no intervention |
| early_extra_turn-025 | original | 1200 | 5 | false | 0.0008 | 0.5017 | 0.2880 | false | no intervention |
| early_extra_turn-025 | equalize_root_priors | 384 | 4 | false | 0.0026 | 0.6042 | 0.2955 | false | equalize corrected reference and selected root priors |
| early_extra_turn-025 | equalize_root_priors | 1200 | 5 | false | 0.0083 | 0.5033 | 0.0885 | false | equalize corrected reference and selected root priors |
| early_extra_turn-025 | uniform_legal_prior | 384 | 5 | false | 0.0182 | 0.8984 | 0.0833 | false | uniform legal prior positive control |
| early_extra_turn-025 | uniform_legal_prior | 1200 | 5 | false | 0.0083 | 0.5350 | 0.0800 | false | uniform legal prior positive control |
| early_extra_turn-025 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.8177 | 0.8161 | false | override first child expansion values with teacher child values |
| early_extra_turn-025 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.4900 | 0.7978 | false | override first child expansion values with teacher child values |
| early_extra_turn-025 | neural_child_value_swap | 384 | 4 | false | 0.0208 | 0.9271 | 0.0353 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-025 | neural_child_value_swap | 1200 | 5 | false | 0.0083 | 0.4983 | 0.0509 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-016 | original | 384 | 5 | false | 0.0234 | 0.9661 | 0.0947 | false | no intervention |
| early_extra_turn-016 | original | 1200 | 5 | false | 0.0108 | 0.9833 | 0.1158 | false | no intervention |
| early_extra_turn-016 | equalize_root_priors | 384 | 5 | false | 0.1536 | 0.8359 | 0.0774 | false | equalize corrected reference and selected root priors |
| early_extra_turn-016 | equalize_root_priors | 1200 | 5 | false | 0.0617 | 0.9350 | 0.1241 | false | equalize corrected reference and selected root priors |
| early_extra_turn-016 | uniform_legal_prior | 384 | 5 | false | 0.0234 | 0.9141 | 0.1049 | false | uniform legal prior positive control |
| early_extra_turn-016 | uniform_legal_prior | 1200 | 5 | false | 0.0117 | 0.6142 | 0.1104 | false | uniform legal prior positive control |
| early_extra_turn-016 | teacher_child_value_override | 384 | 5 | false | 0.0026 | 0.9870 | 0.7031 | false | override first child expansion values with teacher child values |
| early_extra_turn-016 | teacher_child_value_override | 1200 | 5 | false | 0.0075 | 0.9867 | 0.1785 | false | override first child expansion values with teacher child values |
| early_extra_turn-016 | neural_child_value_swap | 384 | 5 | false | 0.0234 | 0.9661 | 0.0812 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-016 | neural_child_value_swap | 1200 | 5 | false | 0.0117 | 0.9825 | 0.1190 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-012 | original | 384 | 5 | false | 0.0026 | 0.9922 | 0.1182 | false | no intervention |
| early_extra_turn-012 | original | 1200 | 5 | false | 0.0008 | 0.9967 | 0.0955 | false | no intervention |
| early_extra_turn-012 | equalize_root_priors | 384 | 5 | false | 0.0547 | 0.9427 | 0.1150 | false | equalize corrected reference and selected root priors |
| early_extra_turn-012 | equalize_root_priors | 1200 | 5 | false | 0.0542 | 0.9442 | 0.1014 | false | equalize corrected reference and selected root priors |
| early_extra_turn-012 | uniform_legal_prior | 384 | 5 | false | 0.0469 | 0.8776 | 0.0938 | false | uniform legal prior positive control |
| early_extra_turn-012 | uniform_legal_prior | 1200 | 5 | false | 0.0158 | 0.9250 | 0.0888 | false | uniform legal prior positive control |
| early_extra_turn-012 | teacher_child_value_override | 384 | 5 | false | 0.0026 | 0.9922 | 0.7082 | false | override first child expansion values with teacher child values |
| early_extra_turn-012 | teacher_child_value_override | 1200 | 5 | false | 0.0008 | 0.9958 | 0.6848 | false | override first child expansion values with teacher child values |
| early_extra_turn-012 | neural_child_value_swap | 384 | 5 | false | 0.0339 | 0.9609 | 0.0503 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-012 | neural_child_value_swap | 1200 | 5 | false | 0.0108 | 0.9867 | 0.0252 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-013 | original | 384 | 1 | false | 0.0104 | 0.8490 | 0.0265 | false | no intervention |
| early_extra_turn-013 | original | 1200 | 5 | false | 0.0033 | 0.7033 | 0.0914 | false | no intervention |
| early_extra_turn-013 | equalize_root_priors | 384 | 1 | false | 0.0417 | 0.9115 | 0.0887 | false | equalize corrected reference and selected root priors |
| early_extra_turn-013 | equalize_root_priors | 1200 | 1 | false | 0.0933 | 0.4742 | 0.0148 | false | equalize corrected reference and selected root priors |
| early_extra_turn-013 | uniform_legal_prior | 384 | 1 | false | 0.0312 | 0.8984 | 0.0791 | false | uniform legal prior positive control |
| early_extra_turn-013 | uniform_legal_prior | 1200 | 4 | false | 0.0117 | 0.5667 | 0.0886 | false | uniform legal prior positive control |
| early_extra_turn-013 | teacher_child_value_override | 384 | 5 | false | 0.0026 | 0.8672 | 0.7306 | false | override first child expansion values with teacher child values |
| early_extra_turn-013 | teacher_child_value_override | 1200 | 5 | false | 0.0008 | 0.9508 | 0.7407 | false | override first child expansion values with teacher child values |
| early_extra_turn-013 | neural_child_value_swap | 384 | 1 | false | 0.0026 | 0.8490 | 0.0671 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-013 | neural_child_value_swap | 1200 | 5 | false | 0.0008 | 0.7108 | 0.1301 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-014 | original | 384 | 5 | false | 0.0885 | 0.8828 | 0.0112 | false | no intervention |
| early_extra_turn-014 | original | 1200 | 5 | false | 0.0333 | 0.9525 | 0.0732 | false | no intervention |
| early_extra_turn-014 | equalize_root_priors | 384 | 5 | false | 0.2188 | 0.7708 | 0.0391 | false | equalize corrected reference and selected root priors |
| early_extra_turn-014 | equalize_root_priors | 1200 | 5 | false | 0.0933 | 0.8975 | 0.0789 | false | equalize corrected reference and selected root priors |
| early_extra_turn-014 | uniform_legal_prior | 384 | 2 | false | 0.0234 | 0.9167 | 0.1077 | false | uniform legal prior positive control |
| early_extra_turn-014 | uniform_legal_prior | 1200 | 2 | false | 0.0142 | 0.9533 | 0.0623 | false | uniform legal prior positive control |
| early_extra_turn-014 | teacher_child_value_override | 384 | 5 | false | 0.0781 | 0.8932 | 0.0331 | false | override first child expansion values with teacher child values |
| early_extra_turn-014 | teacher_child_value_override | 1200 | 5 | false | 0.0300 | 0.9558 | 0.0723 | false | override first child expansion values with teacher child values |
| early_extra_turn-014 | neural_child_value_swap | 384 | 5 | false | 0.1068 | 0.8646 | 0.0180 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-014 | neural_child_value_swap | 1200 | 5 | false | 0.0383 | 0.9475 | 0.0664 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-018 | original | 384 | 5 | false | 0.0078 | 0.9818 | 0.1229 | false | no intervention |
| early_extra_turn-018 | original | 1200 | 5 | false | 0.0108 | 0.9808 | 0.0726 | false | no intervention |
| early_extra_turn-018 | equalize_root_priors | 384 | 5 | false | 0.1484 | 0.8411 | 0.0498 | false | equalize corrected reference and selected root priors |
| early_extra_turn-018 | equalize_root_priors | 1200 | 5 | false | 0.0983 | 0.8983 | 0.0572 | false | equalize corrected reference and selected root priors |
| early_extra_turn-018 | uniform_legal_prior | 384 | 5 | false | 0.3359 | 0.5833 | 0.1109 | false | uniform legal prior positive control |
| early_extra_turn-018 | uniform_legal_prior | 1200 | 5 | false | 0.1075 | 0.8558 | 0.0635 | false | uniform legal prior positive control |
| early_extra_turn-018 | teacher_child_value_override | 384 | 5 | false | 0.0026 | 0.9766 | 0.4326 | false | override first child expansion values with teacher child values |
| early_extra_turn-018 | teacher_child_value_override | 1200 | 5 | false | 0.0025 | 0.9892 | 0.2658 | false | override first child expansion values with teacher child values |
| early_extra_turn-018 | neural_child_value_swap | 384 | 5 | false | 0.0078 | 0.9818 | 0.1084 | false | swap child neural values as diagnostic sanity check |
| early_extra_turn-018 | neural_child_value_swap | 1200 | 5 | false | 0.0108 | 0.9808 | 0.0693 | false | swap child neural values as diagnostic sanity check |

## 11. Row classifications

| row_id | role | row_classification | supporting_evidence | recommended_use | notes |
| --- | --- | --- | --- | --- | --- |
| early_extra_turn-025 | target_candidate | value_head_miscalibration | ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child | target_candidate | root still fails |
| early_extra_turn-016 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| early_extra_turn-012 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| early_extra_turn-013 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| early_extra_turn-014 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| early_extra_turn-018 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| early_extra_turn-017 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| early_extra_turn-007 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| early_extra_turn-011 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| early_extra_turn-002 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| early_extra_turn-009 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |

## 12. Family-level interpretation

| family_classification | value_head_miscalibration_count | puct_child_search_mismatch_count | root_selection_pressure_count | tablebase_reference_conflict_count | tablebase_tie_not_conflict_count | corrected_reference_suspicious_count | backup_perspective_suspect_count | inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_family_uncertain | 1 | 0 | 0 | 0 | 0 | 5 | 0 | 0 | adjudicate early_extra_turn references before training. |

## 13. Exactly one recommended next action

Recommendation: **adjudicate early_extra_turn references before training.**
