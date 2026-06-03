# AlphaZero-lite Sparse Endgame Value/Backup Audit Results

## 1. Context

- This audit evaluates child-afterstate values and backup behavior for the `sparse_endgame` corrected non-opening failure family.
- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references were not mutated.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows: `/tmp/azlite_corrected_non_opening_failure_mining_v6/selected_non_opening_family_rows_v6.jsonl`.

## 2. Why sparse_endgame was selected

- PR #63 selected `sparse_endgame` as the next corrected non-opening failure family.
- Family stats: `{"avg_reference_visit_share_1200": 0.5886, "avg_reference_visit_share_384": 0.5712, "avg_selected_minus_reference_q_margin_1200": 0.1536, "avg_selected_minus_reference_q_margin_384": 0.2444, "classification": "value_or_backup_issue", "conflicting_target_count": 0, "dominant_failure_mode": "value_q", "duplicate_canonical_state_count": 0, "fail_rows": 10, "failure_rate": 0.4167, "family": "sparse_endgame", "high_severity_count": 0, "medium_severity_count": 10, "notes": "persistent failures remain Q/value-dominant", "pass_rows": 14, "persistent_1200_failures": 7, "rank_score": 215.0044, "recovered_at_1200": 1, "stable_corrected_reference_count": 24, "total_rows": 24}`.

## 3. Row validation

| row_id | role | corrected_reference_move | legal | reference_unstable | remaining_seed_count | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | target_candidate | 0 | true | false | 16 | ok | validated, representative target row present |
| sparse_endgame-007 | target_candidate | 0 | true | false | 16 | ok | validated, representative target row present |
| sparse_endgame-009 | target_candidate | 5 | true | false | 14 | ok | validated, representative target row present |
| sparse_endgame-024 | target_candidate | 1 | true | false | 16 | ok | validated, representative target row present |
| sparse_endgame-026 | target_candidate | 4 | true | false | 16 | ok | validated, representative target row present |
| sparse_endgame-025 | target_candidate | 5 | true | false | 16 | ok | validated, representative target row present |
| sparse_endgame-023 | holdout_candidate | 4 | true | false | 16 | ok | validated, representative target row present |
| sparse_endgame-002 | preservation_control | 3 | true | false | 16 | ok | validated, representative control row present |
| sparse_endgame-019 | preservation_control | 5 | true | false | 15 | ok | validated, representative control row present |
| sparse_endgame-020 | preservation_control | 2 | true | false | 11 | ok | validated, representative control row present |
| sparse_endgame-011 | preservation_control | 5 | true | false | 16 | ok | validated, representative control row present |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Implementation rule: `PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player`.

## 4. Root PUCT baseline

| row_id | role | budget | corrected_reference_move | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | reference_q | selected_q | selected_minus_reference_q_margin | reference_policy_probability | selected_policy_probability | remaining_seed_count | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-002 | preservation_control | 384 | 3 | 3 | true | 1.0000 | 1.0000 | -0.8578 | -0.8578 | 0.0000 | 1.0000 | 1.0000 | 16 | preservation_control, selected reference |
| sparse_endgame-002 | preservation_control | 1200 | 3 | 3 | true | 1.0000 | 1.0000 | -0.9411 | -0.9411 | 0.0000 | 1.0000 | 1.0000 | 16 | preservation_control, selected reference |
| sparse_endgame-002 | preservation_control | 2400 | 3 | 3 | true | 1.0000 | 1.0000 | -0.9687 | -0.9687 | 0.0000 | 1.0000 | 1.0000 | 16 | preservation_control, selected reference |
| sparse_endgame-003 | target_candidate | 384 | 0 | 1 | false | 0.0000 | 0.9375 | 0.0000 | 0.9990 | 0.9990 | 0.0008 | 0.0455 | 16 | target_candidate, selected away from reference |
| sparse_endgame-003 | target_candidate | 1200 | 0 | 1 | false | 0.0008 | 0.9642 | 0.6215 | 0.9997 | 0.3782 | 0.0008 | 0.0455 | 16 | target_candidate, selected away from reference |
| sparse_endgame-003 | target_candidate | 2400 | 0 | 1 | false | 0.0004 | 0.9746 | 0.6215 | 0.9999 | 0.3783 | 0.0008 | 0.0455 | 16 | target_candidate, selected away from reference |
| sparse_endgame-007 | target_candidate | 384 | 0 | 3 | false | 0.0000 | 0.9375 | 0.0000 | 0.7795 | 0.7795 | 0.0036 | 0.9332 | 16 | target_candidate, selected away from reference |
| sparse_endgame-007 | target_candidate | 1200 | 0 | 3 | false | 0.0008 | 0.9792 | 0.6576 | 0.8298 | 0.1722 | 0.0036 | 0.9332 | 16 | target_candidate, selected away from reference |
| sparse_endgame-007 | target_candidate | 2400 | 0 | 3 | false | 0.0004 | 0.9896 | 0.6576 | 0.8483 | 0.1907 | 0.0036 | 0.9332 | 16 | target_candidate, selected away from reference |
| sparse_endgame-009 | target_candidate | 384 | 5 | 2 | false | 0.0339 | 0.9219 | -0.2889 | -0.0490 | 0.2399 | 0.4304 | 0.2414 | 14 | target_candidate, selected away from reference |
| sparse_endgame-009 | target_candidate | 1200 | 5 | 2 | false | 0.0325 | 0.9492 | -0.1912 | -0.0569 | 0.1343 | 0.4304 | 0.2414 | 14 | target_candidate, selected away from reference |
| sparse_endgame-009 | target_candidate | 2400 | 5 | 2 | false | 0.0163 | 0.9746 | -0.1912 | -0.0464 | 0.1448 | 0.4304 | 0.2414 | 14 | target_candidate, selected away from reference |
| sparse_endgame-011 | preservation_control | 384 | 5 | 5 | true | 0.9688 | 0.9688 | 0.1684 | 0.1684 | 0.0000 | 0.8077 | 0.8077 | 16 | preservation_control, selected reference |
| sparse_endgame-011 | preservation_control | 1200 | 5 | 5 | true | 0.9833 | 0.9833 | 0.2341 | 0.2341 | 0.0000 | 0.8077 | 0.8077 | 16 | preservation_control, selected reference |
| sparse_endgame-011 | preservation_control | 2400 | 5 | 5 | true | 0.9917 | 0.9917 | 0.2659 | 0.2659 | 0.0000 | 0.8077 | 0.8077 | 16 | preservation_control, selected reference |
| sparse_endgame-019 | preservation_control | 384 | 5 | 5 | true | 0.9792 | 0.9792 | 0.6985 | 0.6985 | 0.0000 | 0.6558 | 0.6558 | 15 | preservation_control, selected reference |
| sparse_endgame-019 | preservation_control | 1200 | 5 | 5 | true | 0.9875 | 0.9875 | 0.8364 | 0.8364 | 0.0000 | 0.6558 | 0.6558 | 15 | preservation_control, selected reference |
| sparse_endgame-019 | preservation_control | 2400 | 5 | 5 | true | 0.9912 | 0.9912 | 0.8843 | 0.8843 | 0.0000 | 0.6558 | 0.6558 | 15 | preservation_control, selected reference |
| sparse_endgame-020 | preservation_control | 384 | 2 | 2 | true | 0.9505 | 0.9505 | -0.0013 | -0.0013 | 0.0000 | 0.6147 | 0.6147 | 11 | preservation_control, selected reference |
| sparse_endgame-020 | preservation_control | 1200 | 2 | 2 | true | 0.9842 | 0.9842 | -0.0035 | -0.0035 | 0.0000 | 0.6147 | 0.6147 | 11 | preservation_control, selected reference |
| sparse_endgame-020 | preservation_control | 2400 | 2 | 2 | true | 0.9904 | 0.9904 | -0.0021 | -0.0021 | 0.0000 | 0.6147 | 0.6147 | 11 | preservation_control, selected reference |
| sparse_endgame-023 | holdout_candidate | 384 | 4 | 5 | false | 0.0000 | 0.7109 | 0.0000 | -0.0891 | -0.0891 | 0.0118 | 0.2256 | 16 | holdout_candidate, selected away from reference |
| sparse_endgame-023 | holdout_candidate | 1200 | 4 | 5 | false | 0.0008 | 0.9067 | -0.2365 | -0.1782 | 0.0582 | 0.0118 | 0.2256 | 16 | holdout_candidate, selected away from reference |
| sparse_endgame-023 | holdout_candidate | 2400 | 4 | 5 | false | 0.0004 | 0.9533 | -0.2365 | -0.0871 | 0.1494 | 0.0118 | 0.2256 | 16 | holdout_candidate, selected away from reference |
| sparse_endgame-024 | target_candidate | 384 | 1 | 5 | false | 0.1510 | 0.8464 | -0.3285 | -0.3008 | 0.0277 | 0.7694 | 0.2175 | 16 | target_candidate, selected away from reference |
| sparse_endgame-024 | target_candidate | 1200 | 1 | 4 | false | 0.0800 | 0.5425 | -0.3590 | -0.2317 | 0.1273 | 0.7694 | 0.0131 | 16 | target_candidate, selected away from reference |
| sparse_endgame-024 | target_candidate | 2400 | 1 | 4 | false | 0.0400 | 0.7712 | -0.3590 | -0.2581 | 0.1009 | 0.7694 | 0.0131 | 16 | target_candidate, selected away from reference |
| sparse_endgame-025 | target_candidate | 384 | 5 | 1 | false | 0.0052 | 0.9922 | -0.4439 | -0.2563 | 0.1876 | 0.1197 | 0.8681 | 16 | target_candidate, selected away from reference |
| sparse_endgame-025 | target_candidate | 1200 | 5 | 1 | false | 0.0158 | 0.9833 | -0.3392 | -0.2527 | 0.0865 | 0.1197 | 0.8681 | 16 | target_candidate, selected away from reference |
| sparse_endgame-025 | target_candidate | 2400 | 5 | 1 | false | 0.0117 | 0.9854 | -0.5516 | -0.5431 | 0.0085 | 0.1197 | 0.8681 | 16 | target_candidate, selected away from reference |
| sparse_endgame-026 | target_candidate | 384 | 4 | 5 | false | 0.0026 | 0.8672 | -0.3943 | -0.0986 | 0.2957 | 0.0133 | 0.2128 | 16 | target_candidate, selected away from reference |
| sparse_endgame-026 | target_candidate | 1200 | 4 | 5 | false | 0.0008 | 0.9017 | -0.3943 | -0.2756 | 0.1187 | 0.0133 | 0.2128 | 16 | target_candidate, selected away from reference |
| sparse_endgame-026 | target_candidate | 2400 | 4 | 5 | false | 0.3700 | 0.5554 | -0.2530 | -0.3938 | -0.1408 | 0.0133 | 0.2128 | 16 | target_candidate, selected away from reference |

- ran 2400 root budget: projected ~0.1s across 6 target rows

## 5. Move consequence and endgame audit

| row_id | move | is_corrected_reference | is_selected | gives_extra_turn | produces_capture | capture_count | immediate_store_delta | side_to_move_after | game_over_after_move | remaining_seed_count_after | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 0 | true | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-003 | 1 | false | true | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-003 | 2 | false | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-003 | 3 | false | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-003 | 4 | false | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-007 | 0 | true | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-007 | 1 | false | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-007 | 2 | false | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-007 | 3 | false | true | false | false | 0 | 1 | 1 | false | 15 | selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| sparse_endgame-007 | 4 | false | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-009 | 0 | false | false | false | false | 0 | 0 | 1 | false | 14 |  |
| sparse_endgame-009 | 2 | false | true | false | true | 2 | 2 | 1 | false | 12 | selected move captures more immediately; selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| sparse_endgame-009 | 4 | false | false | false | false | 0 | 0 | 1 | false | 14 |  |
| sparse_endgame-009 | 5 | true | false | false | false | 0 | 1 | 1 | false | 13 |  |
| sparse_endgame-024 | 1 | true | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-024 | 4 | false | true | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-024 | 5 | false | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-026 | 1 | false | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-026 | 4 | true | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-026 | 5 | false | true | false | false | 0 | 1 | 1 | false | 15 | selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| sparse_endgame-025 | 1 | false | true | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-025 | 4 | false | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-025 | 5 | true | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-023 | 1 | false | false | false | false | 0 | 1 | 1 | false | 15 |  |
| sparse_endgame-023 | 4 | true | false | false | false | 0 | 0 | 1 | false | 16 |  |
| sparse_endgame-023 | 5 | false | true | false | false | 0 | 1 | 1 | false | 15 | selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| sparse_endgame-002 | 3 | true | true | false | false | 0 | 1 | 0 | false | 15 |  |
| sparse_endgame-002 | 5 | false | false | false | false | 0 | 1 | 0 | false | 15 |  |
| sparse_endgame-019 | 0 | false | false | false | false | 0 | 0 | 1 | false | 15 |  |
| sparse_endgame-019 | 1 | false | false | true | false | 0 | 1 | 0 | false | 14 |  |
| sparse_endgame-019 | 4 | false | false | false | false | 0 | 0 | 1 | false | 15 |  |
| sparse_endgame-019 | 5 | true | true | true | false | 0 | 1 | 0 | false | 14 |  |
| sparse_endgame-020 | 0 | false | false | false | false | 0 | 0 | 1 | false | 11 |  |
| sparse_endgame-020 | 2 | true | true | true | false | 0 | 1 | 0 | false | 10 |  |
| sparse_endgame-020 | 4 | false | false | false | false | 0 | 0 | 1 | false | 11 |  |
| sparse_endgame-011 | 0 | false | false | false | false | 0 | 0 | 0 | false | 16 |  |
| sparse_endgame-011 | 1 | false | false | false | false | 0 | 0 | 0 | false | 16 |  |
| sparse_endgame-011 | 5 | true | true | false | false | 0 | 1 | 0 | false | 15 |  |

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_preferred_move | tablebase_value_root | agrees_with_active_reference | agrees_with_puct_selected | agrees_with_classic_teacher | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | root | 16 | true | 0 | 1.0000 | true | false | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-003 | child_after_corrected_reference | 16 | true | 5 | 1.0000 | - | - | true | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-003 | child_after_current_selected | 16 | true | 5 | 1.0000 | - | - | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-007 | root | 16 | true | 0 | 1.0000 | true | false | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-007 | child_after_corrected_reference | 16 | true | 5 | 1.0000 | - | - | true | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-007 | child_after_current_selected | 15 | true | 0 | 1.0000 | - | - | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-009 | root | 14 | true | 0 | -1.0000 | false | false | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-009 | child_after_corrected_reference | 13 | true | 1 | -1.0000 | - | - | true | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-009 | child_after_current_selected | 12 | true | 5 | -1.0000 | - | - | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-024 | root | 16 | true | 1 | -1.0000 | true | false | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-024 | child_after_corrected_reference | 15 | true | 5 | -1.0000 | - | - | true | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-024 | child_after_current_selected | 16 | true | 1 | -1.0000 | - | - | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-026 | root | 16 | true | 1 | -1.0000 | false | false | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-026 | child_after_corrected_reference | 16 | true | 0 | -1.0000 | - | - | true | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-026 | child_after_current_selected | 15 | true | 0 | -1.0000 | - | - | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-025 | root | 16 | true | 1 | -1.0000 | false | true | - | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-025 | child_after_corrected_reference | 15 | true | 5 | -1.0000 | - | - | true | exact tablebase value available under 16 remaining seeds |
| sparse_endgame-025 | child_after_current_selected | 15 | true | 0 | -1.0000 | - | - | - | exact tablebase value available under 16 remaining seeds |

## 7. Neural child-afterstate value audit

| row_id | corrected_reference_move | selected_move | child | raw_value | root_perspective_value | tablebase_value_root_if_available | child_ref_minus_child_selected | neural_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 0 | 1 | corrected_reference_child | -0.6215 | 0.6215 | 1.0000 | -0.0305 | false | sign flip to root perspective |
| sparse_endgame-003 | 0 | 1 | selected_child | -0.6520 | 0.6520 | 1.0000 | -0.0305 | false | sign flip to root perspective |
| sparse_endgame-007 | 0 | 3 | corrected_reference_child | -0.6576 | 0.6576 | 1.0000 | -0.2229 | false | sign flip to root perspective |
| sparse_endgame-007 | 0 | 3 | selected_child | -0.8805 | 0.8805 | 1.0000 | -0.2229 | false | sign flip to root perspective |
| sparse_endgame-009 | 5 | 2 | corrected_reference_child | 0.3834 | -0.3834 | -1.0000 | -0.1620 | false | sign flip to root perspective |
| sparse_endgame-009 | 5 | 2 | selected_child | 0.2214 | -0.2214 | -1.0000 | -0.1620 | false | sign flip to root perspective |
| sparse_endgame-024 | 1 | 4 | corrected_reference_child | 0.2945 | -0.2945 | -1.0000 | 0.0613 | true | sign flip to root perspective |
| sparse_endgame-024 | 1 | 4 | selected_child | 0.3558 | -0.3558 | -1.0000 | 0.0613 | true | sign flip to root perspective |
| sparse_endgame-026 | 4 | 5 | corrected_reference_child | 0.3943 | -0.3943 | -1.0000 | -0.1058 | false | sign flip to root perspective |
| sparse_endgame-026 | 4 | 5 | selected_child | 0.2885 | -0.2885 | -1.0000 | -0.1058 | false | sign flip to root perspective |
| sparse_endgame-025 | 5 | 1 | corrected_reference_child | 0.3568 | -0.3568 | -1.0000 | -0.0935 | false | sign flip to root perspective |
| sparse_endgame-025 | 5 | 1 | selected_child | 0.2633 | -0.2633 | -1.0000 | -0.0935 | false | sign flip to root perspective |

## 8. ClassicMCTS child-afterstate audit

| row_id | budget | seeds | child_ref_value_root_mean | child_selected_value_root_mean | child_ref_minus_child_selected | teacher_prefers_reference | stable | tablebase_agrees_if_available | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 1200 | [11, 23, 37, 42, 101] | 1.0000 | 1.0000 | 0.0000 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-003 | 2400 | [11, 23, 37, 42, 101] | 1.0000 | 1.0000 | 0.0000 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-003 | 5000 | [11, 23, 37, 42, 101] | 1.0000 | 1.0000 | 0.0000 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-007 | 1200 | [11, 23, 37, 42, 101] | 1.0000 | 1.0000 | 0.0000 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-007 | 2400 | [11, 23, 37, 42, 101] | 1.0000 | 1.0000 | 0.0000 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-007 | 5000 | [11, 23, 37, 42, 101] | 1.0000 | 1.0000 | 0.0000 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-009 | 1200 | [11, 23, 37, 42, 101] | -0.9639 | -0.9293 | -0.0346 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-009 | 2400 | [11, 23, 37, 42, 101] | -0.9741 | -0.9405 | -0.0337 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-009 | 5000 | [11, 23, 37, 42, 101] | -0.9790 | -0.9567 | -0.0223 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-024 | 1200 | [11, 23, 37, 42, 101] | -0.9078 | -0.9199 | 0.0121 | true | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-024 | 2400 | [11, 23, 37, 42, 101] | -0.9209 | -0.9294 | 0.0085 | true | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-024 | 5000 | [11, 23, 37, 42, 101] | -0.9369 | -0.9456 | 0.0087 | true | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-026 | 1200 | [11, 23, 37, 42, 101] | -0.9294 | -0.8804 | -0.0491 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-026 | 2400 | [11, 23, 37, 42, 101] | -0.9427 | -0.9016 | -0.0412 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-026 | 5000 | [11, 23, 37, 42, 101] | -0.9520 | -0.9208 | -0.0312 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-025 | 1200 | [11, 23, 37, 42, 101] | -1.0000 | -0.9267 | -0.0733 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-025 | 2400 | [11, 23, 37, 42, 101] | -1.0000 | -0.9242 | -0.0758 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |
| sparse_endgame-025 | 5000 | [11, 23, 37, 42, 101] | -1.0000 | -0.9273 | -0.0727 | false | true | false | ClassicMCTS child-afterstate teacher aggregate |

## 9. PUCT child-afterstate audit

| row_id | budget | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | puct_prefers_reference | tablebase_agreement_if_available | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 384 | 1.0000 | 1.0000 | 0.0000 | false | false | agrees with teacher; agrees with neural |
| sparse_endgame-003 | 1200 | 1.0000 | 1.0000 | 0.0000 | false | false | agrees with teacher; agrees with neural |
| sparse_endgame-007 | 384 | 1.0000 | 0.7698 | 0.2302 | true | false | disagrees with teacher; disagrees with neural |
| sparse_endgame-007 | 1200 | 1.0000 | 0.8286 | 0.1714 | true | false | disagrees with teacher; disagrees with neural |
| sparse_endgame-009 | 384 | -0.3294 | -0.0462 | -0.2833 | false | false | agrees with teacher; agrees with neural |
| sparse_endgame-009 | 1200 | -0.3517 | -0.0561 | -0.2955 | false | false | agrees with teacher; agrees with neural |
| sparse_endgame-024 | 384 | -0.7208 | -0.2016 | -0.5192 | false | false | disagrees with teacher; disagrees with neural |
| sparse_endgame-024 | 1200 | -0.8632 | -0.2365 | -0.6267 | false | false | disagrees with teacher; disagrees with neural |
| sparse_endgame-026 | 384 | -0.1832 | -0.1057 | -0.0775 | false | false | agrees with teacher; agrees with neural |
| sparse_endgame-026 | 1200 | -0.3344 | -0.3380 | 0.0035 | true | false | disagrees with teacher; disagrees with neural |
| sparse_endgame-025 | 384 | -0.9449 | -0.2704 | -0.6744 | false | false | agrees with teacher; agrees with neural |
| sparse_endgame-025 | 1200 | -0.9738 | -0.3788 | -0.5951 | false | false | agrees with teacher; agrees with neural |

## 10. Root counterfactual diagnostics

| row_id | intervention | budget | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | flipped | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | original | 384 | 1 | false | 0.0000 | 0.9375 | 0.9990 | false | no intervention |
| sparse_endgame-003 | original | 1200 | 1 | false | 0.0008 | 0.9642 | 0.3782 | false | no intervention |
| sparse_endgame-003 | equalize_root_priors | 384 | 0 | true | 0.9375 | 0.9375 | 0.0000 | true | equalize corrected reference and selected root priors |
| sparse_endgame-003 | equalize_root_priors | 1200 | 0 | true | 0.7483 | 0.7483 | 0.0000 | true | equalize corrected reference and selected root priors |
| sparse_endgame-003 | uniform_legal_prior | 384 | 2 | false | 0.3125 | 0.3385 | 0.0007 | false | uniform legal prior positive control |
| sparse_endgame-003 | uniform_legal_prior | 1200 | 2 | false | 0.3233 | 0.3342 | 0.0002 | false | uniform legal prior positive control |
| sparse_endgame-003 | teacher_child_value_override | 384 | 1 | false | 0.0000 | 0.9375 | 1.0000 | false | override first child expansion values with teacher child values |
| sparse_endgame-003 | teacher_child_value_override | 1200 | 1 | false | 0.0175 | 0.9475 | 0.0000 | false | override first child expansion values with teacher child values |
| sparse_endgame-003 | neural_child_value_swap | 384 | 1 | false | 0.0000 | 0.9375 | 0.9989 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-003 | neural_child_value_swap | 1200 | 1 | false | 0.0008 | 0.9642 | 0.3476 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-007 | original | 384 | 3 | false | 0.0000 | 0.9375 | 0.7795 | false | no intervention |
| sparse_endgame-007 | original | 1200 | 3 | false | 0.0008 | 0.9792 | 0.1722 | false | no intervention |
| sparse_endgame-007 | equalize_root_priors | 384 | 0 | true | 0.9557 | 0.9557 | 0.0000 | true | equalize corrected reference and selected root priors |
| sparse_endgame-007 | equalize_root_priors | 1200 | 0 | true | 0.9800 | 0.9800 | 0.0000 | true | equalize corrected reference and selected root priors |
| sparse_endgame-007 | uniform_legal_prior | 384 | 2 | false | 0.3281 | 0.3333 | 0.0001 | false | uniform legal prior positive control |
| sparse_endgame-007 | uniform_legal_prior | 1200 | 2 | false | 0.3292 | 0.3308 | 0.0000 | false | uniform legal prior positive control |
| sparse_endgame-007 | teacher_child_value_override | 384 | 3 | false | 0.0000 | 0.9401 | 0.7795 | false | override first child expansion values with teacher child values |
| sparse_endgame-007 | teacher_child_value_override | 1200 | 0 | true | 0.5817 | 0.5817 | 0.0000 | true | override first child expansion values with teacher child values |
| sparse_endgame-007 | neural_child_value_swap | 384 | 3 | false | 0.0000 | 0.9349 | 0.7796 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-007 | neural_child_value_swap | 1200 | 0 | true | 0.5683 | 0.5683 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| sparse_endgame-009 | original | 384 | 2 | false | 0.0339 | 0.9219 | 0.2399 | false | no intervention |
| sparse_endgame-009 | original | 1200 | 2 | false | 0.0325 | 0.9492 | 0.1343 | false | no intervention |
| sparse_endgame-009 | equalize_root_priors | 384 | 2 | false | 0.0208 | 0.9453 | 0.2963 | false | equalize corrected reference and selected root priors |
| sparse_endgame-009 | equalize_root_priors | 1200 | 2 | false | 0.0225 | 0.9633 | 0.1623 | false | equalize corrected reference and selected root priors |
| sparse_endgame-009 | uniform_legal_prior | 384 | 2 | false | 0.0156 | 0.9115 | 0.3679 | false | uniform legal prior positive control |
| sparse_endgame-009 | uniform_legal_prior | 1200 | 2 | false | 0.0083 | 0.9650 | 0.2627 | false | uniform legal prior positive control |
| sparse_endgame-009 | teacher_child_value_override | 384 | 5 | true | 0.7292 | 0.7292 | 0.0000 | true | override first child expansion values with teacher child values |
| sparse_endgame-009 | teacher_child_value_override | 1200 | 2 | false | 0.2333 | 0.6975 | 0.2554 | false | override first child expansion values with teacher child values |
| sparse_endgame-009 | neural_child_value_swap | 384 | 2 | false | 0.0339 | 0.8568 | 0.2251 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-009 | neural_child_value_swap | 1200 | 2 | false | 0.0333 | 0.9300 | 0.1323 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-024 | original | 384 | 5 | false | 0.1510 | 0.8464 | 0.0277 | false | no intervention |
| sparse_endgame-024 | original | 1200 | 4 | false | 0.0800 | 0.5425 | 0.1273 | false | no intervention |
| sparse_endgame-024 | equalize_root_priors | 384 | 4 | false | 0.1380 | 0.4323 | 0.0665 | false | equalize corrected reference and selected root priors |
| sparse_endgame-024 | equalize_root_priors | 1200 | 4 | false | 0.0442 | 0.8175 | 0.0611 | false | equalize corrected reference and selected root priors |
| sparse_endgame-024 | uniform_legal_prior | 384 | 5 | false | 0.1354 | 0.7214 | 0.0138 | false | uniform legal prior positive control |
| sparse_endgame-024 | uniform_legal_prior | 1200 | 4 | false | 0.0442 | 0.7108 | 0.0794 | false | uniform legal prior positive control |
| sparse_endgame-024 | teacher_child_value_override | 384 | 5 | false | 0.2604 | 0.7370 | 0.1143 | false | override first child expansion values with teacher child values |
| sparse_endgame-024 | teacher_child_value_override | 1200 | 5 | false | 0.1875 | 0.8117 | 0.0577 | false | override first child expansion values with teacher child values |
| sparse_endgame-024 | neural_child_value_swap | 384 | 5 | false | 0.1484 | 0.7760 | 0.0257 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-024 | neural_child_value_swap | 1200 | 4 | false | 0.0475 | 0.7042 | 0.1070 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-026 | original | 384 | 5 | false | 0.0026 | 0.8672 | 0.2957 | false | no intervention |
| sparse_endgame-026 | original | 1200 | 5 | false | 0.0008 | 0.9017 | 0.1187 | false | no intervention |
| sparse_endgame-026 | equalize_root_priors | 384 | 5 | false | 0.0104 | 0.8906 | 0.1844 | false | equalize corrected reference and selected root priors |
| sparse_endgame-026 | equalize_root_priors | 1200 | 5 | false | 0.1525 | 0.8042 | -0.0146 | false | equalize corrected reference and selected root priors |
| sparse_endgame-026 | uniform_legal_prior | 384 | 5 | false | 0.0521 | 0.9297 | 0.1034 | false | uniform legal prior positive control |
| sparse_endgame-026 | uniform_legal_prior | 1200 | 5 | false | 0.1525 | 0.8083 | -0.0181 | false | uniform legal prior positive control |
| sparse_endgame-026 | teacher_child_value_override | 384 | 5 | false | 0.0026 | 0.7240 | 0.8384 | false | override first child expansion values with teacher child values |
| sparse_endgame-026 | teacher_child_value_override | 1200 | 5 | false | 0.0008 | 0.8550 | 0.7072 | false | override first child expansion values with teacher child values |
| sparse_endgame-026 | neural_child_value_swap | 384 | 5 | false | 0.0026 | 0.8177 | 0.1982 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-026 | neural_child_value_swap | 1200 | 5 | false | 0.0008 | 0.9050 | 0.0131 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-025 | original | 384 | 1 | false | 0.0052 | 0.9922 | 0.1876 | false | no intervention |
| sparse_endgame-025 | original | 1200 | 1 | false | 0.0158 | 0.9833 | 0.0865 | false | no intervention |
| sparse_endgame-025 | equalize_root_priors | 384 | 1 | false | 0.0521 | 0.9453 | 0.1232 | false | equalize corrected reference and selected root priors |
| sparse_endgame-025 | equalize_root_priors | 1200 | 1 | false | 0.0183 | 0.9808 | 0.1758 | false | equalize corrected reference and selected root priors |
| sparse_endgame-025 | uniform_legal_prior | 384 | 1 | false | 0.0521 | 0.9297 | 0.1274 | false | uniform legal prior positive control |
| sparse_endgame-025 | uniform_legal_prior | 1200 | 1 | false | 0.0208 | 0.9675 | 0.2420 | false | uniform legal prior positive control |
| sparse_endgame-025 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.9922 | 0.5084 | false | override first child expansion values with teacher child values |
| sparse_endgame-025 | teacher_child_value_override | 1200 | 1 | false | 0.0042 | 0.9950 | 0.2079 | false | override first child expansion values with teacher child values |
| sparse_endgame-025 | neural_child_value_swap | 384 | 1 | false | 0.0078 | 0.9896 | 0.1416 | false | swap child neural values as diagnostic sanity check |
| sparse_endgame-025 | neural_child_value_swap | 1200 | 1 | false | 0.0158 | 0.9833 | 0.0815 | false | swap child neural values as diagnostic sanity check |

## 11. Row classifications

| row_id | role | row_classification | supporting_evidence | recommended_use | notes |
| --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| sparse_endgame-007 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| sparse_endgame-009 | target_candidate | tablebase_reference_conflict | tablebase is available at the root and selects a different move than the active corrected reference | target_candidate | root still fails |
| sparse_endgame-024 | target_candidate | puct_child_search_value_mismatch | ClassicMCTS child teacher and neural child values prefer corrected reference child but child PUCT does not | target_candidate | root still fails |
| sparse_endgame-026 | target_candidate | tablebase_reference_conflict | tablebase is available at the root and selects a different move than the active corrected reference | target_candidate | root still fails |
| sparse_endgame-025 | target_candidate | tablebase_reference_conflict | tablebase is available at the root and selects a different move than the active corrected reference | target_candidate | root still fails |
| sparse_endgame-023 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| sparse_endgame-002 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| sparse_endgame-019 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| sparse_endgame-020 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| sparse_endgame-011 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |

## 12. Family-level interpretation

| family_classification | value_head_miscalibration_count | puct_child_search_mismatch_count | root_selection_pressure_count | tablebase_reference_conflict_count | corrected_reference_suspicious_count | backup_perspective_suspect_count | inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| tablebase_reference_patch_needed | 0 | 1 | 0 | 3 | 2 | 0 | 0 | produce a non-mutating tablebase-backed reference patch artifact for sparse_endgame and rerun the audit before training. |

## 13. Exactly one recommended next action

Recommendation: **produce a non-mutating tablebase-backed reference patch artifact for sparse_endgame and rerun the audit before training.**
