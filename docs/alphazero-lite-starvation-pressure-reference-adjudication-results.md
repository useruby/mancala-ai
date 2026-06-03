# AlphaZero-lite Starvation Pressure Reference Adjudication Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay/value-calibration artifacts were created.
- Active corrected references were not mutated.
- Active corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows path: `/tmp/azlite_corrected_non_opening_failure_mining_v5/selected_non_opening_family_rows_v5.jsonl`.

## 2. Why PR #61 blocked value training

- PR #61 did not confirm that `starvation_pressure` is cleanly trainable.
- Family-level classification: `reference_family_uncertain`.
- Mechanism counts: `{"blocked_next_action": "adjudicate starvation_pressure references before training.", "corrected_reference_suspicious_count": 6, "family_classification": "reference_family_uncertain", "inconclusive_count": 1, "puct_child_search_mismatch_count": 0, "root_selection_pressure_count": 0, "value_head_miscalibration_count": 0}`.
- Exact PR #61 recommendation: **adjudicate starvation_pressure references before training.**

## 3. Row validation

| row_id | pr61_classification | active_reference_move | legal | reference_unstable | canonical_state_match | current_selected_384 | current_selected_1200 | current_selected_2400 | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | corrected_reference_suspicious | 0 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| starvation_pressure-026 | corrected_reference_suspicious | 0 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| starvation_pressure-024 | corrected_reference_suspicious | 1 | true | false | true | 3 | 3 | 3 | ok | validated for adjudication |
| starvation_pressure-022 | corrected_reference_suspicious | 2 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| starvation_pressure-023 | corrected_reference_suspicious | 2 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| starvation_pressure-027 | corrected_reference_suspicious | 0 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| starvation_pressure-012 | inconclusive | 3 | true | false | true | 5 | 5 | 5 | ok | validated for adjudication |
| starvation_pressure-015 | holdout_context | 4 | true | false | true | 2 | 2 | 2 | ok | reported as holdout context only |
| starvation_pressure-001 | holdout_context | 1 | true | false | true | 4 | 4 | 4 | ok | reported as holdout context only |
| starvation_pressure-013 | preservation_control | 4 | true | false | true | 4 | 4 | 4 | ok | reported as preservation control only |
| starvation_pressure-003 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |
| starvation_pressure-021 | preservation_control | 0 | true | false | true | 0 | 0 | 0 | ok | reported as preservation control only |
| starvation_pressure-014 | preservation_control | 2 | true | false | true | 2 | 2 | 2 | ok | reported as preservation control only |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Repo convention: `follow state_to_root_perspective_value: child values stay positive for the same player-to-move and are sign-flipped when the turn hands off`.

## 4. Root ClassicMCTS adjudication

| row_id | budget | seeds | active_reference_move | observed_top_moves | majority_move | majority_fraction | reference_selected_fraction | top1_margin_mean | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.2689 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-025 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.3504 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-025 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.3935 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-025 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.4195 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-025 | 30000 | 7 | 0 | 2:7 | 2 | 1.0000 | 0.0000 | 0.0020 | supports_flip_candidate | high-budget majority prefers another move |
| starvation_pressure-026 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.3576 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-026 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.4174 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-026 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.4334 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-026 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 1.2682 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-026 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0021 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-024 | 1200 | 7 | 1 | 1:7 | 1 | 1.0000 | 1.0000 | 0.9478 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-024 | 2400 | 7 | 1 | 1:7 | 1 | 1.0000 | 1.0000 | 1.0200 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-024 | 5000 | 7 | 1 | 1:7 | 1 | 1.0000 | 1.0000 | 1.0579 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-024 | 10000 | 7 | 1 | 1:7 | 1 | 1.0000 | 1.0000 | 1.0767 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-024 | 30000 | 7 | 1 | 1:7 | 1 | 1.0000 | 1.0000 | 1.0904 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-022 | 1200 | 7 | 2 | 1:1, 2:6 | 2 | 0.8571 | 0.8571 | 0.2653 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-022 | 2400 | 7 | 2 | 1:1, 2:6 | 2 | 0.8571 | 0.8571 | 0.6291 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-022 | 5000 | 7 | 2 | 1:1, 2:6 | 2 | 0.8571 | 0.8571 | 0.9985 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-022 | 10000 | 7 | 2 | 1:1, 2:6 | 2 | 0.8571 | 0.8571 | 1.2893 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-022 | 30000 | 7 | 2 | 1:1, 2:6 | 2 | 0.8571 | 0.8571 | 1.5115 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-023 | 1200 | 7 | 2 | 1:2, 2:5 | 2 | 0.7143 | 0.7143 | 0.0711 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-023 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.5525 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-023 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.1485 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-023 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.3770 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-023 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.5238 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-027 | 1200 | 7 | 0 | 0:6, 2:1 | 0 | 0.8571 | 0.8571 | 1.0122 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-027 | 2400 | 7 | 0 | 0:6, 2:1 | 0 | 0.8571 | 0.8571 | 1.0949 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-027 | 5000 | 7 | 0 | 0:6, 2:1 | 0 | 0.8571 | 0.8571 | 1.1436 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-027 | 10000 | 7 | 0 | 0:6, 2:1 | 0 | 0.8571 | 0.8571 | 1.1706 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-027 | 30000 | 7 | 0 | 0:2, 2:5 | 2 | 0.7143 | 0.2857 | 0.1669 | supports_flip_candidate | high-budget majority prefers another move |
| starvation_pressure-012 | 1200 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0647 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-012 | 2400 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0701 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-012 | 5000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0674 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-012 | 10000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0865 | supports_active_reference | high-budget majority stays on the active reference |
| starvation_pressure-012 | 30000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.1234 | supports_active_reference | high-budget majority stays on the active reference |

## 5. Child-afterstate adjudication

| row_id | active_reference_move | comparison_move | budget | active_reference_child_value_root_mean | comparison_child_value_root_mean | active_minus_comparison_value | child_selected_moves | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | 0 | 2 | 1200 | -0.7624 | -0.9040 | 0.1416 | ref[0:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-025 | 0 | 4 | 1200 | -0.7624 | -0.5767 | -0.1857 | ref[0:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-025 | 0 | 2 | 2400 | -0.8426 | -0.9252 | 0.0826 | ref[0:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-025 | 0 | 4 | 2400 | -0.8426 | -0.6828 | -0.1598 | ref[0:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-025 | 0 | 2 | 5000 | -0.8885 | -0.9272 | 0.0387 | ref[0:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-025 | 0 | 4 | 5000 | -0.8885 | -0.8069 | -0.0816 | ref[0:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-026 | 0 | 4 | 1200 | -0.8366 | -0.4408 | -0.3958 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-026 | 0 | 4 | 2400 | -0.8813 | -0.5517 | -0.3296 | ref[5:5] cmp[0:2, 3:3] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-026 | 0 | 4 | 5000 | -0.8951 | -0.7828 | -0.1123 | ref[5:5] cmp[0:3, 3:2] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-024 | 1 | 3 | 1200 | -0.8868 | -0.3861 | -0.5007 | ref[0:5] cmp[0:4, 5:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-024 | 1 | 3 | 2400 | -0.9019 | -0.6422 | -0.2597 | ref[0:5] cmp[0:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-024 | 1 | 3 | 5000 | -0.8983 | -0.7936 | -0.1047 | ref[0:5] cmp[0:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-022 | 2 | 1 | 1200 | -0.6747 | -0.6628 | -0.0119 | ref[1:5] cmp[2:1, 3:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-022 | 2 | 1 | 2400 | -0.7642 | -0.6486 | -0.1156 | ref[1:5] cmp[2:3, 3:2] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-022 | 2 | 1 | 5000 | -0.8440 | -0.7415 | -0.1025 | ref[1:5] cmp[2:1, 3:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-023 | 2 | 1 | 1200 | -0.8274 | -0.8135 | -0.0139 | ref[5:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-023 | 2 | 1 | 2400 | -0.7658 | -0.7691 | 0.0033 | ref[5:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-023 | 2 | 1 | 5000 | -0.8213 | -0.7061 | -0.1152 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-027 | 0 | 2 | 1200 | -0.8436 | -0.9223 | 0.0787 | ref[5:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-027 | 0 | 1 | 1200 | -0.8436 | -0.7912 | -0.0524 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-027 | 0 | 2 | 2400 | -0.8784 | -0.9339 | 0.0555 | ref[5:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-027 | 0 | 1 | 2400 | -0.8784 | -0.8444 | -0.0340 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-027 | 0 | 2 | 5000 | -0.9122 | -0.9372 | 0.0250 | ref[5:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-027 | 0 | 1 | 5000 | -0.9122 | -0.8903 | -0.0219 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-012 | 3 | 5 | 1200 | 0.6210 | 0.3911 | 0.2299 | ref[1:4, 4:1] cmp[1:1, 2:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-012 | 3 | 5 | 2400 | 0.4046 | 0.0070 | 0.3976 | ref[1:3, 2:1, 3:1] cmp[1:1, 2:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| starvation_pressure-012 | 3 | 5 | 5000 | -0.2169 | -0.3586 | 0.1417 | ref[1:3, 2:1, 3:1] cmp[1:1, 2:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_value_root | tablebase_preferred_move | agrees_with_classic_majority | agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | root | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-025 | child_after_move_0 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-025 | child_after_move_2 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-025 | child_after_move_4 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-026 | root | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-026 | child_after_move_0 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-026 | child_after_move_4 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-024 | root | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-024 | child_after_move_1 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-024 | child_after_move_3 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-022 | root | 44 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-022 | child_after_move_2 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-022 | child_after_move_1 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-023 | root | 44 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-023 | child_after_move_2 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-023 | child_after_move_1 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-027 | root | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-027 | child_after_move_0 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-027 | child_after_move_2 | 43 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-027 | child_after_move_1 | 42 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-012 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-012 | child_after_move_3 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| starvation_pressure-012 | child_after_move_5 | 44 | false | - | - | - | - | not solvable under the repo threshold |

## 7. PUCT/artifact teacher comparison

| row_id | budget | active_reference_move | puct_selected_move | puct_reference_visit_share | puct_selected_visit_share | puct_agrees_with_classic_majority | puct_agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-025 | 384 | 0 | 4 | 0.0026 | 0.8125 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-025 | 1200 | 0 | 4 | 0.0008 | 0.8292 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-025 | 2400 | 0 | 4 | 0.0004 | 0.9025 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-025 | 5000 | 0 | 4 | 0.0002 | 0.9406 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-026 | 384 | 0 | 4 | 0.0156 | 0.8698 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-026 | 1200 | 0 | 4 | 0.0050 | 0.9500 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-026 | 2400 | 0 | 4 | 0.0025 | 0.9404 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-026 | 5000 | 0 | 4 | 0.0012 | 0.5928 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-024 | 384 | 1 | 3 | 0.0234 | 0.5807 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-024 | 1200 | 1 | 3 | 0.0075 | 0.6167 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-024 | 2400 | 1 | 3 | 0.0042 | 0.8008 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-024 | 5000 | 1 | 3 | 0.0028 | 0.9012 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-022 | 384 | 2 | 1 | 0.0182 | 0.9427 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-022 | 1200 | 2 | 1 | 0.0067 | 0.9750 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-022 | 2400 | 2 | 1 | 0.0033 | 0.9692 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-022 | 5000 | 2 | 1 | 0.0028 | 0.5302 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-023 | 384 | 2 | 1 | 0.0417 | 0.7630 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-023 | 1200 | 2 | 1 | 0.0225 | 0.9092 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-023 | 2400 | 2 | 1 | 0.0183 | 0.6350 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-023 | 5000 | 2 | 3 | 0.0184 | 0.6088 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-027 | 384 | 0 | 1 | 0.0026 | 0.9505 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-027 | 1200 | 0 | 1 | 0.0033 | 0.9650 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-027 | 2400 | 0 | 1 | 0.4100 | 0.5696 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-027 | 5000 | 0 | 0 | 0.7146 | 0.7146 | false | true | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-012 | 384 | 3 | 5 | 0.0130 | 0.9870 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-012 | 1200 | 3 | 5 | 0.0083 | 0.9917 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-012 | 2400 | 3 | 5 | 0.0058 | 0.9942 | false | false | deterministic artifact PUCT with PR #61 settings |
| starvation_pressure-012 | 5000 | 3 | 5 | 0.0040 | 0.9960 | false | false | deterministic artifact PUCT with PR #61 settings |

## 8. Row decisions

| row_id | active_reference_move | adjudicated_decision | proposed_reference_move | proposed_unstable | evidence_summary | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| starvation_pressure-012 | 3 | still_inconclusive | - | false | highest_root_majority=3 fraction=1.0000; highest_puct_selected=5; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| starvation_pressure-022 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=0.8571; highest_puct_selected=1; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| starvation_pressure-023 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=3; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| starvation_pressure-024 | 1 | still_inconclusive | - | false | highest_root_majority=1 fraction=1.0000; highest_puct_selected=3; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| starvation_pressure-025 | 0 | reference_unstable | - | true | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=supports_active_reference; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| starvation_pressure-026 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| starvation_pressure-027 | 0 | reference_unstable | - | true | highest_root_majority=2 fraction=0.7143; highest_puct_selected=0; child_support=supports_active_reference; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |

## 9. Proposed non-mutating patch artifact

- Proposed review artifact: `/tmp/azlite_starvation_pressure_reference_adjudication/starvation_pressure_reference_review_patch_v1.json`.
- `do_not_auto_apply` is set for every proposed row.
- The active fixture remained unchanged.

## 10. Projected clean targetability

| bucket | row_count | rows | train_target_eligible_count | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| confirmed_active_reference_targets | 0 | - | 0 | still limited to rows with confirmed active references only | usable only if the family still clears the final branch decision |
| reference_flip_candidates | 0 | - | 0 | fixture changes need explicit review before they are safe to use | review the proposed patch artifact before any training |
| unstable_or_excluded | 7 | starvation_pressure-012, starvation_pressure-022, starvation_pressure-023, starvation_pressure-024, starvation_pressure-025, starvation_pressure-026, starvation_pressure-027 | 0 | unstable or mixed labels would contaminate any hard target set | exclude from hard gates and training targets |
| puct_teacher_divergence_rows | 0 | - | 0 | teacher-policy ambiguity remains unresolved for these rows | decide teacher policy before any training use |
| confirmed_preservation_controls | 4 | starvation_pressure-003, starvation_pressure-013, starvation_pressure-014, starvation_pressure-021 | 0 | controls must stay unchanged during any follow-up | preserve unchanged as regression checks |
| holdout_context_rows | 2 | starvation_pressure-001, starvation_pressure-015 | 0 | holdouts should remain context-only during follow-up | report only; do not train on them in this branch |

- Target-candidate rows still eligible under active references: `0`.
- Target-candidate rows that would require reference patching: `0`.
- Target-candidate rows that should stay excluded from training: `7`.
- `starvation_pressure` remains a good training target after adjudication: `false`.
- Projected family decision: `reference_suite_too_noisy_for_starvation_pressure`.

## 11. Exactly one recommended next action

Recommendation: **exclude unstable rows and either target the smaller stable bucket or select the next non-opening family.**
