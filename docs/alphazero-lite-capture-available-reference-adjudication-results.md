# AlphaZero-lite Capture Available Reference Adjudication Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay/value-calibration artifacts were created.
- Active corrected references were not mutated.
- Active corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows path: `/tmp/azlite_corrected_non_opening_failure_mining_v4/selected_non_opening_family_rows_v4.jsonl`.
- Corrected guard rows kept out of training targets: `['capture_available-002', 'capture_available-003', 'capture_available-006', 'capture_available-007', 'capture_available-008']`.

## 2. Why PR #58 blocked value training

- PR #58 did not confirm that `capture_available` is cleanly trainable.
- Family-level classification: `reference_family_uncertain`.
- Mechanism counts: `{"blocked_next_action": "adjudicate capture_available references before training.", "corrected_reference_suspicious_count": 8, "family_classification": "reference_family_uncertain", "inconclusive_count": 2, "puct_child_search_mismatch_count": 1, "root_selection_pressure_count": 0, "value_head_miscalibration_count": 1}`.
- Exact PR #58 recommendation: **adjudicate capture_available references before training.**

## 3. Row validation

| row_id | pr58_classification | active_reference_move | legal | reference_unstable | canonical_state_match | current_selected_384 | current_selected_1200 | current_selected_2400 | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | corrected_reference_suspicious | 3 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| capture_available-013 | corrected_reference_suspicious | 4 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| capture_available-023 | corrected_reference_suspicious | 4 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| capture_available-022 | corrected_reference_suspicious | 3 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| capture_available-016 | corrected_reference_suspicious | 4 | true | false | true | 0 | 0 | 0 | ok | validated for adjudication |
| capture_available-025 | corrected_reference_suspicious | 4 | true | false | true | 0 | 0 | 0 | ok | validated for adjudication |
| capture_available-009 | corrected_reference_suspicious | 4 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| capture_available-001 | corrected_reference_suspicious | 3 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| capture_available-018 | inconclusive | 3 | true | false | true | 2 | 0 | 0 | ok | validated for adjudication |
| capture_available-012 | inconclusive | 5 | true | false | true | 3 | 3 | 3 | ok | validated for adjudication |
| capture_available-019 | value_head_miscalibration | 3 | true | false | true | 0 | 4 | 4 | ok | reported as value-head miscalibration context only |
| capture_available-017 | puct_child_search_mismatch | 3 | true | false | true | 4 | 2 | 2 | ok | reported as child-PUCT mismatch context only |
| capture_available-015 | holdout_context | 3 | true | false | true | 2 | 4 | 4 | ok | reported as holdout context only |
| capture_available-027 | holdout_context | 4 | true | false | true | 0 | 0 | 4 | ok | reported as holdout context only |
| capture_available-021 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |
| capture_available-010 | preservation_control | 2 | true | false | true | 2 | 2 | 2 | ok | reported as preservation control only |
| capture_available-020 | preservation_control | 3 | true | false | true | 3 | 3 | 3 | ok | reported as preservation control only |
| capture_available-011 | preservation_control | 2 | true | false | true | 2 | 2 | 3 | ok | reported as preservation control only |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Repo convention: `follow state_to_root_perspective_value: child values stay positive for the same player-to-move and are sign-flipped when the turn hands off`.

## 4. Root ClassicMCTS adjudication

| row_id | budget | seeds | active_reference_move | observed_top_moves | majority_move | majority_fraction | reference_selected_fraction | top1_margin_mean | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | 1200 | 7 | 3 | 1:1, 3:3, 4:3 | 3 | 0.4286 | 0.4286 | 0.0294 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-024 | 2400 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0297 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-024 | 5000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0378 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-024 | 10000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0704 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-024 | 30000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.1232 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-013 | 1200 | 7 | 4 | 3:1, 4:6 | 4 | 0.8571 | 0.8571 | 0.0412 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-013 | 2400 | 7 | 4 | 2:3, 3:1, 4:3 | 2 | 0.4286 | 0.4286 | 0.1077 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-013 | 5000 | 7 | 4 | 2:5, 3:2 | 2 | 0.7143 | 0.0000 | 0.0678 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-013 | 10000 | 7 | 4 | 2:7 | 2 | 1.0000 | 0.0000 | 0.1251 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-013 | 30000 | 7 | 4 | 2:7 | 2 | 1.0000 | 0.0000 | 0.3144 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-023 | 1200 | 7 | 4 | 1:5, 3:2 | 1 | 0.7143 | 0.0000 | 0.0291 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-023 | 2400 | 7 | 4 | 1:6, 3:1 | 1 | 0.8571 | 0.0000 | 0.0274 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-023 | 5000 | 7 | 4 | 1:1, 3:6 | 3 | 0.8571 | 0.0000 | 0.0204 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-023 | 10000 | 7 | 4 | 3:7 | 3 | 1.0000 | 0.0000 | 0.0714 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-023 | 30000 | 7 | 4 | 3:7 | 3 | 1.0000 | 0.0000 | 0.1677 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-022 | 1200 | 7 | 3 | 1:1, 3:6 | 3 | 0.8571 | 0.8571 | 0.0261 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-022 | 2400 | 7 | 3 | 1:5, 3:2 | 1 | 0.7143 | 0.2857 | 0.0161 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-022 | 5000 | 7 | 3 | 1:1, 3:6 | 3 | 0.8571 | 0.8571 | 0.0313 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-022 | 10000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0614 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-022 | 30000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.1521 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-016 | 1200 | 7 | 4 | 3:1, 4:6 | 4 | 0.8571 | 0.8571 | 0.0500 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-016 | 2400 | 7 | 4 | 2:7 | 2 | 1.0000 | 0.0000 | 0.0477 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-016 | 5000 | 7 | 4 | 2:7 | 2 | 1.0000 | 0.0000 | 0.2195 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-016 | 10000 | 7 | 4 | 2:7 | 2 | 1.0000 | 0.0000 | 0.3081 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-016 | 30000 | 7 | 4 | 2:7 | 2 | 1.0000 | 0.0000 | 0.3812 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-025 | 1200 | 7 | 4 | 3:2, 4:5 | 4 | 0.7143 | 0.7143 | 0.0303 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-025 | 2400 | 7 | 4 | 3:5, 4:2 | 3 | 0.7143 | 0.2857 | 0.0449 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-025 | 5000 | 7 | 4 | 3:5, 4:2 | 3 | 0.7143 | 0.2857 | 0.0202 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-025 | 10000 | 7 | 4 | 3:7 | 3 | 1.0000 | 0.0000 | 0.0393 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-025 | 30000 | 7 | 4 | 3:7 | 3 | 1.0000 | 0.0000 | 0.2730 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-009 | 1200 | 7 | 4 | 1:2, 4:5 | 4 | 0.7143 | 0.7143 | 0.0665 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-009 | 2400 | 7 | 4 | 1:7 | 1 | 1.0000 | 0.0000 | 0.1000 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-009 | 5000 | 7 | 4 | 1:7 | 1 | 1.0000 | 0.0000 | 0.1443 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-009 | 10000 | 7 | 4 | 1:7 | 1 | 1.0000 | 0.0000 | 0.1765 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-009 | 30000 | 7 | 4 | 1:7 | 1 | 1.0000 | 0.0000 | 0.2119 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-001 | 1200 | 7 | 3 | 2:7 | 2 | 1.0000 | 0.0000 | 0.0772 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-001 | 2400 | 7 | 3 | 2:7 | 2 | 1.0000 | 0.0000 | 0.0648 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-001 | 5000 | 7 | 3 | 2:3, 3:4 | 3 | 0.5714 | 0.5714 | 0.0198 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-001 | 10000 | 7 | 3 | 2:4, 3:3 | 2 | 0.5714 | 0.4286 | 0.0170 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-001 | 30000 | 7 | 3 | 2:7 | 2 | 1.0000 | 0.0000 | 0.0820 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-018 | 1200 | 7 | 3 | 2:4, 3:2, 4:1 | 2 | 0.5714 | 0.2857 | 0.0231 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-018 | 2400 | 7 | 3 | 2:4, 3:3 | 2 | 0.5714 | 0.4286 | 0.0697 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-018 | 5000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.0709 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-018 | 10000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.2142 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-018 | 30000 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.3076 | supports_active_reference | high-budget majority stays on the active reference |
| capture_available-012 | 1200 | 7 | 5 | 3:6, 5:1 | 3 | 0.8571 | 0.1429 | 0.0300 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-012 | 2400 | 7 | 5 | 0:1, 3:3, 5:3 | 3 | 0.4286 | 0.4286 | 0.0127 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-012 | 5000 | 7 | 5 | 0:7 | 0 | 1.0000 | 0.0000 | 0.0323 | supports_flip_candidate | high-budget majority prefers another move |
| capture_available-012 | 10000 | 7 | 5 | 0:4, 3:2, 5:1 | 0 | 0.5714 | 0.1429 | 0.0079 | unstable_or_mixed | seed majorities remain mixed at this budget |
| capture_available-012 | 30000 | 7 | 5 | 3:7 | 3 | 1.0000 | 0.0000 | 0.1190 | supports_flip_candidate | high-budget majority prefers another move |

## 5. Child-afterstate adjudication

| row_id | active_reference_move | comparison_move | budget | active_reference_child_value_root_mean | comparison_child_value_root_mean | active_minus_comparison_value | child_selected_moves | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | 3 | 1 | 1200 | 0.0559 | 0.8221 | -0.7662 | ref[2:3, 4:2] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-024 | 3 | 1 | 2400 | -0.1468 | 0.7990 | -0.9458 | ref[2:3, 4:2] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-024 | 3 | 1 | 5000 | -0.3341 | 0.7910 | -1.1251 | ref[2:4, 4:1] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-013 | 4 | 2 | 1200 | -0.0578 | -0.3034 | 0.2456 | ref[1:2, 2:1, 3:1, 5:1] cmp[3:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-013 | 4 | 2 | 2400 | -0.2641 | -0.3760 | 0.1119 | ref[1:4, 2:1] cmp[3:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-013 | 4 | 2 | 5000 | -0.5831 | -0.4346 | -0.1485 | ref[1:4, 2:1] cmp[3:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-023 | 4 | 3 | 1200 | -0.0851 | -0.0655 | -0.0196 | ref[1:3, 3:1, 5:1] cmp[2:3, 3:1, 5:1] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-023 | 4 | 1 | 1200 | -0.0851 | 0.7900 | -0.8751 | ref[1:3, 3:1, 5:1] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-023 | 4 | 3 | 2400 | -0.3269 | -0.2812 | -0.0457 | ref[1:4, 3:1] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-023 | 4 | 1 | 2400 | -0.3269 | 0.7285 | -1.0554 | ref[1:4, 3:1] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-023 | 4 | 3 | 5000 | -0.6077 | -0.4718 | -0.1359 | ref[1:4, 3:1] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-023 | 4 | 1 | 5000 | -0.6077 | 0.7268 | -1.3345 | ref[1:4, 3:1] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-022 | 3 | 1 | 1200 | -0.0687 | 0.7925 | -0.8612 | ref[4:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-022 | 3 | 1 | 2400 | -0.1673 | 0.7599 | -0.9272 | ref[4:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-022 | 3 | 1 | 5000 | -0.4149 | 0.7229 | -1.1378 | ref[4:5] cmp[4:2, 5:3] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-016 | 4 | 2 | 1200 | -0.3840 | -0.5676 | 0.1836 | ref[3:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-016 | 4 | 0 | 1200 | -0.3840 | -0.0969 | -0.2871 | ref[3:5] cmp[1:1, 3:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-016 | 4 | 2 | 2400 | -0.5086 | -0.6576 | 0.1490 | ref[3:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-016 | 4 | 0 | 2400 | -0.5086 | -0.2305 | -0.2781 | ref[3:5] cmp[1:2, 3:3] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-016 | 4 | 2 | 5000 | -0.6365 | -0.7555 | 0.1190 | ref[3:5] cmp[0:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-016 | 4 | 0 | 5000 | -0.6365 | -0.4192 | -0.2173 | ref[3:5] cmp[1:2, 3:3] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-025 | 4 | 3 | 1200 | -0.4745 | -0.2511 | -0.2234 | ref[3:5] cmp[3:1, 4:4] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-025 | 4 | 0 | 1200 | -0.4745 | -0.1041 | -0.3704 | ref[3:5] cmp[1:4, 4:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-025 | 4 | 3 | 2400 | -0.5959 | -0.3595 | -0.2364 | ref[3:5] cmp[3:1, 4:4] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-025 | 4 | 0 | 2400 | -0.5959 | -0.3099 | -0.2860 | ref[3:5] cmp[1:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-025 | 4 | 3 | 5000 | -0.6758 | -0.3707 | -0.3051 | ref[3:5] cmp[4:3, 5:2] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-025 | 4 | 0 | 5000 | -0.6758 | -0.5898 | -0.0860 | ref[3:5] cmp[1:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-009 | 4 | 1 | 1200 | -0.1957 | 0.7079 | -0.9036 | ref[3:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-009 | 4 | 1 | 2400 | -0.3818 | 0.6867 | -1.0685 | ref[3:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-009 | 4 | 1 | 5000 | -0.5507 | 0.7014 | -1.2521 | ref[3:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-001 | 3 | 2 | 1200 | -0.2181 | 0.6937 | -0.9118 | ref[3:5] cmp[3:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-001 | 3 | 2 | 2400 | -0.4801 | 0.6675 | -1.1476 | ref[3:5] cmp[3:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-001 | 3 | 2 | 5000 | -0.6380 | 0.6657 | -1.3037 | ref[3:5] cmp[3:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-018 | 3 | 0 | 1200 | -0.2245 | -0.1332 | -0.0913 | ref[2:4, 5:1] cmp[1:3, 2:2] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-018 | 3 | 0 | 2400 | -0.3181 | -0.2862 | -0.0319 | ref[2:4, 5:1] cmp[1:4, 2:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-018 | 3 | 0 | 5000 | -0.4932 | -0.4676 | -0.0256 | ref[2:5] cmp[1:4, 2:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| capture_available-012 | 5 | 3 | 1200 | -0.1981 | -0.2437 | 0.0456 | ref[2:3, 4:2] cmp[4:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-012 | 5 | 3 | 2400 | -0.3954 | -0.5055 | 0.1101 | ref[2:3, 4:2] cmp[4:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| capture_available-012 | 5 | 3 | 5000 | -0.6122 | -0.6316 | 0.0194 | ref[2:3, 4:2] cmp[4:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_value_root | tablebase_preferred_move | agrees_with_classic_majority | agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-024 | child_after_move_3 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-024 | child_after_move_1 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-013 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-013 | child_after_move_4 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-013 | child_after_move_2 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-023 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-023 | child_after_move_4 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-023 | child_after_move_3 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-023 | child_after_move_1 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-022 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-022 | child_after_move_3 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-022 | child_after_move_1 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-016 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-016 | child_after_move_4 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-016 | child_after_move_2 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-016 | child_after_move_0 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-025 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-025 | child_after_move_4 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-025 | child_after_move_3 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-025 | child_after_move_0 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-009 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-009 | child_after_move_4 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-009 | child_after_move_1 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-001 | root | 47 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-001 | child_after_move_3 | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-001 | child_after_move_2 | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-018 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-018 | child_after_move_3 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-018 | child_after_move_0 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-012 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-012 | child_after_move_5 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| capture_available-012 | child_after_move_3 | 45 | false | - | - | - | - | not solvable under the repo threshold |

## 7. PUCT/artifact teacher comparison

| row_id | budget | active_reference_move | puct_selected_move | puct_reference_visit_share | puct_selected_visit_share | puct_agrees_with_classic_majority | puct_agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-024 | 384 | 3 | 1 | 0.1224 | 0.7812 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-024 | 1200 | 3 | 1 | 0.0392 | 0.9283 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-024 | 2400 | 3 | 1 | 0.0196 | 0.9629 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-024 | 5000 | 3 | 1 | 0.0094 | 0.9816 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-013 | 384 | 4 | 2 | 0.0078 | 0.6484 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-013 | 1200 | 4 | 2 | 0.0025 | 0.7608 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-013 | 2400 | 4 | 2 | 0.0029 | 0.7854 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-013 | 5000 | 4 | 3 | 0.0016 | 0.5096 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-023 | 384 | 4 | 1 | 0.0260 | 0.9219 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-023 | 1200 | 4 | 1 | 0.0083 | 0.9700 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-023 | 2400 | 4 | 1 | 0.0046 | 0.9829 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-023 | 5000 | 4 | 1 | 0.0024 | 0.9904 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-022 | 384 | 3 | 1 | 0.0391 | 0.9375 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-022 | 1200 | 3 | 1 | 0.0125 | 0.9708 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-022 | 2400 | 3 | 1 | 0.0063 | 0.9833 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-022 | 5000 | 3 | 1 | 0.0030 | 0.9914 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-016 | 384 | 4 | 0 | 0.0026 | 0.5234 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-016 | 1200 | 4 | 0 | 0.0017 | 0.7025 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-016 | 2400 | 4 | 0 | 0.0537 | 0.5129 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-016 | 5000 | 4 | 4 | 0.5450 | 0.5450 | false | true | deterministic artifact PUCT with PR #58 settings |
| capture_available-025 | 384 | 4 | 0 | 0.2786 | 0.6120 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-025 | 1200 | 4 | 0 | 0.0892 | 0.8758 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-025 | 2400 | 4 | 0 | 0.0446 | 0.9379 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-025 | 5000 | 4 | 0 | 0.0222 | 0.9692 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-009 | 384 | 4 | 1 | 0.0156 | 0.9714 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-009 | 1200 | 4 | 1 | 0.0050 | 0.9825 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-009 | 2400 | 4 | 1 | 0.0029 | 0.9879 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-009 | 5000 | 4 | 1 | 0.0014 | 0.9936 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-001 | 384 | 3 | 2 | 0.0026 | 0.8880 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-001 | 1200 | 3 | 2 | 0.0183 | 0.9408 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-001 | 2400 | 3 | 2 | 0.0108 | 0.9679 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-001 | 5000 | 3 | 2 | 0.0052 | 0.9822 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-018 | 384 | 3 | 2 | 0.0026 | 0.5443 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-018 | 1200 | 3 | 0 | 0.0008 | 0.6650 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-018 | 2400 | 3 | 0 | 0.0004 | 0.7242 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-018 | 5000 | 3 | 0 | 0.0002 | 0.7914 | false | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-012 | 384 | 5 | 3 | 0.0911 | 0.7995 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-012 | 1200 | 5 | 3 | 0.0325 | 0.8858 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-012 | 2400 | 5 | 3 | 0.0163 | 0.9417 | true | false | deterministic artifact PUCT with PR #58 settings |
| capture_available-012 | 5000 | 5 | 3 | 0.0078 | 0.9708 | true | false | deterministic artifact PUCT with PR #58 settings |

## 8. Row decisions

| row_id | active_reference_move | adjudicated_decision | proposed_reference_move | proposed_unstable | evidence_summary | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| capture_available-001 | 3 | reference_unstable | - | true | highest_root_majority=2 fraction=1.0000; highest_puct_selected=2; child_support=supports_comparison_move; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| capture_available-009 | 4 | reference_should_flip | 1 | false | highest_root_majority=1 fraction=1.0000; highest_puct_selected=1; child_support=supports_comparison_move; tablebase_root_move=- | requires reviewed reference patch before training use | high-budget ClassicMCTS root and child-afterstate evidence consistently prefer another move |
| capture_available-012 | 5 | reference_unstable | - | true | highest_root_majority=3 fraction=1.0000; highest_puct_selected=3; child_support=supports_active_reference; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| capture_available-013 | 4 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=3; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search prefers another move, but child-afterstate evidence stays mixed |
| capture_available-016 | 4 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=supports_active_reference; tablebase_root_move=- | exclude pending more evidence | root search and child-afterstate do not align cleanly enough to flip |
| capture_available-018 | 3 | still_inconclusive | - | false | highest_root_majority=3 fraction=1.0000; highest_puct_selected=0; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| capture_available-022 | 3 | still_inconclusive | - | false | highest_root_majority=3 fraction=1.0000; highest_puct_selected=1; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| capture_available-023 | 4 | reference_should_flip | 3 | false | highest_root_majority=3 fraction=1.0000; highest_puct_selected=1; child_support=supports_comparison_move; tablebase_root_move=- | requires reviewed reference patch before training use | high-budget ClassicMCTS root and child-afterstate evidence consistently prefer another move |
| capture_available-024 | 3 | still_inconclusive | - | false | highest_root_majority=3 fraction=1.0000; highest_puct_selected=1; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| capture_available-025 | 4 | reference_should_flip | 3 | false | highest_root_majority=3 fraction=1.0000; highest_puct_selected=0; child_support=supports_comparison_move; tablebase_root_move=- | requires reviewed reference patch before training use | high-budget ClassicMCTS root and child-afterstate evidence consistently prefer another move |

## 9. Proposed non-mutating patch artifact

- Proposed review artifact: `/tmp/azlite_capture_available_reference_adjudication/capture_available_reference_review_patch_v1.json`.
- `do_not_auto_apply` is set for every proposed row.
- The active fixture remained unchanged.

## 10. Projected clean targetability

| bucket | row_count | rows | train_target_eligible_count | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| confirmed_active_reference_targets | 0 | - | 0 | still limited to rows with confirmed active references only | usable only if the family still clears the final branch decision |
| reference_flip_candidates | 3 | capture_available-009, capture_available-023, capture_available-025 | 0 | fixture changes need explicit review before they are safe to use | review the proposed patch artifact before any training |
| unstable_or_excluded | 7 | capture_available-001, capture_available-012, capture_available-013, capture_available-016, capture_available-018, capture_available-022, capture_available-024 | 0 | unstable or mixed labels would contaminate any hard target set | exclude from hard gates and training targets |
| puct_teacher_divergence_rows | 0 | - | 0 | teacher-policy ambiguity remains unresolved for these rows | decide teacher policy before any training use |
| confirmed_value_head_miscalibration_context | 1 | capture_available-019 | 0 | usable only after references settle and later value calibration is explicitly chosen | reserve for later value-calibration only after reference adjudication settles |
| confirmed_puct_child_mismatch_context | 1 | capture_available-017 | 0 | child-search teacher mismatch remains unresolved for deterministic PUCT | keep diagnostic-only until child-PUCT-specific follow-up is done |
| confirmed_preservation_controls | 3 | capture_available-010, capture_available-020, capture_available-021 | 0 | controls must stay unchanged during any follow-up | preserve unchanged as regression checks |
| holdout_context_rows | 2 | capture_available-015, capture_available-027 | 0 | holdouts should remain context-only during follow-up | report only; do not train on them in this branch |

- Target-candidate rows still eligible under active references: `0`.
- Target-candidate rows that would require reference patching: `3`.
- Target-candidate rows that should stay excluded from training: `7`.
- `capture_available` remains a good training target after adjudication: `false`.
- `capture_available-019` is safe enough to seed a later value-calibration bucket: `false`.
- `capture_available-017` requires child-PUCT-specific follow-up: `true`.
- Projected family decision: `reference_suite_too_noisy_for_capture_available`.

## 11. Exactly one recommended next action

Recommendation: **exclude unstable rows and either target the smaller stable bucket or select the next non-opening family.**
