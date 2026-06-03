# AlphaZero-lite High Value Swing Reference Adjudication Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay/value-calibration artifacts were created.
- Active corrected references were not mutated.
- Active corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows path: `/tmp/azlite_corrected_non_opening_failure_mining_v2/selected_non_opening_family_rows_v2.jsonl`.

## 2. Why PR #52 blocked value training

- PR #52 did not confirm a value-head family gap for `high_value_swing`.
- Family-level classification: `reference_family_uncertain`.
- Mechanism counts: `{"blocked_next_action": "adjudicate high_value_swing references before training.", "corrected_reference_suspicious_count": 12, "family_classification": "reference_family_uncertain", "inconclusive_count": 1, "puct_child_search_mismatch_count": 1, "root_selection_pressure_count": 0, "value_head_miscalibration_count": 0}`.
- Exact PR #52 recommendation: **adjudicate high_value_swing references before training.**

## 3. Row validation

| row_id | pr52_classification | active_reference_move | legal | reference_unstable | canonical_state_match | current_selected_384 | current_selected_1200 | current_selected_2400 | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-007 | corrected_reference_suspicious | 0 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-025 | corrected_reference_suspicious | 0 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-023 | corrected_reference_suspicious | 0 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| high_value_swing-001 | corrected_reference_suspicious | 3 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-021 | corrected_reference_suspicious | 0 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| high_value_swing-013 | corrected_reference_suspicious | 0 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-008 | corrected_reference_suspicious | 0 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| high_value_swing-003 | corrected_reference_suspicious | 0 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-016 | corrected_reference_suspicious | 0 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| high_value_swing-020 | corrected_reference_suspicious | 0 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| high_value_swing-009 | corrected_reference_suspicious | 0 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-015 | corrected_reference_suspicious | 0 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-024 | puct_child_search_mismatch | 1 | true | false | true | 2 | 2 | 2 | ok | validated for adjudication |
| high_value_swing-018 | inconclusive | 0 | true | false | true | 1 | 1 | 1 | ok | validated for adjudication |
| high_value_swing-011 | holdout_context | 0 | true | false | true | 5 | 5 | 5 | ok | reported as holdout context only |
| high_value_swing-022 | holdout_context | 4 | true | false | true | 1 | 1 | 1 | ok | reported as holdout context only |
| high_value_swing-010 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |
| high_value_swing-026 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |
| high_value_swing-027 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |
| high_value_swing-017 | preservation_control | 0 | true | false | true | 0 | 0 | 0 | ok | reported as preservation control only |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Repo convention: `follow state_to_root_perspective_value: child values stay positive for the same player-to-move and are sign-flipped when the turn hands off`.

## 4. Root ClassicMCTS adjudication

| row_id | budget | seeds | active_reference_move | observed_top_moves | majority_move | majority_fraction | reference_selected_fraction | top1_margin_mean | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-007 | 1200 | 7 | 0 | 0:6, 2:1 | 0 | 0.8571 | 0.8571 | 0.0925 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-007 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1111 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-007 | 5000 | 7 | 0 | 0:6, 1:1 | 0 | 0.8571 | 0.8571 | 0.0963 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-007 | 10000 | 7 | 0 | 0:2, 1:5 | 1 | 0.7143 | 0.2857 | 0.1037 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-007 | 30000 | 7 | 0 | 0:2, 1:5 | 1 | 0.7143 | 0.2857 | 0.1893 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-025 | 1200 | 7 | 0 | 0:5, 1:1, 2:1 | 0 | 0.7143 | 0.7143 | 0.0802 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-025 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1532 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-025 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1890 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-025 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2963 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-025 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.4186 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-023 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1100 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-023 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0930 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-023 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0872 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-023 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0597 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-023 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0462 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-001 | 1200 | 7 | 3 | 2:4, 3:3 | 2 | 0.5714 | 0.4286 | 0.0193 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_value_swing-001 | 2400 | 7 | 3 | 0:3, 2:4 | 2 | 0.5714 | 0.0000 | 0.0178 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_value_swing-001 | 5000 | 7 | 3 | 0:7 | 0 | 1.0000 | 0.0000 | 0.0468 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-001 | 10000 | 7 | 3 | 0:7 | 0 | 1.0000 | 0.0000 | 0.0529 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-001 | 30000 | 7 | 3 | 0:7 | 0 | 1.0000 | 0.0000 | 0.0606 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-021 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1613 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-021 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2462 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-021 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2557 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-021 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2384 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-021 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2564 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-013 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0795 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-013 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1147 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-013 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1572 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-013 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1980 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-013 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2418 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-008 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1675 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-008 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1006 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-008 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0243 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-008 | 10000 | 7 | 0 | 1:7 | 1 | 1.0000 | 0.0000 | 0.0412 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-008 | 30000 | 7 | 0 | 1:7 | 1 | 1.0000 | 0.0000 | 0.0978 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-003 | 1200 | 7 | 0 | 0:4, 1:3 | 0 | 0.5714 | 0.5714 | 0.0291 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_value_swing-003 | 2400 | 7 | 0 | 0:6, 1:1 | 0 | 0.8571 | 0.8571 | 0.0362 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-003 | 5000 | 7 | 0 | 0:5, 1:2 | 0 | 0.7143 | 0.7143 | 0.0280 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-003 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0218 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-003 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1177 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-016 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1683 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-016 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1563 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-016 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1506 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-016 | 10000 | 7 | 0 | 0:6, 1:1 | 0 | 0.8571 | 0.8571 | 0.1825 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-016 | 30000 | 7 | 0 | 0:6, 1:1 | 0 | 0.8571 | 0.8571 | 0.2441 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-020 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1302 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-020 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1625 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-020 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1381 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-020 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1508 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-020 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1999 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-009 | 1200 | 7 | 0 | 0:6, 2:1 | 0 | 0.8571 | 0.8571 | 0.0238 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-009 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0351 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-009 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0846 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-009 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0855 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-009 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1074 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-015 | 1200 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1789 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-015 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1815 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-015 | 5000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.1625 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-015 | 10000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2311 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-015 | 30000 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.2899 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-024 | 1200 | 7 | 1 | 1:3, 2:4 | 2 | 0.5714 | 0.4286 | 0.0503 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_value_swing-024 | 2400 | 7 | 1 | 0:1, 2:6 | 2 | 0.8571 | 0.0000 | 0.0298 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-024 | 5000 | 7 | 1 | 1:3, 2:4 | 2 | 0.5714 | 0.4286 | 0.1118 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_value_swing-024 | 10000 | 7 | 1 | 1:3, 2:4 | 2 | 0.5714 | 0.4286 | 0.1849 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_value_swing-024 | 30000 | 7 | 1 | 1:3, 2:4 | 2 | 0.5714 | 0.4286 | 0.3198 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_value_swing-018 | 1200 | 7 | 0 | 0:6, 1:1 | 0 | 0.8571 | 0.8571 | 0.0274 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-018 | 2400 | 7 | 0 | 0:7 | 0 | 1.0000 | 1.0000 | 0.0560 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-018 | 5000 | 7 | 0 | 0:6, 1:1 | 0 | 0.8571 | 0.8571 | 0.0184 | supports_active_reference | high-budget majority stays on the active reference |
| high_value_swing-018 | 10000 | 7 | 0 | 1:7 | 1 | 1.0000 | 0.0000 | 0.0140 | supports_flip_candidate | high-budget majority prefers another move |
| high_value_swing-018 | 30000 | 7 | 0 | 1:7 | 1 | 1.0000 | 0.0000 | 0.0531 | supports_flip_candidate | high-budget majority prefers another move |

## 5. Child-afterstate adjudication

| row_id | active_reference_move | comparison_move | budget | active_reference_child_value_root_mean | comparison_child_value_root_mean | active_minus_comparison_value | child_selected_moves | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-007 | 0 | 1 | 1200 | -0.0840 | 0.7084 | -0.7924 | ref[1:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-007 | 0 | 2 | 1200 | -0.0840 | 0.6301 | -0.7141 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-007 | 0 | 1 | 2400 | -0.3115 | 0.7645 | -1.0760 | ref[1:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-007 | 0 | 2 | 2400 | -0.3115 | 0.6473 | -0.9588 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-007 | 0 | 1 | 5000 | -0.4924 | 0.7839 | -1.2763 | ref[1:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-007 | 0 | 2 | 5000 | -0.4924 | 0.7243 | -1.2167 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-025 | 0 | 2 | 1200 | -0.3850 | 0.5037 | -0.8887 | ref[2:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-025 | 0 | 2 | 2400 | -0.6107 | 0.6244 | -1.2351 | ref[2:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-025 | 0 | 2 | 5000 | -0.7469 | 0.6445 | -1.3914 | ref[2:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-023 | 0 | 1 | 1200 | 0.6668 | 0.8538 | -0.1870 | ref[1:2, 3:3] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-023 | 0 | 1 | 2400 | 0.3307 | 0.8404 | -0.5097 | ref[1:2, 2:1, 3:2] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-023 | 0 | 1 | 5000 | -0.3306 | 0.8497 | -1.1803 | ref[1:2, 2:1, 3:2] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-001 | 3 | 0 | 1200 | 0.1960 | 0.4790 | -0.2830 | ref[1:1, 2:1, 4:3] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-001 | 3 | 2 | 1200 | 0.1960 | 0.8750 | -0.6790 | ref[1:1, 2:1, 4:3] cmp[3:4, 5:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-001 | 3 | 0 | 2400 | -0.0192 | 0.0254 | -0.0446 | ref[1:1, 2:1, 4:3] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-001 | 3 | 2 | 2400 | -0.0192 | 0.8315 | -0.8507 | ref[1:1, 2:1, 4:3] cmp[3:4, 5:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-001 | 3 | 0 | 5000 | -0.2259 | -0.2327 | 0.0068 | ref[1:1, 2:1, 4:3] cmp[2:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-001 | 3 | 2 | 5000 | -0.2259 | 0.8222 | -1.0481 | ref[1:1, 2:1, 4:3] cmp[3:4, 5:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-021 | 0 | 1 | 1200 | -0.1377 | 0.6936 | -0.8313 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-021 | 0 | 1 | 2400 | -0.5197 | 0.7524 | -1.2721 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-021 | 0 | 1 | 5000 | -0.6834 | 0.7935 | -1.4769 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-013 | 0 | 2 | 1200 | -0.2752 | 0.6739 | -0.9491 | ref[3:5] cmp[3:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-013 | 0 | 2 | 2400 | -0.5596 | 0.6577 | -1.2173 | ref[3:5] cmp[3:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-013 | 0 | 2 | 5000 | -0.7400 | 0.6799 | -1.4199 | ref[3:5] cmp[3:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-008 | 0 | 1 | 1200 | 0.2940 | 0.8136 | -0.5196 | ref[1:2, 2:1, 3:2] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-008 | 0 | 1 | 2400 | -0.2245 | 0.8135 | -1.0380 | ref[1:2, 3:3] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-008 | 0 | 1 | 5000 | -0.5269 | 0.8277 | -1.3546 | ref[1:2, 3:3] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-003 | 0 | 2 | 1200 | -0.2599 | 0.7549 | -1.0148 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-003 | 0 | 2 | 2400 | -0.4479 | 0.7393 | -1.1872 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-003 | 0 | 2 | 5000 | -0.5598 | 0.7613 | -1.3211 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-016 | 0 | 1 | 1200 | -0.1536 | 0.6607 | -0.8143 | ref[3:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-016 | 0 | 1 | 2400 | -0.4589 | 0.7482 | -1.2071 | ref[3:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-016 | 0 | 1 | 5000 | -0.6423 | 0.7629 | -1.4052 | ref[3:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-020 | 0 | 1 | 1200 | -0.3228 | 0.7402 | -1.0630 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-020 | 0 | 1 | 2400 | -0.5077 | 0.7571 | -1.2648 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-020 | 0 | 1 | 5000 | -0.6128 | 0.7805 | -1.3933 | ref[1:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-009 | 0 | 2 | 1200 | 0.7056 | 0.9082 | -0.2026 | ref[2:3, 3:1, 4:1] cmp[3:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-009 | 0 | 2 | 2400 | 0.4063 | 0.8667 | -0.4604 | ref[2:3, 3:1, 4:1] cmp[3:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-009 | 0 | 2 | 5000 | -0.1132 | 0.8579 | -0.9711 | ref[2:3, 3:1, 4:1] cmp[3:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-015 | 0 | 2 | 1200 | -0.3465 | 0.6595 | -1.0060 | ref[4:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-015 | 0 | 2 | 2400 | -0.5468 | 0.7075 | -1.2543 | ref[4:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-015 | 0 | 2 | 5000 | -0.6698 | 0.7422 | -1.4120 | ref[4:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-024 | 1 | 2 | 1200 | 0.6162 | 0.5632 | 0.0530 | ref[5:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-024 | 1 | 2 | 2400 | 0.6496 | 0.6426 | 0.0070 | ref[5:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-024 | 1 | 2 | 5000 | 0.6650 | 0.6524 | 0.0126 | ref[5:5] cmp[5:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-018 | 0 | 1 | 1200 | -0.0521 | 0.2389 | -0.2910 | ref[1:5] cmp[3:4, 4:1] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-018 | 0 | 1 | 2400 | -0.3262 | -0.0713 | -0.2549 | ref[1:5] cmp[3:2, 4:3] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_value_swing-018 | 0 | 1 | 5000 | -0.4640 | -0.4164 | -0.0476 | ref[1:5] cmp[3:2, 4:3] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_value_root | tablebase_preferred_move | agrees_with_classic_majority | agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-007 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-007 | child_after_move_0 | 40 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-007 | child_after_move_1 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-007 | child_after_move_2 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-025 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-025 | child_after_move_0 | 39 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-025 | child_after_move_2 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-023 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-023 | child_after_move_0 | 39 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-023 | child_after_move_1 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-001 | root | 47 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-001 | child_after_move_3 | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-001 | child_after_move_0 | 40 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-001 | child_after_move_2 | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-021 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-021 | child_after_move_0 | 39 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-021 | child_after_move_1 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-013 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-013 | child_after_move_0 | 39 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-013 | child_after_move_2 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-008 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-008 | child_after_move_0 | 40 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-008 | child_after_move_1 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-003 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-003 | child_after_move_0 | 40 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-003 | child_after_move_2 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-016 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-016 | child_after_move_0 | 40 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-016 | child_after_move_1 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-020 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-020 | child_after_move_0 | 39 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-020 | child_after_move_1 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-009 | root | 47 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-009 | child_after_move_0 | 39 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-009 | child_after_move_2 | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-015 | root | 46 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-015 | child_after_move_0 | 40 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-015 | child_after_move_2 | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-024 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-024 | child_after_move_1 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-024 | child_after_move_2 | 44 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-018 | root | 45 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-018 | child_after_move_0 | 39 | false | - | - | - | - | not solvable under the repo threshold |
| high_value_swing-018 | child_after_move_1 | 39 | false | - | - | - | - | not solvable under the repo threshold |

## 7. PUCT/artifact teacher comparison

| row_id | budget | active_reference_move | puct_selected_move | puct_reference_visit_share | puct_selected_visit_share | puct_agrees_with_classic_majority | puct_agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-007 | 384 | 0 | 2 | 0.0052 | 0.9635 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-007 | 1200 | 0 | 2 | 0.0050 | 0.9658 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-007 | 2400 | 0 | 2 | 0.0033 | 0.9696 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-007 | 5000 | 0 | 2 | 0.0022 | 0.9818 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-025 | 384 | 0 | 2 | 0.0156 | 0.9583 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-025 | 1200 | 0 | 2 | 0.0108 | 0.9700 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-025 | 2400 | 0 | 2 | 0.0079 | 0.9792 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-025 | 5000 | 0 | 2 | 0.0056 | 0.9868 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-023 | 384 | 0 | 1 | 0.0286 | 0.9635 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-023 | 1200 | 0 | 1 | 0.0192 | 0.9783 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-023 | 2400 | 0 | 1 | 0.0154 | 0.9833 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-023 | 5000 | 0 | 1 | 0.0120 | 0.9874 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-001 | 384 | 3 | 2 | 0.0026 | 0.9844 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-001 | 1200 | 3 | 2 | 0.0008 | 0.9900 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-001 | 2400 | 3 | 2 | 0.0004 | 0.9933 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-001 | 5000 | 3 | 2 | 0.0002 | 0.9958 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-021 | 384 | 0 | 1 | 0.0052 | 0.9948 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-021 | 1200 | 0 | 1 | 0.0050 | 0.9933 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-021 | 2400 | 0 | 1 | 0.0037 | 0.9954 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-021 | 5000 | 0 | 1 | 0.0024 | 0.9972 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-013 | 384 | 0 | 2 | 0.0078 | 0.9427 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-013 | 1200 | 0 | 2 | 0.0033 | 0.9792 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-013 | 2400 | 0 | 2 | 0.0021 | 0.9862 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-013 | 5000 | 0 | 2 | 0.0016 | 0.9910 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-008 | 384 | 0 | 1 | 0.0260 | 0.9453 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-008 | 1200 | 0 | 1 | 0.0158 | 0.9650 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-008 | 2400 | 0 | 1 | 0.0112 | 0.9788 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-008 | 5000 | 0 | 1 | 0.0078 | 0.9868 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-003 | 384 | 0 | 2 | 0.0104 | 0.9427 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-003 | 1200 | 0 | 2 | 0.0125 | 0.9267 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-003 | 2400 | 0 | 2 | 0.0121 | 0.9392 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-003 | 5000 | 0 | 2 | 0.0106 | 0.9448 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-016 | 384 | 0 | 1 | 0.0443 | 0.9219 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-016 | 1200 | 0 | 1 | 0.0142 | 0.9733 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-016 | 2400 | 0 | 1 | 0.0083 | 0.9825 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-016 | 5000 | 0 | 1 | 0.0070 | 0.9862 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-020 | 384 | 0 | 1 | 0.0052 | 0.9896 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-020 | 1200 | 0 | 1 | 0.0042 | 0.9942 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-020 | 2400 | 0 | 1 | 0.0025 | 0.9967 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-020 | 5000 | 0 | 1 | 0.0016 | 0.9980 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-009 | 384 | 0 | 2 | 0.0651 | 0.7995 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-009 | 1200 | 0 | 2 | 0.0208 | 0.9350 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-009 | 2400 | 0 | 2 | 0.0104 | 0.9650 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-009 | 5000 | 0 | 2 | 0.0050 | 0.9796 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-015 | 384 | 0 | 2 | 0.0547 | 0.8333 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-015 | 1200 | 0 | 2 | 0.0175 | 0.9442 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-015 | 2400 | 0 | 2 | 0.0088 | 0.9717 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-015 | 5000 | 0 | 2 | 0.0042 | 0.9848 | false | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-024 | 384 | 1 | 2 | 0.0130 | 0.9818 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-024 | 1200 | 1 | 2 | 0.0100 | 0.9833 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-024 | 2400 | 1 | 2 | 0.0083 | 0.9871 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-024 | 5000 | 1 | 2 | 0.0064 | 0.9904 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-018 | 384 | 0 | 1 | 0.0182 | 0.9766 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-018 | 1200 | 0 | 1 | 0.0108 | 0.9875 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-018 | 2400 | 0 | 1 | 0.0079 | 0.9912 | true | false | deterministic artifact PUCT with PR #52 settings |
| high_value_swing-018 | 5000 | 0 | 1 | 0.0056 | 0.9940 | true | false | deterministic artifact PUCT with PR #52 settings |

## 8. Row decisions

| row_id | active_reference_move | adjudicated_decision | proposed_reference_move | proposed_unstable | evidence_summary | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-001 | 3 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search prefers another move, but child-afterstate evidence stays mixed |
| high_value_swing-003 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-007 | 0 | reference_unstable | - | true | highest_root_majority=1 fraction=0.7143; highest_puct_selected=2; child_support=supports_comparison_move; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| high_value_swing-008 | 0 | reference_unstable | - | true | highest_root_majority=1 fraction=1.0000; highest_puct_selected=1; child_support=supports_comparison_move; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| high_value_swing-009 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-013 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-015 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-016 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=0.8571; highest_puct_selected=1; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-018 | 0 | reference_unstable | - | true | highest_root_majority=1 fraction=1.0000; highest_puct_selected=1; child_support=supports_comparison_move; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| high_value_swing-020 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=1; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-021 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=1; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-023 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=1; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_value_swing-024 | 1 | reference_unstable | - | true | highest_root_majority=2 fraction=0.5714; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| high_value_swing-025 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |

## 9. Proposed non-mutating patch artifact

- Proposed review artifact: `/tmp/azlite_high_value_swing_reference_adjudication/high_value_swing_reference_review_patch_v1.json`.
- `do_not_auto_apply` is set for every proposed row.
- The active fixture remained unchanged.

## 10. Projected clean targetability

| bucket | row_count | rows | train_target_eligible_count | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| confirmed_active_reference_targets | 0 | - | 0 | still limited to rows with confirmed active references only | usable only if the family still clears the final branch decision |
| reference_flip_candidates | 0 | - | 0 | fixture changes need explicit review before they are safe to use | review the proposed patch artifact before any training |
| unstable_or_excluded | 14 | high_value_swing-001, high_value_swing-003, high_value_swing-007, high_value_swing-008, high_value_swing-009, high_value_swing-013, high_value_swing-015, high_value_swing-016, high_value_swing-018, high_value_swing-020, high_value_swing-021, high_value_swing-023, high_value_swing-024, high_value_swing-025 | 0 | unstable or mixed labels would contaminate any hard target set | exclude from hard gates and training targets |
| puct_teacher_divergence_rows | 0 | - | 0 | teacher-policy ambiguity remains unresolved for these rows | decide teacher policy before any training use |
| confirmed_preservation_controls | 4 | high_value_swing-010, high_value_swing-017, high_value_swing-026, high_value_swing-027 | 0 | controls must stay unchanged during any follow-up | preserve unchanged as regression checks |
| holdout_context_rows | 2 | high_value_swing-011, high_value_swing-022 | 0 | holdouts should remain context-only during follow-up | report only; do not train on them in this branch |

- Target-candidate rows still eligible under active references: `0`.
- Target-candidate rows that would require reference patching: `0`.
- Target-candidate rows that should stay excluded from training: `14`.
- `high_value_swing` remains a good training target after adjudication: `false`.
- Projected family decision: `reference_suite_too_noisy_for_high_value_swing`.

## 11. Exactly one recommended next action

Recommendation: **exclude unstable rows and either target the smaller stable bucket or select the next non-opening family.**
