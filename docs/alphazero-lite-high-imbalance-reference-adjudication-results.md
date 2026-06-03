# AlphaZero-lite High Imbalance Reference Adjudication Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay/value-calibration artifacts were created.
- Active corrected references were not mutated.
- Active corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows path: `/tmp/azlite_corrected_non_opening_failure_mining_v3/selected_non_opening_family_rows_v3.jsonl`.

## 2. Why PR #55 blocked value training

- PR #55 did not confirm a value-head family gap for `high_imbalance`.
- Family-level classification: `reference_family_uncertain`.
- Mechanism counts: `{"blocked_next_action": "adjudicate high_imbalance references before training.", "corrected_reference_suspicious_count": 7, "family_classification": "reference_family_uncertain", "inconclusive_count": 0, "puct_child_search_mismatch_count": 0, "root_selection_pressure_count": 1, "value_head_miscalibration_count": 5}`.
- Exact PR #55 recommendation: **adjudicate high_imbalance references before training.**

## 3. Row validation

| row_id | pr55_classification | active_reference_move | legal | reference_unstable | canonical_state_match | current_selected_384 | current_selected_1200 | current_selected_2400 | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-018 | corrected_reference_suspicious | 2 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-020 | corrected_reference_suspicious | 2 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-021 | corrected_reference_suspicious | 2 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-025 | corrected_reference_suspicious | 0 | true | false | true | 3 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-016 | corrected_reference_suspicious | 2 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-027 | corrected_reference_suspicious | 2 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-026 | corrected_reference_suspicious | 2 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-009 | value_head_miscalibration | 4 | true | false | true | 5 | 5 | 5 | ok | validated for adjudication |
| high_imbalance-019 | value_head_miscalibration | 3 | true | false | true | 3 | 5 | 5 | ok | validated for adjudication |
| high_imbalance-023 | value_head_miscalibration | 2 | true | false | true | 3 | 3 | 3 | ok | validated for adjudication |
| high_imbalance-014 | value_head_miscalibration | 0 | true | false | true | 4 | 4 | 4 | ok | validated for adjudication |
| high_imbalance-022 | value_head_miscalibration | 2 | true | false | true | 5 | 5 | 5 | ok | validated for adjudication |
| high_imbalance-024 | root_selection_pressure | 2 | true | false | true | 1 | 1 | 2 | ok | validated for adjudication |
| high_imbalance-008 | holdout_context | 3 | true | false | true | 3 | 2 | 2 | ok | reported as holdout context only |
| high_imbalance-002 | holdout_context | 3 | true | false | true | 5 | 5 | 5 | ok | reported as holdout context only |
| high_imbalance-007 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |
| high_imbalance-012 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |
| high_imbalance-001 | preservation_control | 3 | true | false | true | 3 | 3 | 3 | ok | reported as preservation control only |
| high_imbalance-017 | preservation_control | 1 | true | false | true | 1 | 1 | 1 | ok | reported as preservation control only |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Repo convention: `follow state_to_root_perspective_value: child values stay positive for the same player-to-move and are sign-flipped when the turn hands off`.

## 4. Root ClassicMCTS adjudication

| row_id | budget | seeds | active_reference_move | observed_top_moves | majority_move | majority_fraction | reference_selected_fraction | top1_margin_mean | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-018 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.5499 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-018 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.6246 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-018 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.6703 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-018 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.6934 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-018 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.7115 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-020 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.0573 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-020 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.2074 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-020 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.3626 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-020 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.5351 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-020 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.6807 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-021 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.0725 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-021 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.2261 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-021 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.4270 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-021 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.5818 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-021 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.7081 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-025 | 1200 | 7 | 0 | 0:4, 2:1, 3:2 | 0 | 0.5714 | 0.5714 | 0.0318 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_imbalance-025 | 2400 | 7 | 0 | 0:5, 3:2 | 0 | 0.7143 | 0.7143 | 0.2229 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-025 | 5000 | 7 | 0 | 0:5, 3:2 | 0 | 0.7143 | 0.7143 | 0.7697 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-025 | 10000 | 7 | 0 | 0:5, 3:2 | 0 | 0.7143 | 0.7143 | 1.1007 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-025 | 30000 | 7 | 0 | 0:5, 3:2 | 0 | 0.7143 | 0.7143 | 1.5026 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-016 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.3072 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-016 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.3752 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-016 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.4384 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-016 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.4940 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-016 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.5487 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-027 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.4220 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-027 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.6903 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-027 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.0685 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-027 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.2948 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-027 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.4487 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-026 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.3599 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-026 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.7166 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-026 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.0384 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-026 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.2062 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-026 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.3282 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-009 | 1200 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.0797 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-009 | 2400 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.0570 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-009 | 5000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.0279 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-009 | 10000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.0273 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-009 | 30000 | 7 | 4 | 4:7 | 4 | 1.0000 | 1.0000 | 0.0305 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-019 | 1200 | 7 | 3 | 3:7 | 3 | 1.0000 | 1.0000 | 0.1849 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-019 | 2400 | 7 | 3 | 3:5, 4:2 | 3 | 0.7143 | 0.7143 | 0.4815 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-019 | 5000 | 7 | 3 | 3:4, 4:3 | 3 | 0.5714 | 0.5714 | 1.2282 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_imbalance-019 | 10000 | 7 | 3 | 3:4, 4:3 | 3 | 0.5714 | 0.5714 | 1.4526 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_imbalance-019 | 30000 | 7 | 3 | 3:4, 4:3 | 3 | 0.5714 | 0.5714 | 1.6212 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_imbalance-023 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.1078 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-023 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.0844 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-023 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.1653 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-023 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.6858 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-023 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.4621 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-014 | 1200 | 7 | 0 | 0:5, 4:2 | 0 | 0.7143 | 0.7143 | 1.0936 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-014 | 2400 | 7 | 0 | 0:5, 4:2 | 0 | 0.7143 | 0.7143 | 1.3945 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-014 | 5000 | 7 | 0 | 0:5, 4:2 | 0 | 0.7143 | 0.7143 | 1.5203 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-014 | 10000 | 7 | 0 | 0:5, 4:2 | 0 | 0.7143 | 0.7143 | 1.5757 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-014 | 30000 | 7 | 0 | 0:5, 4:2 | 0 | 0.7143 | 0.7143 | 1.6129 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-022 | 1200 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.3290 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-022 | 2400 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 0.9493 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-022 | 5000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.3758 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-022 | 10000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.5669 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-022 | 30000 | 7 | 2 | 2:7 | 2 | 1.0000 | 1.0000 | 1.7106 | supports_active_reference | high-budget majority stays on the active reference |
| high_imbalance-024 | 1200 | 7 | 2 | 0:3, 2:4 | 2 | 0.5714 | 0.5714 | 0.0304 | unstable_or_mixed | seed majorities remain mixed at this budget |
| high_imbalance-024 | 2400 | 7 | 2 | 0:7 | 0 | 1.0000 | 0.0000 | 0.4695 | supports_flip_candidate | high-budget majority prefers another move |
| high_imbalance-024 | 5000 | 7 | 2 | 0:7 | 0 | 1.0000 | 0.0000 | 1.1641 | supports_flip_candidate | high-budget majority prefers another move |
| high_imbalance-024 | 10000 | 7 | 2 | 0:7 | 0 | 1.0000 | 0.0000 | 1.4183 | supports_flip_candidate | high-budget majority prefers another move |
| high_imbalance-024 | 30000 | 7 | 2 | 0:7 | 0 | 1.0000 | 0.0000 | 1.6104 | supports_flip_candidate | high-budget majority prefers another move |

## 5. Child-afterstate adjudication

| row_id | active_reference_move | comparison_move | budget | active_reference_child_value_root_mean | comparison_child_value_root_mean | active_minus_comparison_value | child_selected_moves | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-018 | 2 | 4 | 1200 | -0.9483 | 0.8218 | -1.7701 | ref[4:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-018 | 2 | 4 | 2400 | -0.9462 | 0.8912 | -1.8374 | ref[4:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-018 | 2 | 4 | 5000 | -0.9454 | 0.9361 | -1.8815 | ref[4:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-020 | 2 | 4 | 1200 | -0.9745 | 0.4159 | -1.3904 | ref[3:1, 5:4] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-020 | 2 | 4 | 2400 | -0.9760 | 0.6210 | -1.5970 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-020 | 2 | 4 | 5000 | -0.9712 | 0.7930 | -1.7642 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-021 | 2 | 4 | 1200 | -0.9784 | 0.4300 | -1.4084 | ref[4:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-021 | 2 | 4 | 2400 | -0.9681 | 0.5607 | -1.5288 | ref[4:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-021 | 2 | 4 | 5000 | -0.9730 | 0.7284 | -1.7014 | ref[4:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-025 | 0 | 4 | 1200 | -0.9699 | -0.9427 | -0.0272 | ref[2:1, 4:4] cmp[4:1, 5:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-025 | 0 | 4 | 2400 | -0.9682 | -0.9420 | -0.0262 | ref[2:4, 4:1] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-025 | 0 | 4 | 5000 | -0.9707 | -0.9329 | -0.0378 | ref[2:5] cmp[2:3, 5:2] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-016 | 2 | 4 | 1200 | -0.9787 | 0.8356 | -1.8143 | ref[3:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-016 | 2 | 4 | 2400 | -0.9710 | 0.9005 | -1.8715 | ref[3:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-016 | 2 | 4 | 5000 | -0.9601 | 0.9414 | -1.9015 | ref[3:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-027 | 2 | 4 | 1200 | -0.9875 | 0.2027 | -1.1902 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-027 | 2 | 4 | 2400 | -0.9800 | 0.4014 | -1.3814 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-027 | 2 | 4 | 5000 | -0.9739 | 0.6457 | -1.6196 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-026 | 2 | 4 | 1200 | -0.9245 | 0.4742 | -1.3987 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-026 | 2 | 4 | 2400 | -0.9209 | 0.6974 | -1.6183 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-026 | 2 | 4 | 5000 | -0.9143 | 0.8280 | -1.7423 | ref[5:5] cmp[2:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-009 | 4 | 5 | 1200 | 0.7640 | -0.6420 | 1.4060 | ref[1:5] cmp[0:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-009 | 4 | 5 | 2400 | 0.2446 | -0.8090 | 1.0536 | ref[1:4, 4:1] cmp[0:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-009 | 4 | 5 | 5000 | -0.3579 | -0.8798 | 0.5219 | ref[1:4, 4:1] cmp[0:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-019 | 3 | 5 | 1200 | -0.8832 | -0.9450 | 0.0618 | ref[4:3, 5:2] cmp[0:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-019 | 3 | 5 | 2400 | -0.8705 | -0.9315 | 0.0610 | ref[5:5] cmp[0:4, 2:1] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-019 | 3 | 5 | 5000 | -0.8750 | -0.9480 | 0.0730 | ref[4:1, 5:4] cmp[0:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-023 | 2 | 3 | 1200 | -0.9493 | -0.9958 | 0.0465 | ref[4:1, 5:4] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-023 | 2 | 3 | 2400 | -0.9403 | -0.9825 | 0.0422 | ref[0:1, 4:4] cmp[4:1, 5:4] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-023 | 2 | 3 | 5000 | -0.9464 | -0.9805 | 0.0341 | ref[4:5] cmp[4:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-014 | 0 | 4 | 1200 | -0.8914 | -0.8831 | -0.0083 | ref[4:5] cmp[4:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-014 | 0 | 4 | 2400 | -0.8976 | -0.8909 | -0.0067 | ref[4:5] cmp[4:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-014 | 0 | 4 | 5000 | -0.9030 | -0.9182 | 0.0152 | ref[4:5] cmp[4:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-022 | 2 | 5 | 1200 | -0.9164 | -0.9907 | 0.0743 | ref[4:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-022 | 2 | 5 | 2400 | -0.9197 | -0.9767 | 0.0570 | ref[4:5] cmp[5:5] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-022 | 2 | 5 | 5000 | -0.9322 | -0.9639 | 0.0317 | ref[4:5] cmp[4:2, 5:3] | comparison=current_puct_2400; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-024 | 2 | 0 | 1200 | -0.9326 | -0.8948 | -0.0378 | ref[5:5] cmp[2:1, 4:4] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-024 | 2 | 0 | 2400 | -0.9227 | -0.9172 | -0.0055 | ref[5:5] cmp[2:1, 4:4] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |
| high_imbalance-024 | 2 | 0 | 5000 | -0.9220 | -0.9322 | 0.0102 | ref[5:5] cmp[4:5] | comparison=classic_majority; same player to move after child => +1, otherwise sign -1 |

## 6. Tablebase availability

| row_id | state_label | remaining_seed_count | tablebase_available | tablebase_value_root | tablebase_preferred_move | agrees_with_classic_majority | agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-018 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-018 | child_after_move_2 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-018 | child_after_move_4 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-020 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-020 | child_after_move_2 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-020 | child_after_move_4 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-021 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-021 | child_after_move_2 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-021 | child_after_move_4 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-025 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-025 | child_after_move_0 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-025 | child_after_move_4 | 34 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-016 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-016 | child_after_move_2 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-016 | child_after_move_4 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-027 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-027 | child_after_move_2 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-027 | child_after_move_4 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-026 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-026 | child_after_move_2 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-026 | child_after_move_4 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-009 | root | 38 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-009 | child_after_move_4 | 37 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-009 | child_after_move_5 | 37 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-019 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-019 | child_after_move_3 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-019 | child_after_move_5 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-023 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-023 | child_after_move_2 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-023 | child_after_move_3 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-014 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-014 | child_after_move_0 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-014 | child_after_move_4 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-022 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-022 | child_after_move_2 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-022 | child_after_move_5 | 35 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-024 | root | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-024 | child_after_move_2 | 36 | false | - | - | - | - | not solvable under the repo threshold |
| high_imbalance-024 | child_after_move_0 | 36 | false | - | - | - | - | not solvable under the repo threshold |

## 7. PUCT/artifact teacher comparison

| row_id | budget | active_reference_move | puct_selected_move | puct_reference_visit_share | puct_selected_visit_share | puct_agrees_with_classic_majority | puct_agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-018 | 384 | 2 | 4 | 0.0026 | 0.8906 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-018 | 1200 | 2 | 4 | 0.0033 | 0.9392 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-018 | 2400 | 2 | 4 | 0.0021 | 0.9542 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-018 | 5000 | 2 | 4 | 0.0010 | 0.9750 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-020 | 384 | 2 | 4 | 0.0078 | 0.8854 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-020 | 1200 | 2 | 4 | 0.0025 | 0.9558 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-020 | 2400 | 2 | 4 | 0.0013 | 0.9742 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-020 | 5000 | 2 | 4 | 0.0006 | 0.9780 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-021 | 384 | 2 | 4 | 0.0026 | 0.7396 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-021 | 1200 | 2 | 4 | 0.0033 | 0.9108 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-021 | 2400 | 2 | 4 | 0.0029 | 0.9400 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-021 | 5000 | 2 | 4 | 0.0014 | 0.9712 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-025 | 384 | 0 | 3 | 0.0026 | 0.7474 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-025 | 1200 | 0 | 4 | 0.0008 | 0.7233 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-025 | 2400 | 0 | 4 | 0.0004 | 0.8279 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-025 | 5000 | 0 | 4 | 0.0014 | 0.7782 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-016 | 384 | 2 | 4 | 0.0078 | 0.8984 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-016 | 1200 | 2 | 4 | 0.0025 | 0.9483 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-016 | 2400 | 2 | 4 | 0.0017 | 0.9604 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-016 | 5000 | 2 | 4 | 0.0010 | 0.9762 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-027 | 384 | 2 | 4 | 0.0078 | 0.8672 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-027 | 1200 | 2 | 4 | 0.0033 | 0.9483 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-027 | 2400 | 2 | 4 | 0.0017 | 0.9688 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-027 | 5000 | 2 | 4 | 0.0008 | 0.9818 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-026 | 384 | 2 | 4 | 0.0234 | 0.9167 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-026 | 1200 | 2 | 4 | 0.0125 | 0.9600 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-026 | 2400 | 2 | 4 | 0.0146 | 0.9663 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-026 | 5000 | 2 | 4 | 0.0378 | 0.9500 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-009 | 384 | 4 | 5 | 0.0130 | 0.9844 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-009 | 1200 | 4 | 5 | 0.0050 | 0.9908 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-009 | 2400 | 4 | 5 | 0.0033 | 0.9938 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-009 | 5000 | 4 | 5 | 0.0024 | 0.9630 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-019 | 384 | 3 | 3 | 0.5573 | 0.5573 | true | true | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-019 | 1200 | 3 | 5 | 0.1783 | 0.7775 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-019 | 2400 | 3 | 5 | 0.0892 | 0.8604 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-019 | 5000 | 3 | 5 | 0.0428 | 0.8180 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-023 | 384 | 2 | 3 | 0.0208 | 0.8802 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-023 | 1200 | 2 | 3 | 0.0125 | 0.9558 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-023 | 2400 | 2 | 3 | 0.0108 | 0.9717 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-023 | 5000 | 2 | 3 | 0.0082 | 0.9752 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-014 | 384 | 0 | 4 | 0.0026 | 0.8984 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-014 | 1200 | 0 | 4 | 0.0100 | 0.9300 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-014 | 2400 | 0 | 4 | 0.0050 | 0.9621 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-014 | 5000 | 0 | 4 | 0.0032 | 0.9766 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-022 | 384 | 2 | 5 | 0.0156 | 0.8906 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-022 | 1200 | 2 | 5 | 0.0058 | 0.8583 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-022 | 2400 | 2 | 5 | 0.2858 | 0.5504 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-022 | 5000 | 2 | 2 | 0.6572 | 0.6572 | true | true | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-024 | 384 | 2 | 1 | 0.0495 | 0.8542 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-024 | 1200 | 2 | 1 | 0.3017 | 0.5758 | false | false | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-024 | 2400 | 2 | 2 | 0.6496 | 0.6496 | false | true | deterministic artifact PUCT with PR #55 settings |
| high_imbalance-024 | 5000 | 2 | 2 | 0.8306 | 0.8306 | false | true | deterministic artifact PUCT with PR #55 settings |

## 8. Row decisions

| row_id | active_reference_move | adjudicated_decision | proposed_reference_move | proposed_unstable | evidence_summary | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | 4 | still_inconclusive | - | false | highest_root_majority=4 fraction=1.0000; highest_puct_selected=5; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-014 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=0.7143; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-016 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-018 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-019 | 3 | reference_unstable | - | true | highest_root_majority=3 fraction=0.5714; highest_puct_selected=5; child_support=mixed; tablebase_root_move=- | exclude from hard gates and training targets | high-budget seeds or budgets do not converge on one stable move |
| high_imbalance-020 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-021 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-022 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-023 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=3; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-024 | 2 | still_inconclusive | - | false | highest_root_majority=0 fraction=1.0000; highest_puct_selected=2; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search prefers another move, but child-afterstate evidence stays mixed |
| high_imbalance-025 | 0 | still_inconclusive | - | false | highest_root_majority=0 fraction=0.7143; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-026 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |
| high_imbalance-027 | 2 | still_inconclusive | - | false | highest_root_majority=2 fraction=1.0000; highest_puct_selected=4; child_support=mixed; tablebase_root_move=- | exclude pending more evidence | root search supports the active reference, but child-afterstate evidence stays mixed |

## 9. Proposed non-mutating patch artifact

- Proposed review artifact: `/tmp/azlite_high_imbalance_reference_adjudication/high_imbalance_reference_review_patch_v1.json`.
- `do_not_auto_apply` is set for every proposed row.
- The active fixture remained unchanged.

## 10. Projected clean targetability

| bucket | row_count | rows | train_target_eligible_count | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| confirmed_active_reference_targets | 0 | - | 0 | still limited to rows with confirmed active references only | usable only if the family still clears the final branch decision |
| reference_flip_candidates | 0 | - | 0 | fixture changes need explicit review before they are safe to use | review the proposed patch artifact before any training |
| unstable_or_excluded | 13 | high_imbalance-009, high_imbalance-014, high_imbalance-016, high_imbalance-018, high_imbalance-019, high_imbalance-020, high_imbalance-021, high_imbalance-022, high_imbalance-023, high_imbalance-024, high_imbalance-025, high_imbalance-026, high_imbalance-027 | 0 | unstable or mixed labels would contaminate any hard target set | exclude from hard gates and training targets |
| puct_teacher_divergence_rows | 0 | - | 0 | teacher-policy ambiguity remains unresolved for these rows | decide teacher policy before any training use |
| confirmed_value_head_miscalibration_context | 5 | high_imbalance-009, high_imbalance-014, high_imbalance-019, high_imbalance-022, high_imbalance-023 | 0 | raw neural child values disagreed with the ClassicMCTS teacher in PR #55; usable only after explicit value calibration | reserve for later value-calibration only after references settle |
| confirmed_preservation_controls | 4 | high_imbalance-001, high_imbalance-007, high_imbalance-012, high_imbalance-017 | 0 | controls must stay unchanged during any follow-up | preserve unchanged as regression checks |
| holdout_context_rows | 2 | high_imbalance-002, high_imbalance-008 | 0 | holdouts should remain context-only during follow-up | report only; do not train on them in this branch |

- Target-candidate rows still eligible under active references: `0`.
- Target-candidate rows that would require reference patching: `0`.
- Target-candidate rows that should stay excluded from training: `13`.
- `high_imbalance` remains a good training target after adjudication: `false`.
- Later value-calibration bucket (5 value_head_miscalibration rows) safe to keep: `false`.
- Projected family decision: `reference_suite_too_noisy_for_high_imbalance`.

## 11. Exactly one recommended next action

Recommendation: **exclude unstable rows and either target the smaller stable bucket or select the next non-opening family.**
