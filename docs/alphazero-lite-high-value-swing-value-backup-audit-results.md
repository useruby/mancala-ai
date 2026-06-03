# AlphaZero-lite High Value Swing Value/Backup Audit Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references were not mutated.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows: `/tmp/azlite_corrected_non_opening_failure_mining_v2/selected_non_opening_family_rows_v2.jsonl`.

## 2. Why high_value_swing was selected

- PR #51 selected `high_value_swing` as the next corrected non-opening family.
- Family stats: `{"avg_reference_visit_share_1200": 0.3097, "avg_reference_visit_share_384": 0.2698, "avg_selected_minus_reference_q_margin_1200": 0.1867, "avg_selected_minus_reference_q_margin_384": 0.1999, "classification": "value_or_backup_issue", "conflicting_target_count": 0, "dominant_failure_mode": "value_q", "duplicate_canonical_state_count": 0, "fail_rows": 17, "failure_rate": 0.7083, "family": "high_value_swing", "high_severity_count": 0, "medium_severity_count": 17, "notes": "persistent failures remain Q/value-dominant", "pass_rows": 7, "persistent_1200_failures": 16, "rank_score": 280.5088, "recovered_at_1200": 1, "stable_corrected_reference_count": 24, "total_rows": 24}`.

## 3. Row validation

| row_id | role | corrected_reference_move | legal | reference_unstable | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-024 | target_candidate | 1 | true | false | ok | validated, required target row present |
| high_value_swing-007 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-025 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-023 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-001 | target_candidate | 3 | true | false | ok | validated, required target row present |
| high_value_swing-021 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-013 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-018 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-008 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-003 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_value_swing-016 | target_candidate | 0 | true | false | ok | validated |
| high_value_swing-020 | target_candidate | 0 | true | false | ok | validated |
| high_value_swing-009 | target_candidate | 0 | true | false | ok | validated |
| high_value_swing-015 | target_candidate | 0 | true | false | ok | validated |
| high_value_swing-011 | holdout_candidate | 0 | true | false | ok | validated |
| high_value_swing-022 | holdout_candidate | 4 | true | false | ok | validated |
| high_value_swing-010 | preservation_control | 1 | true | false | ok | validated, required control row present |
| high_value_swing-026 | preservation_control | 1 | true | false | ok | validated, required control row present |
| high_value_swing-027 | preservation_control | 1 | true | false | ok | validated, required control row present |
| high_value_swing-017 | preservation_control | 0 | true | false | ok | validated, required control row present |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Implementation rule: `PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player`.

## 4. Root PUCT baseline

| row_id | role | budget | corrected_reference_move | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | reference_q | selected_q | selected_minus_reference_q_margin | reference_policy_probability | selected_policy_probability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-001 | target_candidate | 384 | 3 | 2 | false | 0.0026 | 0.9844 | 0.0123 | 0.2658 | 0.2535 | 0.0139 | 0.7608 | target_candidate, selected away from reference |
| high_value_swing-001 | target_candidate | 1200 | 3 | 2 | false | 0.0008 | 0.9900 | 0.0123 | 0.2790 | 0.2667 | 0.0139 | 0.7608 | target_candidate, selected away from reference |
| high_value_swing-001 | target_candidate | 2400 | 3 | 2 | false | 0.0004 | 0.9933 | 0.0123 | 0.3418 | 0.3295 | 0.0139 | 0.7608 | target_candidate, selected away from reference |
| high_value_swing-003 | target_candidate | 384 | 0 | 2 | false | 0.0104 | 0.9427 | 0.0653 | 0.2836 | 0.2183 | 0.1899 | 0.1377 | target_candidate, selected away from reference |
| high_value_swing-003 | target_candidate | 1200 | 0 | 2 | false | 0.0125 | 0.9267 | 0.0910 | 0.2595 | 0.1685 | 0.1899 | 0.1377 | target_candidate, selected away from reference |
| high_value_swing-003 | target_candidate | 2400 | 0 | 2 | false | 0.0121 | 0.9392 | 0.0887 | 0.2014 | 0.1126 | 0.1899 | 0.1377 | target_candidate, selected away from reference |
| high_value_swing-007 | target_candidate | 384 | 0 | 2 | false | 0.0052 | 0.9635 | 0.0455 | 0.3367 | 0.2912 | 0.1009 | 0.4773 | target_candidate, selected away from reference |
| high_value_swing-007 | target_candidate | 1200 | 0 | 2 | false | 0.0050 | 0.9658 | 0.0595 | 0.3664 | 0.3069 | 0.1009 | 0.4773 | target_candidate, selected away from reference |
| high_value_swing-007 | target_candidate | 2400 | 0 | 2 | false | 0.0033 | 0.9696 | 0.0513 | 0.3591 | 0.3079 | 0.1009 | 0.4773 | target_candidate, selected away from reference |
| high_value_swing-008 | target_candidate | 384 | 0 | 1 | false | 0.0260 | 0.9453 | 0.0322 | 0.2057 | 0.1736 | 0.4521 | 0.4344 | target_candidate, selected away from reference |
| high_value_swing-008 | target_candidate | 1200 | 0 | 1 | false | 0.0158 | 0.9650 | 0.0207 | 0.2127 | 0.1920 | 0.4521 | 0.4344 | target_candidate, selected away from reference |
| high_value_swing-008 | target_candidate | 2400 | 0 | 1 | false | 0.0112 | 0.9788 | 0.0186 | 0.2445 | 0.2259 | 0.4521 | 0.4344 | target_candidate, selected away from reference |
| high_value_swing-009 | target_candidate | 384 | 0 | 2 | false | 0.0651 | 0.7995 | 0.0953 | 0.1444 | 0.0491 | 0.0926 | 0.1379 | target_candidate, selected away from reference |
| high_value_swing-009 | target_candidate | 1200 | 0 | 2 | false | 0.0208 | 0.9350 | 0.0953 | 0.1818 | 0.0864 | 0.0926 | 0.1379 | target_candidate, selected away from reference |
| high_value_swing-009 | target_candidate | 2400 | 0 | 2 | false | 0.0104 | 0.9650 | 0.0953 | 0.1851 | 0.0897 | 0.0926 | 0.1379 | target_candidate, selected away from reference |
| high_value_swing-010 | preservation_control | 384 | 1 | 1 | true | 0.9609 | 0.9609 | 0.1198 | 0.1198 | 0.0000 | 0.5311 | 0.5311 | preservation_control, selected reference |
| high_value_swing-010 | preservation_control | 1200 | 1 | 1 | true | 0.9817 | 0.9817 | 0.1406 | 0.1406 | 0.0000 | 0.5311 | 0.5311 | preservation_control, selected reference |
| high_value_swing-010 | preservation_control | 2400 | 1 | 1 | true | 0.9858 | 0.9858 | 0.1481 | 0.1481 | 0.0000 | 0.5311 | 0.5311 | preservation_control, selected reference |
| high_value_swing-011 | holdout_candidate | 384 | 0 | 5 | false | 0.0443 | 0.8750 | -0.0417 | 0.0452 | 0.0870 | 0.1907 | 0.2235 | holdout_candidate, selected away from reference |
| high_value_swing-011 | holdout_candidate | 1200 | 0 | 5 | false | 0.0200 | 0.9058 | -0.0366 | 0.0159 | 0.0524 | 0.1907 | 0.2235 | holdout_candidate, selected away from reference |
| high_value_swing-011 | holdout_candidate | 2400 | 0 | 5 | false | 0.0100 | 0.9387 | -0.0366 | -0.0042 | 0.0324 | 0.1907 | 0.2235 | holdout_candidate, selected away from reference |
| high_value_swing-013 | target_candidate | 384 | 0 | 2 | false | 0.0078 | 0.9427 | -0.0028 | 0.1790 | 0.1818 | 0.0929 | 0.7594 | target_candidate, selected away from reference |
| high_value_swing-013 | target_candidate | 1200 | 0 | 2 | false | 0.0033 | 0.9792 | -0.0036 | 0.1912 | 0.1948 | 0.0929 | 0.7594 | target_candidate, selected away from reference |
| high_value_swing-013 | target_candidate | 2400 | 0 | 2 | false | 0.0021 | 0.9862 | -0.0167 | 0.1764 | 0.1931 | 0.0929 | 0.7594 | target_candidate, selected away from reference |
| high_value_swing-015 | target_candidate | 384 | 0 | 2 | false | 0.0547 | 0.8333 | -0.0104 | 0.0538 | 0.0641 | 0.1550 | 0.3584 | target_candidate, selected away from reference |
| high_value_swing-015 | target_candidate | 1200 | 0 | 2 | false | 0.0175 | 0.9442 | -0.0104 | 0.0654 | 0.0758 | 0.1550 | 0.3584 | target_candidate, selected away from reference |
| high_value_swing-015 | target_candidate | 2400 | 0 | 2 | false | 0.0088 | 0.9717 | -0.0104 | 0.0806 | 0.0910 | 0.1550 | 0.3584 | target_candidate, selected away from reference |
| high_value_swing-016 | target_candidate | 384 | 0 | 1 | false | 0.0443 | 0.9219 | -0.0375 | 0.0803 | 0.1179 | 0.2494 | 0.3810 | target_candidate, selected away from reference |
| high_value_swing-016 | target_candidate | 1200 | 0 | 1 | false | 0.0142 | 0.9733 | -0.0375 | 0.0918 | 0.1294 | 0.2494 | 0.3810 | target_candidate, selected away from reference |
| high_value_swing-016 | target_candidate | 2400 | 0 | 1 | false | 0.0083 | 0.9825 | -0.0361 | 0.0919 | 0.1279 | 0.2494 | 0.3810 | target_candidate, selected away from reference |
| high_value_swing-017 | preservation_control | 384 | 0 | 0 | true | 0.9453 | 0.9453 | 0.0048 | 0.0048 | 0.0000 | 0.7133 | 0.7133 | preservation_control, selected reference |
| high_value_swing-017 | preservation_control | 1200 | 0 | 0 | true | 0.9575 | 0.9575 | 0.0161 | 0.0161 | 0.0000 | 0.7133 | 0.7133 | preservation_control, selected reference |
| high_value_swing-017 | preservation_control | 2400 | 0 | 0 | true | 0.9717 | 0.9717 | 0.0075 | 0.0075 | 0.0000 | 0.7133 | 0.7133 | preservation_control, selected reference |
| high_value_swing-018 | target_candidate | 384 | 0 | 1 | false | 0.0182 | 0.9766 | 0.0901 | 0.3158 | 0.2257 | 0.1874 | 0.7978 | target_candidate, selected away from reference |
| high_value_swing-018 | target_candidate | 1200 | 0 | 1 | false | 0.0108 | 0.9875 | 0.1042 | 0.2976 | 0.1933 | 0.1874 | 0.7978 | target_candidate, selected away from reference |
| high_value_swing-018 | target_candidate | 2400 | 0 | 1 | false | 0.0079 | 0.9912 | 0.1159 | 0.3172 | 0.2013 | 0.1874 | 0.7978 | target_candidate, selected away from reference |
| high_value_swing-020 | target_candidate | 384 | 0 | 1 | false | 0.0052 | 0.9896 | 0.0462 | 0.2126 | 0.1664 | 0.0975 | 0.8953 | target_candidate, selected away from reference |
| high_value_swing-020 | target_candidate | 1200 | 0 | 1 | false | 0.0042 | 0.9942 | 0.0847 | 0.1792 | 0.0945 | 0.0975 | 0.8953 | target_candidate, selected away from reference |
| high_value_swing-020 | target_candidate | 2400 | 0 | 1 | false | 0.0025 | 0.9967 | 0.0776 | 0.1960 | 0.1185 | 0.0975 | 0.8953 | target_candidate, selected away from reference |
| high_value_swing-021 | target_candidate | 384 | 0 | 1 | false | 0.0052 | 0.9948 | 0.0548 | 0.3167 | 0.2619 | 0.1159 | 0.8807 | target_candidate, selected away from reference |
| high_value_swing-021 | target_candidate | 1200 | 0 | 1 | false | 0.0050 | 0.9933 | 0.0889 | 0.3403 | 0.2514 | 0.1159 | 0.8807 | target_candidate, selected away from reference |
| high_value_swing-021 | target_candidate | 2400 | 0 | 1 | false | 0.0037 | 0.9954 | 0.1076 | 0.3468 | 0.2392 | 0.1159 | 0.8807 | target_candidate, selected away from reference |
| high_value_swing-022 | holdout_candidate | 384 | 4 | 1 | false | 0.0026 | 0.6771 | -0.0287 | 0.2961 | 0.3248 | 0.0534 | 0.3126 | holdout_candidate, selected away from reference |
| high_value_swing-022 | holdout_candidate | 1200 | 4 | 1 | false | 0.0108 | 0.8692 | 0.1579 | 0.2051 | 0.0472 | 0.0534 | 0.3126 | holdout_candidate, selected away from reference |
| high_value_swing-022 | holdout_candidate | 2400 | 4 | 1 | false | 0.0063 | 0.9287 | 0.1427 | 0.2058 | 0.0631 | 0.0534 | 0.3126 | holdout_candidate, selected away from reference |
| high_value_swing-023 | target_candidate | 384 | 0 | 1 | false | 0.0286 | 0.9635 | 0.0852 | 0.3653 | 0.2802 | 0.5002 | 0.4644 | target_candidate, selected away from reference |
| high_value_swing-023 | target_candidate | 1200 | 0 | 1 | false | 0.0192 | 0.9783 | 0.1125 | 0.3871 | 0.2745 | 0.5002 | 0.4644 | target_candidate, selected away from reference |
| high_value_swing-023 | target_candidate | 2400 | 0 | 1 | false | 0.0154 | 0.9833 | 0.1478 | 0.4091 | 0.2613 | 0.5002 | 0.4644 | target_candidate, selected away from reference |
| high_value_swing-024 | target_candidate | 384 | 1 | 2 | false | 0.0130 | 0.9818 | 0.0731 | 0.3903 | 0.3172 | 0.1582 | 0.7335 | target_candidate, selected away from reference |
| high_value_swing-024 | target_candidate | 1200 | 1 | 2 | false | 0.0100 | 0.9833 | 0.0706 | 0.4382 | 0.3676 | 0.1582 | 0.7335 | target_candidate, selected away from reference |
| high_value_swing-024 | target_candidate | 2400 | 1 | 2 | false | 0.0083 | 0.9871 | 0.1302 | 0.4562 | 0.3261 | 0.1582 | 0.7335 | target_candidate, selected away from reference |
| high_value_swing-025 | target_candidate | 384 | 0 | 2 | false | 0.0156 | 0.9583 | 0.0519 | 0.3647 | 0.3128 | 0.2305 | 0.5127 | target_candidate, selected away from reference |
| high_value_swing-025 | target_candidate | 1200 | 0 | 2 | false | 0.0108 | 0.9700 | 0.0862 | 0.3709 | 0.2847 | 0.2305 | 0.5127 | target_candidate, selected away from reference |
| high_value_swing-025 | target_candidate | 2400 | 0 | 2 | false | 0.0079 | 0.9792 | 0.0910 | 0.3965 | 0.3055 | 0.2305 | 0.5127 | target_candidate, selected away from reference |
| high_value_swing-026 | preservation_control | 384 | 1 | 1 | true | 0.9531 | 0.9531 | 0.1851 | 0.1851 | 0.0000 | 0.3925 | 0.3925 | preservation_control, selected reference |
| high_value_swing-026 | preservation_control | 1200 | 1 | 1 | true | 0.9767 | 0.9767 | 0.2276 | 0.2276 | 0.0000 | 0.3925 | 0.3925 | preservation_control, selected reference |
| high_value_swing-026 | preservation_control | 2400 | 1 | 1 | true | 0.9838 | 0.9838 | 0.2402 | 0.2402 | 0.0000 | 0.3925 | 0.3925 | preservation_control, selected reference |
| high_value_swing-027 | preservation_control | 384 | 1 | 1 | true | 0.9375 | 0.9375 | 0.2112 | 0.2112 | 0.0000 | 0.3190 | 0.3190 | preservation_control, selected reference |
| high_value_swing-027 | preservation_control | 1200 | 1 | 1 | true | 0.9717 | 0.9717 | 0.2508 | 0.2508 | 0.0000 | 0.3190 | 0.3190 | preservation_control, selected reference |
| high_value_swing-027 | preservation_control | 2400 | 1 | 1 | true | 0.9817 | 0.9817 | 0.2612 | 0.2612 | 0.0000 | 0.3190 | 0.3190 | preservation_control, selected reference |

- ran 2400 root budget: projected ~4.1s across 14 target rows

## 5. Move consequence audit

| row_id | move | is_corrected_reference | is_selected | gives_extra_turn | produces_capture | capture_count | immediate_store_delta | side_to_move_after | game_over_after_move | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-024 | 0 | false | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-024 | 1 | true | false | true | false | 0 | 1 | 0 | false |  |
| high_value_swing-024 | 2 | false | true | true | false | 0 | 1 | 0 | false |  |
| high_value_swing-024 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-024 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-007 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-007 | 1 | false | false | true | false | 0 | 1 | 0 | false |  |
| high_value_swing-007 | 2 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-007 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-007 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-025 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-025 | 1 | false | false | true | false | 0 | 1 | 0 | false |  |
| high_value_swing-025 | 2 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-025 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-025 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-023 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-023 | 1 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-023 | 2 | false | false | false | false | 0 | 0 | 1 | false |  |
| high_value_swing-023 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-023 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-001 | 0 | false | false | false | true | 7 | 7 | 1 | false |  |
| high_value_swing-001 | 1 | false | false | false | false | 0 | 0 | 1 | false |  |
| high_value_swing-001 | 2 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-001 | 3 | true | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-001 | 5 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-021 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-021 | 1 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-021 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-021 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-013 | 0 | true | false | false | true | 7 | 7 | 1 | false |  |
| high_value_swing-013 | 1 | false | false | false | false | 0 | 0 | 1 | false |  |
| high_value_swing-013 | 2 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-013 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-013 | 5 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-018 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-018 | 1 | false | true | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-018 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-018 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-008 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-008 | 1 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-008 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-008 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-008 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-003 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-003 | 1 | false | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-003 | 2 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-003 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-003 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-016 | 0 | true | false | false | true | 6 | 6 | 0 | false |  |
| high_value_swing-016 | 1 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn |
| high_value_swing-016 | 2 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-016 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-016 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-020 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-020 | 1 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-020 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-020 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-009 | 0 | true | false | false | true | 8 | 8 | 0 | false |  |
| high_value_swing-009 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_value_swing-009 | 2 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn |
| high_value_swing-009 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-009 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-015 | 0 | true | false | false | true | 6 | 6 | 0 | false |  |
| high_value_swing-015 | 1 | false | false | true | false | 0 | 1 | 1 | false |  |
| high_value_swing-015 | 2 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn |
| high_value_swing-015 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-015 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-011 | 0 | true | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-011 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-011 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-011 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-011 | 5 | false | true | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-022 | 0 | false | false | false | false | 0 | 0 | 1 | false |  |
| high_value_swing-022 | 1 | false | true | true | false | 0 | 1 | 0 | false | selected move gains extra turn |
| high_value_swing-022 | 2 | false | false | false | true | 5 | 5 | 1 | false |  |
| high_value_swing-022 | 4 | true | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-022 | 5 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-010 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_value_swing-010 | 1 | true | true | false | true | 6 | 6 | 0 | false |  |
| high_value_swing-010 | 2 | false | false | true | false | 0 | 1 | 1 | false |  |
| high_value_swing-010 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-010 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-026 | 0 | false | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-026 | 1 | true | true | true | false | 0 | 1 | 0 | false |  |
| high_value_swing-026 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-026 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-026 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-027 | 0 | false | false | false | true | 6 | 6 | 1 | false |  |
| high_value_swing-027 | 1 | true | true | true | false | 0 | 1 | 0 | false |  |
| high_value_swing-027 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-027 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-027 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_value_swing-017 | 0 | true | true | false | true | 6 | 6 | 0 | false |  |
| high_value_swing-017 | 1 | false | false | true | false | 0 | 1 | 1 | false |  |
| high_value_swing-017 | 2 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-017 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_value_swing-017 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |

## 6. Neural child-afterstate value audit

| row_id | corrected_reference_move | selected_move | child | raw_value | root_perspective_value | child_ref_minus_child_selected | neural_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-024 | 1 | 2 | corrected_reference_child | 0.2194 | 0.2194 | 0.0489 | true | identity conversion |
| high_value_swing-024 | 1 | 2 | selected_child | 0.1705 | 0.1705 | 0.0489 | true | identity conversion |
| high_value_swing-007 | 0 | 2 | corrected_reference_child | -0.0690 | 0.0690 | -0.1943 | false | sign flip to root perspective |
| high_value_swing-007 | 0 | 2 | selected_child | 0.2633 | 0.2633 | -0.1943 | false | identity conversion |
| high_value_swing-025 | 0 | 2 | corrected_reference_child | 0.0052 | -0.0052 | -0.1726 | false | sign flip to root perspective |
| high_value_swing-025 | 0 | 2 | selected_child | 0.1675 | 0.1675 | -0.1726 | false | identity conversion |
| high_value_swing-023 | 0 | 1 | corrected_reference_child | -0.1063 | 0.1063 | -0.1250 | false | sign flip to root perspective |
| high_value_swing-023 | 0 | 1 | selected_child | 0.2313 | 0.2313 | -0.1250 | false | identity conversion |
| high_value_swing-001 | 3 | 2 | corrected_reference_child | -0.0123 | 0.0123 | -0.0768 | false | sign flip to root perspective |
| high_value_swing-001 | 3 | 2 | selected_child | 0.0891 | 0.0891 | -0.0768 | false | identity conversion |
| high_value_swing-021 | 0 | 1 | corrected_reference_child | -0.0795 | 0.0795 | -0.1893 | false | sign flip to root perspective |
| high_value_swing-021 | 0 | 1 | selected_child | 0.2688 | 0.2688 | -0.1893 | false | identity conversion |
| high_value_swing-013 | 0 | 2 | corrected_reference_child | -0.0392 | 0.0392 | -0.0683 | false | sign flip to root perspective |
| high_value_swing-013 | 0 | 2 | selected_child | 0.1075 | 0.1075 | -0.0683 | false | identity conversion |
| high_value_swing-018 | 0 | 1 | corrected_reference_child | -0.1391 | 0.1391 | -0.0682 | false | sign flip to root perspective |
| high_value_swing-018 | 0 | 1 | selected_child | -0.2073 | 0.2073 | -0.0682 | false | sign flip to root perspective |
| high_value_swing-008 | 0 | 1 | corrected_reference_child | -0.0535 | 0.0535 | -0.1730 | false | sign flip to root perspective |
| high_value_swing-008 | 0 | 1 | selected_child | 0.2264 | 0.2264 | -0.1730 | false | identity conversion |
| high_value_swing-003 | 0 | 2 | corrected_reference_child | -0.1181 | 0.1181 | -0.1266 | false | sign flip to root perspective |
| high_value_swing-003 | 0 | 2 | selected_child | 0.2447 | 0.2447 | -0.1266 | false | identity conversion |
| high_value_swing-016 | 0 | 1 | corrected_reference_child | -0.1011 | 0.1011 | 0.0515 | true | sign flip to root perspective |
| high_value_swing-016 | 0 | 1 | selected_child | 0.0496 | 0.0496 | 0.0515 | true | identity conversion |
| high_value_swing-020 | 0 | 1 | corrected_reference_child | -0.1077 | 0.1077 | -0.1729 | false | sign flip to root perspective |
| high_value_swing-020 | 0 | 1 | selected_child | 0.2805 | 0.2805 | -0.1729 | false | identity conversion |
| high_value_swing-009 | 0 | 2 | corrected_reference_child | -0.1982 | 0.1982 | 0.0790 | true | sign flip to root perspective |
| high_value_swing-009 | 0 | 2 | selected_child | 0.1192 | 0.1192 | 0.0790 | true | identity conversion |
| high_value_swing-015 | 0 | 2 | corrected_reference_child | -0.1124 | 0.1124 | 0.0092 | true | sign flip to root perspective |
| high_value_swing-015 | 0 | 2 | selected_child | 0.1032 | 0.1032 | 0.0092 | true | identity conversion |

## 7. ClassicMCTS child-afterstate audit

| row_id | budget | seeds | child_ref_value_root_mean | child_selected_value_root_mean | child_ref_minus_child_selected | teacher_prefers_reference | stable | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-024 | 1200 | [11, 23, 37, 42, 101] | 0.6162 | 0.5632 | 0.0531 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-024 | 2400 | [11, 23, 37, 42, 101] | 0.6496 | 0.6426 | 0.0071 | true | false | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-024 | 5000 | [11, 23, 37, 42, 101] | 0.6650 | 0.6524 | 0.0126 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-007 | 1200 | [11, 23, 37, 42, 101] | -0.0840 | 0.6301 | -0.7141 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-007 | 2400 | [11, 23, 37, 42, 101] | -0.3115 | 0.6473 | -0.9588 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-007 | 5000 | [11, 23, 37, 42, 101] | -0.4924 | 0.7243 | -1.2167 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-025 | 1200 | [11, 23, 37, 42, 101] | -0.3850 | 0.5037 | -0.8888 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-025 | 2400 | [11, 23, 37, 42, 101] | -0.6107 | 0.6244 | -1.2350 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-025 | 5000 | [11, 23, 37, 42, 101] | -0.7469 | 0.6445 | -1.3914 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-023 | 1200 | [11, 23, 37, 42, 101] | 0.6668 | 0.8538 | -0.1870 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-023 | 2400 | [11, 23, 37, 42, 101] | 0.3307 | 0.8404 | -0.5097 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-023 | 5000 | [11, 23, 37, 42, 101] | -0.3306 | 0.8497 | -1.1803 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-001 | 1200 | [11, 23, 37, 42, 101] | 0.1960 | 0.8750 | -0.6790 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-001 | 2400 | [11, 23, 37, 42, 101] | -0.0192 | 0.8315 | -0.8507 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-001 | 5000 | [11, 23, 37, 42, 101] | -0.2259 | 0.8222 | -1.0481 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-021 | 1200 | [11, 23, 37, 42, 101] | -0.1377 | 0.6936 | -0.8314 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-021 | 2400 | [11, 23, 37, 42, 101] | -0.5197 | 0.7524 | -1.2720 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-021 | 5000 | [11, 23, 37, 42, 101] | -0.6834 | 0.7935 | -1.4770 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-013 | 1200 | [11, 23, 37, 42, 101] | -0.2752 | 0.6739 | -0.9491 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-013 | 2400 | [11, 23, 37, 42, 101] | -0.5596 | 0.6577 | -1.2174 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-013 | 5000 | [11, 23, 37, 42, 101] | -0.7400 | 0.6799 | -1.4199 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-018 | 1200 | [11, 23, 37, 42, 101] | -0.0521 | 0.2389 | -0.2910 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-018 | 2400 | [11, 23, 37, 42, 101] | -0.3262 | -0.0713 | -0.2549 | false | false | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-018 | 5000 | [11, 23, 37, 42, 101] | -0.4640 | -0.4164 | -0.0475 | true | false | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-008 | 1200 | [11, 23, 37, 42, 101] | 0.2940 | 0.8136 | -0.5196 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-008 | 2400 | [11, 23, 37, 42, 101] | -0.2245 | 0.8135 | -1.0380 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-008 | 5000 | [11, 23, 37, 42, 101] | -0.5269 | 0.8277 | -1.3545 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-003 | 1200 | [11, 23, 37, 42, 101] | -0.2599 | 0.7549 | -1.0149 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-003 | 2400 | [11, 23, 37, 42, 101] | -0.4479 | 0.7393 | -1.1872 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-003 | 5000 | [11, 23, 37, 42, 101] | -0.5598 | 0.7613 | -1.3211 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-016 | 1200 | [11, 23, 37, 42, 101] | -0.1536 | 0.6607 | -0.8143 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-016 | 2400 | [11, 23, 37, 42, 101] | -0.4589 | 0.7482 | -1.2071 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-016 | 5000 | [11, 23, 37, 42, 101] | -0.6423 | 0.7629 | -1.4052 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-020 | 1200 | [11, 23, 37, 42, 101] | -0.3228 | 0.7402 | -1.0630 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-020 | 2400 | [11, 23, 37, 42, 101] | -0.5077 | 0.7571 | -1.2648 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-020 | 5000 | [11, 23, 37, 42, 101] | -0.6128 | 0.7805 | -1.3932 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-009 | 1200 | [11, 23, 37, 42, 101] | 0.7056 | 0.9082 | -0.2026 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-009 | 2400 | [11, 23, 37, 42, 101] | 0.4063 | 0.8667 | -0.4604 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-009 | 5000 | [11, 23, 37, 42, 101] | -0.1132 | 0.8579 | -0.9710 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-015 | 1200 | [11, 23, 37, 42, 101] | -0.3465 | 0.6595 | -1.0061 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-015 | 2400 | [11, 23, 37, 42, 101] | -0.5468 | 0.7075 | -1.2543 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_value_swing-015 | 5000 | [11, 23, 37, 42, 101] | -0.6698 | 0.7422 | -1.4121 | false | true | ClassicMCTS child-afterstate teacher aggregate |

## 8. PUCT child-afterstate audit

| row_id | budget | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | puct_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-024 | 384 | 0.1666 | 0.3903 | -0.2237 | false | artifact PUCT child-afterstate audit |
| high_value_swing-024 | 1200 | 0.1539 | 0.4382 | -0.2842 | false | artifact PUCT child-afterstate audit |
| high_value_swing-007 | 384 | 0.0345 | 0.3343 | -0.2999 | false | artifact PUCT child-afterstate audit |
| high_value_swing-007 | 1200 | 0.0517 | 0.3666 | -0.3149 | false | artifact PUCT child-afterstate audit |
| high_value_swing-025 | 384 | 0.0841 | 0.3579 | -0.2738 | false | artifact PUCT child-afterstate audit |
| high_value_swing-025 | 1200 | 0.1086 | 0.3721 | -0.2634 | false | artifact PUCT child-afterstate audit |
| high_value_swing-023 | 384 | 0.1874 | 0.3701 | -0.1827 | false | artifact PUCT child-afterstate audit |
| high_value_swing-023 | 1200 | 0.2410 | 0.3864 | -0.1454 | false | artifact PUCT child-afterstate audit |
| high_value_swing-001 | 384 | 0.1714 | 0.2689 | -0.0975 | false | artifact PUCT child-afterstate audit |
| high_value_swing-001 | 1200 | 0.2049 | 0.2788 | -0.0739 | false | artifact PUCT child-afterstate audit |
| high_value_swing-021 | 384 | 0.1911 | 0.3180 | -0.1270 | false | artifact PUCT child-afterstate audit |
| high_value_swing-021 | 1200 | 0.1816 | 0.3404 | -0.1587 | false | artifact PUCT child-afterstate audit |
| high_value_swing-013 | 384 | 0.0444 | 0.1763 | -0.1319 | false | artifact PUCT child-afterstate audit |
| high_value_swing-013 | 1200 | 0.0902 | 0.1895 | -0.0993 | false | artifact PUCT child-afterstate audit |
| high_value_swing-018 | 384 | 0.1933 | 0.3162 | -0.1229 | false | artifact PUCT child-afterstate audit |
| high_value_swing-018 | 1200 | 0.1363 | 0.2945 | -0.1581 | false | artifact PUCT child-afterstate audit |
| high_value_swing-008 | 384 | 0.1218 | 0.2066 | -0.0848 | false | artifact PUCT child-afterstate audit |
| high_value_swing-008 | 1200 | 0.1289 | 0.2145 | -0.0855 | false | artifact PUCT child-afterstate audit |
| high_value_swing-003 | 384 | 0.0962 | 0.2807 | -0.1845 | false | artifact PUCT child-afterstate audit |
| high_value_swing-003 | 1200 | 0.0348 | 0.2636 | -0.2288 | false | artifact PUCT child-afterstate audit |
| high_value_swing-016 | 384 | 0.0267 | 0.0860 | -0.0593 | false | artifact PUCT child-afterstate audit |
| high_value_swing-016 | 1200 | -0.0073 | 0.0938 | -0.1012 | false | artifact PUCT child-afterstate audit |
| high_value_swing-020 | 384 | 0.1411 | 0.2141 | -0.0729 | false | artifact PUCT child-afterstate audit |
| high_value_swing-020 | 1200 | 0.1808 | 0.1800 | 0.0008 | true | artifact PUCT child-afterstate audit |
| high_value_swing-009 | 384 | 0.1312 | 0.1687 | -0.0375 | false | artifact PUCT child-afterstate audit |
| high_value_swing-009 | 1200 | 0.1453 | 0.1871 | -0.0418 | false | artifact PUCT child-afterstate audit |
| high_value_swing-015 | 384 | -0.0123 | 0.0640 | -0.0762 | false | artifact PUCT child-afterstate audit |
| high_value_swing-015 | 1200 | 0.0149 | 0.0629 | -0.0480 | false | artifact PUCT child-afterstate audit |

## 9. Root counterfactual diagnostics

| row_id | intervention | budget | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | flipped | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_value_swing-024 | original | 384 | 2 | false | 0.0130 | 0.9818 | 0.3172 | false | no intervention |
| high_value_swing-024 | original | 1200 | 2 | false | 0.0100 | 0.9833 | 0.3676 | false | no intervention |
| high_value_swing-024 | equalize_root_priors | 384 | 2 | false | 0.0469 | 0.9505 | 0.2728 | false | equalize corrected reference and selected root priors |
| high_value_swing-024 | equalize_root_priors | 1200 | 2 | false | 0.0433 | 0.9517 | 0.2536 | false | equalize corrected reference and selected root priors |
| high_value_swing-024 | uniform_legal_prior | 384 | 2 | false | 0.0156 | 0.9427 | 0.3354 | false | uniform legal prior positive control |
| high_value_swing-024 | uniform_legal_prior | 1200 | 2 | false | 0.0083 | 0.9675 | 0.3846 | false | uniform legal prior positive control |
| high_value_swing-024 | teacher_child_value_override | 384 | 2 | false | 0.0156 | 0.9792 | 0.2669 | false | override first child expansion values with teacher child values |
| high_value_swing-024 | teacher_child_value_override | 1200 | 2 | false | 0.0108 | 0.9825 | 0.3354 | false | override first child expansion values with teacher child values |
| high_value_swing-024 | neural_child_value_swap | 384 | 2 | false | 0.0104 | 0.9844 | 0.3123 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-024 | neural_child_value_swap | 1200 | 2 | false | 0.0100 | 0.9833 | 0.3717 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-007 | original | 384 | 2 | false | 0.0052 | 0.9635 | 0.2912 | false | no intervention |
| high_value_swing-007 | original | 1200 | 2 | false | 0.0050 | 0.9658 | 0.3069 | false | no intervention |
| high_value_swing-007 | equalize_root_priors | 384 | 2 | false | 0.0208 | 0.9557 | 0.2855 | false | equalize corrected reference and selected root priors |
| high_value_swing-007 | equalize_root_priors | 1200 | 2 | false | 0.0158 | 0.9642 | 0.3299 | false | equalize corrected reference and selected root priors |
| high_value_swing-007 | uniform_legal_prior | 384 | 2 | false | 0.0104 | 0.9427 | 0.2969 | false | uniform legal prior positive control |
| high_value_swing-007 | uniform_legal_prior | 1200 | 2 | false | 0.0067 | 0.9700 | 0.3158 | false | uniform legal prior positive control |
| high_value_swing-007 | teacher_child_value_override | 384 | 2 | false | 0.0052 | 0.9115 | 0.5745 | false | override first child expansion values with teacher child values |
| high_value_swing-007 | teacher_child_value_override | 1200 | 2 | false | 0.0033 | 0.9633 | 0.4547 | false | override first child expansion values with teacher child values |
| high_value_swing-007 | neural_child_value_swap | 384 | 2 | false | 0.0104 | 0.9583 | 0.2499 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-007 | neural_child_value_swap | 1200 | 2 | false | 0.0050 | 0.9658 | 0.2758 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-025 | original | 384 | 2 | false | 0.0156 | 0.9583 | 0.3128 | false | no intervention |
| high_value_swing-025 | original | 1200 | 2 | false | 0.0108 | 0.9700 | 0.2847 | false | no intervention |
| high_value_swing-025 | equalize_root_priors | 384 | 2 | false | 0.0339 | 0.9453 | 0.2797 | false | equalize corrected reference and selected root priors |
| high_value_swing-025 | equalize_root_priors | 1200 | 2 | false | 0.0217 | 0.9650 | 0.2598 | false | equalize corrected reference and selected root priors |
| high_value_swing-025 | uniform_legal_prior | 384 | 2 | false | 0.0104 | 0.9375 | 0.3296 | false | uniform legal prior positive control |
| high_value_swing-025 | uniform_legal_prior | 1200 | 2 | false | 0.0067 | 0.9658 | 0.3207 | false | uniform legal prior positive control |
| high_value_swing-025 | teacher_child_value_override | 384 | 2 | false | 0.0130 | 0.9531 | 0.4747 | false | override first child expansion values with teacher child values |
| high_value_swing-025 | teacher_child_value_override | 1200 | 2 | false | 0.0083 | 0.9725 | 0.3699 | false | override first child expansion values with teacher child values |
| high_value_swing-025 | neural_child_value_swap | 384 | 2 | false | 0.0182 | 0.9557 | 0.2879 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-025 | neural_child_value_swap | 1200 | 2 | false | 0.0117 | 0.9692 | 0.2694 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-023 | original | 384 | 1 | false | 0.0286 | 0.9635 | 0.2802 | false | no intervention |
| high_value_swing-023 | original | 1200 | 1 | false | 0.0192 | 0.9783 | 0.2745 | false | no intervention |
| high_value_swing-023 | equalize_root_priors | 384 | 1 | false | 0.0286 | 0.9635 | 0.2802 | false | equalize corrected reference and selected root priors |
| high_value_swing-023 | equalize_root_priors | 1200 | 1 | false | 0.0175 | 0.9800 | 0.2856 | false | equalize corrected reference and selected root priors |
| high_value_swing-023 | uniform_legal_prior | 384 | 1 | false | 0.0156 | 0.8958 | 0.2898 | false | uniform legal prior positive control |
| high_value_swing-023 | uniform_legal_prior | 1200 | 1 | false | 0.0075 | 0.9592 | 0.3009 | false | uniform legal prior positive control |
| high_value_swing-023 | teacher_child_value_override | 384 | 1 | false | 0.0286 | 0.9635 | 0.3216 | false | override first child expansion values with teacher child values |
| high_value_swing-023 | teacher_child_value_override | 1200 | 1 | false | 0.0175 | 0.9800 | 0.3069 | false | override first child expansion values with teacher child values |
| high_value_swing-023 | neural_child_value_swap | 384 | 1 | false | 0.0312 | 0.9609 | 0.2659 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-023 | neural_child_value_swap | 1200 | 1 | false | 0.0200 | 0.9775 | 0.2733 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-001 | original | 384 | 2 | false | 0.0026 | 0.9844 | 0.2535 | false | no intervention |
| high_value_swing-001 | original | 1200 | 2 | false | 0.0008 | 0.9900 | 0.2667 | false | no intervention |
| high_value_swing-001 | equalize_root_priors | 384 | 2 | false | 0.0312 | 0.9583 | 0.2552 | false | equalize corrected reference and selected root priors |
| high_value_swing-001 | equalize_root_priors | 1200 | 2 | false | 0.0200 | 0.9750 | 0.2343 | false | equalize corrected reference and selected root priors |
| high_value_swing-001 | uniform_legal_prior | 384 | 2 | false | 0.0104 | 0.9349 | 0.2976 | false | uniform legal prior positive control |
| high_value_swing-001 | uniform_legal_prior | 1200 | 2 | false | 0.0067 | 0.9633 | 0.2791 | false | uniform legal prior positive control |
| high_value_swing-001 | teacher_child_value_override | 384 | 2 | false | 0.0026 | 0.9714 | 0.4937 | false | override first child expansion values with teacher child values |
| high_value_swing-001 | teacher_child_value_override | 1200 | 2 | false | 0.0008 | 0.9842 | 0.5057 | false | override first child expansion values with teacher child values |
| high_value_swing-001 | neural_child_value_swap | 384 | 2 | false | 0.0026 | 0.9844 | 0.1764 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-001 | neural_child_value_swap | 1200 | 2 | false | 0.0008 | 0.9900 | 0.1898 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-021 | original | 384 | 1 | false | 0.0052 | 0.9948 | 0.2619 | false | no intervention |
| high_value_swing-021 | original | 1200 | 1 | false | 0.0050 | 0.9933 | 0.2514 | false | no intervention |
| high_value_swing-021 | equalize_root_priors | 384 | 1 | false | 0.0286 | 0.9714 | 0.2436 | false | equalize corrected reference and selected root priors |
| high_value_swing-021 | equalize_root_priors | 1200 | 1 | false | 0.0233 | 0.9758 | 0.2385 | false | equalize corrected reference and selected root priors |
| high_value_swing-021 | uniform_legal_prior | 384 | 1 | false | 0.0156 | 0.9427 | 0.2254 | false | uniform legal prior positive control |
| high_value_swing-021 | uniform_legal_prior | 1200 | 1 | false | 0.0083 | 0.9700 | 0.2597 | false | uniform legal prior positive control |
| high_value_swing-021 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.9948 | 0.6447 | false | override first child expansion values with teacher child values |
| high_value_swing-021 | teacher_child_value_override | 1200 | 1 | false | 0.0033 | 0.9950 | 0.4065 | false | override first child expansion values with teacher child values |
| high_value_swing-021 | neural_child_value_swap | 384 | 1 | false | 0.0078 | 0.9922 | 0.1949 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-021 | neural_child_value_swap | 1200 | 1 | false | 0.0058 | 0.9925 | 0.2308 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-013 | original | 384 | 2 | false | 0.0078 | 0.9427 | 0.1818 | false | no intervention |
| high_value_swing-013 | original | 1200 | 2 | false | 0.0033 | 0.9792 | 0.1948 | false | no intervention |
| high_value_swing-013 | equalize_root_priors | 384 | 2 | false | 0.0339 | 0.9505 | 0.1917 | false | equalize corrected reference and selected root priors |
| high_value_swing-013 | equalize_root_priors | 1200 | 2 | false | 0.0200 | 0.9675 | 0.2060 | false | equalize corrected reference and selected root priors |
| high_value_swing-013 | uniform_legal_prior | 384 | 2 | false | 0.0104 | 0.8958 | 0.1813 | false | uniform legal prior positive control |
| high_value_swing-013 | uniform_legal_prior | 1200 | 2 | false | 0.0067 | 0.9483 | 0.2134 | false | uniform legal prior positive control |
| high_value_swing-013 | teacher_child_value_override | 384 | 2 | false | 0.0052 | 0.9193 | 0.5670 | false | override first child expansion values with teacher child values |
| high_value_swing-013 | teacher_child_value_override | 1200 | 2 | false | 0.0025 | 0.9717 | 0.4541 | false | override first child expansion values with teacher child values |
| high_value_swing-013 | neural_child_value_swap | 384 | 2 | false | 0.0078 | 0.9427 | 0.1502 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-013 | neural_child_value_swap | 1200 | 2 | false | 0.0033 | 0.9792 | 0.1737 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-018 | original | 384 | 1 | false | 0.0182 | 0.9766 | 0.2257 | false | no intervention |
| high_value_swing-018 | original | 1200 | 1 | false | 0.0108 | 0.9875 | 0.1933 | false | no intervention |
| high_value_swing-018 | equalize_root_priors | 384 | 1 | false | 0.0521 | 0.9427 | 0.2134 | false | equalize corrected reference and selected root priors |
| high_value_swing-018 | equalize_root_priors | 1200 | 1 | false | 0.0308 | 0.9675 | 0.1903 | false | equalize corrected reference and selected root priors |
| high_value_swing-018 | uniform_legal_prior | 384 | 1 | false | 0.0156 | 0.8828 | 0.2052 | false | uniform legal prior positive control |
| high_value_swing-018 | uniform_legal_prior | 1200 | 1 | false | 0.0083 | 0.9475 | 0.1990 | false | uniform legal prior positive control |
| high_value_swing-018 | teacher_child_value_override | 384 | 1 | false | 0.0104 | 0.9844 | 0.3467 | false | override first child expansion values with teacher child values |
| high_value_swing-018 | teacher_child_value_override | 1200 | 1 | false | 0.0083 | 0.9900 | 0.2577 | false | override first child expansion values with teacher child values |
| high_value_swing-018 | neural_child_value_swap | 384 | 1 | false | 0.0182 | 0.9766 | 0.2158 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-018 | neural_child_value_swap | 1200 | 1 | false | 0.0117 | 0.9867 | 0.1868 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-008 | original | 384 | 1 | false | 0.0260 | 0.9453 | 0.1736 | false | no intervention |
| high_value_swing-008 | original | 1200 | 1 | false | 0.0158 | 0.9650 | 0.1920 | false | no intervention |
| high_value_swing-008 | equalize_root_priors | 384 | 1 | false | 0.0260 | 0.9453 | 0.1736 | false | equalize corrected reference and selected root priors |
| high_value_swing-008 | equalize_root_priors | 1200 | 1 | false | 0.0150 | 0.9658 | 0.1896 | false | equalize corrected reference and selected root priors |
| high_value_swing-008 | uniform_legal_prior | 384 | 1 | false | 0.0104 | 0.8776 | 0.2066 | false | uniform legal prior positive control |
| high_value_swing-008 | uniform_legal_prior | 1200 | 1 | false | 0.0067 | 0.9517 | 0.1951 | false | uniform legal prior positive control |
| high_value_swing-008 | teacher_child_value_override | 384 | 1 | false | 0.0260 | 0.9167 | 0.2369 | false | override first child expansion values with teacher child values |
| high_value_swing-008 | teacher_child_value_override | 1200 | 1 | false | 0.0158 | 0.9642 | 0.2463 | false | override first child expansion values with teacher child values |
| high_value_swing-008 | neural_child_value_swap | 384 | 1 | false | 0.0260 | 0.9635 | 0.1559 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-008 | neural_child_value_swap | 1200 | 1 | false | 0.0158 | 0.9708 | 0.1837 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-003 | original | 384 | 2 | false | 0.0104 | 0.9427 | 0.2183 | false | no intervention |
| high_value_swing-003 | original | 1200 | 2 | false | 0.0125 | 0.9267 | 0.1685 | false | no intervention |
| high_value_swing-003 | equalize_root_priors | 384 | 2 | false | 0.0104 | 0.9427 | 0.2183 | false | equalize corrected reference and selected root priors |
| high_value_swing-003 | equalize_root_priors | 1200 | 2 | false | 0.0125 | 0.9275 | 0.1684 | false | equalize corrected reference and selected root priors |
| high_value_swing-003 | uniform_legal_prior | 384 | 2 | false | 0.0104 | 0.9010 | 0.2222 | false | uniform legal prior positive control |
| high_value_swing-003 | uniform_legal_prior | 1200 | 2 | false | 0.0067 | 0.9608 | 0.1792 | false | uniform legal prior positive control |
| high_value_swing-003 | teacher_child_value_override | 384 | 2 | false | 0.0104 | 0.7578 | 0.4345 | false | override first child expansion values with teacher child values |
| high_value_swing-003 | teacher_child_value_override | 1200 | 2 | false | 0.0083 | 0.9167 | 0.2523 | false | override first child expansion values with teacher child values |
| high_value_swing-003 | neural_child_value_swap | 384 | 2 | false | 0.0104 | 0.9453 | 0.1865 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-003 | neural_child_value_swap | 1200 | 2 | false | 0.0133 | 0.9258 | 0.1651 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-016 | original | 384 | 1 | false | 0.0443 | 0.9219 | 0.1179 | false | no intervention |
| high_value_swing-016 | original | 1200 | 1 | false | 0.0142 | 0.9733 | 0.1294 | false | no intervention |
| high_value_swing-016 | equalize_root_priors | 384 | 1 | false | 0.0495 | 0.9219 | 0.1208 | false | equalize corrected reference and selected root priors |
| high_value_swing-016 | equalize_root_priors | 1200 | 1 | false | 0.0158 | 0.9725 | 0.1324 | false | equalize corrected reference and selected root priors |
| high_value_swing-016 | uniform_legal_prior | 384 | 1 | false | 0.0469 | 0.9089 | 0.1197 | false | uniform legal prior positive control |
| high_value_swing-016 | uniform_legal_prior | 1200 | 1 | false | 0.0150 | 0.9625 | 0.1317 | false | uniform legal prior positive control |
| high_value_swing-016 | teacher_child_value_override | 384 | 1 | false | 0.0156 | 0.9479 | 0.2371 | false | override first child expansion values with teacher child values |
| high_value_swing-016 | teacher_child_value_override | 1200 | 1 | false | 0.0083 | 0.9792 | 0.1635 | false | override first child expansion values with teacher child values |
| high_value_swing-016 | neural_child_value_swap | 384 | 1 | false | 0.0417 | 0.9245 | 0.1177 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-016 | neural_child_value_swap | 1200 | 1 | false | 0.0133 | 0.9742 | 0.1292 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-020 | original | 384 | 1 | false | 0.0052 | 0.9896 | 0.1664 | false | no intervention |
| high_value_swing-020 | original | 1200 | 1 | false | 0.0042 | 0.9942 | 0.0945 | false | no intervention |
| high_value_swing-020 | equalize_root_priors | 384 | 1 | false | 0.0286 | 0.9714 | 0.1491 | false | equalize corrected reference and selected root priors |
| high_value_swing-020 | equalize_root_priors | 1200 | 1 | false | 0.2500 | 0.7483 | 0.0060 | false | equalize corrected reference and selected root priors |
| high_value_swing-020 | uniform_legal_prior | 384 | 1 | false | 0.0156 | 0.9297 | 0.1343 | false | uniform legal prior positive control |
| high_value_swing-020 | uniform_legal_prior | 1200 | 1 | false | 0.0083 | 0.7975 | 0.1072 | false | uniform legal prior positive control |
| high_value_swing-020 | teacher_child_value_override | 384 | 1 | false | 0.0052 | 0.9896 | 0.5279 | false | override first child expansion values with teacher child values |
| high_value_swing-020 | teacher_child_value_override | 1200 | 1 | false | 0.0033 | 0.9950 | 0.3274 | false | override first child expansion values with teacher child values |
| high_value_swing-020 | neural_child_value_swap | 384 | 1 | false | 0.0078 | 0.9870 | 0.0847 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-020 | neural_child_value_swap | 1200 | 1 | false | 0.0050 | 0.9933 | 0.0734 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-009 | original | 384 | 2 | false | 0.0651 | 0.7995 | 0.0491 | false | no intervention |
| high_value_swing-009 | original | 1200 | 2 | false | 0.0208 | 0.9350 | 0.0864 | false | no intervention |
| high_value_swing-009 | equalize_root_priors | 384 | 2 | false | 0.0677 | 0.7969 | 0.0555 | false | equalize corrected reference and selected root priors |
| high_value_swing-009 | equalize_root_priors | 1200 | 2 | false | 0.0217 | 0.9350 | 0.0919 | false | equalize corrected reference and selected root priors |
| high_value_swing-009 | uniform_legal_prior | 384 | 2 | false | 0.0651 | 0.8594 | 0.0588 | false | uniform legal prior positive control |
| high_value_swing-009 | uniform_legal_prior | 1200 | 2 | false | 0.0208 | 0.9500 | 0.0885 | false | uniform legal prior positive control |
| high_value_swing-009 | teacher_child_value_override | 384 | 2 | false | 0.0052 | 0.8464 | 0.1564 | false | override first child expansion values with teacher child values |
| high_value_swing-009 | teacher_child_value_override | 1200 | 2 | false | 0.0100 | 0.9425 | 0.0593 | false | override first child expansion values with teacher child values |
| high_value_swing-009 | neural_child_value_swap | 384 | 2 | false | 0.0625 | 0.8021 | 0.0490 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-009 | neural_child_value_swap | 1200 | 2 | false | 0.0200 | 0.9358 | 0.0849 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-015 | original | 384 | 2 | false | 0.0547 | 0.8333 | 0.0641 | false | no intervention |
| high_value_swing-015 | original | 1200 | 2 | false | 0.0175 | 0.9442 | 0.0758 | false | no intervention |
| high_value_swing-015 | equalize_root_priors | 384 | 2 | false | 0.1380 | 0.7526 | 0.0683 | false | equalize corrected reference and selected root priors |
| high_value_swing-015 | equalize_root_priors | 1200 | 2 | false | 0.0442 | 0.9192 | 0.0758 | false | equalize corrected reference and selected root priors |
| high_value_swing-015 | uniform_legal_prior | 384 | 1 | false | 0.1667 | 0.7214 | 0.1060 | false | uniform legal prior positive control |
| high_value_swing-015 | uniform_legal_prior | 1200 | 1 | false | 0.0533 | 0.9075 | 0.1233 | false | uniform legal prior positive control |
| high_value_swing-015 | teacher_child_value_override | 384 | 2 | false | 0.0078 | 0.8672 | 0.1813 | false | override first child expansion values with teacher child values |
| high_value_swing-015 | teacher_child_value_override | 1200 | 2 | false | 0.0050 | 0.9550 | 0.1417 | false | override first child expansion values with teacher child values |
| high_value_swing-015 | neural_child_value_swap | 384 | 2 | false | 0.0547 | 0.8333 | 0.0646 | false | swap child neural values as diagnostic sanity check |
| high_value_swing-015 | neural_child_value_swap | 1200 | 2 | false | 0.0175 | 0.9442 | 0.0762 | false | swap child neural values as diagnostic sanity check |

## 10. Row classifications

| row_id | role | row_classification | supporting_evidence | recommended_use | notes |
| --- | --- | --- | --- | --- | --- |
| high_value_swing-024 | target_candidate | puct_child_search_value_mismatch | ClassicMCTS child teacher and neural child values prefer corrected reference child but child PUCT does not | target_candidate | root still fails |
| high_value_swing-007 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-025 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-023 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-001 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-021 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-013 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-018 | target_candidate | inconclusive | ClassicMCTS child teacher is unstable across seeds | target_candidate | root still fails |
| high_value_swing-008 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-003 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-016 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-020 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-009 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-015 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_value_swing-011 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| high_value_swing-022 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| high_value_swing-010 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| high_value_swing-026 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| high_value_swing-027 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| high_value_swing-017 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |

## 11. Family-level interpretation

| family_classification | value_head_miscalibration_count | puct_child_search_mismatch_count | root_selection_pressure_count | corrected_reference_suspicious_count | inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- |
| reference_family_uncertain | 0 | 1 | 0 | 12 | 1 | adjudicate high_value_swing references before training. |

## 12. Exactly one recommended next action

Recommendation: **adjudicate high_value_swing references before training.**
