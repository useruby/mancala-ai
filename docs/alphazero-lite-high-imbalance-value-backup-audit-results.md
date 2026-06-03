# AlphaZero-lite High Imbalance Value/Backup Audit Results

## 1. Context

- This audit evaluates child-afterstate values and backup behavior for the `high_imbalance` corrected non-opening failure family.
- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references were not mutated.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Selected family rows: `/tmp/azlite_corrected_non_opening_failure_mining_v3/selected_non_opening_family_rows_v3.jsonl`.

## 2. Why high_imbalance was selected

- PR #54 selected `high_imbalance` as the next corrected non-opening failure family.
- Family stats: `{"avg_reference_visit_share_1200": 0.2873, "avg_reference_visit_share_384": 0.205, "avg_selected_minus_reference_q_margin_1200": 0.0975, "avg_selected_minus_reference_q_margin_384": 0.1077, "classification": "value_or_backup_issue", "conflicting_target_count": 0, "dominant_failure_mode": "value_q", "duplicate_canonical_state_count": 0, "fail_rows": 16, "failure_rate": 0.6667, "family": "high_imbalance", "high_severity_count": 0, "medium_severity_count": 16, "notes": "persistent failures remain Q/value-dominant", "pass_rows": 8, "persistent_1200_failures": 15, "rank_score": 272.2512, "recovered_at_1200": 1, "stable_corrected_reference_count": 24, "total_rows": 24}`.

## 3. Row validation

| row_id | role | corrected_reference_move | legal | reference_unstable | status | notes |
| --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | target_candidate | 4 | true | false | ok | validated, required target row present |
| high_imbalance-018 | target_candidate | 2 | true | false | ok | validated, required target row present |
| high_imbalance-020 | target_candidate | 2 | true | false | ok | validated, required target row present |
| high_imbalance-021 | target_candidate | 2 | true | false | ok | validated, required target row present |
| high_imbalance-025 | target_candidate | 0 | true | false | ok | validated, required target row present |
| high_imbalance-024 | target_candidate | 2 | true | false | ok | validated, required target row present |
| high_imbalance-016 | target_candidate | 2 | true | false | ok | validated, required target row present |
| high_imbalance-027 | target_candidate | 2 | true | false | ok | validated, required target row present |
| high_imbalance-019 | target_candidate | 3 | true | false | ok | validated, required target row present |
| high_imbalance-023 | target_candidate | 2 | true | false | ok | validated, required target row present |
| high_imbalance-026 | target_candidate | 2 | true | false | ok | validated |
| high_imbalance-014 | target_candidate | 0 | true | false | ok | validated |
| high_imbalance-022 | target_candidate | 2 | true | false | ok | validated |
| high_imbalance-008 | holdout_candidate | 3 | true | false | ok | validated |
| high_imbalance-002 | holdout_candidate | 3 | true | false | ok | validated |
| high_imbalance-007 | preservation_control | 1 | true | false | ok | validated, required control row present |
| high_imbalance-012 | preservation_control | 1 | true | false | ok | validated, required control row present |
| high_imbalance-001 | preservation_control | 3 | true | false | ok | validated, required control row present |
| high_imbalance-017 | preservation_control | 1 | true | false | ok | validated, required control row present |

- Perspective conversion: `+1 when child current_player == root current_player, else -1`.
- Implementation rule: `PUCT _search negates the returned child value exactly when child.game.current_player != parent.game.current_player`.

## 4. Root PUCT baseline

| row_id | role | budget | corrected_reference_move | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | reference_q | selected_q | selected_minus_reference_q_margin | reference_policy_probability | selected_policy_probability | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-001 | preservation_control | 384 | 3 | 3 | true | 0.5990 | 0.5990 | -0.1681 | -0.1681 | 0.0000 | 0.3073 | 0.3073 | preservation_control, selected reference |
| high_imbalance-001 | preservation_control | 1200 | 3 | 3 | true | 0.8717 | 0.8717 | -0.1586 | -0.1586 | 0.0000 | 0.3073 | 0.3073 | preservation_control, selected reference |
| high_imbalance-001 | preservation_control | 2400 | 3 | 3 | true | 0.9083 | 0.9083 | -0.1986 | -0.1986 | 0.0000 | 0.3073 | 0.3073 | preservation_control, selected reference |
| high_imbalance-002 | holdout_candidate | 384 | 3 | 5 | false | 0.0417 | 0.8281 | -0.1543 | -0.1275 | 0.0269 | 0.1735 | 0.4198 | holdout_candidate, selected away from reference |
| high_imbalance-002 | holdout_candidate | 1200 | 3 | 5 | false | 0.0292 | 0.8892 | -0.1645 | -0.1437 | 0.0208 | 0.1735 | 0.4198 | holdout_candidate, selected away from reference |
| high_imbalance-002 | holdout_candidate | 2400 | 3 | 5 | false | 0.0167 | 0.7042 | -0.1797 | -0.1636 | 0.0161 | 0.1735 | 0.4198 | holdout_candidate, selected away from reference |
| high_imbalance-007 | preservation_control | 384 | 1 | 1 | true | 0.9870 | 0.9870 | 0.0970 | 0.0970 | 0.0000 | 0.8161 | 0.8161 | preservation_control, selected reference |
| high_imbalance-007 | preservation_control | 1200 | 1 | 1 | true | 0.9925 | 0.9925 | 0.1109 | 0.1109 | 0.0000 | 0.8161 | 0.8161 | preservation_control, selected reference |
| high_imbalance-007 | preservation_control | 2400 | 1 | 1 | true | 0.9950 | 0.9950 | 0.0783 | 0.0783 | 0.0000 | 0.8161 | 0.8161 | preservation_control, selected reference |
| high_imbalance-008 | holdout_candidate | 384 | 3 | 3 | true | 0.4427 | 0.4427 | -0.1812 | -0.1812 | 0.0000 | 0.1705 | 0.1705 | holdout_candidate, selected reference |
| high_imbalance-008 | holdout_candidate | 1200 | 3 | 2 | false | 0.1950 | 0.7158 | -0.2116 | -0.1720 | 0.0396 | 0.1705 | 0.5839 | holdout_candidate, selected away from reference |
| high_imbalance-008 | holdout_candidate | 2400 | 3 | 2 | false | 0.0975 | 0.8579 | -0.2116 | -0.1871 | 0.0245 | 0.1705 | 0.5839 | holdout_candidate, selected away from reference |
| high_imbalance-009 | target_candidate | 384 | 4 | 5 | false | 0.0130 | 0.9844 | 0.0343 | 0.2587 | 0.2245 | 0.1478 | 0.7743 | target_candidate, selected away from reference |
| high_imbalance-009 | target_candidate | 1200 | 4 | 5 | false | 0.0050 | 0.9908 | 0.0324 | 0.2973 | 0.2649 | 0.1478 | 0.7743 | target_candidate, selected away from reference |
| high_imbalance-009 | target_candidate | 2400 | 4 | 5 | false | 0.0033 | 0.9938 | 0.0203 | 0.2772 | 0.2569 | 0.1478 | 0.7743 | target_candidate, selected away from reference |
| high_imbalance-012 | preservation_control | 384 | 1 | 1 | true | 0.9609 | 0.9609 | -0.3437 | -0.3437 | 0.0000 | 0.5180 | 0.5180 | preservation_control, selected reference |
| high_imbalance-012 | preservation_control | 1200 | 1 | 1 | true | 0.9267 | 0.9267 | -0.3132 | -0.3132 | 0.0000 | 0.5180 | 0.5180 | preservation_control, selected reference |
| high_imbalance-012 | preservation_control | 2400 | 1 | 1 | true | 0.9342 | 0.9342 | -0.3143 | -0.3143 | 0.0000 | 0.5180 | 0.5180 | preservation_control, selected reference |
| high_imbalance-014 | target_candidate | 384 | 0 | 4 | false | 0.0026 | 0.8984 | -0.2266 | -0.0947 | 0.1319 | 0.0495 | 0.3145 | target_candidate, selected away from reference |
| high_imbalance-014 | target_candidate | 1200 | 0 | 4 | false | 0.0100 | 0.9300 | -0.1416 | -0.0958 | 0.0458 | 0.0495 | 0.3145 | target_candidate, selected away from reference |
| high_imbalance-014 | target_candidate | 2400 | 0 | 4 | false | 0.0050 | 0.9621 | -0.1416 | -0.0624 | 0.0792 | 0.0495 | 0.3145 | target_candidate, selected away from reference |
| high_imbalance-016 | target_candidate | 384 | 2 | 4 | false | 0.0078 | 0.8984 | -0.2401 | -0.1012 | 0.1390 | 0.0377 | 0.2526 | target_candidate, selected away from reference |
| high_imbalance-016 | target_candidate | 1200 | 2 | 4 | false | 0.0025 | 0.9483 | -0.2401 | -0.1438 | 0.0963 | 0.0377 | 0.2526 | target_candidate, selected away from reference |
| high_imbalance-016 | target_candidate | 2400 | 2 | 4 | false | 0.0017 | 0.9604 | -0.2446 | -0.1645 | 0.0801 | 0.0377 | 0.2526 | target_candidate, selected away from reference |
| high_imbalance-017 | preservation_control | 384 | 1 | 1 | true | 0.8542 | 0.8542 | -0.2916 | -0.2916 | 0.0000 | 0.4291 | 0.4291 | preservation_control, selected reference |
| high_imbalance-017 | preservation_control | 1200 | 1 | 1 | true | 0.8625 | 0.8625 | -0.3126 | -0.3126 | 0.0000 | 0.4291 | 0.4291 | preservation_control, selected reference |
| high_imbalance-017 | preservation_control | 2400 | 1 | 1 | true | 0.8725 | 0.8725 | -0.3356 | -0.3356 | 0.0000 | 0.4291 | 0.4291 | preservation_control, selected reference |
| high_imbalance-018 | target_candidate | 384 | 2 | 4 | false | 0.0026 | 0.8906 | -0.3890 | -0.0914 | 0.2976 | 0.0526 | 0.2512 | target_candidate, selected away from reference |
| high_imbalance-018 | target_candidate | 1200 | 2 | 4 | false | 0.0033 | 0.9392 | -0.3240 | -0.0960 | 0.2281 | 0.0526 | 0.2512 | target_candidate, selected away from reference |
| high_imbalance-018 | target_candidate | 2400 | 2 | 4 | false | 0.0021 | 0.9542 | -0.3700 | -0.1041 | 0.2659 | 0.0526 | 0.2512 | target_candidate, selected away from reference |
| high_imbalance-019 | target_candidate | 384 | 3 | 3 | true | 0.5573 | 0.5573 | -0.2482 | -0.2482 | 0.0000 | 0.3610 | 0.3610 | target_candidate, selected reference |
| high_imbalance-019 | target_candidate | 1200 | 3 | 5 | false | 0.1783 | 0.7775 | -0.2482 | -0.1552 | 0.0930 | 0.3610 | 0.0939 | target_candidate, selected away from reference |
| high_imbalance-019 | target_candidate | 2400 | 3 | 5 | false | 0.0892 | 0.8604 | -0.2482 | -0.1754 | 0.0728 | 0.3610 | 0.0939 | target_candidate, selected away from reference |
| high_imbalance-020 | target_candidate | 384 | 2 | 4 | false | 0.0078 | 0.8854 | -0.2903 | -0.0993 | 0.1911 | 0.0233 | 0.2531 | target_candidate, selected away from reference |
| high_imbalance-020 | target_candidate | 1200 | 2 | 4 | false | 0.0025 | 0.9558 | -0.2903 | -0.0785 | 0.2119 | 0.0233 | 0.2531 | target_candidate, selected away from reference |
| high_imbalance-020 | target_candidate | 2400 | 2 | 4 | false | 0.0013 | 0.9742 | -0.2903 | -0.0504 | 0.2399 | 0.0233 | 0.2531 | target_candidate, selected away from reference |
| high_imbalance-021 | target_candidate | 384 | 2 | 4 | false | 0.0026 | 0.7396 | -0.4216 | -0.0490 | 0.3726 | 0.0349 | 0.2769 | target_candidate, selected away from reference |
| high_imbalance-021 | target_candidate | 1200 | 2 | 4 | false | 0.0033 | 0.9108 | -0.2811 | -0.1136 | 0.1675 | 0.0349 | 0.2769 | target_candidate, selected away from reference |
| high_imbalance-021 | target_candidate | 2400 | 2 | 4 | false | 0.0029 | 0.9400 | -0.2460 | -0.1206 | 0.1254 | 0.0349 | 0.2769 | target_candidate, selected away from reference |
| high_imbalance-022 | target_candidate | 384 | 2 | 5 | false | 0.0156 | 0.8906 | -0.2498 | -0.2080 | 0.0418 | 0.1563 | 0.7017 | target_candidate, selected away from reference |
| high_imbalance-022 | target_candidate | 1200 | 2 | 5 | false | 0.0058 | 0.8583 | -0.2881 | -0.2463 | 0.0418 | 0.1563 | 0.7017 | target_candidate, selected away from reference |
| high_imbalance-022 | target_candidate | 2400 | 2 | 5 | false | 0.2858 | 0.5504 | -0.1689 | -0.2717 | -0.1028 | 0.1563 | 0.7017 | target_candidate, selected away from reference |
| high_imbalance-023 | target_candidate | 384 | 2 | 3 | false | 0.0208 | 0.8802 | -0.4059 | -0.2635 | 0.1424 | 0.3619 | 0.3056 | target_candidate, selected away from reference |
| high_imbalance-023 | target_candidate | 1200 | 2 | 3 | false | 0.0125 | 0.9558 | -0.3502 | -0.2632 | 0.0870 | 0.3619 | 0.3056 | target_candidate, selected away from reference |
| high_imbalance-023 | target_candidate | 2400 | 2 | 3 | false | 0.0108 | 0.9717 | -0.3394 | -0.2878 | 0.0516 | 0.3619 | 0.3056 | target_candidate, selected away from reference |
| high_imbalance-024 | target_candidate | 384 | 2 | 1 | false | 0.0495 | 0.8542 | -0.1909 | -0.1576 | 0.0332 | 0.0190 | 0.2403 | target_candidate, selected away from reference |
| high_imbalance-024 | target_candidate | 1200 | 2 | 1 | false | 0.3017 | 0.5758 | -0.1037 | -0.2025 | -0.0988 | 0.0190 | 0.2403 | target_candidate, selected away from reference |
| high_imbalance-024 | target_candidate | 2400 | 2 | 2 | true | 0.6496 | 0.6496 | -0.1239 | -0.1239 | 0.0000 | 0.0190 | 0.0190 | target_candidate, selected reference |
| high_imbalance-025 | target_candidate | 384 | 0 | 3 | false | 0.0026 | 0.7474 | -0.3543 | -0.3400 | 0.0143 | 0.0262 | 0.3690 | target_candidate, selected away from reference |
| high_imbalance-025 | target_candidate | 1200 | 0 | 4 | false | 0.0008 | 0.7233 | -0.3543 | -0.2391 | 0.1152 | 0.0262 | 0.2597 | target_candidate, selected away from reference |
| high_imbalance-025 | target_candidate | 2400 | 0 | 4 | false | 0.0004 | 0.8279 | -0.3543 | -0.2794 | 0.0749 | 0.0262 | 0.2597 | target_candidate, selected away from reference |
| high_imbalance-026 | target_candidate | 384 | 2 | 4 | false | 0.0234 | 0.9167 | -0.2126 | -0.1062 | 0.1064 | 0.1773 | 0.1858 | target_candidate, selected away from reference |
| high_imbalance-026 | target_candidate | 1200 | 2 | 4 | false | 0.0125 | 0.9600 | -0.2042 | -0.1509 | 0.0532 | 0.1773 | 0.1858 | target_candidate, selected away from reference |
| high_imbalance-026 | target_candidate | 2400 | 2 | 4 | false | 0.0146 | 0.9663 | -0.1981 | -0.1400 | 0.0581 | 0.1773 | 0.1858 | target_candidate, selected away from reference |
| high_imbalance-027 | target_candidate | 384 | 2 | 4 | false | 0.0078 | 0.8672 | -0.1197 | -0.0545 | 0.0652 | 0.0284 | 0.2586 | target_candidate, selected away from reference |
| high_imbalance-027 | target_candidate | 1200 | 2 | 4 | false | 0.0033 | 0.9483 | -0.1669 | -0.0713 | 0.0956 | 0.0284 | 0.2586 | target_candidate, selected away from reference |
| high_imbalance-027 | target_candidate | 2400 | 2 | 4 | false | 0.0017 | 0.9688 | -0.1669 | -0.0881 | 0.0788 | 0.0284 | 0.2586 | target_candidate, selected away from reference |

- ran 2400 root budget: projected ~3.0s across 13 target rows

## 5. Move consequence audit

| row_id | move | is_corrected_reference | is_selected | gives_extra_turn | produces_capture | capture_count | immediate_store_delta | side_to_move_after | game_over_after_move | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | 3 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_imbalance-009 | 4 | true | false | false | false | 0 | 1 | 1 | false |  |
| high_imbalance-009 | 5 | false | true | false | false | 0 | 1 | 1 | false |  |
| high_imbalance-018 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-018 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-018 | 2 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-018 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-018 | 4 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| high_imbalance-018 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-020 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-020 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-020 | 2 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-020 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-020 | 4 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| high_imbalance-020 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-021 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-021 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-021 | 2 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-021 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-021 | 4 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| high_imbalance-021 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-025 | 0 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-025 | 1 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-025 | 2 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-025 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-025 | 4 | false | true | false | true | 2 | 2 | 0 | false | selected move captures more immediately; selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| high_imbalance-024 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-024 | 1 | false | true | false | false | 0 | 1 | 0 | false | selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| high_imbalance-024 | 2 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-024 | 3 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-024 | 4 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-024 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-016 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-016 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-016 | 2 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-016 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-016 | 4 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| high_imbalance-016 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-027 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-027 | 1 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-027 | 2 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-027 | 3 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-027 | 4 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn; selected move has larger immediate store gain; selected move leaves fewer seeds in pits |
| high_imbalance-027 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-019 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-019 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-019 | 2 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-019 | 3 | true | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-019 | 4 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-019 | 5 | false | true | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-023 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-023 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-023 | 2 | true | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-023 | 3 | false | true | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-023 | 4 | false | false | false | true | 2 | 2 | 0 | false |  |
| high_imbalance-026 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-026 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-026 | 2 | true | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-026 | 3 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-026 | 4 | false | true | true | false | 0 | 1 | 1 | false | selected move gains extra turn |
| high_imbalance-026 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-014 | 0 | true | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-014 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-014 | 2 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-014 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-014 | 4 | false | true | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-014 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-022 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-022 | 1 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-022 | 2 | true | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-022 | 3 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-022 | 4 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-022 | 5 | false | true | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-008 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-008 | 2 | false | true | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-008 | 3 | true | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-008 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-002 | 2 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_imbalance-002 | 3 | true | false | false | false | 0 | 1 | 1 | false |  |
| high_imbalance-002 | 4 | false | false | false | false | 0 | 1 | 1 | false |  |
| high_imbalance-002 | 5 | false | true | false | false | 0 | 1 | 1 | false |  |
| high_imbalance-007 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-007 | 1 | true | true | true | false | 0 | 1 | 1 | false |  |
| high_imbalance-007 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-007 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-012 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-012 | 1 | true | true | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-012 | 3 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-012 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-001 | 2 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-001 | 3 | true | true | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-001 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-001 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-017 | 0 | false | false | false | false | 0 | 0 | 0 | false |  |
| high_imbalance-017 | 1 | true | true | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-017 | 4 | false | false | false | false | 0 | 1 | 0 | false |  |
| high_imbalance-017 | 5 | false | false | false | false | 0 | 1 | 0 | false |  |

## 6. Neural child-afterstate value audit

| row_id | corrected_reference_move | selected_move | child | raw_value | root_perspective_value | child_ref_minus_child_selected | neural_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | 4 | 5 | corrected_reference_child | 0.0800 | -0.0800 | -0.0679 | false | sign flip to root perspective |
| high_imbalance-009 | 4 | 5 | selected_child | 0.0120 | -0.0120 | -0.0679 | false | sign flip to root perspective |
| high_imbalance-018 | 2 | 4 | corrected_reference_child | 0.3890 | -0.3890 | -0.4056 | false | sign flip to root perspective |
| high_imbalance-018 | 2 | 4 | selected_child | 0.0167 | 0.0167 | -0.4056 | false | identity conversion |
| high_imbalance-020 | 2 | 4 | corrected_reference_child | 0.2112 | -0.2112 | -0.1758 | false | sign flip to root perspective |
| high_imbalance-020 | 2 | 4 | selected_child | -0.0354 | -0.0354 | -0.1758 | false | identity conversion |
| high_imbalance-021 | 2 | 4 | corrected_reference_child | 0.4216 | -0.4216 | -0.4009 | false | sign flip to root perspective |
| high_imbalance-021 | 2 | 4 | selected_child | -0.0207 | -0.0207 | -0.4009 | false | identity conversion |
| high_imbalance-025 | 0 | 4 | corrected_reference_child | 0.3543 | -0.3543 | 0.0228 | true | sign flip to root perspective |
| high_imbalance-025 | 0 | 4 | selected_child | 0.3771 | -0.3771 | 0.0228 | true | sign flip to root perspective |
| high_imbalance-024 | 2 | 1 | corrected_reference_child | 0.0337 | -0.0337 | 0.1558 | true | sign flip to root perspective |
| high_imbalance-024 | 2 | 1 | selected_child | 0.1895 | -0.1895 | 0.1558 | true | sign flip to root perspective |
| high_imbalance-016 | 2 | 4 | corrected_reference_child | 0.2009 | -0.2009 | -0.2658 | false | sign flip to root perspective |
| high_imbalance-016 | 2 | 4 | selected_child | 0.0648 | 0.0648 | -0.2658 | false | identity conversion |
| high_imbalance-027 | 2 | 4 | corrected_reference_child | 0.1332 | -0.1332 | -0.1671 | false | sign flip to root perspective |
| high_imbalance-027 | 2 | 4 | selected_child | 0.0339 | 0.0339 | -0.1671 | false | identity conversion |
| high_imbalance-019 | 3 | 5 | corrected_reference_child | 0.1985 | -0.1985 | -0.1555 | false | sign flip to root perspective |
| high_imbalance-019 | 3 | 5 | selected_child | 0.0430 | -0.0430 | -0.1555 | false | sign flip to root perspective |
| high_imbalance-023 | 2 | 3 | corrected_reference_child | 0.2650 | -0.2650 | -0.0110 | false | sign flip to root perspective |
| high_imbalance-023 | 2 | 3 | selected_child | 0.2540 | -0.2540 | -0.0110 | false | sign flip to root perspective |
| high_imbalance-026 | 2 | 4 | corrected_reference_child | 0.1156 | -0.1156 | -0.1830 | false | sign flip to root perspective |
| high_imbalance-026 | 2 | 4 | selected_child | 0.0674 | 0.0674 | -0.1830 | false | identity conversion |
| high_imbalance-014 | 0 | 4 | corrected_reference_child | 0.2266 | -0.2266 | -0.1332 | false | sign flip to root perspective |
| high_imbalance-014 | 0 | 4 | selected_child | 0.0934 | -0.0934 | -0.1332 | false | sign flip to root perspective |
| high_imbalance-022 | 2 | 5 | corrected_reference_child | 0.2596 | -0.2596 | -0.2283 | false | sign flip to root perspective |
| high_imbalance-022 | 2 | 5 | selected_child | 0.0312 | -0.0312 | -0.2283 | false | sign flip to root perspective |

## 7. ClassicMCTS child-afterstate audit

| row_id | budget | seeds | child_ref_value_root_mean | child_selected_value_root_mean | child_ref_minus_child_selected | teacher_prefers_reference | stable | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | 1200 | [11, 23, 37, 42, 101] | 0.7640 | -0.6420 | 1.4060 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-009 | 2400 | [11, 23, 37, 42, 101] | 0.2446 | -0.8090 | 1.0536 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-009 | 5000 | [11, 23, 37, 42, 101] | -0.3579 | -0.8798 | 0.5219 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-018 | 1200 | [11, 23, 37, 42, 101] | -0.9483 | 0.8218 | -1.7701 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-018 | 2400 | [11, 23, 37, 42, 101] | -0.9462 | 0.8912 | -1.8374 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-018 | 5000 | [11, 23, 37, 42, 101] | -0.9454 | 0.9361 | -1.8815 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-020 | 1200 | [11, 23, 37, 42, 101] | -0.9745 | 0.4159 | -1.3904 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-020 | 2400 | [11, 23, 37, 42, 101] | -0.9760 | 0.6210 | -1.5970 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-020 | 5000 | [11, 23, 37, 42, 101] | -0.9712 | 0.7930 | -1.7641 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-021 | 1200 | [11, 23, 37, 42, 101] | -0.9784 | 0.4300 | -1.4083 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-021 | 2400 | [11, 23, 37, 42, 101] | -0.9681 | 0.5607 | -1.5288 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-021 | 5000 | [11, 23, 37, 42, 101] | -0.9730 | 0.7284 | -1.7014 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-025 | 1200 | [11, 23, 37, 42, 101] | -0.9699 | -0.9427 | -0.0272 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-025 | 2400 | [11, 23, 37, 42, 101] | -0.9682 | -0.9420 | -0.0262 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-025 | 5000 | [11, 23, 37, 42, 101] | -0.9707 | -0.9329 | -0.0379 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-024 | 1200 | [11, 23, 37, 42, 101] | -0.9326 | -0.9570 | 0.0244 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-024 | 2400 | [11, 23, 37, 42, 101] | -0.9227 | -0.9489 | 0.0262 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-024 | 5000 | [11, 23, 37, 42, 101] | -0.9220 | -0.9580 | 0.0360 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-016 | 1200 | [11, 23, 37, 42, 101] | -0.9787 | 0.8356 | -1.8144 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-016 | 2400 | [11, 23, 37, 42, 101] | -0.9710 | 0.9005 | -1.8715 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-016 | 5000 | [11, 23, 37, 42, 101] | -0.9601 | 0.9414 | -1.9014 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-027 | 1200 | [11, 23, 37, 42, 101] | -0.9875 | 0.2027 | -1.1903 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-027 | 2400 | [11, 23, 37, 42, 101] | -0.9800 | 0.4014 | -1.3814 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-027 | 5000 | [11, 23, 37, 42, 101] | -0.9739 | 0.6457 | -1.6196 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-019 | 1200 | [11, 23, 37, 42, 101] | -0.8832 | -0.9450 | 0.0618 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-019 | 2400 | [11, 23, 37, 42, 101] | -0.8705 | -0.9315 | 0.0610 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-019 | 5000 | [11, 23, 37, 42, 101] | -0.8750 | -0.9480 | 0.0730 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-023 | 1200 | [11, 23, 37, 42, 101] | -0.9493 | -0.9958 | 0.0464 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-023 | 2400 | [11, 23, 37, 42, 101] | -0.9403 | -0.9825 | 0.0422 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-023 | 5000 | [11, 23, 37, 42, 101] | -0.9464 | -0.9805 | 0.0341 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-026 | 1200 | [11, 23, 37, 42, 101] | -0.9245 | 0.4742 | -1.3987 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-026 | 2400 | [11, 23, 37, 42, 101] | -0.9209 | 0.6974 | -1.6182 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-026 | 5000 | [11, 23, 37, 42, 101] | -0.9143 | 0.8280 | -1.7423 | false | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-014 | 1200 | [11, 23, 37, 42, 101] | -0.8914 | -0.8831 | -0.0084 | false | false | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-014 | 2400 | [11, 23, 37, 42, 101] | -0.8976 | -0.8909 | -0.0066 | false | false | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-014 | 5000 | [11, 23, 37, 42, 101] | -0.9030 | -0.9182 | 0.0152 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-022 | 1200 | [11, 23, 37, 42, 101] | -0.9164 | -0.9907 | 0.0742 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-022 | 2400 | [11, 23, 37, 42, 101] | -0.9197 | -0.9767 | 0.0569 | true | true | ClassicMCTS child-afterstate teacher aggregate |
| high_imbalance-022 | 5000 | [11, 23, 37, 42, 101] | -0.9322 | -0.9639 | 0.0317 | true | true | ClassicMCTS child-afterstate teacher aggregate |

## 8. PUCT child-afterstate audit

| row_id | budget | child_ref_value_root | child_selected_value_root | child_ref_minus_child_selected | puct_prefers_reference | notes |
| --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | 384 | 0.2523 | 0.2611 | -0.0088 | false | artifact PUCT child-afterstate audit |
| high_imbalance-009 | 1200 | 0.2735 | 0.2972 | -0.0237 | false | artifact PUCT child-afterstate audit |
| high_imbalance-018 | 384 | -0.2965 | -0.1051 | -0.1914 | false | artifact PUCT child-afterstate audit |
| high_imbalance-018 | 1200 | -0.3354 | -0.0947 | -0.2408 | false | artifact PUCT child-afterstate audit |
| high_imbalance-020 | 384 | -0.3084 | -0.0931 | -0.2153 | false | artifact PUCT child-afterstate audit |
| high_imbalance-020 | 1200 | -0.3418 | -0.0791 | -0.2627 | false | artifact PUCT child-afterstate audit |
| high_imbalance-021 | 384 | -0.2902 | -0.0580 | -0.2322 | false | artifact PUCT child-afterstate audit |
| high_imbalance-021 | 1200 | -0.3286 | -0.1228 | -0.2057 | false | artifact PUCT child-afterstate audit |
| high_imbalance-025 | 384 | -0.3963 | -0.2143 | -0.1821 | false | artifact PUCT child-afterstate audit |
| high_imbalance-025 | 1200 | -0.4147 | -0.2594 | -0.1554 | false | artifact PUCT child-afterstate audit |
| high_imbalance-024 | 384 | -0.1156 | -0.1603 | 0.0447 | true | artifact PUCT child-afterstate audit |
| high_imbalance-024 | 1200 | -0.1263 | -0.2250 | 0.0987 | true | artifact PUCT child-afterstate audit |
| high_imbalance-016 | 384 | -0.2643 | -0.1128 | -0.1515 | false | artifact PUCT child-afterstate audit |
| high_imbalance-016 | 1200 | -0.2819 | -0.1478 | -0.1341 | false | artifact PUCT child-afterstate audit |
| high_imbalance-027 | 384 | -0.2827 | -0.0544 | -0.2283 | false | artifact PUCT child-afterstate audit |
| high_imbalance-027 | 1200 | -0.3141 | -0.0785 | -0.2356 | false | artifact PUCT child-afterstate audit |
| high_imbalance-019 | 384 | -0.2427 | -0.1914 | -0.0512 | false | artifact PUCT child-afterstate audit |
| high_imbalance-019 | 1200 | -0.2659 | -0.1607 | -0.1053 | false | artifact PUCT child-afterstate audit |
| high_imbalance-023 | 384 | -0.2972 | -0.2643 | -0.0329 | false | artifact PUCT child-afterstate audit |
| high_imbalance-023 | 1200 | -0.2902 | -0.2685 | -0.0217 | false | artifact PUCT child-afterstate audit |
| high_imbalance-026 | 384 | -0.1936 | -0.1041 | -0.0895 | false | artifact PUCT child-afterstate audit |
| high_imbalance-026 | 1200 | -0.2656 | -0.1513 | -0.1143 | false | artifact PUCT child-afterstate audit |
| high_imbalance-014 | 384 | -0.1672 | -0.1120 | -0.0552 | false | artifact PUCT child-afterstate audit |
| high_imbalance-014 | 1200 | -0.1969 | -0.0926 | -0.1043 | false | artifact PUCT child-afterstate audit |
| high_imbalance-022 | 384 | -0.1462 | -0.2784 | 0.1322 | true | artifact PUCT child-afterstate audit |
| high_imbalance-022 | 1200 | -0.1410 | -0.2842 | 0.1432 | true | artifact PUCT child-afterstate audit |

## 9. Root counterfactual diagnostics

| row_id | intervention | budget | selected_move | selected_is_reference | reference_visit_share | selected_visit_share | selected_minus_reference_q_margin | flipped | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | original | 384 | 5 | false | 0.0130 | 0.9844 | 0.2245 | false | no intervention |
| high_imbalance-009 | original | 1200 | 5 | false | 0.0050 | 0.9908 | 0.2649 | false | no intervention |
| high_imbalance-009 | equalize_root_priors | 384 | 5 | false | 0.1510 | 0.8464 | 0.0649 | false | equalize corrected reference and selected root priors |
| high_imbalance-009 | equalize_root_priors | 1200 | 4 | true | 0.5700 | 0.5700 | 0.0000 | true | equalize corrected reference and selected root priors |
| high_imbalance-009 | uniform_legal_prior | 384 | 5 | false | 0.0182 | 0.9375 | 0.2166 | false | uniform legal prior positive control |
| high_imbalance-009 | uniform_legal_prior | 1200 | 5 | false | 0.0117 | 0.8500 | 0.2404 | false | uniform legal prior positive control |
| high_imbalance-009 | teacher_child_value_override | 384 | 5 | false | 0.0078 | 0.9896 | 0.3213 | false | override first child expansion values with teacher child values |
| high_imbalance-009 | teacher_child_value_override | 1200 | 5 | false | 0.0050 | 0.9908 | 0.2976 | false | override first child expansion values with teacher child values |
| high_imbalance-009 | neural_child_value_swap | 384 | 5 | false | 0.0130 | 0.9844 | 0.2107 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-009 | neural_child_value_swap | 1200 | 5 | false | 0.0050 | 0.9917 | 0.2534 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-018 | original | 384 | 4 | false | 0.0026 | 0.8906 | 0.2976 | false | no intervention |
| high_imbalance-018 | original | 1200 | 4 | false | 0.0033 | 0.9392 | 0.2281 | false | no intervention |
| high_imbalance-018 | equalize_root_priors | 384 | 4 | false | 0.0234 | 0.8776 | 0.2270 | false | equalize corrected reference and selected root priors |
| high_imbalance-018 | equalize_root_priors | 1200 | 4 | false | 0.0233 | 0.9275 | 0.1218 | false | equalize corrected reference and selected root priors |
| high_imbalance-018 | uniform_legal_prior | 384 | 4 | false | 0.0104 | 0.9115 | 0.2325 | false | uniform legal prior positive control |
| high_imbalance-018 | uniform_legal_prior | 1200 | 4 | false | 0.0058 | 0.9367 | 0.2121 | false | uniform legal prior positive control |
| high_imbalance-018 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.7839 | 0.8716 | false | override first child expansion values with teacher child values |
| high_imbalance-018 | teacher_child_value_override | 1200 | 4 | false | 0.0017 | 0.8675 | 0.4347 | false | override first child expansion values with teacher child values |
| high_imbalance-018 | neural_child_value_swap | 384 | 4 | false | 0.0104 | 0.8828 | 0.1308 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-018 | neural_child_value_swap | 1200 | 4 | false | 0.0042 | 0.9383 | 0.1926 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-020 | original | 384 | 4 | false | 0.0078 | 0.8854 | 0.1911 | false | no intervention |
| high_imbalance-020 | original | 1200 | 4 | false | 0.0025 | 0.9558 | 0.2119 | false | no intervention |
| high_imbalance-020 | equalize_root_priors | 384 | 4 | false | 0.0130 | 0.8854 | 0.1964 | false | equalize corrected reference and selected root priors |
| high_imbalance-020 | equalize_root_priors | 1200 | 4 | false | 0.0067 | 0.9542 | 0.2449 | false | equalize corrected reference and selected root priors |
| high_imbalance-020 | uniform_legal_prior | 384 | 4 | false | 0.0104 | 0.8906 | 0.1731 | false | uniform legal prior positive control |
| high_imbalance-020 | uniform_legal_prior | 1200 | 4 | false | 0.0067 | 0.9525 | 0.2452 | false | uniform legal prior positive control |
| high_imbalance-020 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.7240 | 0.8813 | false | override first child expansion values with teacher child values |
| high_imbalance-020 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.8600 | 0.8942 | false | override first child expansion values with teacher child values |
| high_imbalance-020 | neural_child_value_swap | 384 | 4 | false | 0.0130 | 0.8776 | 0.1640 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-020 | neural_child_value_swap | 1200 | 4 | false | 0.0042 | 0.9533 | 0.1824 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-021 | original | 384 | 4 | false | 0.0026 | 0.7396 | 0.3726 | false | no intervention |
| high_imbalance-021 | original | 1200 | 4 | false | 0.0033 | 0.9108 | 0.1675 | false | no intervention |
| high_imbalance-021 | equalize_root_priors | 384 | 4 | false | 0.0573 | 0.6875 | 0.2168 | false | equalize corrected reference and selected root priors |
| high_imbalance-021 | equalize_root_priors | 1200 | 4 | false | 0.0233 | 0.8942 | 0.1749 | false | equalize corrected reference and selected root priors |
| high_imbalance-021 | uniform_legal_prior | 384 | 4 | false | 0.0417 | 0.6875 | 0.1878 | false | uniform legal prior positive control |
| high_imbalance-021 | uniform_legal_prior | 1200 | 4 | false | 0.0133 | 0.8942 | 0.1318 | false | uniform legal prior positive control |
| high_imbalance-021 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.7552 | 0.9244 | false | override first child expansion values with teacher child values |
| high_imbalance-021 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.8608 | 0.8652 | false | override first child expansion values with teacher child values |
| high_imbalance-021 | neural_child_value_swap | 384 | 4 | false | 0.0417 | 0.6823 | 0.1593 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-021 | neural_child_value_swap | 1200 | 4 | false | 0.0133 | 0.8950 | 0.1062 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-025 | original | 384 | 3 | false | 0.0026 | 0.7474 | 0.0143 | false | no intervention |
| high_imbalance-025 | original | 1200 | 4 | false | 0.0008 | 0.7233 | 0.1152 | false | no intervention |
| high_imbalance-025 | equalize_root_priors | 384 | 3 | false | 0.0391 | 0.8776 | 0.0391 | false | equalize corrected reference and selected root priors |
| high_imbalance-025 | equalize_root_priors | 1200 | 4 | false | 0.0125 | 0.6475 | 0.1205 | false | equalize corrected reference and selected root priors |
| high_imbalance-025 | uniform_legal_prior | 384 | 3 | false | 0.0365 | 0.8854 | 0.0258 | false | uniform legal prior positive control |
| high_imbalance-025 | uniform_legal_prior | 1200 | 4 | false | 0.0125 | 0.6542 | 0.1200 | false | uniform legal prior positive control |
| high_imbalance-025 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.6406 | 0.7646 | false | override first child expansion values with teacher child values |
| high_imbalance-025 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.7183 | 0.7308 | false | override first child expansion values with teacher child values |
| high_imbalance-025 | neural_child_value_swap | 384 | 3 | false | 0.0026 | 0.7240 | 0.0406 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-025 | neural_child_value_swap | 1200 | 4 | false | 0.0008 | 0.7292 | 0.1374 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-024 | original | 384 | 1 | false | 0.0495 | 0.8542 | 0.0332 | false | no intervention |
| high_imbalance-024 | original | 1200 | 1 | false | 0.3017 | 0.5758 | -0.0988 | false | no intervention |
| high_imbalance-024 | equalize_root_priors | 384 | 1 | false | 0.1146 | 0.8047 | 0.0420 | false | equalize corrected reference and selected root priors |
| high_imbalance-024 | equalize_root_priors | 1200 | 1 | false | 0.3975 | 0.5325 | -0.0691 | false | equalize corrected reference and selected root priors |
| high_imbalance-024 | uniform_legal_prior | 384 | 1 | false | 0.1146 | 0.7734 | 0.0405 | false | uniform legal prior positive control |
| high_imbalance-024 | uniform_legal_prior | 1200 | 1 | false | 0.4542 | 0.4992 | -0.0567 | false | uniform legal prior positive control |
| high_imbalance-024 | teacher_child_value_override | 384 | 1 | false | 0.0026 | 0.5807 | 0.7562 | false | override first child expansion values with teacher child values |
| high_imbalance-024 | teacher_child_value_override | 1200 | 1 | false | 0.0008 | 0.5700 | 0.7199 | false | override first child expansion values with teacher child values |
| high_imbalance-024 | neural_child_value_swap | 384 | 1 | false | 0.0026 | 0.9010 | 0.0300 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-024 | neural_child_value_swap | 1200 | 1 | false | 0.1025 | 0.7533 | -0.0301 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-016 | original | 384 | 4 | false | 0.0078 | 0.8984 | 0.1390 | false | no intervention |
| high_imbalance-016 | original | 1200 | 4 | false | 0.0025 | 0.9483 | 0.0963 | false | no intervention |
| high_imbalance-016 | equalize_root_priors | 384 | 4 | false | 0.0130 | 0.9010 | 0.1719 | false | equalize corrected reference and selected root priors |
| high_imbalance-016 | equalize_root_priors | 1200 | 4 | false | 0.0092 | 0.9450 | 0.1473 | false | equalize corrected reference and selected root priors |
| high_imbalance-016 | uniform_legal_prior | 384 | 4 | false | 0.0130 | 0.8854 | 0.1724 | false | uniform legal prior positive control |
| high_imbalance-016 | uniform_legal_prior | 1200 | 4 | false | 0.0058 | 0.9458 | 0.1422 | false | uniform legal prior positive control |
| high_imbalance-016 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.7865 | 0.8687 | false | override first child expansion values with teacher child values |
| high_imbalance-016 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.8108 | 0.8215 | false | override first child expansion values with teacher child values |
| high_imbalance-016 | neural_child_value_swap | 384 | 4 | false | 0.0130 | 0.7760 | 0.1261 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-016 | neural_child_value_swap | 1200 | 4 | false | 0.0042 | 0.9217 | 0.0786 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-027 | original | 384 | 4 | false | 0.0078 | 0.8672 | 0.0652 | false | no intervention |
| high_imbalance-027 | original | 1200 | 4 | false | 0.0033 | 0.9483 | 0.0956 | false | no intervention |
| high_imbalance-027 | equalize_root_priors | 384 | 4 | false | 0.0208 | 0.8698 | 0.1281 | false | equalize corrected reference and selected root priors |
| high_imbalance-027 | equalize_root_priors | 1200 | 4 | false | 0.0108 | 0.9450 | 0.1450 | false | equalize corrected reference and selected root priors |
| high_imbalance-027 | uniform_legal_prior | 384 | 4 | false | 0.0130 | 0.8750 | 0.1382 | false | uniform legal prior positive control |
| high_imbalance-027 | uniform_legal_prior | 1200 | 4 | false | 0.0092 | 0.9425 | 0.1273 | false | uniform legal prior positive control |
| high_imbalance-027 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.7370 | 0.9104 | false | override first child expansion values with teacher child values |
| high_imbalance-027 | teacher_child_value_override | 1200 | 4 | false | 0.0008 | 0.8542 | 0.9003 | false | override first child expansion values with teacher child values |
| high_imbalance-027 | neural_child_value_swap | 384 | 4 | false | 0.0104 | 0.8646 | 0.0692 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-027 | neural_child_value_swap | 1200 | 4 | false | 0.0033 | 0.9483 | 0.0537 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-019 | original | 384 | 3 | true | 0.5573 | 0.5573 | 0.0000 | true | no intervention |
| high_imbalance-019 | original | 1200 | 5 | false | 0.1783 | 0.7775 | 0.0930 | false | no intervention |
| high_imbalance-019 | equalize_root_priors | 384 | 5 | false | 0.3932 | 0.5651 | 0.0428 | false | equalize corrected reference and selected root priors |
| high_imbalance-019 | equalize_root_priors | 1200 | 5 | false | 0.1258 | 0.8525 | 0.0841 | false | equalize corrected reference and selected root priors |
| high_imbalance-019 | uniform_legal_prior | 384 | 0 | false | 0.0573 | 0.8516 | 0.0801 | false | uniform legal prior positive control |
| high_imbalance-019 | uniform_legal_prior | 1200 | 0 | false | 0.0900 | 0.4633 | 0.0017 | false | uniform legal prior positive control |
| high_imbalance-019 | teacher_child_value_override | 384 | 0 | false | 0.1901 | 0.4531 | 0.1057 | false | override first child expansion values with teacher child values |
| high_imbalance-019 | teacher_child_value_override | 1200 | 0 | false | 0.1775 | 0.4742 | 0.0085 | false | override first child expansion values with teacher child values |
| high_imbalance-019 | neural_child_value_swap | 384 | 3 | true | 0.7995 | 0.7995 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| high_imbalance-019 | neural_child_value_swap | 1200 | 1 | false | 0.2558 | 0.4100 | 0.0436 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-023 | original | 384 | 3 | false | 0.0208 | 0.8802 | 0.1424 | false | no intervention |
| high_imbalance-023 | original | 1200 | 3 | false | 0.0125 | 0.9558 | 0.0870 | false | no intervention |
| high_imbalance-023 | equalize_root_priors | 384 | 3 | false | 0.0208 | 0.8828 | 0.1431 | false | equalize corrected reference and selected root priors |
| high_imbalance-023 | equalize_root_priors | 1200 | 3 | false | 0.0125 | 0.9558 | 0.0870 | false | equalize corrected reference and selected root priors |
| high_imbalance-023 | uniform_legal_prior | 384 | 3 | false | 0.0130 | 0.6979 | 0.0965 | false | uniform legal prior positive control |
| high_imbalance-023 | uniform_legal_prior | 1200 | 3 | false | 0.0067 | 0.8858 | 0.1474 | false | uniform legal prior positive control |
| high_imbalance-023 | teacher_child_value_override | 384 | 3 | false | 0.0208 | 0.5000 | 0.2138 | false | override first child expansion values with teacher child values |
| high_imbalance-023 | teacher_child_value_override | 1200 | 3 | false | 0.0125 | 0.8342 | 0.1410 | false | override first child expansion values with teacher child values |
| high_imbalance-023 | neural_child_value_swap | 384 | 3 | false | 0.0208 | 0.8698 | 0.1409 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-023 | neural_child_value_swap | 1200 | 3 | false | 0.0125 | 0.9525 | 0.0864 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-026 | original | 384 | 4 | false | 0.0234 | 0.9167 | 0.1064 | false | no intervention |
| high_imbalance-026 | original | 1200 | 4 | false | 0.0125 | 0.9600 | 0.0532 | false | no intervention |
| high_imbalance-026 | equalize_root_priors | 384 | 4 | false | 0.0234 | 0.9167 | 0.1064 | false | equalize corrected reference and selected root priors |
| high_imbalance-026 | equalize_root_priors | 1200 | 4 | false | 0.0125 | 0.9600 | 0.0532 | false | equalize corrected reference and selected root priors |
| high_imbalance-026 | uniform_legal_prior | 384 | 4 | false | 0.0208 | 0.9349 | 0.0894 | false | uniform legal prior positive control |
| high_imbalance-026 | uniform_legal_prior | 1200 | 4 | false | 0.0258 | 0.9258 | 0.0442 | false | uniform legal prior positive control |
| high_imbalance-026 | teacher_child_value_override | 384 | 4 | false | 0.0104 | 0.8984 | 0.2407 | false | override first child expansion values with teacher child values |
| high_imbalance-026 | teacher_child_value_override | 1200 | 4 | false | 0.0058 | 0.9633 | 0.1262 | false | override first child expansion values with teacher child values |
| high_imbalance-026 | neural_child_value_swap | 384 | 4 | false | 0.0234 | 0.9167 | 0.0855 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-026 | neural_child_value_swap | 1200 | 4 | false | 0.0200 | 0.9525 | 0.0315 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-014 | original | 384 | 4 | false | 0.0026 | 0.8984 | 0.1319 | false | no intervention |
| high_imbalance-014 | original | 1200 | 4 | false | 0.0100 | 0.9300 | 0.0458 | false | no intervention |
| high_imbalance-014 | equalize_root_priors | 384 | 4 | false | 0.0443 | 0.9115 | 0.0574 | false | equalize corrected reference and selected root priors |
| high_imbalance-014 | equalize_root_priors | 1200 | 4 | false | 0.0142 | 0.9650 | 0.0570 | false | equalize corrected reference and selected root priors |
| high_imbalance-014 | uniform_legal_prior | 384 | 4 | false | 0.0443 | 0.8125 | 0.0574 | false | uniform legal prior positive control |
| high_imbalance-014 | uniform_legal_prior | 1200 | 4 | false | 0.0150 | 0.9283 | 0.0652 | false | uniform legal prior positive control |
| high_imbalance-014 | teacher_child_value_override | 384 | 4 | false | 0.0026 | 0.7344 | 0.8102 | false | override first child expansion values with teacher child values |
| high_imbalance-014 | teacher_child_value_override | 1200 | 4 | false | 0.0017 | 0.8492 | 0.2767 | false | override first child expansion values with teacher child values |
| high_imbalance-014 | neural_child_value_swap | 384 | 4 | false | 0.0443 | 0.8932 | 0.0482 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-014 | neural_child_value_swap | 1200 | 4 | false | 0.0142 | 0.9500 | 0.0476 | false | swap child neural values as diagnostic sanity check |
| high_imbalance-022 | original | 384 | 5 | false | 0.0156 | 0.8906 | 0.0418 | false | no intervention |
| high_imbalance-022 | original | 1200 | 5 | false | 0.0058 | 0.8583 | 0.0418 | false | no intervention |
| high_imbalance-022 | equalize_root_priors | 384 | 5 | false | 0.0260 | 0.8984 | 0.0327 | false | equalize corrected reference and selected root priors |
| high_imbalance-022 | equalize_root_priors | 1200 | 2 | true | 0.5083 | 0.5083 | 0.0000 | true | equalize corrected reference and selected root priors |
| high_imbalance-022 | uniform_legal_prior | 384 | 5 | false | 0.0156 | 0.8672 | 0.0449 | false | uniform legal prior positive control |
| high_imbalance-022 | uniform_legal_prior | 1200 | 5 | false | 0.0058 | 0.7850 | 0.0431 | false | uniform legal prior positive control |
| high_imbalance-022 | teacher_child_value_override | 384 | 5 | false | 0.0078 | 0.8698 | 0.2193 | false | override first child expansion values with teacher child values |
| high_imbalance-022 | teacher_child_value_override | 1200 | 5 | false | 0.0050 | 0.6842 | 0.0991 | false | override first child expansion values with teacher child values |
| high_imbalance-022 | neural_child_value_swap | 384 | 2 | true | 0.8984 | 0.8984 | 0.0000 | true | swap child neural values as diagnostic sanity check |
| high_imbalance-022 | neural_child_value_swap | 1200 | 2 | true | 0.9558 | 0.9558 | 0.0000 | true | swap child neural values as diagnostic sanity check |

## 10. Row classifications

| row_id | role | row_classification | supporting_evidence | recommended_use | notes |
| --- | --- | --- | --- | --- | --- |
| high_imbalance-009 | target_candidate | value_head_miscalibration | ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child | target_candidate | root still fails |
| high_imbalance-018 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_imbalance-020 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_imbalance-021 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_imbalance-025 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_imbalance-024 | target_candidate | root_selection_pressure | neural, ClassicMCTS child teacher, and child PUCT support corrected reference child but root PUCT still selects another move | target_candidate | root still fails |
| high_imbalance-016 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_imbalance-027 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_imbalance-019 | target_candidate | value_head_miscalibration | ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child | target_candidate | root still fails |
| high_imbalance-023 | target_candidate | value_head_miscalibration | ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child | target_candidate | root still fails |
| high_imbalance-026 | target_candidate | corrected_reference_suspicious | ClassicMCTS child teacher does not support the corrected reference child | target_candidate | root still fails |
| high_imbalance-014 | target_candidate | value_head_miscalibration | ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child | target_candidate | root still fails |
| high_imbalance-022 | target_candidate | value_head_miscalibration | ClassicMCTS child teacher prefers corrected reference child but raw neural child values prefer selected child | target_candidate | root still fails |
| high_imbalance-008 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| high_imbalance-002 | holdout_candidate | inconclusive | holdout row reserved for out-of-sample follow-up rather than mechanism counting | holdout | root still fails |
| high_imbalance-007 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| high_imbalance-012 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| high_imbalance-001 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |
| high_imbalance-017 | preservation_control | inconclusive | preservation control retained to guard against regression | preserve_control | baseline pass |

## 11. Family-level interpretation

| family_classification | value_head_miscalibration_count | puct_child_search_mismatch_count | root_selection_pressure_count | corrected_reference_suspicious_count | inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- |
| reference_family_uncertain | 5 | 0 | 1 | 7 | 0 | adjudicate high_imbalance references before training. |

## 12. Exactly one recommended next action

Recommendation: **adjudicate high_imbalance references before training.**
