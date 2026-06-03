# AlphaZero-lite Corrected Non-Opening Failure Family Mining v2 Results

## 1. Context

- This v2 mining pass selects the next corrected non-opening failure family without training, arena, promotion, or replay-artifact creation.
- Corrected references default to `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact evaluated: `storage/ai/alphazero_lite/current`.

## 2. Why Start-State Fairness Did Not Advance

- PR #50 left default standard Kalah(6,4) self-play unchanged and treated new start modes as diagnostic-only curriculum variants.
- The fairness scan did not find genuinely near-zero symmetric total-24 starts; the best observed absolute first-player margin stayed materially non-zero.
- Tiny traces improved some local disagreement metrics but worsened standard-start calibration, so no production-scale candidate advanced.
- The fairness branch is therefore closed as `start_bias_not_primary_bottleneck` for immediate next-family selection.

## 3. Why Opening and Incumbent_proxy Branches Are Excluded

- Opening replay is closed after the prior guard-safe lane failed to produce strength gains; opening rows remain context only.
- Corrected guard confirmation rows remain validation context, not training-family targets.
- `incumbent_proxy_disagreement` remains diagnostic-only after cross-teacher interference and low-LR/soft-policy follow-ups failed the strict non-regression gate.

## 4. Corrected Failure Inventory Source

- Inventory path: `/tmp/azlite_forensic_reference_rebaseline/corrected_failure_inventory.json`.
- Inventory mode: `loaded_existing`.
- Inventory freshness reason: `fresh`.
- Effective corrected reference path: `/tmp/azlite_forensic_reference_rebaseline/incumbent_forensic_references_v1_rebased.json`.
- Forensic validation path: `/tmp/azlite_forensic_reference_rebaseline/forensic_suite_validation.json`.

## 5. Exclusion Summary

| exclusion_group | row_count | reason | notes |
| --- | ---: | --- | --- |
| opening branch | 48 | opening replay branch is closed | includes opening_plies_1_8 and named opening subfamilies |
| corrected guard rows | 4 | corrected guard confirmations stay context-only | exclude capture_available 002/003/006/007/008 |
| incumbent_proxy branch | 32 | incumbent_proxy branch is diagnostic-only/exhausted here | do not continue cross-teacher or soft-policy follow-ups in this task |

## 6. Remaining Family Ranking

| family | rows | failures | failure_rate | persistent_1200_failures | recovered_at_1200 | high_severity | medium_severity | avg_reference_visit_share_1200 | avg_selected_minus_reference_q_margin_1200 | dominant_failure_mode | classification | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| high_value_swing | 24 | 17 | 0.708 | 16 | 1 | 0 | 17 | 0.3097 | 0.1867 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| high_imbalance | 24 | 16 | 0.667 | 15 | 1 | 0 | 16 | 0.2873 | 0.0975 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| capture_available | 20 | 14 | 0.700 | 14 | 0 | 14 | 0 | 0.2082 | 0.0651 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| starvation_pressure | 24 | 11 | 0.458 | 9 | 1 | 0 | 11 | 0.2910 | 0.0998 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| sparse_endgame | 24 | 10 | 0.417 | 7 | 1 | 0 | 10 | 0.5886 | 0.1536 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| early_extra_turn | 24 | 8 | 0.333 | 7 | 0 | 0 | 8 | 0.6359 | 0.1097 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |

## 7. Representative Row Diagnostics

| family | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | selected_move_2400 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | failure_mode | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| high_value_swing | high_value_swing-024 | 1 | 2 | 2 | 2 | 0.0130 | 0.0100 | 0.3172 | 0.3676 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-007 | 0 | 2 | 2 | 2 | 0.0052 | 0.0050 | 0.2912 | 0.3069 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-025 | 0 | 2 | 2 | 2 | 0.0156 | 0.0108 | 0.3128 | 0.2847 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-023 | 0 | 1 | 1 | 1 | 0.0286 | 0.0192 | 0.2801 | 0.2746 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-001 | 3 | 2 | 2 | 2 | 0.0026 | 0.0008 | 0.2535 | 0.2667 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-021 | 0 | 1 | 1 | 1 | 0.0052 | 0.0050 | 0.2619 | 0.2514 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-013 | 0 | 2 | 2 | 2 | 0.0078 | 0.0033 | 0.1818 | 0.1948 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-018 | 0 | 1 | 1 | 1 | 0.0182 | 0.0108 | 0.2257 | 0.1934 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-008 | 0 | 1 | 1 | 1 | 0.0260 | 0.0158 | 0.1735 | 0.1920 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-003 | 0 | 2 | 2 | 2 | 0.0104 | 0.0125 | 0.2183 | 0.1685 | value_q | persists_at_1200,persists_at_2400 |
| high_value_swing | high_value_swing-010 | 1 | 1 | 1 | 1 | 0.9609 | 0.9817 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| high_value_swing | high_value_swing-026 | 1 | 1 | 1 | 1 | 0.9531 | 0.9767 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| high_value_swing | high_value_swing-027 | 1 | 1 | 1 | 1 | 0.9375 | 0.9717 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| high_value_swing | high_value_swing-017 | 0 | 0 | 0 | 0 | 0.9453 | 0.9575 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| high_imbalance | high_imbalance-009 | 4 | 5 | 5 | - | 0.0130 | 0.0050 | 0.2244 | 0.2649 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-018 | 2 | 4 | 4 | - | 0.0026 | 0.0033 | 0.2976 | 0.2280 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-020 | 2 | 4 | 4 | - | 0.0078 | 0.0025 | 0.1910 | 0.2118 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-021 | 2 | 4 | 4 | - | 0.0026 | 0.0033 | 0.3726 | 0.1675 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-025 | 0 | 3 | 4 | - | 0.0026 | 0.0008 | 0.0143 | 0.1152 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-024 | 2 | 1 | 1 | - | 0.0495 | 0.3017 | 0.0333 | -0.0988 | search_selection | persists_at_1200 |
| high_imbalance | high_imbalance-016 | 2 | 4 | 4 | - | 0.0078 | 0.0025 | 0.1389 | 0.0963 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-027 | 2 | 4 | 4 | - | 0.0078 | 0.0033 | 0.0652 | 0.0956 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-019 | 3 | 3 | 5 | - | 0.5573 | 0.1783 | 0.0000 | 0.0930 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-023 | 2 | 3 | 3 | - | 0.0208 | 0.0125 | 0.1424 | 0.0870 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-007 | 1 | 1 | 1 | - | 0.9870 | 0.9925 | 0.0000 | 0.0000 | pass | passing_control |
| high_imbalance | high_imbalance-012 | 1 | 1 | 1 | - | 0.9609 | 0.9267 | 0.0000 | 0.0000 | pass | passing_control |
| high_imbalance | high_imbalance-001 | 3 | 3 | 3 | - | 0.5990 | 0.8717 | 0.0000 | 0.0000 | pass | passing_control |
| high_imbalance | high_imbalance-017 | 1 | 1 | 1 | - | 0.8542 | 0.8625 | 0.0000 | 0.0000 | pass | passing_control |
| capture_available | capture_available-024 | 3 | 1 | 1 | - | 0.1224 | 0.0392 | 0.1330 | 0.2069 | value_q | persists_at_1200 |
| capture_available | capture_available-017 | 3 | 4 | 2 | - | 0.0026 | 0.0008 | 0.0952 | 0.1542 | value_q | persists_at_1200 |
| capture_available | capture_available-019 | 3 | 0 | 4 | - | 0.0026 | 0.0008 | 0.1117 | 0.1149 | value_q | persists_at_1200 |
| capture_available | capture_available-018 | 3 | 2 | 0 | - | 0.0026 | 0.0008 | 0.1161 | 0.0917 | value_q | persists_at_1200 |
| capture_available | capture_available-013 | 4 | 2 | 2 | - | 0.0078 | 0.0025 | 0.0769 | 0.0649 | value_q | persists_at_1200 |
| capture_available | capture_available-023 | 4 | 1 | 1 | - | 0.0260 | 0.0083 | 0.0362 | 0.0454 | value_q | persists_at_1200 |
| capture_available | capture_available-022 | 3 | 1 | 1 | - | 0.0391 | 0.0125 | 0.0196 | 0.0453 | value_q | persists_at_1200 |
| capture_available | capture_available-016 | 4 | 0 | 0 | - | 0.0026 | 0.0017 | 0.0705 | 0.0448 | value_q | persists_at_1200 |
| capture_available | capture_available-025 | 4 | 0 | 0 | - | 0.2786 | 0.0892 | 0.0113 | 0.0440 | value_q | persists_at_1200 |
| capture_available | capture_available-009 | 4 | 1 | 1 | - | 0.0156 | 0.0050 | 0.0294 | 0.0355 | value_q | persists_at_1200 |
| capture_available | capture_available-021 | 1 | 1 | 1 | - | 0.9479 | 0.9817 | 0.0000 | 0.0000 | pass | passing_control |
| capture_available | capture_available-010 | 2 | 2 | 2 | - | 0.9740 | 0.9767 | 0.0000 | 0.0000 | pass | passing_control |
| capture_available | capture_available-020 | 3 | 3 | 3 | - | 0.7448 | 0.9142 | 0.0000 | 0.0000 | pass | passing_control |
| capture_available | capture_available-011 | 2 | 2 | 2 | - | 0.8099 | 0.5225 | 0.0000 | 0.0000 | pass | passing_control |
| starvation_pressure | starvation_pressure-025 | 0 | 4 | 4 | - | 0.0026 | 0.0008 | 0.2204 | 0.2063 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-026 | 0 | 4 | 4 | - | 0.0156 | 0.0050 | 0.2148 | 0.1868 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-024 | 1 | 3 | 3 | - | 0.0234 | 0.0075 | 0.1357 | 0.1654 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-022 | 2 | 1 | 1 | - | 0.0182 | 0.0067 | 0.1113 | 0.1118 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-012 | 3 | 5 | 5 | - | 0.0130 | 0.0083 | 0.0752 | 0.0986 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-023 | 2 | 1 | 1 | - | 0.0417 | 0.0225 | 0.0739 | 0.0653 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-027 | 0 | 1 | 1 | - | 0.0026 | 0.0033 | 0.0493 | 0.0417 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-015 | 4 | 2 | 2 | - | 0.1979 | 0.4800 | -0.0627 | 0.0196 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-001 | 1 | 4 | 4 | - | 0.0156 | 0.3558 | 0.1275 | 0.0029 | value_q | persists_at_1200 |
| starvation_pressure | starvation_pressure-010 | 4 | 1 | 4 | - | 0.1693 | 0.7225 | -0.0185 | 0.0000 | pass | 384_only_recovers_at_1200 |
| starvation_pressure | starvation_pressure-013 | 4 | 4 | 4 | - | 0.9974 | 0.9992 | 0.0000 | 0.0000 | pass | passing_control |
| starvation_pressure | starvation_pressure-003 | 1 | 1 | 1 | - | 0.8958 | 0.9642 | 0.0000 | 0.0000 | pass | passing_control |
| starvation_pressure | starvation_pressure-021 | 0 | 0 | 0 | - | 0.8984 | 0.9608 | 0.0000 | 0.0000 | pass | passing_control |
| starvation_pressure | starvation_pressure-014 | 2 | 2 | 2 | - | 0.9844 | 0.9467 | 0.0000 | 0.0000 | pass | passing_control |
| sparse_endgame | sparse_endgame-003 | 0 | 1 | 1 | - | 0.0000 | 0.0008 | 0.9990 | 0.3782 | value_q | persists_at_1200 |
| sparse_endgame | sparse_endgame-007 | 0 | 3 | 3 | - | 0.0000 | 0.0008 | 0.7795 | 0.1722 | value_q | persists_at_1200 |
| sparse_endgame | sparse_endgame-009 | 5 | 2 | 2 | - | 0.0339 | 0.0325 | 0.2399 | 0.1343 | value_q | persists_at_1200 |
| sparse_endgame | sparse_endgame-024 | 1 | 5 | 4 | - | 0.1510 | 0.0800 | 0.0277 | 0.1273 | value_q | persists_at_1200 |
| sparse_endgame | sparse_endgame-026 | 4 | 5 | 5 | - | 0.0026 | 0.0008 | 0.2957 | 0.1187 | value_q | persists_at_1200 |
| sparse_endgame | sparse_endgame-025 | 5 | 1 | 1 | - | 0.0052 | 0.0158 | 0.1876 | 0.0865 | value_q | persists_at_1200 |
| sparse_endgame | sparse_endgame-023 | 4 | 5 | 5 | - | 0.0000 | 0.0008 | -0.0891 | 0.0583 | value_q | persists_at_1200 |
| sparse_endgame | sparse_endgame-021 | 0 | 0 | 0 | - | 0.4922 | 0.5842 | 0.0000 | 0.0000 | pass | 384_only_recovers_at_1200 |
| sparse_endgame | sparse_endgame-013 | 5 | 4 | 5 | - | 0.3411 | 0.7650 | 0.0036 | 0.0000 | pass | 384_only_recovers_at_1200 |
| sparse_endgame | sparse_endgame-017 | 4 | 4 | 4 | - | 0.8698 | 0.9575 | 0.0000 | 0.0000 | pass | 384_only_recovers_at_1200 |
| sparse_endgame | sparse_endgame-002 | 3 | 3 | 3 | - | 1.0000 | 1.0000 | 0.0000 | 0.0000 | pass | passing_control |
| sparse_endgame | sparse_endgame-019 | 5 | 5 | 5 | - | 0.9792 | 0.9875 | 0.0000 | 0.0000 | pass | passing_control |
| sparse_endgame | sparse_endgame-020 | 2 | 2 | 2 | - | 0.9505 | 0.9842 | 0.0000 | 0.0000 | pass | passing_control |
| sparse_endgame | sparse_endgame-011 | 5 | 5 | 5 | - | 0.9688 | 0.9833 | 0.0000 | 0.0000 | pass | passing_control |

## 8. Targetability Classification

- `high_value_swing`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `high_imbalance`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `capture_available`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `starvation_pressure`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `sparse_endgame`: `value_or_backup_issue`. persistent failures remain Q/value-dominant

## 9. Selected Next Family

| selected_family | target_rows | control_rows | holdout_rows | reason_selected | risks | next_action |
| --- | ---: | ---: | ---: | --- | --- | --- |
| high_value_swing | 14 | 4 | 2 | highest-ranked remaining corrected non-opening family after exclusions; persistent failures remain Q/value-dominant | dominant mechanism: value_q | run child-afterstate value/backup audit for `high_value_swing` before training. |

## 10. Exactly One Recommended Next Action

Recommendation: **run child-afterstate value/backup audit for `high_value_swing` before training.**

Run classification: `next_family_value_backup_audit_ready`.
