# AlphaZero-lite Corrected Non-Opening Failure Family Mining v3 Results

## 1. Context

- This v3 mining pass selects the next corrected non-opening failure family without training, arena, promotion, or replay-artifact creation.
- Corrected references default to `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact evaluated: `storage/ai/alphazero_lite/current`.
- No training is run.
- No arena is run.
- No model is promoted.
- No replay artifacts are created.
- Active corrected reference fixtures are not mutated.

## 2. Why high_value_swing is excluded after PR #53

- PR #53 adjudicated the `high_value_swing` reference suite without mutating the active fixture.
- Projected family decision: `reference_suite_too_noisy_for_high_value_swing`.
- Of the 14 target-candidate rows that PR #51 had selected, the PR #53 adjudication classified every one as either `reference_unstable` or `still_inconclusive`.
- Target-candidate rows still eligible under active references: `0`.
- Target-candidate rows requiring reference patching: `0`.
- Target-candidate rows that should stay excluded from training: `14`.
- The PR #53 recommendation was to exclude unstable rows and either target a smaller stable bucket or select the next non-opening family; since no eligible target rows remain, v3 selects the next non-opening family.
- `high_value_swing` is therefore excluded from this run's target-family selection, but every row is kept in the exclusion summary for context.

## 3. Other excluded branches

- Opening replay is closed after the prior guard-safe lane failed to produce strength gains; opening rows remain context only.
- Corrected guard confirmation rows (`capture_available-002/003/006/007/008`) remain validation context, not training-family targets.
- `incumbent_proxy_disagreement` remains diagnostic-only after cross-teacher interference and low-LR/soft-policy follow-ups failed the strict non-regression gate.
- Reference-data exclusions apply to any remaining row that is reference-unstable, has a reference-integrity error, is train-only, has a legacy/stale reference source, or is explicitly do-not-train or excluded-diagnostic.

## 4. Corrected failure inventory source

- Inventory path: `/tmp/azlite_forensic_reference_rebaseline/corrected_failure_inventory.json`.
- Inventory mode: `loaded_existing`.
- Inventory freshness reason: `fresh`.
- Effective corrected reference path: `/tmp/azlite_forensic_reference_rebaseline/incumbent_forensic_references_v1_rebased.json`.
- Forensic validation path: `/tmp/azlite_forensic_reference_rebaseline/forensic_suite_validation.json`.

## 5. Exclusion summary

| exclusion_group | row_count | reason | notes |
| --- | ---: | --- | --- |
| opening branch | 48 | opening replay branch is closed | includes opening_plies_1_8 and named opening subfamilies |
| corrected guard rows | 4 | corrected guard confirmations stay context-only | exclude capture_available 002/003/006/007/008 |
| high_value_swing branch | 24 | high_value_swing reference suite too noisy after PR #53 adjudication | every target-candidate row was classified reference_unstable or still_inconclusive; no eligible training-target rows remain |
| incumbent_proxy branch | 32 | incumbent_proxy branch is diagnostic-only/exhausted here | do not continue cross-teacher or soft-policy follow-ups in this task |

## 6. Remaining family ranking

| family | rows | failures | failure_rate | persistent_1200_failures | recovered_at_1200 | high_severity | medium_severity | avg_reference_visit_share_1200 | avg_selected_minus_reference_q_margin_1200 | dominant_failure_mode | classification | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| high_imbalance | 24 | 16 | 0.667 | 15 | 1 | 0 | 16 | 0.2873 | 0.0975 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| capture_available | 20 | 14 | 0.700 | 14 | 0 | 14 | 0 | 0.2082 | 0.0651 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| starvation_pressure | 24 | 11 | 0.458 | 9 | 1 | 0 | 11 | 0.2910 | 0.0998 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| sparse_endgame | 24 | 10 | 0.417 | 7 | 1 | 0 | 10 | 0.5886 | 0.1536 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |
| early_extra_turn | 24 | 8 | 0.333 | 7 | 0 | 0 | 8 | 0.6359 | 0.1097 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |

## 7. Representative row diagnostics

| family | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | selected_move_2400 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | failure_mode | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| high_imbalance | high_imbalance-009 | 4 | 5 | 5 | 5 | 0.0130 | 0.0050 | 0.2244 | 0.2649 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-018 | 2 | 4 | 4 | 4 | 0.0026 | 0.0033 | 0.2976 | 0.2280 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-020 | 2 | 4 | 4 | 4 | 0.0078 | 0.0025 | 0.1910 | 0.2118 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-021 | 2 | 4 | 4 | 4 | 0.0026 | 0.0033 | 0.3726 | 0.1675 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-025 | 0 | 3 | 4 | 4 | 0.0026 | 0.0008 | 0.0143 | 0.1152 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-024 | 2 | 1 | 1 | 2 | 0.0495 | 0.3017 | 0.0333 | -0.0988 | search_selection | persists_at_1200,recovers_at_2400 |
| high_imbalance | high_imbalance-016 | 2 | 4 | 4 | 4 | 0.0078 | 0.0025 | 0.1389 | 0.0963 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-027 | 2 | 4 | 4 | 4 | 0.0078 | 0.0033 | 0.0652 | 0.0956 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-019 | 3 | 3 | 5 | 5 | 0.5573 | 0.1783 | 0.0000 | 0.0930 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-023 | 2 | 3 | 3 | 3 | 0.0208 | 0.0125 | 0.1424 | 0.0870 | value_q | persists_at_1200,persists_at_2400 |
| high_imbalance | high_imbalance-007 | 1 | 1 | 1 | 1 | 0.9870 | 0.9925 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| high_imbalance | high_imbalance-012 | 1 | 1 | 1 | 1 | 0.9609 | 0.9267 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| high_imbalance | high_imbalance-001 | 3 | 3 | 3 | 3 | 0.5990 | 0.8717 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| high_imbalance | high_imbalance-017 | 1 | 1 | 1 | 1 | 0.8542 | 0.8625 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
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
| early_extra_turn | early_extra_turn-025 | 0 | 4 | 5 | - | 0.0026 | 0.0008 | 0.2951 | 0.2881 | value_q | persists_at_1200 |
| early_extra_turn | early_extra_turn-016 | 2 | 5 | 5 | - | 0.0234 | 0.0108 | 0.0947 | 0.1158 | value_q | persists_at_1200 |
| early_extra_turn | early_extra_turn-012 | 3 | 5 | 5 | - | 0.0026 | 0.0008 | 0.1182 | 0.0955 | value_q | persists_at_1200 |
| early_extra_turn | early_extra_turn-013 | 3 | 1 | 5 | - | 0.0104 | 0.0033 | 0.0265 | 0.0914 | value_q | persists_at_1200 |
| early_extra_turn | early_extra_turn-014 | 1 | 5 | 5 | - | 0.0885 | 0.0333 | 0.0113 | 0.0732 | value_q | persists_at_1200 |
| early_extra_turn | early_extra_turn-018 | 2 | 5 | 5 | - | 0.0078 | 0.0108 | 0.1229 | 0.0726 | value_q | persists_at_1200 |
| early_extra_turn | early_extra_turn-017 | 2 | 5 | 5 | - | 0.0521 | 0.0208 | 0.0036 | 0.0311 | value_q | persists_at_1200 |
| early_extra_turn | early_extra_turn-015 | 2 | 2 | 2 | - | 0.7422 | 0.8975 | 0.0000 | 0.0000 | pass | 384_only_recovers_at_1200 |
| early_extra_turn | early_extra_turn-007 | 5 | 5 | 5 | - | 0.9974 | 0.9975 | 0.0000 | 0.0000 | pass | passing_control |
| early_extra_turn | early_extra_turn-011 | 5 | 5 | 5 | - | 0.9505 | 0.9817 | 0.0000 | 0.0000 | pass | passing_control |
| early_extra_turn | early_extra_turn-002 | 4 | 4 | 4 | - | 0.9375 | 0.9767 | 0.0000 | 0.0000 | pass | passing_control |
| early_extra_turn | early_extra_turn-009 | 4 | 4 | 4 | - | 0.9297 | 0.9717 | 0.0000 | 0.0000 | pass | passing_control |

## 8. Targetability classification

- `high_imbalance`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `capture_available`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `starvation_pressure`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `sparse_endgame`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `early_extra_turn`: `value_or_backup_issue`. persistent failures remain Q/value-dominant

## 9. Selected next family

| selected_family | target_rows | control_rows | holdout_rows | reason_selected | risks | next_action |
| --- | ---: | ---: | ---: | --- | --- | --- |
| high_imbalance | 13 | 4 | 2 | highest-ranked remaining corrected non-opening family after exclusions (including high_value_swing after PR #53); persistent failures remain Q/value-dominant | dominant mechanism: value_q | run child-afterstate value/backup audit for `high_imbalance` before training. |

## 10. Exactly one recommended next action

Recommendation: **run child-afterstate value/backup audit for `high_imbalance` before training.**

Run classification: `next_family_value_backup_audit_ready`.
