# AlphaZero-lite Corrected Non-Opening Failure Family Mining Results

## 1. Context

- This run mines corrected non-opening failure families against `storage/ai/alphazero_lite/current`.
- Corrected references default to `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact evaluated: `storage/ai/alphazero_lite/current`.

## 2. Why Opening Replay Is Closed

- PR #41 closed the denoised opening replay lane as `guard_safe_no_strength_gain`.
- Guard-safe candidates lost all 60 arena games against the current artifact.
- Opening replay rows are therefore context only and are excluded from next-family selection.

## 3. Corrected Failure Inventory Source

- Inventory path: `/tmp/azlite_forensic_reference_rebaseline/corrected_failure_inventory.json`.
- Inventory mode: `loaded_existing`.
- Inventory freshness reason: `fresh`.
- Forensic validation path: `/tmp/azlite_forensic_reference_rebaseline/forensic_suite_validation.json`.
- Opening subfamily diagnostic path: `/tmp/azlite_forensic_reference_rebaseline/opening_plies_subfamily_diagnostic.json`.

## 4. Exclusions

- Excluded rows kept only as context: `52`.
- Closed opening rows: `48`.
- Guard blocker rows excluded from targeting: `4`.
- Reference instability or integrity exclusions: `0`.

## 5. Non-Opening Family Ranking

| family | rows | failures | failure_rate | high_severity | medium_severity | stable_reference_rows | avg_reference_visit_share | avg_selected_minus_reference_q_margin | classification | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| incumbent_proxy_disagreement | 32 | 23 | 0.719 | 0 | 23 | 32 | 0.3708 | 0.0660 | value_or_backup_issue | persistent failures remain Q/value-dominant |
| high_value_swing | 24 | 17 | 0.708 | 0 | 17 | 24 | 0.2698 | 0.1999 | value_or_backup_issue | persistent failures remain Q/value-dominant |
| high_imbalance | 24 | 16 | 0.667 | 0 | 16 | 24 | 0.2050 | 0.1077 | value_or_backup_issue | persistent failures remain Q/value-dominant |
| capture_available | 20 | 14 | 0.700 | 14 | 0 | 20 | 0.2473 | 0.0709 | too_sparse | not enough persistent repeated failures after sampling |
| starvation_pressure | 24 | 11 | 0.458 | 0 | 11 | 24 | 0.2281 | 0.0843 | too_sparse | not enough persistent repeated failures after sampling |
| sparse_endgame | 24 | 10 | 0.417 | 0 | 10 | 24 | 0.5712 | 0.2444 | too_sparse | not enough persistent repeated failures after sampling |
| early_extra_turn | 24 | 8 | 0.333 | 0 | 8 | 24 | 0.6000 | 0.0840 | too_sparse | not enough persistent repeated failures after sampling |

## 6. Representative Row Diagnostics

| family | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | failure_mode | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-009 | 0 | 3 | 3 | 0.0026 | 0.0008 | 0.2088 | 0.2139 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-021 | 2 | 5 | 5 | 0.0078 | 0.0042 | 0.1542 | 0.2335 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-035 | 3 | 4 | 4 | 0.0104 | 0.0058 | 0.1514 | 0.1402 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-022 | 3 | 5 | 5 | 0.0052 | 0.0033 | 0.1457 | 0.0540 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-024 | 2 | 1 | 1 | 0.0104 | 0.0067 | 0.1257 | 0.1008 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-011 | 4 | 5 | 5 | 0.0339 | 0.0475 | 0.1196 | 0.0714 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-014 | 4 | 5 | 5 | 0.0026 | 0.3392 | 0.1030 | -0.0033 | search_selection | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-007 | 0 | 5 | 3 | 0.0052 | 0.0025 | 0.1018 | 0.0733 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-025 | 2 | 4 | 4 | 0.0000 | 0.2650 | 0.1011 | -0.0592 | search_selection | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-023 | 2 | 1 | 1 | 0.0130 | 0.0050 | 0.0918 | 0.0965 | value_q | persists_at_1200 |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 0.9922 | 0.9650 | 0.0000 | 0.0000 | pass | passing_control |
| incumbent_proxy_disagreement | incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 0.9323 | 0.9250 | 0.0000 | 0.0000 | pass | passing_control |
| high_value_swing | high_value_swing-022 | 4 | 1 | 1 | 0.0026 | 0.0108 | 0.3248 | 0.0472 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-024 | 1 | 2 | 2 | 0.0130 | 0.0100 | 0.3172 | 0.3676 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-025 | 0 | 2 | 2 | 0.0156 | 0.0108 | 0.3128 | 0.2847 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-007 | 0 | 2 | 2 | 0.0052 | 0.0050 | 0.2912 | 0.3069 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-023 | 0 | 1 | 1 | 0.0286 | 0.0192 | 0.2801 | 0.2746 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-021 | 0 | 1 | 1 | 0.0052 | 0.0050 | 0.2619 | 0.2514 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-001 | 3 | 2 | 2 | 0.0026 | 0.0008 | 0.2535 | 0.2667 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-018 | 0 | 1 | 1 | 0.0182 | 0.0108 | 0.2257 | 0.1934 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-003 | 0 | 2 | 2 | 0.0104 | 0.0125 | 0.2183 | 0.1685 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-013 | 0 | 2 | 2 | 0.0078 | 0.0033 | 0.1818 | 0.1948 | value_q | persists_at_1200 |
| high_value_swing | high_value_swing-010 | 1 | 1 | 1 | 0.9609 | 0.9817 | 0.0000 | 0.0000 | pass | passing_control |
| high_value_swing | high_value_swing-026 | 1 | 1 | 1 | 0.9531 | 0.9767 | 0.0000 | 0.0000 | pass | passing_control |
| high_imbalance | high_imbalance-021 | 2 | 4 | 4 | 0.0026 | 0.0033 | 0.3726 | 0.1675 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-018 | 2 | 4 | 4 | 0.0026 | 0.0033 | 0.2976 | 0.2280 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-009 | 4 | 5 | 5 | 0.0130 | 0.0050 | 0.2244 | 0.2649 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-020 | 2 | 4 | 4 | 0.0078 | 0.0025 | 0.1910 | 0.2118 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-023 | 2 | 3 | 3 | 0.0208 | 0.0125 | 0.1424 | 0.0870 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-016 | 2 | 4 | 4 | 0.0078 | 0.0025 | 0.1389 | 0.0963 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-014 | 0 | 4 | 4 | 0.0026 | 0.0100 | 0.1319 | 0.0458 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-026 | 2 | 4 | 4 | 0.0234 | 0.0125 | 0.1064 | 0.0533 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-027 | 2 | 4 | 4 | 0.0078 | 0.0033 | 0.0652 | 0.0956 | value_q | persists_at_1200 |
| high_imbalance | high_imbalance-015 | 4 | 5 | 4 | 0.1094 | 0.7150 | -0.0635 | 0.0000 | search_selection | 384_only_recovers_at_1200 |
| high_imbalance | high_imbalance-007 | 1 | 1 | 1 | 0.9870 | 0.9925 | 0.0000 | 0.0000 | pass | passing_control |
| high_imbalance | high_imbalance-012 | 1 | 1 | 1 | 0.9609 | 0.9267 | 0.0000 | 0.0000 | pass | passing_control |

## 7. Targetability Classification

- `incumbent_proxy_disagreement`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `high_value_swing`: `value_or_backup_issue`. persistent failures remain Q/value-dominant
- `high_imbalance`: `value_or_backup_issue`. persistent failures remain Q/value-dominant

## 8. Selected Next Candidate Family

| selected_family | target_rows | control_rows | holdout_rows | reason_selected | risks | next_action |
| --- | ---: | ---: | ---: | --- | --- | --- |
| incumbent_proxy_disagreement | 21 | 4 | 2 | highest-ranked non-opening family; persistent failures remain Q/value-dominant | dominant mechanism: value_q | run a child-afterstate value/backup audit on `incumbent_proxy_disagreement`. |

## 9. Exactly One Recommended Next Action

Recommendation: **run a child-afterstate value/backup audit on `incumbent_proxy_disagreement`.**

Run classification: `value_backup_gap`.
