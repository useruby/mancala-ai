# AlphaZero-lite Corrected Non-Opening Failure Family Mining v6 Results

## 1. Context

- This v6 mining pass selects the next corrected non-opening failure family after PR #62 without training, arena, promotion, replay-artifact creation, or reference mutation.
- Corrected references default to `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact evaluated: `storage/ai/alphazero_lite/current`.
- No training is run.
- No arena is run.
- No model is promoted.
- No replay artifacts are created.
- Active corrected reference fixtures are not mutated.

## 2. Why starvation_pressure is excluded after PR #62

- PR #62 adjudicated `starvation_pressure` without training, arena, promotion, replay/value-calibration artifacts, or corrected-reference mutation.
- Projected family decision: `reference_suite_too_noisy_for_starvation_pressure`.
- Clean-enough rows: `0`.
- Reference-should-flip rows: `0`.
- Reference-unstable rows: `starvation_pressure-025`, `starvation_pressure-027`.
- Still-inconclusive rows: `starvation_pressure-012`, `starvation_pressure-022`, `starvation_pressure-023`, `starvation_pressure-024`, `starvation_pressure-026`.
- Preservation controls confirmed: `starvation_pressure-003`, `starvation_pressure-013`, `starvation_pressure-014`, `starvation_pressure-021`.
- Recommendation from PR #62: exclude unstable rows and either target the smaller stable bucket or select the next non-opening family.
- Because `starvation_pressure` has no clean trainable target bucket, v6 excludes it from selection and recomputes the next family from the remaining corrected non-opening inventory.

## 3. Other excluded branches

- Opening replay remains closed; opening rows are context only.
- Corrected guard confirmation rows (`capture_available-002/003/006/007/008`) remain validation context, not training-family targets.
- `capture_available` remains closed after PR #59 left no safe target rows.
- `high_imbalance` remains closed after PR #56 left no safe target rows.
- `high_value_swing` remains closed after PR #53 left no safe target rows.
- `incumbent_proxy_disagreement` remains diagnostic-only after prior follow-ups failed the strict non-regression path.
- Reference-data exclusions also remove rows that are unstable, have integrity errors, are train-only, come from stale/legacy sources, or are explicitly do-not-train / excluded-diagnostic.

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
| corrected guard rows | 4 | corrected guard confirmations stay context-only | exclude capture_available 002/003/007/008 in the active inventory; historical guard id 006 remains non-selectable if reintroduced |
| capture_available branch | 20 | capture_available was adjudicated as not safe for training targets after PR #59 | 0 eligible target rows remain under active corrected references; 3 rows would require reviewed fixture patching and 7 rows stay excluded |
| high_imbalance branch | 24 | high_imbalance reference suite is too noisy after PR #56 adjudication | 0 eligible target rows remain under active corrected references; keep all excluded from training |
| high_value_swing branch | 24 | high_value_swing reference suite too noisy after PR #53 adjudication | every target-candidate row was classified reference_unstable or still_inconclusive; no eligible training-target rows remain |
| starvation_pressure branch | 24 | starvation_pressure reference suite too noisy after PR #62 adjudication | 0 clean-enough target rows remain; 0 reference-should-flip rows; 2 reference-unstable rows (starvation_pressure-025, starvation_pressure-027) and 5 still-inconclusive rows stay excluded |
| sparse_endgame branch | 24 | sparse_endgame is dominated by forced/tied positions; 83% of rows have arbitrary policy labels | Excluded after exact-tablebase targetability split (PR #67). 16 of 24 rows are forced_all_moves_equivalent. Only 1 row (sparse_endgame-023) has unique optimal + PUCT-suboptimal signal, but family-level training is not justified. |
| incumbent_proxy branch | 32 | incumbent_proxy branch is diagnostic-only/exhausted here | do not continue cross-teacher or soft-policy follow-ups in this task |

## 6. Remaining family ranking

| family | rows | failures | failure_rate | persistent_1200_failures | recovered_at_1200 | high_severity | medium_severity | avg_reference_visit_share_1200 | avg_selected_minus_reference_q_margin_1200 | dominant_failure_mode | classification | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| early_extra_turn | 24 | 8 | 0.333 | 7 | 0 | 0 | 8 | 0.6359 | 0.1097 | value_q | value_or_backup_issue | persistent failures remain Q/value-dominant |

## 7. Representative row diagnostics

| family | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | selected_move_2400 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | failure_mode | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |
| early_extra_turn | early_extra_turn-025 | 0 | 4 | 5 | 4 | 0.0026 | 0.0008 | 0.2951 | 0.2881 | value_q | persists_at_1200,persists_at_2400 |
| early_extra_turn | early_extra_turn-016 | 2 | 5 | 5 | 5 | 0.0234 | 0.0108 | 0.0947 | 0.1158 | value_q | persists_at_1200,persists_at_2400 |
| early_extra_turn | early_extra_turn-012 | 3 | 5 | 5 | 5 | 0.0026 | 0.0008 | 0.1182 | 0.0955 | value_q | persists_at_1200,persists_at_2400 |
| early_extra_turn | early_extra_turn-013 | 3 | 1 | 5 | 5 | 0.0104 | 0.0033 | 0.0265 | 0.0914 | value_q | persists_at_1200,persists_at_2400 |
| early_extra_turn | early_extra_turn-014 | 1 | 5 | 5 | 5 | 0.0885 | 0.0333 | 0.0113 | 0.0732 | value_q | persists_at_1200,persists_at_2400 |
| early_extra_turn | early_extra_turn-018 | 2 | 5 | 5 | 5 | 0.0078 | 0.0108 | 0.1229 | 0.0726 | value_q | persists_at_1200,persists_at_2400 |
| early_extra_turn | early_extra_turn-017 | 2 | 5 | 5 | 5 | 0.0521 | 0.0208 | 0.0036 | 0.0311 | value_q | persists_at_1200,persists_at_2400 |
| early_extra_turn | early_extra_turn-015 | 2 | 2 | 2 | 2 | 0.7422 | 0.8975 | 0.0000 | 0.0000 | pass | 384_only_recovers_at_1200,recovers_at_2400 |
| early_extra_turn | early_extra_turn-007 | 5 | 5 | 5 | 5 | 0.9974 | 0.9975 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| early_extra_turn | early_extra_turn-011 | 5 | 5 | 5 | 5 | 0.9505 | 0.9817 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| early_extra_turn | early_extra_turn-002 | 4 | 4 | 4 | 4 | 0.9375 | 0.9767 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |
| early_extra_turn | early_extra_turn-009 | 4 | 4 | 4 | 4 | 0.9297 | 0.9717 | 0.0000 | 0.0000 | pass | passing_control,recovers_at_2400 |

## 8. Targetability classification

- `early_extra_turn`: `value_or_backup_issue`. persistent failures remain Q/value-dominant

## 9. Selected next family

| selected_family | target_rows | control_rows | holdout_rows | reason_selected | risks | next_action |
| --- | ---: | ---: | ---: | --- | --- | --- |
| early_extra_turn | 6 | 4 | 1 | highest-ranked remaining corrected non-opening family after excluding starvation_pressure after PR #62; persistent failures remain Q/value-dominant | dominant mechanism: value_q | run child-afterstate value/backup audit for `early_extra_turn` before training. |

## 10. Exactly one recommended next action

Recommendation: **run child-afterstate value/backup audit for `early_extra_turn` before training.**

Run classification: `next_family_value_backup_audit_ready`.
