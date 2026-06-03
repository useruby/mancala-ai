# AlphaZero-lite Corrected Non-Opening Failure Family Mining v8 Results

## 1. Context

- This v8 mining pass selects the next corrected non-opening failure family after PR #67 without training, arena, promotion, replay-artifact creation, or reference mutation.
- Corrected references default to `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact evaluated: `storage/ai/alphazero_lite/current`.
- No training is run.
- No arena is run.
- No model is promoted.
- No replay artifacts are created.
- Active corrected reference fixtures are not mutated.

## 2. Why early_extra_turn is excluded after reference adjudication

- The early_extra_turn reference adjudication (PR follow-up) evaluated all 5 corrected_reference_suspicious rows.
- Projected family decision: `reference_suite_too_noisy_for_early_extra_turn`.
- Confirmed reference targets: `0`.
- Reference-should-flip rows: `early_extra_turn-013` (ref 3 -> proposed 4), `early_extra_turn-018` (ref 2 -> proposed 5).
- Reference-unstable rows: `early_extra_turn-014`, `early_extra_turn-016`.
- Still-inconclusive rows: `early_extra_turn-012`.
- Preservation controls confirmed: `early_extra_turn-002`, `early_extra_turn-007`, `early_extra_turn-009`, `early_extra_turn-011`.
- Reference documentation: `docs/alphazero-lite-early-extra-turn-reference-adjudication-results.md`.
- Because `early_extra_turn` has no clean trainable target bucket and all corrected non-opening families are now exhausted, v8 excludes it and reports the final state.

## 3. Other excluded branches

- Opening replay remains closed; opening rows are context only.
- Corrected guard confirmation rows (`capture_available-002/003/006/007/008`) remain validation context, not training-family targets.
- `capture_available` remains closed after PR #59 left no safe target rows.
- `high_imbalance` remains closed after PR #56 left no safe target rows.
- `high_value_swing` remains closed after PR #53 left no safe target rows.
- `starvation_pressure` remains closed after PR #62 left no safe target rows.
- `sparse_endgame` remains closed after PR #67 forced/tied exclusion.
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
| early_extra_turn branch | 24 | early_extra_turn reference suite is too noisy after reference adjudication | 0 confirmed reference targets remain; 2 reference-should-flip rows (early_extra_turn-013, early_extra_turn-018) require patch review; 3 rows are unstable or inconclusive. |
| incumbent_proxy branch | 32 | incumbent_proxy branch is diagnostic-only/exhausted here | do not continue cross-teacher or soft-policy follow-ups in this task |

## 6. Remaining family ranking

| family | rows | failures | failure_rate | persistent_1200_failures | recovered_at_1200 | high_severity | medium_severity | avg_reference_visit_share_1200 | avg_selected_minus_reference_q_margin_1200 | dominant_failure_mode | classification | notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |

## 7. Representative row diagnostics

| family | row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | selected_move_2400 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | failure_mode | notes |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |

## 8. Targetability classification


## 9. Selected next family

| selected_family | target_rows | control_rows | holdout_rows | reason_selected | risks | next_action |
| --- | ---: | ---: | ---: | --- | --- | --- |
| none_safe | 0 | 0 | 0 | no coherent non-opening family remained safe after all exclusions | all 6 corrected non-opening branches are exhausted under current active references | improve mining/scoring or revisit teacher-policy architecture, not replay. |

## 10. Exactly one recommended next action

Recommendation: **improve mining/scoring or revisit teacher-policy architecture, not replay.**

Run classification: `diffuse_failure_inventory_after_exclusions`.
