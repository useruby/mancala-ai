# AlphaZero-lite Fresh Hard-State Mining (Teacher-Filtered) Results

## 1. Context

- Run classification: fresh_family_tablebase_exact_ready
- Selected family: fresh_endgame_tablebase_unique
- Current artifact: storage/ai/alphazero_lite/current
- Active references: ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json
- No training was run.
- No arena was run.
- No model was promoted.
- Active references were not mutated.

## 2. Why PR #70 stopped incumbent_proxy training

PR #70 concluded that teacher-policy conflict within the incumbent_proxy_disagreement
family is architectural — no single-head or teacher-conditioned probe variant
resolved the cross-teacher interference. PR #69 patch bundle was not applied.
The recommendation was to stop training on incumbent_proxy, use teacher-policy
split only for evaluation, and improve mining/scoring from fresh positions.

This run implements that recommendation by mining fresh positions from
current-model play, not from the exhausted fixture inventory.

## 3. Exclusion policy

| exclusion_group | excluded_count | reason | notes |
|---|---|---|---|
| capture_available | 24 | no safe target rows after PR #59 | excluded from training; 24 rows |
| corrected_guard_rows | 4 | corrected guard confirmations stay context-only | excluded from training; 4 rows |
| early_extra_turn | 24 | reference suite too noisy after reference adjudication | excluded from training; 24 rows |
| high_imbalance | 24 | reference suite too noisy after PR #56 | excluded from training; 24 rows |
| high_value_swing | 24 | reference suite too noisy after PR #53 | excluded from training; 24 rows |
| incumbent_proxy_disagreement | 32 | teacher-conflict family; stopped after PR #70 | excluded from training; 32 rows |
| opening_plies_1_8 | 48 | opening replay branch is closed | excluded from training; 48 rows |
| reference_unstable_rows | 0 | rows marked reference_unstable in active references | no rows in active inventory |
| sparse_endgame | 24 | dominated by forced/tied positions after PR #67 | excluded from training; 24 rows |
| starvation_pressure | 24 | reference suite too noisy after PR #62 | excluded from training; 24 rows |

## 4. Fresh candidate generation

| source | raw_candidates | deduplicated_candidates | known_fixture_overlaps | excluded_overlaps | remaining_candidates | notes |
|---|---|---|---|---|---|---|
| self_play | 8360 | 8120 | 7 | 7 | 233 | fresh positions from current-model PUCT |

## 5. Deduplication and fixture overlap

- Raw candidates: 8360
- Duplicate candidates: 8120
- Known-fixture overlaps: 7
- Excluded overlaps: 7
- Remaining novel candidates: 233

## 6. Teacher labeling / adjudication-lite

| candidate_id | provisional_family | ply | puct_1200_move | classic_2400_move | tablebase_decision | teacher_agreement | teacher_conflict | reference_confidence | status | notes |
|---|---|---|---|---|---|---|---|---|---|---|
| fresh_42_4 | fresh_search_selection_pressure | 4 | 4 | 3 | N/A | disagree | True | low | conflict |  |
| fresh_42_5 | fresh_value_q_disagreement | 5 | 5 | 4 | N/A | disagree | True | low | conflict |  |
| fresh_42_9 | fresh_value_q_disagreement | 9 | 5 | 1 | N/A | disagree | True | low | conflict |  |
| fresh_42_10 | fresh_value_q_disagreement | 10 | 1 | 0 | N/A | disagree | True | low | conflict |  |
| fresh_42_11 | fresh_midgame_capture_swing | 11 | 5 | 5 | N/A | agree | False | high | candidate |  |
| fresh_42_12 | fresh_capture_swing | 12 | 1 | 3 | N/A | disagree | True | low | conflict |  |
| fresh_42_13 | fresh_value_q_disagreement | 13 | 4 | 2 | N/A | disagree | True | low | conflict |  |
| fresh_42_14 | fresh_extra_turn_handoff | 14 | 5 | 3 | N/A | disagree | True | low | conflict |  |
| fresh_42_15 | fresh_midgame_capture_swing | 15 | 3 | 3 | N/A | agree | False | high | candidate |  |
| fresh_42_16 | fresh_capture_swing | 16 | 5 | 2 | N/A | disagree | True | low | conflict |  |
| fresh_42_17 | fresh_midgame_capture_swing | 17 | 2 | 2 | N/A | agree | False | high | candidate |  |
| fresh_42_18 | fresh_capture_swing | 18 | 5 | 2 | N/A | disagree | True | low | conflict |  |
| fresh_42_19 | fresh_value_q_disagreement | 19 | 4 | 0 | N/A | disagree | True | low | conflict |  |
| fresh_42_20 | fresh_capture_swing | 20 | 3 | 2 | N/A | disagree | True | low | conflict |  |
| fresh_42_21 | fresh_capture_swing | 21 | 2 | 5 | N/A | disagree | True | low | conflict |  |
| fresh_42_22 | fresh_midgame_capture_swing | 22 | 2 | 2 | N/A | agree | False | high | candidate |  |
| fresh_42_23 | fresh_value_q_disagreement | 23 | 5 | 3 | N/A | disagree | True | low | conflict |  |
| fresh_42_24 | fresh_capture_swing | 24 | 4 | 1 | N/A | disagree | True | low | conflict |  |
| fresh_42_25 | fresh_capture_swing | 25 | 0 | 1 | N/A | disagree | True | low | conflict |  |
| fresh_42_26 | fresh_capture_swing | 26 | 0 | 1 | N/A | disagree | True | low | conflict |  |
| fresh_42_27 | fresh_capture_swing | 27 | 0 | 1 | N/A | disagree | True | low | conflict |  |
| fresh_42_28 | fresh_midgame_capture_swing | 28 | 1 | 1 | N/A | agree | False | high | candidate |  |
| fresh_42_29 | fresh_endgame_tablebase_unique | 29 | 3 | 3 | [3] | tablebase_confirms | False | high | candidate |  |
| fresh_42_30 | fresh_endgame_tablebase_tie | 30 | 5 | 5 | [4, 5] | tablebase_tie | False | medium | candidate |  |
| fresh_42_31 | fresh_endgame_tablebase_unique | 31 | 4 | 4 | [4] | tablebase_confirms | False | high | candidate |  |
| fresh_42_32 | fresh_endgame_tablebase_tie | 32 | 5 | 2 | [2, 5] | tablebase_tie | False | medium | candidate |  |
| fresh_42_33 | fresh_endgame_tablebase_unique | 33 | 2 | 2 | [2] | tablebase_confirms | False | high | candidate |  |
| fresh_42_34 | fresh_endgame_tablebase_tie | 34 | 5 | 5 | [0, 4, 5] | tablebase_tie | False | medium | candidate |  |
| fresh_42_35 | fresh_endgame_tablebase_tie | 35 | 4 | 4 | [0, 4] | tablebase_tie | False | medium | candidate |  |
| fresh_42_36 | fresh_endgame_tablebase_unique | 36 | 1 | 1 | [1] | tablebase_confirms | False | high | candidate |  |
| ... and 203 more rows | ... | ... | ... | ... | ... | ... | ... | ... | ... | ... |

## 7. Provisional family clustering

| provisional_family | rows | stable_teacher_rows | rejected_rows | persistent_1200_failures | teacher_conflict_rate | tablebase_confirmed_rows | dominant_failure_mode | targetability | notes |
|---|---|---|---|---|---|---|---|---|---|
| fresh_endgame_tablebase_unique | 14 | 14 | 0 | 0 | 0.0 | 14 | tablebase_exact | fresh_family_tablebase_exact_ready |  |
| fresh_search_selection_pressure | 8 | 3 | 0 | 8 | 0.625 | 0 | teacher_conflict | teacher_conflict_dominant |  |
| fresh_value_q_disagreement | 21 | 0 | 0 | 0 | 1.0 | 0 | teacher_conflict | teacher_conflict_dominant |  |
| fresh_capture_swing | 38 | 0 | 0 | 8 | 1.0 | 0 | teacher_conflict | teacher_conflict_dominant |  |
| fresh_extra_turn_handoff | 17 | 0 | 0 | 1 | 1.0 | 0 | teacher_conflict | teacher_conflict_dominant |  |
| fresh_midgame_capture_swing | 55 | 37 | 0 | 0 | 0.3273 | 0 | mixed | too_noisy |  |
| fresh_endgame_tablebase_tie | 80 | 80 | 0 | 5 | 0.0125 | 0 | mixed | no_clean_family |  |

## 8. Targetability scoring

- fresh_family_tablebase_exact_ready: 1 family/ies
- no_clean_family: 1 family/ies
- teacher_conflict_dominant: 4 family/ies
- too_noisy: 1 family/ies

## 9. Selected fresh family

| selected_family | target_rows | control_rows | holdout_rows | teacher_source | reason_selected | risks | next_action |
|---|---|---|---|---|---|---|---|
| fresh_endgame_tablebase_unique | 12 | 1 | 1 | tablebase | selected_family_fresh_family_tablebase_exact_ready | teacher_conflict may reappear at higher budget; monitor holdout rows | run tablebase-backed local search/value diagnostics; do not train until target/control split is clean |

## 10. Exactly one recommended next action

Recommendation: **run tablebase-backed local search/value diagnostics; do not train until target/control split is clean**

### Acceptance criteria

- No training was run.
- No arena was run.
- No model was promoted.
- Active references were not mutated.
- Exhausted families were excluded from selection.
- Teacher-conflict filtering from PR #70 was used.
- Selected rows are metadata candidates only, not replay artifacts.
- Final report recommends exactly one next branch.