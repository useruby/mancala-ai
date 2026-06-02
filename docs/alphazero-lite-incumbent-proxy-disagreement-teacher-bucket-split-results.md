# AlphaZero-lite Incumbent Proxy Disagreement Teacher Bucket Split Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- Active references stayed read-only at `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- No production replay artifact was created.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Teacher decisions were loaded from `/tmp/azlite_incumbent_proxy_teacher_policy_decision/teacher_policy_decision_summary.json`.

## 2. Why PR #46 blocked training

- PR #46 concluded that this family required a teacher-policy split before any training target could be constructed.
- Family decision stayed `teacher_policy_split_required` with counts Classic=`6`, PUCT=`7`, excluded=`8`.
- Mixing Classic-teacher and PUCT-teacher rows into one target would still blend incompatible labels.

## 3. Bucket definitions

- Classic-teacher bucket: `incumbent_proxy_disagreement-008, incumbent_proxy_disagreement-014, incumbent_proxy_disagreement-022, incumbent_proxy_disagreement-024, incumbent_proxy_disagreement-025, incumbent_proxy_disagreement-035`.
- PUCT-teacher bucket: `incumbent_proxy_disagreement-007, incumbent_proxy_disagreement-009, incumbent_proxy_disagreement-021, incumbent_proxy_disagreement-026, incumbent_proxy_disagreement-028, incumbent_proxy_disagreement-032, incumbent_proxy_disagreement-033`.
- Excluded diagnostic bucket: `incumbent_proxy_disagreement-003, incumbent_proxy_disagreement-010, incumbent_proxy_disagreement-012, incumbent_proxy_disagreement-018, incumbent_proxy_disagreement-020, incumbent_proxy_disagreement-023, incumbent_proxy_disagreement-027, incumbent_proxy_disagreement-029`.
- Any validation or integrity failure is forced into the excluded diagnostic bucket.

## 4. Reference and state validation

| row_id | bucket | canonical_state_match | active_reference_legal | preferred_move_legal | duplicate_state_conflict | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | excluded_diagnostic | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-007 | puct_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-008 | classic_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-009 | puct_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-010 | excluded_diagnostic | true | true | true | false | ok | row 010 remains excluded |
| incumbent_proxy_disagreement-012 | excluded_diagnostic | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-014 | classic_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-018 | excluded_diagnostic | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-020 | excluded_diagnostic | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-021 | puct_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-022 | classic_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-023 | excluded_diagnostic | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-024 | classic_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-025 | classic_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-026 | puct_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-027 | excluded_diagnostic | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-028 | puct_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-029 | excluded_diagnostic | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-032 | puct_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-033 | puct_teacher | true | true | true | false | ok | validated against active references |
| incumbent_proxy_disagreement-035 | classic_teacher | true | true | true | false | ok | validated against active references |

## 5. Classic-teacher bucket

| row_id | active_reference_move | current_selected_384 | current_selected_1200 | pass_384 | pass_1200 | reference_visit_share_384 | reference_visit_share_1200 | recommended_role | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-008 | 2 | 2 | 2 | true | true | 0.9297 | 0.9708 | preservation_control | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-014 | 4 | 5 | 5 | false | false | 0.0026 | 0.3392 | target_candidate | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.1667 favors move 4; child delta -0.0141 remains near zero |
| incumbent_proxy_disagreement-022 | 3 | 5 | 5 | false | false | 0.0052 | 0.0033 | target_candidate | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-024 | 2 | 1 | 1 | false | false | 0.0104 | 0.0067 | target_candidate | paired continuations support classic/reference under both policies |
| incumbent_proxy_disagreement-025 | 2 | 4 | 4 | false | false | 0.0000 | 0.2650 | target_candidate | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-035 | 3 | 4 | 4 | false | false | 0.0104 | 0.0058 | target_candidate | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.0356 favors move 3; child delta -0.0197 remains near zero |

## 6. PUCT-teacher bucket

| row_id | active_reference_move | puct_preferred_move | current_selected_384 | current_selected_1200 | puct_preference_still_reproduced | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-007 | 0 | 3 | 5 | 3 | true | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child decisive; puct-child near-zero split; child delta -0.2914 favors move 3; child delta -0.0173 remains near zero |
| incumbent_proxy_disagreement-009 | 0 | 3 | 3 | 3 | true | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child: child delta -0.1368 favors move 3; puct-child: child delta -0.1350 favors move 3 |
| incumbent_proxy_disagreement-021 | 2 | 5 | 5 | 5 | true | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child: child delta -0.0540 favors move 5; puct-child: child delta -0.0851 favors move 5 |
| incumbent_proxy_disagreement-026 | 5 | 5 | 5 | 5 | true | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; classic-child: child delta -0.1901 favors move 5; puct-child: child delta -0.1858 favors move 5 |
| incumbent_proxy_disagreement-028 | 5 | 5 | 5 | 5 | true | do_not_train_until_reference_policy_changes | paired continuations support the PUCT-selected branch under both policies |
| incumbent_proxy_disagreement-032 | 4 | 1 | 0 | 1 | true | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; puct-child decisive; classic-child near-zero split; child delta -0.0105 remains near zero; child delta -0.0220 favors move 1 |
| incumbent_proxy_disagreement-033 | 4 | 0 | 0 | 0 | true | do_not_train_until_reference_policy_changes | downstream child evidence favors PUCT; puct-child decisive; classic-child near-zero split; child delta -0.0180 remains near zero; child delta -0.1262 favors move 0 |

## 7. Excluded diagnostic bucket

| row_id | bucket | row_decision | preferred_teacher | preferred_move | active_reference_move | recommended_role | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | excluded_diagnostic | unstable_or_inconclusive | none | 4 | 4 | excluded_diagnostic | teacher_evidence_mixed |
| incumbent_proxy_disagreement-010 | excluded_diagnostic | unstable_or_inconclusive | excluded | 4 | 4 | excluded_diagnostic | excluded_unstable |
| incumbent_proxy_disagreement-012 | excluded_diagnostic | unstable_or_inconclusive | none | 3 | 3 | excluded_diagnostic | teacher_evidence_mixed |
| incumbent_proxy_disagreement-018 | excluded_diagnostic | unstable_or_inconclusive | none | 2 | 2 | excluded_diagnostic | teacher_evidence_mixed |
| incumbent_proxy_disagreement-020 | excluded_diagnostic | unstable_or_inconclusive | none | 3 | 3 | excluded_diagnostic | teacher_evidence_mixed |
| incumbent_proxy_disagreement-023 | excluded_diagnostic | unstable_or_inconclusive | none | 2 | 2 | excluded_diagnostic | teacher_evidence_mixed |
| incumbent_proxy_disagreement-027 | excluded_diagnostic | unstable_or_inconclusive | none | 3 | 3 | excluded_diagnostic | teacher_evidence_mixed |
| incumbent_proxy_disagreement-029 | excluded_diagnostic | unstable_or_inconclusive | none | 4 | 4 | excluded_diagnostic | teacher_evidence_mixed |

## 8. Lightweight bucket validation

| bucket | row_count | train_target_eligible_count | preservation_control_count | excluded_count | integrity_errors | targetability | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| classic_teacher | 6 | 5 | 1 | 0 | 0 | classic_diagnostic_ready | use target candidates plus controls for a tiny Classic-teacher diagnostic artifact |
| puct_teacher | 7 | 0 | 0 | 0 | 0 | reference_policy_only | use only for a future separate PUCT-teacher reference artifact decision |
| excluded_diagnostic | 8 | 0 | 0 | 8 | 0 | not_targetable | keep diagnostic-only and exclude from training and gates |

## 9. Targetability decision

| classification | supporting_evidence | rejected_alternatives | next_action |
| --- | --- | --- | --- |
| split_teacher_policy_required | Classic bucket has 5 stable target candidates plus 1 controls, while PUCT still has 7 reproduced rows that require a separate reference-policy branch; excluded diagnostic bucket remains isolated at 8 rows | rejected a single Classic-only branch because useful PUCT-preferred rows remain incompatible; rejected a PUCT-first branch because it would require a separate reference artifact before any safe training target exists | keep separate Classic-target and PUCT-reference branches; run Classic diagnostic artifact first because it does not require mutating references. |

## 10. Exactly one recommended next action

Recommendation: **keep separate Classic-target and PUCT-reference branches; run Classic diagnostic artifact first because it does not require mutating references.**

## Bucket Assignment Table

| row_id | bucket | row_decision | preferred_teacher | preferred_move | active_reference_move | recommended_role | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | excluded_diagnostic | unstable_or_inconclusive | none | 4 | 4 | excluded_diagnostic | teacher evidence remains mixed; classic=4 puct=5 high_sim=5; classic-child: child delta 0.3466 favors move 4; puct-child: child delta -0.0517 favors move 5 |
| incumbent_proxy_disagreement-007 | puct_teacher | puct_reference_preferred | artifact_puct | 3 | 0 | reference_policy_decision_only | downstream child evidence favors PUCT; classic-child decisive; puct-child near-zero split; child delta -0.2914 favors move 3; child delta -0.0173 remains near zero |
| incumbent_proxy_disagreement-008 | classic_teacher | classic_reference_confirmed | classic_mcts | 2 | 2 | preservation_control | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-009 | puct_teacher | puct_reference_preferred | artifact_puct | 3 | 0 | reference_policy_decision_only | downstream child evidence favors PUCT; classic-child: child delta -0.1368 favors move 3; puct-child: child delta -0.1350 favors move 3 |
| incumbent_proxy_disagreement-010 | excluded_diagnostic | unstable_or_inconclusive | excluded | 4 | 4 | excluded_diagnostic | row 010 remains excluded/unstable regardless of downstream comparisons |
| incumbent_proxy_disagreement-012 | excluded_diagnostic | unstable_or_inconclusive | none | 3 | 3 | excluded_diagnostic | teacher evidence remains mixed; classic=1 puct=5 high_sim=5; classic-child: child delta 0.2481 favors move 3; puct-child: child delta -0.1530 favors move 5 |
| incumbent_proxy_disagreement-014 | classic_teacher | classic_reference_confirmed | classic_mcts | 4 | 4 | target_candidate | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.1667 favors move 4; child delta -0.0141 remains near zero |
| incumbent_proxy_disagreement-018 | excluded_diagnostic | unstable_or_inconclusive | none | 2 | 2 | excluded_diagnostic | teacher evidence remains mixed; classic=2 puct=3 high_sim=3; classic-child: child delta -0.0634 favors move 3; puct-child: child delta 0.0350 favors move 2 |
| incumbent_proxy_disagreement-020 | excluded_diagnostic | unstable_or_inconclusive | none | 3 | 3 | excluded_diagnostic | teacher evidence remains mixed; classic=0 puct=5 high_sim=5; classic-child: child delta 0.4567 favors move 3; puct-child: child delta -0.1048 favors move 5 |
| incumbent_proxy_disagreement-021 | puct_teacher | puct_reference_preferred | artifact_puct | 5 | 2 | reference_policy_decision_only | downstream child evidence favors PUCT; classic-child: child delta -0.0540 favors move 5; puct-child: child delta -0.0851 favors move 5 |
| incumbent_proxy_disagreement-022 | classic_teacher | classic_reference_confirmed | classic_mcts | 3 | 3 | target_candidate | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-023 | excluded_diagnostic | unstable_or_inconclusive | none | 2 | 2 | excluded_diagnostic | teacher evidence remains mixed; classic=2 puct=1 high_sim=1; classic-child: child delta -0.2700 favors move 1; puct-child: child delta 0.0847 favors move 2 |
| incumbent_proxy_disagreement-024 | classic_teacher | classic_reference_confirmed | classic_mcts | 2 | 2 | target_candidate | paired continuations support classic/reference under both policies |
| incumbent_proxy_disagreement-025 | classic_teacher | classic_reference_confirmed | classic_mcts | 2 | 2 | target_candidate | ClassicMCTS, artifact PUCT, and active reference all agree |
| incumbent_proxy_disagreement-026 | puct_teacher | puct_reference_preferred | artifact_puct | 5 | 5 | reference_policy_decision_only | downstream child evidence favors PUCT; classic-child: child delta -0.1901 favors move 5; puct-child: child delta -0.1858 favors move 5 |
| incumbent_proxy_disagreement-027 | excluded_diagnostic | unstable_or_inconclusive | none | 3 | 3 | excluded_diagnostic | teacher evidence remains mixed; classic=3 puct=5 high_sim=5; classic-child: child delta 0.1175 favors move 3; puct-child: child delta -0.0300 favors move 5 |
| incumbent_proxy_disagreement-028 | puct_teacher | puct_reference_preferred | artifact_puct | 5 | 5 | reference_policy_decision_only | paired continuations support the PUCT-selected branch under both policies |
| incumbent_proxy_disagreement-029 | excluded_diagnostic | unstable_or_inconclusive | none | 4 | 4 | excluded_diagnostic | teacher evidence remains mixed; classic=3 puct=4 high_sim=4; classic-child: child delta 0.0000 remains near zero; puct-child: child delta 0.0000 remains near zero |
| incumbent_proxy_disagreement-032 | puct_teacher | puct_reference_preferred | artifact_puct | 1 | 4 | reference_policy_decision_only | downstream child evidence favors PUCT; puct-child decisive; classic-child near-zero split; child delta -0.0105 remains near zero; child delta -0.0220 favors move 1 |
| incumbent_proxy_disagreement-033 | puct_teacher | puct_reference_preferred | artifact_puct | 0 | 4 | reference_policy_decision_only | downstream child evidence favors PUCT; puct-child decisive; classic-child near-zero split; child delta -0.0180 remains near zero; child delta -0.1262 favors move 0 |
| incumbent_proxy_disagreement-035 | classic_teacher | classic_reference_confirmed | classic_mcts | 3 | 3 | target_candidate | downstream child evidence favors classic/reference; classic-child decisive; puct-child near-zero split; child delta 0.0356 favors move 3; child delta -0.0197 remains near zero |