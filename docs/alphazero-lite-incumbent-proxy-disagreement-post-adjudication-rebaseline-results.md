# AlphaZero-lite Incumbent Proxy Disagreement Post-Adjudication Rebaseline Results

## 1. Context

- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.

## 2. Why PR #43 made the previous audit stale

- PR #43’s checked-in value/backup audit used the pre-adjudication `incumbent_proxy_disagreement-021` label with reference move `3`.
- The merged approved fixture update changed row `021` to corrected reference move `2`, so the previous family failure inventory and mechanism counts could no longer be trusted as current.
- This rerun recomputes the family baseline and the mechanism audit against the live corrected fixture rather than reusing the stale pre-adjudication report.

## 3. Fixture verification

| row_id | active_reference_move | expected_reference_move | observed_reference_moves | reference_unstable | canonical_state_match | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-021 | 2 | 2 | [2] | false | true | ok | active fixture row now points at move 2; no active stale reference_move 3 remains for row 021; observed reference moves are consistent with the approved patch |

## 4. Post-adjudication family baseline

| row_id | corrected_reference_move | selected_384 | selected_1200 | pass_384 | pass_1200 | reference_visit_share_384 | reference_visit_share_1200 | selected_minus_reference_q_margin_384 | selected_minus_reference_q_margin_1200 | severity | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-001 | 3 | 3 | 3 | true | true | 0.7161 | 0.8950 | 0.0000 | 0.0000 | medium | role=target_candidate; failure_mode=pass |
| incumbent_proxy_disagreement-002 | 3 | 3 | 3 | true | true | 0.5547 | 0.8558 | 0.0000 | 0.0000 | medium | role=target_candidate; failure_mode=pass |
| incumbent_proxy_disagreement-003 | 4 | 4 | 5 | true | false | 0.5807 | 0.2175 | 0.0000 | 0.0012 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-007 | 0 | 5 | 3 | false | false | 0.0052 | 0.0025 | 0.1018 | 0.0733 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-008 | 2 | 2 | 2 | true | true | 0.9297 | 0.9708 | 0.0000 | 0.0000 | none | role=preservation_control; failure_mode=pass |
| incumbent_proxy_disagreement-009 | 0 | 3 | 3 | false | false | 0.0026 | 0.0008 | 0.2088 | 0.2139 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-010 | 4 | 5 | 5 | false | false | 0.0417 | 0.1183 | 0.0565 | 0.0055 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-011 | 4 | 5 | 5 | false | false | 0.0339 | 0.0475 | 0.1195 | 0.0714 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-012 | 3 | 5 | 5 | false | false | 0.0911 | 0.0633 | 0.0475 | 0.0191 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-013 | 2 | 2 | 2 | true | true | 0.9089 | 0.9617 | 0.0000 | 0.0000 | none | role=pass; failure_mode=pass |
| incumbent_proxy_disagreement-014 | 4 | 5 | 5 | false | false | 0.0026 | 0.3392 | 0.1030 | -0.0033 | medium | role=target_candidate; failure_mode=search_selection |
| incumbent_proxy_disagreement-015 | 4 | 5 | 4 | false | true | 0.0938 | 0.6258 | 0.0263 | 0.0000 | medium | role=target_candidate; failure_mode=pass |
| incumbent_proxy_disagreement-016 | 5 | 5 | 5 | true | true | 0.8854 | 0.9392 | 0.0000 | 0.0000 | none | role=pass; failure_mode=pass |
| incumbent_proxy_disagreement-017 | 4 | 5 | 4 | false | true | 0.4297 | 0.8158 | -0.0686 | 0.0000 | medium | role=target_candidate; failure_mode=pass |
| incumbent_proxy_disagreement-018 | 2 | 5 | 3 | false | false | 0.0182 | 0.0108 | 0.0609 | 0.1286 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-019 | 3 | 3 | 3 | true | true | 0.7292 | 0.8925 | 0.0000 | 0.0000 | medium | role=holdout_candidate; failure_mode=pass |
| incumbent_proxy_disagreement-020 | 3 | 5 | 5 | false | false | 0.0312 | 0.1308 | 0.0646 | -0.0004 | medium | role=target_candidate; failure_mode=policy_only |
| incumbent_proxy_disagreement-021 | 2 | 5 | 5 | false | false | 0.0078 | 0.0042 | 0.1543 | 0.2335 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-022 | 3 | 5 | 5 | false | false | 0.0052 | 0.0033 | 0.1457 | 0.0540 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-023 | 2 | 1 | 1 | false | false | 0.0130 | 0.0050 | 0.0918 | 0.0965 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-024 | 2 | 1 | 1 | false | false | 0.0104 | 0.0067 | 0.1256 | 0.1009 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-025 | 2 | 4 | 4 | false | false | 0.0000 | 0.2650 | 0.1011 | -0.0592 | medium | role=target_candidate; failure_mode=search_selection |
| incumbent_proxy_disagreement-026 | 5 | 5 | 5 | true | true | 0.9922 | 0.9650 | 0.0000 | 0.0000 | none | role=preservation_control; failure_mode=pass |
| incumbent_proxy_disagreement-027 | 3 | 5 | 5 | false | false | 0.3542 | 0.1333 | -0.0080 | 0.0407 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-028 | 5 | 5 | 5 | true | true | 0.9323 | 0.9250 | 0.0000 | 0.0000 | none | role=preservation_control; failure_mode=pass |
| incumbent_proxy_disagreement-029 | 4 | 4 | 4 | true | true | 0.9323 | 0.9767 | 0.0000 | 0.0000 | none | role=preservation_control; failure_mode=pass |
| incumbent_proxy_disagreement-030 | 3 | 3 | 3 | true | true | 0.7318 | 0.8875 | 0.0000 | 0.0000 | medium | role=holdout_candidate; failure_mode=pass |
| incumbent_proxy_disagreement-031 | 4 | 4 | 4 | true | true | 0.7292 | 0.9050 | 0.0000 | 0.0000 | none | role=pass; failure_mode=pass |
| incumbent_proxy_disagreement-032 | 4 | 0 | 1 | false | false | 0.0859 | 0.0275 | 0.0179 | 0.0278 | none | role=pass; failure_mode=value_q |
| incumbent_proxy_disagreement-033 | 4 | 0 | 0 | false | false | 0.2578 | 0.0825 | 0.0347 | 0.0417 | medium | role=target_candidate; failure_mode=value_q |
| incumbent_proxy_disagreement-034 | 4 | 4 | 4 | true | true | 0.7474 | 0.9000 | 0.0000 | 0.0000 | none | role=pass; failure_mode=pass |
| incumbent_proxy_disagreement-035 | 3 | 4 | 4 | false | false | 0.0104 | 0.0058 | 0.1515 | 0.1401 | medium | role=target_candidate; failure_mode=value_q |

## 5. Row 021 pre-vs-post comparison

| row_id | old_reference_move | new_reference_move | current_selected_384 | current_selected_1200 | old_label_status | new_label_status | recommended_role | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-021 | 3 | 2 | 5 | 5 | fail_corrected_reference | fail_corrected_reference | target_candidate | row 021 remains a corrected-reference failure after the approved move-2 update |

## 6. Post-adjudication value/backup audit

- Audited rows: `['incumbent_proxy_disagreement-003', 'incumbent_proxy_disagreement-007', 'incumbent_proxy_disagreement-008', 'incumbent_proxy_disagreement-009', 'incumbent_proxy_disagreement-010', 'incumbent_proxy_disagreement-011', 'incumbent_proxy_disagreement-012', 'incumbent_proxy_disagreement-014', 'incumbent_proxy_disagreement-018', 'incumbent_proxy_disagreement-019', 'incumbent_proxy_disagreement-020', 'incumbent_proxy_disagreement-021', 'incumbent_proxy_disagreement-022', 'incumbent_proxy_disagreement-023', 'incumbent_proxy_disagreement-024', 'incumbent_proxy_disagreement-025', 'incumbent_proxy_disagreement-026', 'incumbent_proxy_disagreement-027', 'incumbent_proxy_disagreement-028', 'incumbent_proxy_disagreement-029', 'incumbent_proxy_disagreement-030', 'incumbent_proxy_disagreement-032', 'incumbent_proxy_disagreement-033', 'incumbent_proxy_disagreement-035']`.
- Teacher budgets used: `[1200, 2400, 5000]`.

## 7. Mechanism-specific subfamilies

| row_id | selected_move | corrected_reference_move | row_classification | teacher_child_prefers_reference | neural_child_prefers_reference | child_puct_prefers_reference | root_puct_prefers_reference | reference_suspicious | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| incumbent_proxy_disagreement-003 | 5 | 4 | value_head_miscalibration | true | false | false | false | false | target_candidate | teacher child values prefer corrected reference while raw neural child values prefer selected move |
| incumbent_proxy_disagreement-007 | 3 | 0 | corrected_reference_suspicious | false | false | true | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-008 | 2 | 2 | controls_stable | - | - | - | true | false | preserve_control | preservation control still passes at 1200 |
| incumbent_proxy_disagreement-009 | 3 | 0 | corrected_reference_suspicious | false | false | false | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-010 | 5 | 4 | corrected_reference_suspicious | false | true | true | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-011 | 5 | 4 | puct_child_search_value_mismatch | true | true | false | false | false | target_candidate | teacher and raw neural child values prefer corrected reference while child PUCT does not |
| incumbent_proxy_disagreement-012 | 5 | 3 | value_head_miscalibration | true | false | false | false | false | target_candidate | teacher child values prefer corrected reference while raw neural child values prefer selected move |
| incumbent_proxy_disagreement-014 | 5 | 4 | root_prior_or_selection_pressure | true | true | true | false | false | target_candidate | child teacher, neural, and child PUCT all support corrected reference but root still selects away |
| incumbent_proxy_disagreement-018 | 3 | 2 | corrected_reference_suspicious | false | true | false | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-019 | 3 | 3 | pass_after_021_update | - | - | - | true | false | holdout | row passes under the corrected post-adjudication reference |
| incumbent_proxy_disagreement-020 | 5 | 3 | value_head_miscalibration | true | false | false | false | false | target_candidate | teacher child values prefer corrected reference while raw neural child values prefer selected move |
| incumbent_proxy_disagreement-021 | 5 | 2 | corrected_reference_suspicious | false | false | false | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-022 | 5 | 3 | value_head_miscalibration | true | false | true | false | false | target_candidate | teacher child values prefer corrected reference while raw neural child values prefer selected move |
| incumbent_proxy_disagreement-023 | 1 | 2 | corrected_reference_suspicious | false | true | false | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-024 | 1 | 2 | corrected_reference_suspicious | false | true | false | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-025 | 4 | 2 | value_head_miscalibration | true | false | true | false | false | target_candidate | teacher child values prefer corrected reference while raw neural child values prefer selected move |
| incumbent_proxy_disagreement-026 | 5 | 5 | controls_stable | - | - | - | true | false | preserve_control | preservation control still passes at 1200 |
| incumbent_proxy_disagreement-027 | 5 | 3 | value_head_miscalibration | true | false | false | false | false | target_candidate | teacher child values prefer corrected reference while raw neural child values prefer selected move |
| incumbent_proxy_disagreement-028 | 5 | 5 | controls_stable | - | - | - | true | false | preserve_control | preservation control still passes at 1200 |
| incumbent_proxy_disagreement-029 | 4 | 4 | controls_stable | - | - | - | true | false | preserve_control | preservation control still passes at 1200 |
| incumbent_proxy_disagreement-030 | 3 | 3 | pass_after_021_update | - | - | - | true | false | holdout | row passes under the corrected post-adjudication reference |
| incumbent_proxy_disagreement-032 | 1 | 4 | corrected_reference_suspicious | false | false | false | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-033 | 0 | 4 | corrected_reference_suspicious | false | false | true | false | true | exclude_pending_adjudication | deepest child teacher does not prefer corrected reference |
| incumbent_proxy_disagreement-035 | 4 | 3 | value_head_miscalibration | true | false | false | false | false | target_candidate | teacher child values prefer corrected reference while raw neural child values prefer selected move |


| mechanism | row_count | rows | train_target_eligible_count | holdout_candidate_count | risks | next_action |
| --- | --- | --- | --- | --- | --- | --- |
| stable_value_head_miscalibration | 7 | incumbent_proxy_disagreement-003, incumbent_proxy_disagreement-012, incumbent_proxy_disagreement-020, incumbent_proxy_disagreement-022, incumbent_proxy_disagreement-025, incumbent_proxy_disagreement-027, incumbent_proxy_disagreement-035 | 7 | 0 | value drift if targets are too small or references regress | eligible for train-only value calibration artifact |
| stable_root_selection_pressure | 1 | incumbent_proxy_disagreement-014 | 1 | 0 | root prior pressure may mask correct child values | root cpuct/prior diagnostics |
| stable_puct_child_mismatch | 1 | incumbent_proxy_disagreement-011 | 1 | 0 | child search mismatch may indicate backup or child-selection issues | search-stack diagnostics before training |
| residual_reference_suspicious | 9 | incumbent_proxy_disagreement-007, incumbent_proxy_disagreement-009, incumbent_proxy_disagreement-010, incumbent_proxy_disagreement-018, incumbent_proxy_disagreement-021, incumbent_proxy_disagreement-023, incumbent_proxy_disagreement-024, incumbent_proxy_disagreement-032, incumbent_proxy_disagreement-033 | 0 | 0 | incorrect labels would contaminate any training target set | focused non-mutating adjudication |
| pass_after_021_update | 2 | incumbent_proxy_disagreement-019, incumbent_proxy_disagreement-030 | 0 | 2 | pass rows should not be treated as failures | keep as validation/pass rows |
| controls_stable | 4 | incumbent_proxy_disagreement-008, incumbent_proxy_disagreement-026, incumbent_proxy_disagreement-028, incumbent_proxy_disagreement-029 | 0 | 0 | controls must remain stable during any future experiment | preserve as regression gates |

## 8. Remaining suspicious references

- `incumbent_proxy_disagreement-007` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-009` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-010` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-018` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-021` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-023` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-024` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-032` remains excluded from training targets pending focused adjudication.
- `incumbent_proxy_disagreement-033` remains excluded from training targets pending focused adjudication.

## 9. Targetability decision

- Decision: `needs_more_reference_adjudication`.
- Notes: residual suspicious references remain the largest post-adjudication bucket

## 10. Exactly one recommended next action

Recommendation: **run focused adjudication on the residual suspicious rows before any training.**
