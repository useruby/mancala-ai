# AlphaZero-lite Sparse Endgame Tablebase Convention Audit Results

## 1. Context

- This document reports a tablebase preferred-move convention audit for the `sparse_endgame` corrected non-opening failure family.
- No training was run.
- No arena was run.
- No model was promoted.
- Active references were not mutated.
- PR #65 patch artifact was not applied.
- Active references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Tablebase convention audit output: `/tmp/azlite_sparse_endgame_tablebase_convention_audit/tablebase_convention_audit_summary.json`.
- Corrected reaudit summary: `/tmp/azlite_sparse_endgame_tablebase_convention_audit/sparse_endgame_reaudit_corrected_tablebase_summary.json`.

## 2. Why PR #65 blocked reference changes

PR #65 concluded with:

> **Recommendation: Audit tablebase perspective/convention before touching sparse_endgame references.**

The key finding was that three `tablebase_reference_conflict` rows (sparse_endgame-009, sparse_endgame-025, sparse_endgame-026) had tablebase root value -1.0 (forced loss). In these positions the tablebase "preferred move" was a first-legal-move tie-breaker, not a genuine improvement. ClassicMCTS at 10000 simulations with 7 seeds consistently disagreed with the tablebase tie-breaker and preferred the active reference.

This audit resolves that recommendation by documenting the tablebase convention, adding legal-move value enumeration, reclassifying the false conflicts, and re-running the value/backup audit with corrected tablebase interpretation.

## 3. Tablebase API semantics

The `EndgameTablebase` class in `ml/alphazero_lite/endgame_tablebase.py` was inspected. The `tablebase_preferred_move` helper in `build_sparse_endgame_tablebase_reference_patch.py` and `run_sparse_endgame_value_backup_audit.py` was documented:

| field | observed_behavior | implication | notes |
| --- | --- | --- | --- |
| value convention | current-player perspective minimax; returned as win-rate in [0.0, 1.0] from `perspective_player` argument | Values are from the `perspective_player`'s perspective, not the current game player | Converted to [-1.0, 1.0] via `root_value = 2 * win_rate - 1` |
| preferred move convention | first legal move (by move index) among optimal child values from root-player perspective | In forced positions (|value| = 1.0), all children have identical win-rate; preferred move is a move-index tie-breaker | Tie-breaker is NOT a genuine policy improvement |
| child value computation | `child_win_rate = tablebase.lookup(child_game, root_player)` for each legal move | Children are evaluated from the same root-player perspective | Values are exact minimax solutions (not neural estimates) |
| optimal move set | all moves whose child value equals `max` child value within tolerance | Can be multiple tied optimal moves in forcing or draw positions | Not previously exposed by the API |
| value gap | `best_child_value - active_reference_child_value` | Zero for all sparse_endgame rows (all references are optimal) | Gap > EPS indicates a real conflict |

**Critical finding**: The `tablebase_preferred_move` implementation selects the first move with the highest child win-rate, with a move-index tie-breaker when win-rates are equal. In forcing positions (root value ±1.0), ALL legal moves have identical child win-rates, so the "preferred move" is determined solely by the tie-breaker (lowest move index), not by any game-theoretic distinction.

## 4. Legal-move value enumeration

A diagnostic helper `tablebase_legal_move_values()` was added in both:
- `ml/alphazero_lite/run_sparse_endgame_tablebase_convention_audit.py` (new)
- `ml/alphazero_lite/run_sparse_endgame_value_backup_audit.py` (updated)

For each tablebase-solvable root, it computes:
- `legal_moves`: all legal moves
- `root_value`: root-perspective value in [-1.0, 1.0]
- `child_value_by_move`: root-perspective value for each legal move
- `best_value`: max child value
- `optimal_moves`: all moves whose child value equals best_value within `eps = 1e-9`
- `preferred_move`: first optimal move by move index
- `preferred_move_is_tiebreak`: `len(optimal_moves) > 1`
- `active_reference_is_optimal`: whether the active reference is in optimal_moves

Default production behavior remains unchanged.

## 5. Corrected tablebase conflict definition

Previous logic classified any row where `tablebase_preferred_move != active_reference_move` as `tablebase_reference_conflict`. This did not account for tie-breakers in forcing positions.

Corrected logic:

| condition | classification | meaning |
| --- | --- | --- |
| tablebase not available | tablebase_unavailable | position exceeds solve threshold |
| active reference NOT in optimal_moves AND best_value - ref_value > EPS | tablebase_real_conflict | genuine reference error; tablebase identifies a strictly better move |
| active reference in optimal_moves AND preferred_move != ref_move | tablebase_tie_not_conflict | reference is optimal; preferred differs only by tie-breaking; no change needed |
| active reference in optimal_moves AND preferred_move == ref_move | tablebase_confirmed | reference matches tablebase preference exactly |
| other | tablebase_ambiguous_or_error | edge case; review required |

The corrected logic is implemented in `tablebase_audit_for_row()` in `run_sparse_endgame_value_backup_audit.py` and in `classify_tablebase_row()` in `run_sparse_endgame_tablebase_convention_audit.py`.

No active fixture mutation occurs. No patch artifact is created or applied. Default behavior remains compatible.

## 6. Sparse_endgame row results

All 11 sparse_endgame rows (6 target + 5 control) were audited with legal-move value enumeration:

| row_id | legal_moves | root_value | active_reference_move | preferred_move | optimal_moves | active_reference_is_optimal | puct_selected_is_optimal | preferred_move_is_tiebreak | corrected_tablebase_decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | [0,1,2,3,4] | 1.0000 | 0 | 0 | [0,1,2,3,4] | true | true | true | tablebase_confirmed | forced win; all moves optimal; ref matches preferred |
| sparse_endgame-007 | [0,1,2,3,4] | 1.0000 | 0 | 0 | [0,1,2,3,4] | true | true | true | tablebase_confirmed | forced win; all moves optimal; ref matches preferred |
| sparse_endgame-009 | [0,2,4,5] | -1.0000 | 5 | 0 | [0,2,4,5] | true | true | true | tablebase_tie_not_conflict | forced loss; all moves optimal; tie-break only |
| sparse_endgame-024 | [1,4,5] | -1.0000 | 1 | 1 | [1,4,5] | true | true | true | tablebase_confirmed | forced loss; ref matches preferred |
| sparse_endgame-025 | [1,4,5] | -1.0000 | 5 | 1 | [1,4,5] | true | true | true | tablebase_tie_not_conflict | forced loss; all moves optimal; tie-break only |
| sparse_endgame-026 | [1,4,5] | -1.0000 | 4 | 1 | [1,4,5] | true | true | true | tablebase_tie_not_conflict | forced loss; all moves optimal; tie-break only |
| sparse_endgame-002 | [3,5] | -1.0000 | 3 | 3 | [3,5] | true | true | true | tablebase_confirmed | control; forced loss; ref matches preferred |
| sparse_endgame-011 | [0,1,5] | 1.0000 | 5 | 0 | [0,1,5] | true | true | true | tablebase_tie_not_conflict | control; forced win; all optimal; tie-break only |
| sparse_endgame-019 | [0,1,4,5] | 1.0000 | 5 | 1 | [1,5] | true | true | true | tablebase_tie_not_conflict | control; forced win; move 0,4 suboptimal; ref optimal |
| sparse_endgame-020 | [0,2,4] | 0.0000 | 2 | 0 | [0,2,4] | true | true | true | tablebase_tie_not_conflict | control; draw; all optimal; tie-break only |
| sparse_endgame-023 | [1,4,5] | 0.0000 | 4 | 4 | [4] | true | false | false | tablebase_confirmed | holdout; only move 4 is optimal; PUCT selects suboptimal 5 |

Key findings:
- **All 11 active references are optimal** under the tablebase enumeration.
- **All three former "conflict" rows** (009, 025, 026) are forced losses where `preferred_move_is_tiebreak = True`. The active reference is optimal.
- **sparse_endgame-023** is the only non-tie case: move 4 is uniquely optimal, and the active reference (4) matches it exactly. PUCT selects move 5 (suboptimal).

## 7. Sparse_endgame reaudit with corrected tablebase interpretation

The value/backup audit (`run_sparse_endgame_value_backup_audit.py`) was updated with the corrected tablebase conflict logic and re-run.

Reaudit command:
```
.venv/bin/python ml/alphazero_lite/run_sparse_endgame_value_backup_audit.py \
  --summary-out /tmp/azlite_sparse_endgame_tablebase_convention_audit/sparse_endgame_reaudit_corrected_tablebase_summary.json \
  --report-out /dev/null
```

Reaudit row classifications:

| row_id | role | row_classification | recommended_use |
| --- | --- | --- | --- |
| sparse_endgame-003 | target_candidate | corrected_reference_suspicious | target_candidate |
| sparse_endgame-007 | target_candidate | corrected_reference_suspicious | target_candidate |
| sparse_endgame-009 | target_candidate | tablebase_tie_not_conflict | target_candidate |
| sparse_endgame-024 | target_candidate | puct_child_search_value_mismatch | target_candidate |
| sparse_endgame-025 | target_candidate | tablebase_tie_not_conflict | target_candidate |
| sparse_endgame-026 | target_candidate | tablebase_tie_not_conflict | target_candidate |
| sparse_endgame-023 | holdout_candidate | inconclusive | holdout |
| sparse_endgame-002 | preservation_control | inconclusive | preserve_control |
| sparse_endgame-019 | preservation_control | inconclusive | preserve_control |
| sparse_endgame-020 | preservation_control | inconclusive | preserve_control |
| sparse_endgame-011 | preservation_control | inconclusive | preserve_control |

Family-level summary:

| family_classification | value_head_miscalibration_count | puct_child_search_mismatch_count | root_selection_pressure_count | tablebase_reference_conflict_count | tablebase_tie_not_conflict_count | corrected_reference_suspicious_count | backup_perspective_suspect_count | inconclusive_count | next_action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| reference_family_uncertain | 0 | 1 | 0 | 0 | 3 | 2 | 0 | 0 | adjudicate sparse_endgame references before training |

## 8. Old vs corrected classification comparison

| row_id | pr64_classification | pr65_patch_classification | corrected_tablebase_classification | active_reference_move | corrected_reference_change_needed | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | corrected_reference_suspicious | corrected_reference_suspicious | tablebase_confirmed | 0 | false | target_candidate | non-tablebase issue; teacher/neural do not prefer reference |
| sparse_endgame-007 | corrected_reference_suspicious | corrected_reference_suspicious | tablebase_confirmed | 0 | false | target_candidate | non-tablebase issue; teacher/neural do not prefer reference |
| sparse_endgame-009 | tablebase_reference_conflict | corrected_reference_suspicious (patch) | tablebase_tie_not_conflict | 5 | false | target_candidate | **RESOLVED**: tie-break artifact; ref is optimal |
| sparse_endgame-024 | puct_child_search_value_mismatch | puct_child_search_value_mismatch | tablebase_confirmed | 1 | false | target_candidate | non-tablebase issue; child PUCT mismatch |
| sparse_endgame-025 | tablebase_reference_conflict | corrected_reference_suspicious (patch) | tablebase_tie_not_conflict | 5 | false | target_candidate | **RESOLVED**: tie-break artifact; ref is optimal |
| sparse_endgame-026 | tablebase_reference_conflict | corrected_reference_suspicious (patch) | tablebase_tie_not_conflict | 4 | false | target_candidate | **RESOLVED**: tie-break artifact; ref is optimal |
| sparse_endgame-002 | inconclusive | inconclusive | tablebase_confirmed | 3 | false | preserve_control | stable control |
| sparse_endgame-011 | inconclusive | inconclusive | tablebase_tie_not_conflict | 5 | false | preserve_control | control; tie-break only |
| sparse_endgame-019 | inconclusive | inconclusive | tablebase_tie_not_conflict | 5 | false | preserve_control | control; tie-break only |
| sparse_endgame-020 | inconclusive | inconclusive | tablebase_tie_not_conflict | 2 | false | preserve_control | control; tie-break only |
| sparse_endgame-023 | inconclusive | inconclusive | tablebase_confirmed | 4 | false | holdout | stable holdout |

Key changes:
- **3 tablebase_reference_conflict rows** (009, 025, 026) are now `tablebase_tie_not_conflict` — no reference change needed.
- **0 tablebase_reference_conflict rows remain** after the correction.
- **2 corrected_reference_suspicious rows** (003, 007) persist as non-tablebase issues.
- **1 puct_child_search_value_mismatch** (024) persists as a non-tablebase issue.

## 9. Targetability decision

Decision rules applied (from task specification):

| classification | stable_target_rows | preservation_controls | excluded_rows | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| **tablebase_tie_logic_fixed_sparse_endgame_reauditable** | 3 (009, 025, 026 — optimal refs, no tie-break no change) | 4 (002, 011, 019, 020) | 0 real conflicts excluded; 3 non-tablebase (003, 007 suspicious, 024 puct mismatch) remain targetable | Corrected_reference_suspicious and puct mismatch mechanisms need separate investigation; tablebase integration is now clean | Review corrected audit; address corrected_reference_suspicious (003/007) via value-calibration diagnostic or exclude/reselect family |

The apparent tablebase conflicts from PR #64 are all tie-break artifacts:
- All 3 conflict rows have `abs(root_value) = 1.0` (forcing positions).
- All legal moves have identical child values in these positions.
- All active references are optimal under corrected logic.
- No tablebase_real_conflict exists in sparse_endgame.

Three stable target rows (009, 025, 026) are fully resolved: the tablebase confirms their active references are optimal. No reference change is needed.

Three remaining target rows have non-tablebase issues:
- 003 and 007: `corrected_reference_suspicious` — ClassicMCTS child teacher does not support the corrected reference. This is a reference policy/value issue, not a tablebase issue.
- 024: `puct_child_search_value_mismatch` — teacher and neural prefer reference, but child PUCT selects away. This is a search/value normalization issue.

## 10. Exactly one recommended next action

Recommendation: **Review corrected sparse_endgame audit and address remaining corrected_reference_suspicious (003, 007) and puct_child_search_value_mismatch (024) mechanisms. Tablebase integration is clean; no tablebase-backed patch artifact is needed.**

Rationale:
1. The tablebase convention audit has resolved the PR #65 recommendation. The tablebase API semantics are documented, and the tie-break logic is corrected.
2. All three former `tablebase_reference_conflict` rows are now `tablebase_tie_not_conflict`. No reference changes are needed for these rows.
3. The value/backup audit has been re-run with corrected tablebase logic, producing `reference_family_uncertain` with 2 corrected_reference_suspicious, 1 puct_child_search_value_mismatch, and 3 tablebase_tie_not_conflict target rows.
4. The remaining non-tablebase issues (003, 007, 024) do not stem from tablebase integration and should be addressed through value-calibration diagnostics or family reselection.
5. The active reference fixture should not be mutated for any sparse_endgame row based on tablebase data.
6. PR #65's proposed patch artifact remains `do_not_auto_apply = true` and should not be applied.

**Do not:**
- Train
- Run arena
- Promote a model
- Create replay or value-calibration artifacts
- Mutate the active reference fixture
- Apply the PR #65 patch artifact (all entries have `do_not_auto_apply = true`)
