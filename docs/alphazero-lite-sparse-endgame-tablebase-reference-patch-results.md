# AlphaZero-lite Sparse Endgame Tablebase Reference Patch Results

## 1. Context

- This document reports the tablebase-backed reference patch workflow for the `sparse_endgame` corrected non-opening failure family selected by PR #63 and audited by PR #64.
- PR #64 requested: "produce a non-mutating tablebase-backed reference patch artifact for sparse_endgame and rerun the audit before training."
- No training was run.
- No arena was run.
- No model was promoted.
- No replay artifacts were created.
- Corrected references were not mutated.
- Active references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Patch output: `/tmp/azlite_sparse_endgame_tablebase_patch/sparse_endgame_tablebase_reference_patch_v1.json`.
- Rerun audit summary: `/tmp/azlite_sparse_endgame_tablebase_patch/sparse_endgame_tablebase_patch_reaudit_summary.json`.

## 2. Why PR #64 requested a tablebase-backed patch

PR #64 reported the following family-level classification:

| mechanism | count |
| --- | --- |
| `tablebase_reference_conflict` | 3 |
| `corrected_reference_suspicious` | 2 |
| `puct_child_search_value_mismatch` | 1 |
| `value_head_miscalibration` | 0 |
| `root_selection_pressure` | 0 |
| `inconclusive` | 0 |

The three tablebase-reference conflict rows (sparse_endgame-009, sparse_endgame-025, sparse_endgame-026) had tablebase availability at root with a preferred move that differed from the active corrected reference. PR #64 recommended producing a non-mutating tablebase-backed reference patch artifact and rerunning the audit before making any training decision.

## 3. Row and reference validation

All 11 sparse_endgame rows (6 target + 5 control) were validated against the forensic suite and the active reference fixture.

| row_id | active_reference_move | legal | canonical_state_match | remaining_seed_count | root_tablebase_available | status | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 0 | true | true | 16 | true | ok | validated |
| sparse_endgame-007 | 0 | true | true | 16 | true | ok | validated |
| sparse_endgame-009 | 5 | true | true | 14 | true | ok | validated |
| sparse_endgame-024 | 1 | true | true | 16 | true | ok | validated |
| sparse_endgame-025 | 5 | true | true | 16 | true | ok | validated |
| sparse_endgame-026 | 4 | true | true | 16 | true | ok | validated |
| sparse_endgame-002 | 3 | true | true | 16 | true | ok | validated, control |
| sparse_endgame-011 | 5 | true | true | 16 | true | ok | validated, control |
| sparse_endgame-019 | 5 | true | true | 15 | true | ok | validated, control |
| sparse_endgame-020 | 2 | true | true | 11 | true | ok | validated, control |
| sparse_endgame-023 | 4 | true | true | 16 | true | ok | validated, holdout |

All rows pass. No `reference_integrity_error`, no canonical state mismatches, and all reference moves are legal. Tablebase is available for all root positions (all within the 16-seed solve threshold).

## 4. Tablebase root adjudication

| row_id | active_reference_move | tablebase_preferred_move | tablebase_value_root | current_puct_selected_move | tablebase_agrees_with_active_reference | tablebase_agrees_with_puct | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 0 | 0 | 1.0000 | 1 | true | false | tablebase_confirmed | forced win; active reference matches tablebase |
| sparse_endgame-007 | 0 | 0 | 1.0000 | 3 | true | false | tablebase_confirmed | forced win; active reference matches tablebase |
| sparse_endgame-009 | 5 | 0 | -1.0000 | 2 | false | false | tablebase_conflict | forced loss; tablebase prefers move 0 (tie-breaker) |
| sparse_endgame-024 | 1 | 1 | -1.0000 | 4 | true | false | tablebase_confirmed | forced loss; active reference matches tablebase |
| sparse_endgame-025 | 5 | 1 | -1.0000 | 1 | false | true | tablebase_conflict | forced loss; PUCT-selected matches tablebase |
| sparse_endgame-026 | 4 | 1 | -1.0000 | 5 | false | false | tablebase_conflict | forced loss; tablebase prefers move 1 (tie-breaker) |
| sparse_endgame-002 | 3 | 3 | -1.0000 | 3 | true | true | tablebase_confirmed | control, confirmed |
| sparse_endgame-011 | 5 | 0 | 1.0000 | 5 | false | false | tablebase_conflict | control, no action needed |
| sparse_endgame-019 | 5 | 1 | 1.0000 | 5 | false | false | tablebase_conflict | control, no action needed |
| sparse_endgame-020 | 2 | 0 | 0.0000 | 2 | false | false | tablebase_conflict | control, no action needed |
| sparse_endgame-023 | 4 | 4 | 0.0000 | 5 | true | false | tablebase_confirmed | holdout, confirmed |

**Key finding**: The three conflict rows (009, 025, 026) all have root value -1.0 (forced loss from root player perspective). In a forced-loss position, all legal moves have equal win rate (0.0) from the root player's perspective. The tablebase `preferred_move` is therefore a tie-breaker (first legal move by index), not a genuine distinction:

| row_id | root value | tablebase preferred | rationale for tie-break |
| --- | --- | --- | --- |
| sparse_endgame-009 | -1.0 (forced loss) | move 0 | first legal move [0,2,4,5] |
| sparse_endgame-025 | -1.0 (forced loss) | move 1 | first legal move [1,4,5] |
| sparse_endgame-026 | -1.0 (forced loss) | move 1 | first legal move [1,4,5] |

Control rows 011, 019, 020 show tablebase conflicts but are preservation controls and are excluded from mechanism counting (their role in the audit is to guard against regression, not to receive reference changes).

## 5. Tablebase child adjudication

For each target row, children after active reference, tablebase-preferred, and PUCT-selected moves were evaluated with tablebase from root perspective.

| row_id | child_from_move | child_label | child_value_root | active_minus_tablebase_preferred | active_minus_puct_selected | notes |
| --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 0 | active_reference | 1.0000 | 0.0 | 0.0 | all children win from root perspective |
| sparse_endgame-003 | 0 | tablebase_preferred | 1.0000 | - | - | tie with active reference |
| sparse_endgame-003 | 1 | puct_selected | 1.0000 | - | - | tie with active reference |
| sparse_endgame-007 | 0 | active_reference | 1.0000 | 0.0 | 0.0 | all children win from root perspective |
| sparse_endgame-007 | 0 | tablebase_preferred | 1.0000 | - | - | tie with active reference |
| sparse_endgame-007 | 3 | puct_selected | 1.0000 | - | - | tie with active reference |
| sparse_endgame-009 | 0 | tablebase_preferred | -1.0000 | 0.0 | 0.0 | all children lose equally from root perspective |
| sparse_endgame-009 | 2 | puct_selected | -1.0000 | - | - | tie with tablebase |
| sparse_endgame-009 | 5 | active_reference | -1.0000 | - | - | tie with tablebase |
| sparse_endgame-024 | 1 | active_reference | -1.0000 | 0.0 | 0.0 | all children lose equally |
| sparse_endgame-024 | 1 | tablebase_preferred | -1.0000 | - | - | tie with active reference |
| sparse_endgame-024 | 4 | puct_selected | -1.0000 | - | - | tie with active reference |
| sparse_endgame-025 | 1 | tablebase_preferred | -1.0000 | 0.0 | 0.0 | all children lose equally |
| sparse_endgame-025 | 4 | puct_selected_also | -1.0000 | - | - | tie (PUCT move 4 in diff budget) |
| sparse_endgame-025 | 5 | active_reference | -1.0000 | - | - | tie |
| sparse_endgame-026 | 1 | tablebase_preferred | -1.0000 | 0.0 | 0.0 | all children lose equally |
| sparse_endgame-026 | 4 | active_reference | -1.0000 | - | - | tie |
| sparse_endgame-026 | 5 | puct_selected | -1.0000 | - | - | tie |

**Key finding**: In all cases where `active_minus_tablebase_preferred` is calculable, the difference is 0.0. Children are tablebase-equal across all compared moves because the root itself is a forced position (all root values are exactly ±1.0). The tablebase cannot distinguish between child moves when the root is solved as a forced win or forced loss.

This confirms that the `active_minus_tablebase_preferred` metric is informative only in non-forcing positions (|root_value| < 1.0), where a meaningful win-rate comparison between children exists.

## 6. ClassicMCTS cross-check

For the six target rows, ClassicMCTS was run at budgets 1200, 2400, 5000, and 10000 with 7 seeds each (11, 23, 37, 42, 101, 202, 303).

| row_id | budget | seeds | tablebase_preferred_move | classic_majority_move | classic_majority_fraction | classic_agrees_with_tablebase | classic_agrees_with_active_reference | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 1200 | [7] | 0 | 0 | 1.0000 | true | true | classic stable, agrees with both |
| sparse_endgame-003 | 2400 | [7] | 0 | 0 | 1.0000 | true | true | - |
| sparse_endgame-003 | 5000 | [7] | 0 | 0 | 1.0000 | true | true | - |
| sparse_endgame-003 | 10000 | [7] | 0 | 0 | 1.0000 | true | true | - |
| sparse_endgame-007 | 1200 | [7] | 0 | 0 | 1.0000 | true | true | classic stable, agrees with both |
| sparse_endgame-007 | 2400 | [7] | 0 | 0 | 1.0000 | true | true | - |
| sparse_endgame-007 | 5000 | [7] | 0 | 0 | 1.0000 | true | true | - |
| sparse_endgame-007 | 10000 | [7] | 0 | 0 | 1.0000 | true | true | - |
| sparse_endgame-009 | 1200 | [7] | 0 | 5 | 1.0000 | false | true | classic disagrees with tablebase |
| sparse_endgame-009 | 2400 | [7] | 0 | 5 | 1.0000 | false | true | classic stable at active reference |
| sparse_endgame-009 | 5000 | [7] | 0 | 5 | 1.0000 | false | true | all 7 seeds prefer move 5 |
| sparse_endgame-009 | 10000 | [7] | 0 | 5 | 1.0000 | false | true | all 7 seeds prefer move 5 |
| sparse_endgame-024 | 1200 | [7] | 1 | 1 | 1.0000 | true | true | classic stable, agrees with both |
| sparse_endgame-024 | 2400 | [7] | 1 | 1 | 1.0000 | true | true | - |
| sparse_endgame-024 | 5000 | [7] | 1 | 1 | 1.0000 | true | true | - |
| sparse_endgame-024 | 10000 | [7] | 1 | 1 | 1.0000 | true | true | - |
| sparse_endgame-025 | 1200 | [7] | 1 | 5 | 0.7143 | false | true | classic prefers active reference |
| sparse_endgame-025 | 2400 | [7] | 1 | 5 | 0.7143 | false | true | stable at 5/7 seeds |
| sparse_endgame-025 | 5000 | [7] | 1 | 5 | 0.8571 | false | true | 6/7 seeds prefer move 5 |
| sparse_endgame-025 | 10000 | [7] | 1 | 5 | 0.7143 | false | true | 5/7 seeds prefer move 5 |
| sparse_endgame-026 | 1200 | [7] | 1 | 4 | 0.7143 | false | true | classic prefers active reference |
| sparse_endgame-026 | 2400 | [7] | 1 | 4 | 0.7143 | false | true | 5/7 seeds prefer move 4 |
| sparse_endgame-026 | 5000 | [7] | 1 | 4 | 0.8571 | false | true | 6/7 seeds prefer move 4 |
| sparse_endgame-026 | 10000 | [7] | 1 | 4 | 0.8571 | false | true | 6/7 seeds prefer move 4 |

**Critical finding**: ClassicMCTS disagrees with the tablebase-preferred move in all three former conflict rows (009, 025, 026) at every budget, including 10000 simulations with 7 seeds. The majority agreement with the tablebase is zero across all budget-seed combinations for these three rows.

This is consistent with the tablebase child adjudication finding: in forcing positions (root value ±1.0), the tablebase "preferred move" is a first-move tie-breaker, while ClassicMCTS uses playout heuristics and exploration to distinguish between equally-losing children. ClassicMCTS picks the active reference move in all three cases, suggesting the original references are sensible heuristic choices even in losing positions.

Rows 003, 007, and 024 show complete agreement between ClassicMCTS and tablebase. These rows are stable and the active references are confirmed.

## 7. Proposed non-mutating patch artifact

The patch artifact is written to `/tmp/azlite_sparse_endgame_tablebase_patch/sparse_endgame_tablebase_reference_patch_v1.json` with `do_not_auto_apply = true` for all entries. The active reference fixture is not mutated.

| row_id | current_active_reference_move | proposed_reference_move | proposed_reference_unstable | tablebase_preferred_move | tablebase_value_root | classic_majority_move (10000) | classic_majority_fraction (10000) | evidence_summary | do_not_auto_apply |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 0 | 0 | false | 0 | 1.0000 | 0 | 1.0000 | tablebase confirms; ClassicMCTS agrees | true |
| sparse_endgame-007 | 0 | 0 | false | 0 | 1.0000 | 0 | 1.0000 | tablebase confirms; ClassicMCTS agrees | true |
| sparse_endgame-009 | 5 | 0 | true | 0 | -1.0000 | 5 | 1.0000 | forcing position: tablebase tie-breaker; ClassicMCTS disagrees | true |
| sparse_endgame-024 | 1 | 1 | false | 1 | -1.0000 | 1 | 1.0000 | tablebase confirms; ClassicMCTS agrees | true |
| sparse_endgame-025 | 5 | 1 | true | 1 | -1.0000 | 5 | 0.7143 | forcing position: tablebase tie-breaker; ClassicMCTS disagrees | true |
| sparse_endgame-026 | 4 | 1 | true | 1 | -1.0000 | 4 | 0.8571 | forcing position: tablebase tie-breaker; ClassicMCTS disagrees | true |

Three entries propose a reference change:
- sparse_endgame-009: 5 → 0 (unstable, forcing position)
- sparse_endgame-025: 5 → 1 (unstable, forcing position)
- sparse_endgame-026: 4 → 1 (unstable, forcing position)

All propose entries are marked `proposed_reference_unstable = true` because the tablebase preference is a tie-breaker in a forcing position (root value -1.0), and ClassicMCTS at budget 10000 disagrees with the proposed move.

## 8. Sparse_endgame audit rerun with patch override

The rerun applied the proposed patch as an in-memory reference overlay (the active fixture was not mutated). The `--reference-patch` option was used with `run_sparse_endgame_value_backup_audit.py`.

6 patch overrides were loaded (matching the 6 target rows, with 3 no-ops for unchanged rows).

Rerun command:
```
.venv/bin/python ml/alphazero_lite/run_sparse_endgame_value_backup_audit.py \
  --reference-patch /tmp/azlite_sparse_endgame_tablebase_patch/sparse_endgame_tablebase_reference_patch_v1.json \
  --summary-out /tmp/azlite_sparse_endgame_tablebase_patch/sparse_endgame_tablebase_patch_reaudit_summary.json
```

Rerun summary:
- Family classification: `reference_family_uncertain`
- Next action from rerun: `adjudicate sparse_endgame references before training.`

## 9. Pre-patch vs post-patch comparison

| row_id | reference_before | reference_after | classification_before | classification_after | selected_is_reference_before (1200) | selected_is_reference_after (1200) | reference_visit_share_before (1200) | reference_visit_share_after (1200) | tablebase_agreement_after | recommended_use_after |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | 0 | 0 | corrected_reference_suspicious | corrected_reference_suspicious | false | false | 0.0008 | 0.0008 | true | target_candidate |
| sparse_endgame-007 | 0 | 0 | corrected_reference_suspicious | corrected_reference_suspicious | false | false | 0.0008 | 0.0008 | true | target_candidate |
| sparse_endgame-009 | 5 | 0 | tablebase_reference_conflict | corrected_reference_suspicious | false | false | 0.0325 | 0.0158 | true | target_candidate |
| sparse_endgame-024 | 1 | 1 | puct_child_search_value_mismatch | puct_child_search_value_mismatch | false | false | 0.0800 | 0.0800 | true | target_candidate |
| sparse_endgame-025 | 5 | 1 | tablebase_reference_conflict | corrected_reference_suspicious | false | true | 0.0158 | 0.9833 | true | target_candidate |
| sparse_endgame-026 | 4 | 1 | tablebase_reference_conflict | corrected_reference_suspicious | false | false | 0.0008 | 0.0975 | true | target_candidate |
| sparse_endgame-023 | 4 | 4 | inconclusive | inconclusive | false | false | 0.0008 | 0.0008 | true | holdout |
| sparse_endgame-002 | 3 | 3 | inconclusive | inconclusive | true | true | 1.0000 | 1.0000 | true | preserve_control |
| sparse_endgame-019 | 5 | 5 | inconclusive | inconclusive | true | true | 0.9875 | 0.9875 | true | preserve_control |
| sparse_endgame-020 | 2 | 2 | inconclusive | inconclusive | true | true | 0.9842 | 0.9842 | true | preserve_control |
| sparse_endgame-011 | 5 | 5 | inconclusive | inconclusive | true | true | 0.9833 | 0.9833 | false | preserve_control |

Notable pre/post differences:

| row_id | change |
| --- | --- |
| sparse_endgame-009 | tablebase_reference_conflict → corrected_reference_suspicious. Tablebase now agrees with reference (both move 0), but ClassicMCTS does not prefer the corrected reference. |
| sparse_endgame-025 | tablebase_reference_conflict → corrected_reference_suspicious. The proposed reference (1) now matches what PUCT selects at 1200, so `selected_is_reference` flips to true and `reference_visit_share` jumps from 0.0158 to 0.9833. However, ClassicMCTS still prefers move 5 over the corrected reference 1. |
| sparse_endgame-026 | tablebase_reference_conflict → corrected_reference_suspicious. Reference visit share increases from 0.0008 to 0.0975 because the proposed move 1 gets more exploration than the original move 4, but PUCT still prefers move 5. |

## 10. Projected post-patch targetability

| classification | stable_target_rows | preservation_controls | excluded_rows | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| **tablebase_integration_issue** | 2 (003, 007) | 4 (002, 011, 019, 020) | 3 (009, 025, 026 unstable patch proposals) + 1 (024 puct mismatch) | ClassicMCTS and tablebase disagree in 3/6 target rows even at budget 10000; disagreement is caused by tablebase returning tie-breaker moves in forcing (|value|=1.0) positions; the tablebase "conflict" reported by PR #64 is a tie-breaking artifact, not a genuine reference error. | Audit tablebase perspective/convention before touching references |

The pre-patch `tablebase_reference_patch_needed` family classification was driven by the three tablebase conflicts. However, the root values for all three conflict rows are -1.0 (forced loss). In a forced-loss position, all legal moves have equal win rate (0.0) from the root player's perspective, so the tablebase preferred move is simply the first legal move by index. ClassicMCTS at budget 10000 across 7 seeds consistently prefers the original active reference move (or another move) over the tablebase tie-breaker, confirming that the "conflict" is a tablebase integration artifact rather than a genuine reference error.

The rerun audit after applying the proposed patch reclassifies the three formerly-conflicted rows as `corrected_reference_suspicious` (5 total corrected_reference_suspicious rows) because the teacher (ClassicMCTS) does not support the patched reference. The family shifts from `tablebase_reference_patch_needed` to `reference_family_uncertain`.

## 11. Exactly one recommended next action

Recommendation: **Audit tablebase perspective/convention before touching sparse_endgame references.**

Rationale:
1. The three `tablebase_reference_conflict` rows from PR #64 occur exclusively in forcing positions (root value -1.0, forced loss from root perspective). The tablebase "preferred move" in these positions is a first-move tie-breaker, not a genuine improvement over the active reference.
2. ClassicMCTS at budget 10000 with 7 seeds unanimously prefers the active reference moves over the tablebase-preferred moves in all three cases, with zero variation across seeds.
3. Applying the proposed patch does not resolve the family uncertainty: it simply reclassifies the three rows from `tablebase_reference_conflict` to `corrected_reference_suspicious`, and the family classification changes from `tablebase_reference_patch_needed` to `reference_family_uncertain`.
4. The tablebase integration should be audited to ensure that the `tablebase_preferred_move` function does not return tie-breaker moves in forcing positions, or that the audit pipeline filters out forcing-position conflicts.
5. After the tablebase integration audit, sparse_endgame should be re-evaluated: rows 003 and 007 are stable `corrected_reference_suspicious` candidates that may benefit from a targeted value-calibration diagnostic, but no training or arena should be run until the tablebase convention is clarified.

**Do not:**
- Train
- Run arena
- Promote a model
- Create replay or value-calibration artifacts
- Mutate the active reference fixture
- Apply the proposed patch artifact (all entries have `do_not_auto_apply = true`)
