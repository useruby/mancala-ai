# AlphaZero-lite Sparse Endgame Exact Tablebase Targetability Results

## 1. Context

- This report evaluates whether `sparse_endgame` has meaningful trainable signal after correcting tablebase tie-break semantics.
- All 24 sparse_endgame rows from the active suite and references were loaded and enumerated with exact tablebase move values.
- No training was run.
- No arena was run.
- No model was promoted.
- No replay or value-calibration artifacts were created.
- Active references were not mutated.
- Active references: `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`.
- Current artifact: `storage/ai/alphazero_lite/current`.
- Output directory: `/tmp/azlite_sparse_endgame_exact_tablebase_targetability/`.
- This analysis was done by: `ml/alphazero_lite/run_sparse_endgame_exact_tablebase_targetability_split.py`.
- Schema: `azlite_sparse_endgame_exact_tablebase_targetability_v1`.

## 2. Why PR #66 changed the sparse_endgame interpretation

PR #66 (tablebase convention audit) corrected the tablebase preferred-move definition from "first legal move by index among all children" to "first optimal move among children with the highest win rate from root-player perspective." This change:

- Reclassified 3 `tablebase_reference_conflict` rows (sparse_endgame-009, -025, -026) as `tablebase_tie_not_conflict` — the active reference was already optimal; the "preferred move" was only a move-index tie-breaker in forced-loss positions.
- Left 2 `corrected_reference_suspicious` rows (sparse_endgame-003, -007) as non-tablebase issues.
- Left 1 `puct_child_search_value_mismatch` (sparse_endgame-024) as a non-tablebase issue.
- All 11 audited rows have tablebase-optimal active references.

**Critical implication**: In forcing positions (root value ±1.0 with all child values identical), ALL legal moves have identical game-theoretic value. The tablebase does not distinguish among them. Policy labels for these rows are arbitrary tie-breakers, not meaningful training targets.

PR #66's final recommendation was to "address remaining non-tablebase mechanisms before training." The present analysis goes one step further: it asks whether _any_ meaningful trainable signal remains after the tablebase fix.

## 3. Row validation

All 24 sparse_endgame rows (3 seed + 21 generated) were loaded from the forensic suite and cross-validated against reference fixtures:

| check | result |
|---|---|
| Suite rows loaded | 224 total (24 sparse_endgame) |
| Reference rows matched | 24/24 (100%) |
| Canonical state hash match | 24/24 (100%) |
| Reference move legal | 24/24 (100%) |
| reference_unstable | 0/24 (0%) |
| Active fixture mutated | No |
| Rows with PUCT selected data | 11 (from PR #63 selected rows) |
| Rows without PUCT data | 13 (non-selected context rows) |
| Tablebase solvable (<=16 seeds) | 24/24 (100%) |

## 4. Exact tablebase move-value enumeration

For each of the 24 rows, the `EndgameTablebase` solver computed:
- Root value from root-player perspective ([-1.0, 1.0], current-player convention)
- Child value by legal move
- Best child value
- Optimal move set (all moves within EPS 1e-9 of best)
- Whether all legal moves are equivalent
- Whether the position is forcing (abs(root_value) == 1.0)

### Exact tablebase table

| row_id | role | legal_moves | root_value | active_reference_move | current_puct_selected_move | optimal_moves | active_reference_is_optimal | puct_selected_is_optimal | unique_optimal_move | all_legal_moves_equivalent | exact_signal_class | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-001 | unselected | [4,5] | -1.0 | 4 | - | [4,5] | true | - | - | true | forced_all_moves_equivalent | forced loss, opponent current player |
| sparse_endgame-002 | preservation_control | [3,5] | -1.0 | 3 | 3 | [3,5] | true | true | - | true | forced_all_moves_equivalent | forced loss, ref optimal, PUCT selects ref |
| sparse_endgame-003 | target_candidate | [0,1,2,3,4] | 1.0 | 0 | 1 | [0,1,2,3,4] | true | true | - | true | forced_all_moves_equivalent | forced win, all moves optimal |
| sparse_endgame-007 | target_candidate | [0,1,2,3,4] | 1.0 | 0 | 3 | [0,1,2,3,4] | true | true | - | true | forced_all_moves_equivalent | forced win, all moves optimal |
| sparse_endgame-008 | unselected | [0,1,4,5] | 0.0 | 5 | - | [0,1,4,5] | true | - | - | false | exact_value_only_tie | draw, all moves optimal |
| sparse_endgame-009 | target_candidate | [0,2,4,5] | -1.0 | 5 | 2 | [0,2,4,5] | true | true | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-010 | unselected | [0,1,3] | -1.0 | 3 | - | [0,1,3] | true | - | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-011 | preservation_control | [0,1,5] | 1.0 | 5 | 5 | [0,1,5] | true | true | - | true | forced_all_moves_equivalent | forced win, all moves optimal |
| sparse_endgame-012 | unselected | [0,1,5] | 0.0 | 5 | - | [0,1,5] | true | - | - | false | exact_value_only_tie | draw, all moves optimal |
| sparse_endgame-013 | unselected | [0,3,4,5] | -1.0 | 5 | - | [0,3,4,5] | true | - | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-014 | unselected | [0,1,2,5] | 0.0 | 0 | - | [0,1,5] | true | - | - | false | exact_value_only_tie | draw; move 2 suboptimal (child value != 0) |
| sparse_endgame-015 | unselected | [0,2,4,5] | 0.0 | 5 | - | [5] | true | - | 5 | false | exact_unique_policy_target | draw; only move 5 is optimal |
| sparse_endgame-016 | unselected | [0,1,3] | -1.0 | 3 | - | [0,1,3] | true | - | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-017 | unselected | [0,2,3,4,5] | 0.0 | 4 | - | [4] | true | - | 4 | false | exact_unique_policy_target | draw; only move 4 is optimal |
| sparse_endgame-018 | unselected | [0,1,5] | -1.0 | 5 | - | [0,1,5] | true | - | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-019 | preservation_control | [0,1,4,5] | 1.0 | 5 | 5 | [1,5] | true | true | - | false | exact_value_only_tie | forced win; moves 0,4 suboptimal |
| sparse_endgame-020 | preservation_control | [0,2,4] | 0.0 | 2 | 2 | [0,2,4] | true | true | - | true | forced_all_moves_equivalent | draw; all moves optimal |
| sparse_endgame-021 | unselected | [0,1,4] | -1.0 | 0 | - | [0,1,4] | true | - | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-022 | unselected | [1,3,5] | 0.0 | 1 | - | [1] | true | - | 1 | false | exact_unique_policy_target | draw; only move 1 is optimal |
| sparse_endgame-023 | holdout_candidate | [1,4,5] | 0.0 | 4 | 5 | [4] | true | false | 4 | false | exact_unique_policy_target | draw; only move 4 is optimal; PUCT selects suboptimal 5 |
| sparse_endgame-024 | target_candidate | [1,4,5] | -1.0 | 1 | 4 | [1,4,5] | true | true | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-025 | target_candidate | [1,4,5] | -1.0 | 5 | 1 | [1,4,5] | true | true | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-026 | target_candidate | [1,4,5] | -1.0 | 4 | 5 | [1,4,5] | true | true | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |
| sparse_endgame-027 | unselected | [1,4,5] | -1.0 | 5 | - | [1,4,5] | true | - | - | true | forced_all_moves_equivalent | forced loss, all moves optimal |

Key findings:
- **All 24 active references are tablebase-optimal** — no reference integrity errors.
- **16 rows (67%) are forced_all_moves_equivalent** — all legal moves have identical game-theoretic value. Policy labels are arbitrary.
- **4 rows (17%) are exact_value_only_tie** — multiple moves optimal but not all equivalent; value labels are correct but policy targets are ambiguous.
- **4 rows (17%) are exact_unique_policy_target** — exactly one move is optimal; these are the only rows with meaningful policy signal.
- **1 row (4%) has PUCT-suboptimal behavior** — sparse_endgame-023, where PUCT selects move 5 but move 4 is uniquely optimal.
- **0 rows are tablebase_unavailable** — all 24 rows are within the 16-seed solve threshold.
- **0 rows have reference_suboptimal** — every active reference is tablebase-optimal.

## 5. PR #66 row reinterpretation

The 7 PR #66 rows are re-evaluated under the exact tablebase signal classification:

| row_id | pr66_classification | exact_signal_class | meaningful_policy_target | meaningful_value_target | recommended_use | notes |
| --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-003 | corrected_reference_suspicious | forced_all_moves_equivalent | false | false | exclude_from_training | Forced win; all 5 moves optimal. Teacher/neural disagreement is about which arbitrary tie-breaker to select, not about a meaningful policy distinction. |
| sparse_endgame-007 | corrected_reference_suspicious | forced_all_moves_equivalent | false | false | exclude_from_training | Same as 003. The corrected_reference_suspicious flag was a non-tablebase issue; but the issue is moot because all moves are equivalent. |
| sparse_endgame-009 | tablebase_tie_not_conflict | forced_all_moves_equivalent | false | false | exclude_from_training | Forced loss; all 4 moves optimal. The former "conflict" was a tie-break artifact. No training signal. |
| sparse_endgame-024 | puct_child_search_value_mismatch | forced_all_moves_equivalent | false | false | exclude_from_training | Forced loss; all 3 moves optimal. The child PUCT disagreement is about which equally-good move to select. No meaningful policy error. |
| sparse_endgame-025 | tablebase_tie_not_conflict | forced_all_moves_equivalent | false | false | exclude_from_training | Forced loss; all 3 moves optimal. Tie-break artifact. |
| sparse_endgame-026 | tablebase_tie_not_conflict | forced_all_moves_equivalent | false | false | exclude_from_training | Forced loss; all 3 moves optimal. Tie-break artifact. |
| sparse_endgame-023 | inconclusive (holdout) | exact_unique_policy_target | true | true | holdout_only | Only row with both unique optimal move AND PUCT suboptimal behavior. Single-row bucket; too small for isolated training. |

**Critical reinterpretation**: The 6 PR #66 target rows (003, 007, 009, 024, 025, 026) are all `forced_all_moves_equivalent`. Their "failures" (corrected_reference_suspicious, puct_child_search_value_mismatch, tablebase_tie_not_conflict) are all because the system selects a different move among equally-optimal choices. These are NOT policy-error rows — they are noise rows where policy targets are fundamentally ambiguous.

## 6. Targetability buckets

| bucket | row_count | rows | recommended_use | risks | next_action |
| --- | --- | --- | --- | --- | --- |
| forced_all_moves_equivalent | 16 | 001, 002, 003, 007, 009, 010, 011, 013, 016, 018, 020, 021, 024, 025, 026, 027 | exclude_from_training | No meaningful policy or value signal; all moves lead to same game-theoretic outcome | Exclude from training selection; preserve as context only |
| exact_value_only_ties | 4 | 008, 012, 014, 019 | value_only_candidate | Multiple optimal moves means policy target ambiguous; value labels correct | Assess whether >=5 rows exist for value-only diagnostic (fails at 4) |
| exact_unique_policy_targets | 4 | 015, 017, 022, 023 | policy_target_candidate | 3 of 4 rows lack PUCT data; single-row (023) has PUCT suboptimal behavior; insufficient for family-level mining | Holdout only; too small for isolated training |
| puct_suboptimal_exact_rows | 1 | 023 | search_diagnostic | PUCT selects suboptimal move despite exact tablebase optimal; search/value issue not policy | Optional: run tiny PUCT diagnostic on 023 only |
| preservation_controls | 4 | 002, 011, 019, 020 | preservation_control | Controls must remain passing; no training target | Retain in test/holdout set for regression detection |
| exact_reference_suboptimal | 0 | — | exclude_from_training | — | — |
| tablebase_unavailable_or_ambiguous | 0 | — | exclude_from_training | — | — |
| tablebase_integrity_error | 0 | — | exclude_from_training | — | — |

Cross-cutting note: `puct_suboptimal_exact_rows` is a subset of `exact_unique_policy_targets` (sparse_endgame-023 appears in both).

## 7. Value-only candidate assessment

| row_id | exact_value | neural_value | value_error | sign_error | abs_error | forced_win_or_loss | value_only_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-008 | 0.0 | — | — | — | — | false | true | Exact-value tie; neural value not computed in this split |
| sparse_endgame-012 | 0.0 | — | — | — | — | false | true | Same |
| sparse_endgame-014 | 0.0 | — | — | — | — | false | true | Same |
| sparse_endgame-019 | 1.0 | — | — | — | — | true | true | Forced win; neural value not computed in this split |

Value-only candidate count: **4** (neural values not computed in this split; no threshold test run).

Neural values are available from the PR #66 value backup audit for selected rows only. Of the 4 value-only ties:
- sparse_endgame-019 (preservation control) has neural child values in the audit but not raw root neural value.
- The other 3 (008, 012, 014) are not in the selected set and have no neural data.

**Verdict**: < 5 value-only candidate rows with available neural error. Not enough to justify a value-only diagnostic artifact. The 4 ties are draw positions where all moves lead to the same outcome — even if value error exists, the training signal is trivially flat.

## 8. Policy/search candidate assessment

| row_id | exact_optimal_move | puct_selected_move | puct_selected_is_optimal | reference_visit_share_1200 | selected_minus_reference_q_margin_1200 | policy_or_search_candidate | notes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| sparse_endgame-015 | 5 | — | — | — | — | true | Unique policy target; no PUCT data (unselected row) |
| sparse_endgame-017 | 4 | — | — | — | — | true | Unique policy target; no PUCT data (unselected row) |
| sparse_endgame-022 | 1 | — | — | — | — | true | Unique policy target; no PUCT data (unselected row) |
| sparse_endgame-023 | 4 | 5 | false | 0.0008 (1200) | 0.0582 (1200) | true | Unique policy target WITH PUCT-suboptimal behavior. Reference visit share is 0.08%, near-zero. Selected-minus-reference Q margin is positive (PUCT prefers suboptimal 5 over optimal 4). Root-prior mismatch likely: reference policy probability is 0.0118, selected is 0.2256. |

Policy/search candidate count: **4 total** (3 unique targets without PUCT data + 1 with PUCT-suboptimal behavior).

Of the 4 unique policy targets:
- sparse_endgame-023 is the only row with actual PUCT data showing suboptimal behavior. Its reference visit share at 1200 is 0.08% (effectively zero) and the PUCT-selected move (5) has Q margin +0.0582 over the optimal reference (4). This is consistent with a root-prior mismatch: the neural policy assigns 1.18% to the optimal move vs 22.56% to the suboptimal move.
- sparse_endgame-015, -017, -022 are unique policy targets but lack PUCT, child-value, and neural data. They would need MCTS runs to be useful diagnostics.

**Verdict**: Only 1 row (023) has actionable policy/search diagnostic data. This is insufficient for a training family. It could serve as a single-row search diagnostic context item, but not as a training target.

## 9. Final targetability decision

| classification | supporting_evidence | rejected_alternatives | next_action |
| --- | --- | --- | --- |
| **sparse_endgame_not_trainable_as_policy_family** | 83% of 24 rows are forced/tied (16 forced_all_moves_equivalent + 4 exact_value_only_ties). Policy labels are arbitrary in all 16 forced-all-equivalent rows. 4 unique policy targets exist but only 1 has PUCT data. 1 PUCT-suboptimal row exists but family-level training is not justified at 83% forced noise. | **A (value_only_diagnostic_ready)**: Rejected — only 4 value-only candidates exist, neural values not computed in this split, <5 threshold. **B (policy_search_diagnostic_ready)**: Rejected — 4 unique targets exist but 3 lack PUCT data; only 1 row (023) has actionable data. **D (tablebase_coverage_insufficient)**: Rejected — all 24 rows have tablebase coverage. **E (too_small)**: Rejected — 24 rows is not small; the problem is signal quality, not quantity. | **Exclude sparse_endgame from training-family selection and rerun non-opening family mining v7.** Optional: run a tiny PUCT diagnostic on sparse_endgame-023 only (unique policy target with PUCT-suboptimal behavior) if search diagnosis is desired, but do not treat as a training family. |

### Decision rationale summary

**Primary reason**: 16 of 24 rows (67%) are forced-loss or forced-win positions where ALL legal moves have identical game-theoretic value. In these positions, any policy target is a tie-breaker artifact, not a meaningful training signal. This is not a defect in the references or the search — it is a fundamental property of the position. Training on arbitrary tie-breakers would teach the model to memorize noise.

**Secondary reason**: Of the remaining 8 rows, 4 are value-only ties (multiple optimal moves, ambiguous policy), and 4 are unique policy targets. But only 1 unique target (sparse_endgame-023) has PUCT data showing suboptimal behavior. The other 3 unique targets are unselected context rows without PUCT, neural, or child-value data. A training family of 4 rows (1 with real data) is too small for any practical training run.

**Do not:**
- Train on sparse_endgame as a family
- Run arena
- Promote a model
- Create replay or value-calibration artifacts
- Mutate the active reference fixture
- Apply the PR #65 patch artifact (all entries have `do_not_auto_apply = true`)

## 10. Exactly one recommended next action

**Recommendation: Exclude sparse_endgame from training-family selection and rerun non-opening family mining v7 to select the next candidate family.**

Rationale:
1. 67% of sparse_endgame rows are forced positions with arbitrary policy labels.
2. Only 4 unique policy targets exist; 3 lack PUCT data.
3. Only 1 row (023) has both unique optimal move and PUCT-suboptimal behavior — too small for a training family.
4. The remaining corrected_reference_suspicious (003, 007) and puct_child_search_value_mismatch (024) rows are not fixable — they are forced/tied positions where any "error" is a tie-breaker artifact.
5. The tablebase integration is clean: all 24 active references are optimal and 0 rows have tablebase conflicts. The tablebase is not the source of any remaining issue.
6. The time spent on sparse_endgame across PR #63, #64, #65, and #66 has thoroughly documented that this family is structurally unsuitable for policy training.

**Optional side action**: If the next family also proves problematic, sparse_endgame-023 could be included as a single-row PUCT search diagnostic in a broader search-quality evaluation — but NOT as a training target.
