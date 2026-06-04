# Fresh Endgame Tablebase Unique — Diagnostics Results

**Date:** 2026-06-04
**Family:** `fresh_endgame_tablebase_unique`
**Source:** 14 rows from `/tmp/azlite_fresh_hard_state_mining_teacher_filtered/selected_fresh_family_rows.jsonl`
**Artifact:** `storage/ai/alphazero_lite/current`
**Script:** `ml/alphazero_lite/run_fresh_endgame_tablebase_unique_diagnostics.py`

## Summary

| Metric | Count |
|---|---|
| Selected rows | 14 |
| Valid | 14 |
| Target candidates | 0 |
| Preservation controls | 2 |
| Holdouts | 12 |
| Excluded | 0 |
| **Decision** | `tablebase_unique_too_small` |
| **Next action** | Mine more fresh exact-tablebase unique rows |

All 14 rows pass PUCT at all search budgets (128–5000). No rows exhibit value rank errors, sign errors, or child PUCT failures. With zero target candidates, the family is too small to support a diagnostic artifact.

## Exact Tablebase Enumeration

| Signal Class | Count |
|---|---|
| `exact_unique_clear_margin` | 6 |
| `exact_unique_but_forced_all_bad` | 8 |

All 14 rows have a unique tablebase-optimal move. 8 rows have only one legal move after recursion (all children produce the same value as the lone legal move — "forced all bad"). 6 rows show a clear value margin between the optimal move and the second-best alternative (margins of 1.0 or 2.0).

## PUCT Baseline (5 budgets × 14 rows)

**100% pass rate across all budgets.**

Row-level classification:

| Classification | Count |
|---|---|
| `passes_at_high_budget` | 14 |
| `persistent_failure` | 0 |
| `low_budget_only_failure` | 0 |

Even at budget 128, every row selects the tablebase-optimal move. For the 8 "forced all bad" rows the visit share is 100% (only one distinct child value), and for the 6 "clear margin" rows the optimal visit share is 92–99% with the remainder going to equivalent-value moves.

## Neural Value Audit

| Classification | Count |
|---|---|
| `value_error_small` | 14 |
| `value_head_prefers_wrong_child` | 0 |
| `value_head_underestimates_optimal_child` | 0 |
| Value rank errors | 0 |
| Sign errors | 0 |

Root value errors range from 0.28 to 0.99 (in absolute terms). These are not negligible, but they never cause a rank inversion or sign error, and PUCT's search compensates for them fully.

## Child PUCT Audit

Not run — no row has a persistent PUCT failure at the root.

## Root Counterfactual Diagnostics

Not run — no row has a persistent PUCT failure at the root.

## Clean Split

| Role | Count |
|---|---|
| Target candidate | 0 |
| Preservation control | 2 |
| Holdout | 12 |
| Excluded | 0 |

## Decision

**`tablebase_unique_too_small`** — Dominant mechanism: **`too_few_targets`**.

No row in the current 14-row sample fails PUCT. All are trivially solvable by the existing network and search. To find rows where PUCT selects a suboptimal move, the miner must be adjusted to find harder endgame-tablebase-unique states — likely deeper into the game tree where the value head's errors are large enough to flip the search outcome.

## Guardrail Compliance

- Created replay artifacts: No
- Mutated active fixture: No
- Promoted model: No
- Ran arena: No
- Ran training: No

## Outputs

- `/tmp/azlite_fresh_endgame_tablebase_unique_diagnostics/fresh_endgame_tablebase_unique_diagnostics_summary.json`
- `/tmp/azlite_fresh_endgame_tablebase_unique_diagnostics/fresh_endgame_tablebase_unique_row_diagnostics.jsonl`
- `/tmp/azlite_fresh_endgame_tablebase_unique_diagnostics/fresh_endgame_tablebase_unique_clean_split.json`
