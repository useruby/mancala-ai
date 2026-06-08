# AlphaZero-Lite Seat-Aware Promotion Gate Results

**Date:** 2026-06-08

## Summary

The seat-aware promotion gate evaluates candidates using forced seat-splits and
a search-budget sentinel matrix to distinguish true model strength from Kalah's
deterministic first-player advantage. This replaces the misleading single
alternating-seat arena score (always 0.50 under seat dominance) with a
multi-dimensional evaluation.

## Classification

The seat-aware protocol classifies candidates based on their disadvantaged-seat
(challenger as player 1) performance across four budget pairs:

| Budget Pair | Challenger Sims | Current Sims | Label |
|------------|-----------------|-------------|-------|
| 384:256 | 384 | 256 | standard (local promotion budget) |
| 1200:1200 | 1200 | 1200 | equal_high |
| 1200:256 | 1200 | 256 | challenger_high (challenger has search advantage) |
| 256:768 | 256 | 768 | current_high_asymmetry (current has search advantage) |

### Classification Categories

| Classification | Criteria |
|---------------|----------|
| `seat_artifact_only` | Alternating score is 0.50; disadvantaged-seat score is 0.00 at all budgets. No evidence of strength beyond seat advantage. |
| `high_search_breakthrough` | Disadvantaged-seat score improves only at 1200:1200 or 1200:256. Policy improvement exists but requires more search budget than the standard promotion protocol provides. |
| `standard_budget_breakthrough` | Disadvantaged-seat score improves at 384:256. Candidate is strong enough to overcome seat disadvantage at practical budgets. |
| `regression_masked_by_seat` | Alternating score is not 0.50 but margins, low-budget score, or current-high asymmetry are worse than current/baseline. Seat alternation masks a regression. |

## Re-Ranked Candidates (PR #93/#94)

### Ranking Table

| Rank | Candidate | Classification | Std Alt Score | Std DS Score | 1200:1200 DS | 1200:256 DS | 256:768 DS | Margin Mean | Latency p95 ms |
|------|-----------|---------------|---------------|--------------|-------------|------------|-----------|-------------|---------------|
| 1 | iter0_reference | high_search_breakthrough | 0.50 | 0.00 | **1.00** | 0.00 | 0.00 | -4.9 | 46.2 |
| 2 | iter1_continue_no_new_data | high_search_breakthrough | 0.50 | 0.00 | 0.50 | **1.00** | 0.00 | -6.8 | 46.5 |
| 3 | iter1_candidate_random_replay | seat_artifact_only | 0.50 | 0.00 | 0.00 | 0.00 | 0.00 | -6.8 | 46.5 |

### Detailed per-Candidate Analysis

#### iter0_reference: Leading Candidate
- **Artifact SHA256:** `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd`
- **Classification:** `high_search_breakthrough`
- **Evidence:**
  - At 1200:1200 equal budget: wins from BOTH seats (disadvantaged-seat score = 1.00)
  - At 384:256 standard: seat artifact dominates (disadvantaged-seat score = 0.00)
  - At 256:768 asymmetry: wins from seat 0 despite being out-searched (disadvantaged-seat score = 0.00)
  - At 256:256 equal: survives without collapse (vs iter1 candidates that collapse)
  - Margin mean: -4.9 (smallest negative margin of all candidates)
  - Latency p95: 46.2 ms (acceptable)
- **Decision:** Leading candidate. Retains PR #97 1200:1200 both-seat breakthrough. Only candidate that survives low-budget (256:256) without total collapse. Acceptable latency.
- **Caveat:** Breakthrough requires 1200:1200 search — 4.7x the standard challenger budget. The standard 384:256 budget does not reveal this strength.

#### iter1_continue_no_new_data: Promising
- **Artifact SHA256:** `a4a86dceb064a2be63d186785e9e3e30e5e5bed52bd4be2fb878d42edc45a5a9`
- **Classification:** `high_search_breakthrough`
- **Evidence:**
  - At 1200:256 asymmetric: wins from BOTH seats (disadvantaged-seat score = 1.00)
  - At 1200:1200 equal: partial breakthrough (disadvantaged-seat score = 0.50)
  - At 384:256 standard: seat artifact dominates (disadvantaged-seat score = 0.00)
  - At 256:768 asymmetry: loses from both seats (0.00 score)
  - At 256:256 equal: collapses completely (0.00 score)
  - Margin mean: -6.8 (worse than iter0_reference)
  - Latency p95: 46.5 ms
- **Decision:** Promising but behind iter0_reference. Its 1200:256 both-seat breakthrough is reproducible (matches PR #97). However, low-budget collapses at 256:256 and 256:768 are worse than iter0_reference. Second-ranked.
- **Caveat:** Only wins from disadvantaged seat when given search advantage (1200 vs 256). At equal budgets, only iter0_reference breaks through.

#### iter1_candidate_random_replay: Rejected
- **Artifact SHA256:** `777b25d4c5a601ff4b3cdc3a750550f2fee2756da296ad15c300b0d749b61b44`
- **Classification:** `seat_artifact_only`
- **Evidence:**
  - All disadvantaged-seat scores are 0.00 across all four budget pairs
  - Never exceeds 0.50 alternating score at any budget
  - At 256:768: collapses to 0.00 from both seats
  - Margin mean: -6.8 (worst, tied with iter1_continue)
  - No breakthrough signal at any budget
- **Decision:** Rejected. Adding candidate-mined data made the model strictly worse than continuing on teacher data alone. Shows no disadvantaged-seat breakthrough.
- **Caveat:** Raw policy differs from current (38% top-move disagreement), but this difference does not translate to arena strength.

## Search-Budget Sentinel Matrix

### How the matrix works

Four budget pairs span the evaluation space:
1. **standard (384:256):** The current local promotion protocol. Tests whether
   policy improvement is detectable at practical search budgets.
2. **equal_high (1200:1200):** Tests whether high-quality search reveals
   underlying policy improvements masked by budget constraints.
3. **challenger_high (1200:256):** Tests whether a search advantage lets the
   new policy overcome the seat disadvantage. A positive signal here means the
   policy is directionally better but not strong enough for equal budgets.
4. **current_high_asymmetry (256:768):** Sentinel for regressions. If the
   candidate collapses from both seats when current has a search advantage, the
   learned policy may be fragile.

### Decision Rules

| Rule | Application |
|------|------------|
| Promote iter0_reference? | Only if it retains 1200:1200 both-seat breakthrough AND survives 256:768 asymmetry. ✓ Met. |
| Consider iter1_continue? | Only if 1200:256 both-seat breakthrough is reproducible AND low-budget collapses are not worse than iter0_reference by enough to matter. Partial — collapses are worse. |
| Reject iter1_candidate_random_replay? | Unless it shows a disadvantaged-seat breakthrough that was absent in PR #97. ✗ No breakthrough. Rejected. |

## Legacy Metric Warning

The alternating-seat arena score (always 0.50 under seat dominance) is reported
for backward compatibility but should not be used as a promotion criterion. The
0.55 threshold is unreachable at 384:256 under deterministic first-player
dominance — see [PR #97 findings](alphazero-lite-arena-ceiling-diagnostic-results.md).

## Artifacts

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| iter1_continue_no_new_data | /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data/artifact | `a4a86dceb064a2be63d186785e9e3e30e5e5bed52bd4be2fb878d42edc45a5a9` |
| iter1_candidate_random_replay | /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay/artifact | `777b25d4c5a601ff4b3cdc3a750550f2fee2756da296ad15c300b0d749b61b44` |

## Protocol Changes

### New Tools

1. **`script/ai/seat_aware_promotion_gate`** — Standalone seat-aware evaluation
   script. Runs arenas at multiple budget pairs with forced seat-splits,
   computes disadvantaged-seat scores, and classifies candidates.

2. **`script/ai/seat_aware_rerank`** — Re-ranks candidates from cached arena
   JSON/JSONL data. Does not run new arenas; reads existing diagnostic output.

3. **`script/ai/local_promotion_gate --seat-aware`** — Optional flag on the
   existing local promotion gate. When set, runs seat-aware evaluation as a
   subprocess and embeds results in the report. Default behavior unchanged.

### New Module

4. **`ml/alphazero_lite/seat_aware_arena.py`** — Shared module with:
   - `compute_seat_split_metrics()` — per-game JSONL → seat-aware aggregates
   - `classify_candidate()` — budget matrix → classification label
   - `build_seat_aware_report()` — arena results → structured report
   - `build_candidate_ranking()` — multi-candidate reports → ranked table

### Tests

5. **`ml/alphazero_lite/tests/test_seat_aware_arena.py`** — 26 tests covering:
   - Forced-seat metrics computed correctly
   - Alternating-seat score remains backward compatible
   - Disadvantaged-seat score computed correctly
   - Candidate classification on synthetic arena summaries
   - Ranking sorts by disadvantaged-seat score
   - Budget pair label mapping

## Guardrails

- No training performed.
- No model promoted.
- `model-artifact/current` and `storage/ai/alphazero_lite/current` not overwritten.
- Default `local_promotion_gate` behavior unchanged without `--seat-aware` flag.
- No promotion thresholds changed globally.
- Legacy 0.55 alternating-seat metric preserved and flagged as misleading under seat dominance.
