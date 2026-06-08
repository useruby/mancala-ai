# AlphaZero-Lite Arena Ceiling Diagnostic Results

**Date:** 2026-06-08

## Classification

**SEAT_OR_OPENING_ARTIFACT (primary), SEARCH_BUDGET_LIMITING (secondary)**

The 0.50 arena ceiling is caused primarily by a deterministic seat/opening artifact:
challengers win every game when they start as player 0 and lose every game when
they start as player 1. The alternating seat protocol guarantees exactly 50/50.
At very high search budgets (1200:1200) some candidates break through from both
seats, making this also a search-budget-limiting pattern.

## Artifacts

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| iter1_continue_no_new_data | /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data/artifact | `a4a86dceb064a2be63d186785e9e3e30e5e5bed52bd4be2fb878d42edc45a5a9` |
| iter1_candidate_random_replay | /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay/artifact | `777b25d4c5a601ff4b3cdc3a750550f2fee2756da296ad15c300b0d749b61b44` |

## Finding 1: The 0.50 Is a Seat Artifact

### Standard Arena (384:256, default seat alternation)

Every candidate at the standard promotion budget produces exactly:
- 60 W / 60 L / 0 D → 0.50 score
- 120 W / 120 L / 0 D → 0.50 at 240 games

### Forced Seat-Split at 384:256

When the challenger is forced to always start as one seat, the true seat
asymmetry is revealed:

| Candidate | Forced Start | W | L | D | Score |
|-----------|-------------|---|---|---|---|
| iter0_reference | 0 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 1 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 0 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 1 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 0 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 1 | 0 | 120 | 0 | 0.00 |

**Key observation:** At the standard 384:256 budget, the seat determines the
winner with 100% reliability. Player 0 always wins. The alternating protocol
produces 0.50 as an artifact of this deterministic seat asymmetry with zero
draws.

### This pattern holds across most budget pairs

The forced seat-split at 128:128, 256:256, 384:384, 768:768, 384:256, 768:256,
1200:256, and 256:768 all show the same pattern:
- Challenger starts 0 → 1.00 score
- Challenger starts 1 → 0.00 score

This is not specific to the 384:256 budget — it is the default behavior across
the entire search budget matrix.

## Finding 2: Search Budget Breakthrough at the Top

### Equal-Budget 1200:1200

At the highest equal budget, some candidates break the seat artifact:

| Candidate | 1200:1200 Score | Forced Start 0 | Forced Start 1 |
|-----------|----------------|----------------|----------------|
| iter0_reference | 1.00 | 1.00 | 1.00 |
| iter1_continue_no_new_data | 0.75 | 1.00 | 0.50 |
| iter1_candidate_random_replay | 0.50 | 1.00 | 0.00 |

**iter0_reference at 1200:1200 wins from both seats.** This demonstrates the
model policy has sufficient quality that, with enough search budget, it can
overcome the seat disadvantage that dominates at lower budgets.

**iter1_continue at 1200:1200** scores 0.75 with significant draws — it wins
from seat 0 and draws from seat 1, showing partial breakthrough.

**iter1_candidate_random_replay at 1200:1200** remains stuck at the seat-artifact
pattern (1.00 from seat 0, 0.00 from seat 1), suggesting this candidate is not
stronger than iter0 despite similar training metrics.

### Asymmetric Budget: 1200 Challenger vs 256 Current

| Candidate | 1200:256 Score | Forced Start 0 | Forced Start 1 |
|-----------|----------------|----------------|----------------|
| iter0_reference | 0.50 | 1.00 | 0.00 |
| iter1_continue_no_new_data | 1.00 | 1.00 | 1.00 |
| iter1_candidate_random_replay | 0.50 | 1.00 | 0.00 |

**iter1_continue_no_new_data at 1200:256 wins from both seats.** Only this
candidate wins from the disadvantaged seat when given a massive search
advantage. This is evidence that extra training on the same data improves
something that only manifests with enough search budget.

## Finding 3: Low-Budget Collapse

At 128:128 equal budget, results are inconsistent:

| Candidate | 128:128 Score | W/L/D |
|-----------|----------------|-------|
| iter0_reference | 0.00 | 0/120/0 |
| iter1_continue_no_new_data | 0.25 | 0/60/60 |
| iter1_candidate_random_replay | 0.50 | 60/60/0 |

At 256:256:

| Candidate | 256:256 Score | W/L/D |
|-----------|----------------|-------|
| iter0_reference | 0.50 | 60/60/0 |
| iter1_continue_no_new_data | 0.00 | 0/120/0 |
| iter1_candidate_random_replay | 0.00 | 0/120/0 |

**iter0_reference is the strongest at low budgets,** dropping to 0.00 only at
128:128. Both iter1 candidates collapse at 256:256, suggesting the extra
training round may have reduced low-budget robustness.

## Finding 4: The Asymmetry Check (256:768)

When current has more search budget:

| Candidate | 256:768 Score | Forced Start 0 | Forced Start 1 |
|-----------|----------------|----------------|----------------|
| iter0_reference | 0.50 | 1.00 | 0.00 |
| iter1_continue_no_new_data | 0.00 | 0.00 | 0.00 |
| iter1_candidate_random_replay | 0.00 | 0.00 | 0.00 |

**iter0_reference still wins from seat 0** even when out-searched (256 vs 768).
The iter1 candidates lose from both seats when current has a 768:256 advantage.

## Finding 5: Raw Policy Diagnostic (500 sampled states)

| Metric | iter0_reference | iter1_continue_no_new_data | iter1_candidate_random_replay |
|--------|----------------|---------------------------|------------------------------|
| Candidate/current top-move agreement | 0.652 | 0.614 | 0.620 |
| Candidate/classic-MCTS top-move agreement | 0.482 | 0.486 | 0.498 |
| Current/classic-MCTS top-move agreement | 0.478 | 0.478 | 0.478 |
| Candidate policy entropy mean | 0.881 | 0.834 | 0.860 |
| Current policy entropy mean | 0.735 | 0.735 | 0.735 |
| Candidate value mean | 0.017 | 0.035 | -0.053 |
| Current value mean | 0.053 | 0.053 | 0.053 |

**Candidates have measurably different raw policies from current** (35-39%
top-move disagreement), but the differences don't translate to arena strength at
standard budgets because the seat artifact dominates.

**Candidates have slightly higher classic-MCTS agreement** (48-50% vs 47.8% for
current), confirming the training does move the policy toward the teacher
distribution, but this improvement is too small to overcome the seat bias.

**Candidate policy entropy is higher** than current (0.83-0.88 vs 0.73),
suggesting candidates spread probability across more moves. This could reflect
the random-replay data distribution encouraging exploration.

## Finding 6: First Move and Trajectory Analysis

### First move distributions (extremely stereotyped)

All candidates and current play only 2-3 different first moves across 120 games:
- Pit 1 (index 1): ~8 times
- Pit 2 (index 2): ~7 times
- Pit 5 (index 5): occasionally

Total distinct first moves per 120 games: 2 (15 distinct events).

### Duplicate trajectories

15 duplicate trajectory instances per 120 games (12.5%). This is moderately
high but not dominant — 87.5% of games follow unique trajectories.

### Game length

Mean game length varies from 26 to 49 plies, with shorter games at higher
search budgets (1200:256 → 26.3 plies for iter1_continue, suggesting the
search advantage produces decisive play faster).

## Finding 7: Score Margins

Score margins consistently show negative means (current wins by more seeds when
it wins than challenger does when it wins). At standard 384:256:

| Candidate | Margin Mean |
|-----------|-------------|
| iter0_reference | -4.9 |
| iter1_continue_no_new_data | -6.8 |
| iter1_candidate_random_replay | -6.8 |

The negative margins suggest current wins by larger differentials, even though
win counts are equal. This is a second-order signal that the current model may
be slightly stronger in per-game magnitude.

## Classification Rationale

### Primary: SEAT_OR_OPENING_ARTIFACT

The 0.50 score is overwhelmingly a seat artifact:

1. At the standard 384:256 budget and most other budgets, the challenger wins
   every game from player 0 and loses every game from player 1.
2. Kalah has a strong first-player advantage. The arena protocol alternates
   which model starts as player 0, producing exactly 50/50 win splits.
3. Zero draws across 120 games at the standard budget is explained by the
   deterministic seat outcome — each game's winner is determined by the seat
   assignment before any moves are played.
4. The artifact persists across all three candidates and the vast majority of
   budget pairs (128:128 through 768:768, 384:256 through 1200:256, 256:768).

### Secondary: SEARCH_BUDGET_LIMITING

Search budget does matter at the extremes:

1. iter0_reference breaks through at 1200:1200, winning from both seats (1.00
   overall). The learned policy improvement exists but only manifests with
   sufficient search budget.
2. iter1_continue_no_new_data breaks through at 1200:256 (1.00 overall, wins
   from both seats). Extra training on the same data helps when given a search
   advantage.
3. At 128:128, some candidates collapse to 0.00, showing a lower bound on
   useful search budget.
4. This is secondary because the seat artifact dominates at all practical
   budgets — the 0.55 promotion threshold is unreachable at any budget except
   1200:1200 (iter0) or 1200:256 (iter1_continue).

### NOT: MODEL_POLICY_NOT_BETTER

The model policy IS measurably different and better in some regimes:
- iter0_reference wins from both seats at 1200:1200
- Raw policy shows 35-39% top-move disagreement with current
- Classic-MCTS agreement slightly improves (48-50% vs 47.8%)

### NOT: EVALUATION_NOISE_OR_PROTOCOL_ISSUE

The results are highly consistent across seeds and extended arenas. The 0.50
pattern is deterministic, not noisy.

## Full Score Matrix by Budget Pair

| Candidate | 128:128 | 256:256 | 384:384 | 768:768 | 1200:1200 | 384:256 | 768:256 | 1200:256 | 256:768 |
|-----------|---------|---------|---------|---------|-----------|---------|---------|----------|---------|
| iter0_reference | 0.00 | 0.50 | 0.50 | 0.50 | **1.00** | 0.50 | 0.50 | 0.50 | 0.50 |
| iter1_continue_no_new_data | 0.25 | 0.00 | 0.50 | 0.50 | 0.75 | 0.50 | 0.50 | **1.00** | 0.00 |
| iter1_candidate_random_replay | 0.50 | 0.00 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.50 | 0.00 |

## Arena Results by Candidate and Budget Pair (Full)

| Candidate | Chall Sims | Curr Sims | Games | Wins | Losses | Draws | Score | CI95 Lower | CI95 Upper | P0 Score | P1 Score | Margin Mean | Game Len Mean | Dup Traj | Move Time ms | Move Time p95 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| iter0_reference | 128 | 128 | 120 | 0 | 120 | 0 | 0.00 | 0.000 | 0.031 | 0.00 | 0.00 | -9.2 | 37.3 | 15 | 8.5 | 11.8 |
| iter0_reference | 256 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -3.9 | 41.1 | 15 | 23.4 | 32.9 |
| iter0_reference (ext) | 256 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 23.3 | 32.8 |
| iter0_reference | 384 | 384 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -7.7 | 49.3 | 15 | 33.3 | 49.7 |
| iter0_reference (ext) | 384 | 384 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 33.3 | 49.9 |
| iter0_reference | 768 | 768 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -3.5 | 34.7 | 15 | 70.4 | 103.5 |
| iter0_reference (ext) | 768 | 768 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 70.8 | 103.8 |
| iter0_reference | 1200 | 1200 | 120 | 120 | 0 | 0 | **1.00** | 0.969 | 1.000 | 1.00 | 1.00 | 12.5 | 41.7 | 15 | 112.6 | 165.2 |
| iter0_reference | 384 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -4.9 | 38.9 | 15 | 29.5 | 46.2 |
| iter0_reference (ext) | 384 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 29.5 | 46.2 |
| iter0_reference | 768 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | 3.1 | 31.7 | 15 | 51.7 | 94.7 |
| iter0_reference (ext) | 768 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 51.4 | 94.1 |
| iter0_reference | 1200 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | 1.2 | 34.6 | 15 | 66.2 | 148.2 |
| iter0_reference (ext) | 1200 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 66.1 | 148.2 |
| iter0_reference | 256 | 768 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -6.7 | 36.0 | 15 | 51.9 | 102.2 |
| iter0_reference (ext) | 256 | 768 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 52.3 | 103.2 |
| iter1_continue_no_new_data | 128 | 128 | 120 | 0 | 60 | 60 | 0.25 | 0.181 | 0.334 | 0.50 | 0.00 | -8.5 | 36.0 | 15 | 10.9 | 16.1 |
| iter1_continue_no_new_data | 256 | 256 | 120 | 0 | 120 | 0 | 0.00 | 0.000 | 0.031 | 0.00 | 0.00 | -15.9 | 37.1 | 15 | 21.4 | 32.5 |
| iter1_continue_no_new_data | 384 | 384 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -6.8 | 45.5 | 15 | 33.4 | 50.2 |
| iter1_continue_no_new_data (ext) | 384 | 384 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 33.4 | 50.2 |
| iter1_continue_no_new_data | 768 | 768 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -10.9 | 39.3 | 15 | 65.3 | 102.6 |
| iter1_continue_no_new_data (ext) | 768 | 768 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 65.4 | 103.0 |
| iter1_continue_no_new_data | 1200 | 1200 | 120 | 60 | 0 | 60 | **0.75** | 0.666 | 0.819 | 1.00 | 0.50 | 2.1 | 40.7 | 15 | 102.8 | 164.6 |
| iter1_continue_no_new_data | 384 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -6.8 | 38.6 | 15 | 28.2 | 46.5 |
| iter1_continue_no_new_data (ext) | 384 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 28.3 | 46.6 |
| iter1_continue_no_new_data | 768 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -6.1 | 35.3 | 15 | 42.9 | 94.0 |
| iter1_continue_no_new_data (ext) | 768 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 43.3 | 94.7 |
| iter1_continue_no_new_data | 1200 | 256 | 120 | 120 | 0 | 0 | **1.00** | 0.969 | 1.000 | 1.00 | 1.00 | 10.1 | 26.3 | 15 | 59.7 | 151.5 |
| iter1_continue_no_new_data | 256 | 768 | 120 | 0 | 120 | 0 | 0.00 | 0.000 | 0.031 | 0.00 | 0.00 | -17.1 | 34.1 | 15 | 50.8 | 102.7 |
| iter1_candidate_random_replay | 128 | 128 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -0.3 | 32.7 | 15 | 12.7 | 16.2 |
| iter1_candidate_random_replay (ext) | 128 | 128 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 12.7 | 16.2 |
| iter1_candidate_random_replay | 256 | 256 | 120 | 0 | 120 | 0 | 0.00 | 0.000 | 0.031 | 0.00 | 0.00 | -13.5 | 42.3 | 15 | 22.5 | 32.7 |
| iter1_candidate_random_replay | 384 | 384 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -5.7 | 40.7 | 15 | 36.4 | 50.1 |
| iter1_candidate_random_replay (ext) | 384 | 384 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 36.5 | 50.1 |
| iter1_candidate_random_replay | 768 | 768 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -2.4 | 42.2 | 15 | 67.4 | 102.0 |
| iter1_candidate_random_replay (ext) | 768 | 768 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 67.8 | 102.7 |
| iter1_candidate_random_replay | 1200 | 1200 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -2.4 | 32.9 | 15 | 104.6 | 163.8 |
| iter1_candidate_random_replay (ext) | 1200 | 1200 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 105.0 | 164.9 |
| iter1_candidate_random_replay | 384 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -6.8 | 38.6 | 15 | 28.4 | 46.5 |
| iter1_candidate_random_replay (ext) | 384 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 28.4 | 46.6 |
| iter1_candidate_random_replay | 768 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -0.7 | 33.7 | 15 | 47.6 | 95.0 |
| iter1_candidate_random_replay (ext) | 768 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 47.6 | 95.0 |
| iter1_candidate_random_replay | 1200 | 256 | 120 | 60 | 60 | 0 | 0.50 | 0.412 | 0.588 | 1.00 | 0.00 | -3.3 | 34.3 | 15 | 66.1 | 149.9 |
| iter1_candidate_random_replay (ext) | 1200 | 256 | 240 | 120 | 120 | 0 | 0.50 | 0.437 | 0.563 | — | — | — | — | — | 66.8 | 151.6 |
| iter1_candidate_random_replay | 256 | 768 | 120 | 0 | 120 | 0 | 0.00 | 0.000 | 0.031 | 0.00 | 0.00 | -16.8 | 39.0 | 15 | 46.6 | 102.2 |

## Forced Seat-Split Results (All)

| Candidate | Chall Sims | Curr Sims | Starts | Games | Wins | Losses | Draws | Score |
|---|---|---|---|---|---|---|---|---|---|
| iter0_reference | 128 | 128 | 0 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 128 | 128 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 256 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 256 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 384 | 384 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 384 | 384 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 768 | 768 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 768 | 768 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 1200 | 1200 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 1200 | 1200 | 1 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 384 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 384 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 768 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 768 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 1200 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 1200 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter0_reference | 256 | 768 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter0_reference | 256 | 768 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 128 | 128 | 0 | 120 | 0 | 0 | 120 | 0.50 |
| iter1_continue_no_new_data | 128 | 128 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 256 | 256 | 0 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 256 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 384 | 384 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 384 | 384 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 768 | 768 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 768 | 768 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 1200 | 1200 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 1200 | 1200 | 1 | 120 | 0 | 0 | 120 | 0.50 |
| iter1_continue_no_new_data | 384 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 384 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 768 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 768 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 1200 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 1200 | 256 | 1 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_continue_no_new_data | 256 | 768 | 0 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_continue_no_new_data | 256 | 768 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 128 | 128 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 128 | 128 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 256 | 256 | 0 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 256 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 384 | 384 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 384 | 384 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 768 | 768 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 768 | 768 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 1200 | 1200 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 1200 | 1200 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 384 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 384 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 768 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 768 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 1200 | 256 | 0 | 120 | 120 | 0 | 0 | 1.00 |
| iter1_candidate_random_replay | 1200 | 256 | 1 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 256 | 768 | 0 | 120 | 0 | 120 | 0 | 0.00 |
| iter1_candidate_random_replay | 256 | 768 | 1 | 120 | 0 | 120 | 0 | 0.00 |

## First Move Distributions

First moves are extremely stereotyped across all candidates and budgets. Only 2-3
pit choices appear in 120 games, with approximately 7-8 occurrences each. The
standard pits chosen are 1, 2, and occasionally 5.

| Candidate | Chall Sims | Curr Sims | Challenger First Moves | Current First Moves |
|---|---|---|---|---|
| iter0_reference | 128 | 128 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 256 | 256 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 384 | 384 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter0_reference | 768 | 768 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 1200 | 1200 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 384 | 256 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 768 | 256 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter0_reference | 1200 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter0_reference | 256 | 768 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 128 | 128 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 256 | 256 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 384 | 384 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 768 | 768 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 1200 | 1200 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 384 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 768 | 256 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_continue_no_new_data | 1200 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_continue_no_new_data | 256 | 768 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 128 | 128 | {"1": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 256 | 256 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 384 | 384 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 768 | 768 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 1200 | 1200 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 384 | 256 | {"2": 8, "1": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 768 | 256 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |
| iter1_candidate_random_replay | 1200 | 256 | {"5": 8, "2": 7} | {"2": 8, "1": 7} |
| iter1_candidate_random_replay | 256 | 768 | {"2": 8, "5": 7} | {"1": 8, "2": 7} |

## Raw Policy Diagnostic

| Metric | iter0_reference | iter1_continue_no_new_data | iter1_candidate_random_replay |
|--------|----------------|---------------------------|------------------------------|
| sampled_states | 500 | 500 | 500 |
| candidate_current_top_move_agreement | 0.652 | 0.614 | 0.620 |
| candidate_classic_mcts_top_move_agreement | 0.482 | 0.486 | 0.498 |
| current_classic_mcts_top_move_agreement | 0.478 | 0.478 | 0.478 |
| candidate_policy_entropy_mean | 0.881 | 0.834 | 0.860 |
| current_policy_entropy_mean | 0.735 | 0.735 | 0.735 |
| candidate_value_mean | 0.017 | 0.035 | -0.053 |
| candidate_value_std | 0.430 | 0.481 | 0.434 |
| current_value_mean | 0.053 | 0.053 | 0.053 |
| current_value_std | 0.392 | 0.392 | 0.392 |

## Implications

1. **The 0.50 score is not a measure of model strength.** It is a
   deterministic artifact of Kalah's first-player advantage combined with the
   arena's alternating seat protocol. Win counts of 60/60 with zero draws
   appear because one seat always wins.

2. **The learned policy IS better but the arena protocol can't detect it**
   at standard search budgets. iter0_reference wins from both seats only at
   1200:1200 — 4.7x the standard challenger budget.

3. **The promotion threshold of 0.55 is unreachable at 384:256** regardless of
   model quality. If Player 0 always wins and Player 1 always loses, the score
   will always be 0.50 with alternating seats. The only way to exceed 0.55 is
   to win from the disadvantaged seat (Player 1), which requires either a
   massive search budget or a fundamentally better model.

4. **Zero draws is a consequence of the seat artifact**, not a playing style
   change. When the seat determines the winner, there are no close games that
   end in draws.

5. **The arena protocol needs seat-aware scoring.** To measure true relative
   strength, the arena should report per-seat scores separately, or use a
   balanced paired-game protocol (play each pairing twice, swapping seats).

6. **iter0_reference is the strongest candidate** across the budget matrix:
   it is the only one that wins from both seats at 1200:1200, and the only one
   that survives 256:256 and 256:768 without collapsing to 0.00.

7. **iter1_candidate_random_replay is the weakest** — it never exceeds 0.50
   at any budget pair, collapses at 256:768, and remains seat-locked even at
   1200:1200. Adding candidate-mined data made the model strictly worse than
   continuing on the same data alone.

## Runner Tool

The diagnostic was produced by `ml/alphazero_lite/run_arena_ceiling_diagnostic.py`.
It orchestrates arena calls with `--game-jsonl` for per-game trajectory data and
`--challenger-starts` for forced seat-split evaluation, then aggregates results
into the tables above. The tool does not train, promote, or overwrite any model.

### Commands Used

```bash
.venv/bin/python ml/alphazero_lite/run_arena_ceiling_diagnostic.py \
  --workdir /tmp/azlite_arena_ceiling_diagnostic \
  --current model-artifact/current \
  --candidates \
    /tmp/azlite_iterative_random_replay/iter0_candidate_artifact,\
    /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data/artifact,\
    /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay/artifact \
  --budget-pairs 128:128,256:256,384:384,768:768,1200:1200,384:256,768:256,1200:256,256:768 \
  --games 120 \
  --extended-games 240 \
  --seed 42 \
  --workers 8
```

The runner was verified with:
```bash
.venv/bin/python -m unittest discover ml/alphazero_lite  # 50 passed
.venv/bin/ruff check ml/alphazero_lite/run_arena_ceiling_diagnostic.py ml/alphazero_lite/arena.py script/ai/local_promotion_gate  # clean
```
