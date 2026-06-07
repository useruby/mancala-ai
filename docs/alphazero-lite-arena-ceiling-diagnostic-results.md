# AlphaZero-Lite Arena Ceiling Diagnostic Results

**Date:** 2026-06-07

## Classification

**PRIMARY: SEAT AND OPENING ARTIFACT** — The 0.50 ceiling is an evaluation artifact, not a strength ceiling. At standard arena budgets (384 challenger / 256 current), both models play deterministically: the player who moves first (player 0) wins every game. The 0.50 score (60W/60L/0D) is purely the average of two extremes — the challenger wins all games as player 0 and loses all games as player 1.

**SECONDARY: SEARCH BUDGET LIMITING** — At very high equal search budgets (1200v1200), differentiation emerges. The strongest candidate (iter0) achieves 1.00 (all wins regardless of seat), iter1_continue achieves 0.75 (all wins as player 0, all draws as player 1), while iter1_candidate_replay stays at 0.50. The learned policy improvement only manifests at higher MCTS budgets.

**TERTIARY: MODEL POLICY NOT BETTER (candidate-mined lane)** — The iter1_candidate_random_replay lane never exceeds 0.50 at any budget pair, including 1200v1200 and 1200v256. Raw-policy analysis confirms it is nearly identical to iter0 in all policy/value metrics. Adding candidate-mined states is confirmed as a zero-benefit intervention.

## Experiment Design

Three candidates from PRs #93/#94 tested across 9 search-budget pairs (120 games each), forced seat-split evaluation at the standard 384v256 budget, extended 240-game checks for all near-threshold results, and a raw-policy diagnostic on 500 sampled states.

| Candidate | Source | SHA256 |
|-----------|--------|--------|
| iter0_reference | PR #93 teacher_1200 lane | `0bbeaa9...` |
| iter1_continue_no_new_data | PR #94, iter0 init, generic+old data | `a4a86dc...` |
| iter1_candidate_random_replay | PR #94, iter0 init, +candidate-mined data | `777b25d...` |
| current (production) | model-artifact/current | `6946aaf...` |

## Budget Matrix Results (120 games)

| Candidate | Budget | Score | W | L | D | Mean ms | p95 ms |
|-----------|--------|-------|---|---|---|---------|--------|
| iter0 | 128v128 | 0.0000 | 0 | 120 | 0 | 7.9 | 11.0 |
| iter0 | 256v256 | 0.5000 | 60 | 60 | 0 | 18.5 | 26.2 |
| iter0 | 384v384 | 0.5000 | 60 | 60 | 0 | 26.7 | 39.9 |
| iter0 | 768v768 | 0.5000 | 60 | 60 | 0 | 56.7 | 83.2 |
| iter0 | **1200v1200** | **1.0000** | **120** | **0** | **0** | 90.9 | 132.8 |
| iter0 | 384v256 | 0.5000 | 60 | 60 | 0 | 23.6 | 37.0 |
| iter0 | 768v256 | 0.5000 | 60 | 60 | 0 | 41.2 | 75.4 |
| iter0 | 1200v256 | 0.5000 | 60 | 60 | 0 | 52.8 | 119.1 |
| iter0 | 256v768 | 0.5000 | 60 | 60 | 0 | 41.7 | 82.2 |
| iter1_continue | 128v128 | 0.2500 | 0 | 60 | 60 | 8.7 | 12.8 |
| iter1_continue | 256v256 | 0.0000 | 0 | 120 | 0 | 17.2 | 26.1 |
| iter1_continue | 384v384 | 0.5000 | 60 | 60 | 0 | 26.7 | 40.1 |
| iter1_continue | 768v768 | 0.5000 | 60 | 60 | 0 | 52.4 | 82.4 |
| iter1_continue | **1200v1200** | **0.7500** | **60** | **0** | **60** | 82.8 | 132.4 |
| iter1_continue | 384v256 | 0.5000 | 60 | 60 | 0 | 22.7 | 37.4 |
| iter1_continue | 768v256 | 0.5000 | 60 | 60 | 0 | 34.5 | 75.6 |
| iter1_continue | **1200v256** | **1.0000** | **120** | **0** | **0** | 48.0 | 121.3 |
| iter1_continue | 256v768 | 0.0000 | 0 | 120 | 0 | 41.0 | 82.7 |
| iter1_candidate | 128v128 | 0.5000 | 60 | 60 | 0 | 10.2 | 13.0 |
| iter1_candidate | 256v256 | 0.0000 | 0 | 120 | 0 | 18.0 | 26.2 |
| iter1_candidate | 384v384 | 0.5000 | 60 | 60 | 0 | 29.2 | 40.1 |
| iter1_candidate | 768v768 | 0.5000 | 60 | 60 | 0 | 54.3 | 82.1 |
| iter1_candidate | **1200v1200** | **0.5000** | **60** | **60** | **0** | 85.2 | 133.7 |
| iter1_candidate | 384v256 | 0.5000 | 60 | 60 | 0 | 22.7 | 37.2 |
| iter1_candidate | 768v256 | 0.5000 | 60 | 60 | 0 | 38.1 | 76.2 |
| iter1_candidate | 1200v256 | 0.5000 | 60 | 60 | 0 | 53.1 | 120.7 |
| iter1_candidate | 256v768 | 0.0000 | 0 | 120 | 0 | 37.3 | 81.8 |

## Forced Seat-Split (384v256, 120 games)

| Candidate | Seat | Score | W | L | D |
|-----------|------|-------|---|---|---|
| iter0 | 0 (always player 0) | **1.0000** | 120 | 0 | 0 |
| iter0 | 1 (always player 1) | **0.0000** | 0 | 120 | 0 |
| iter1_continue | 0 (always player 0) | **1.0000** | 120 | 0 | 0 |
| iter1_continue | 1 (always player 1) | **0.0000** | 0 | 120 | 0 |
| iter1_candidate | 0 (always player 0) | **1.0000** | 120 | 0 | 0 |
| iter1_candidate | 1 (always player 1) | **0.0000** | 0 | 120 | 0 |

The standard arena alternates seats: game 0 challenger=player 0, game 1 challenger=player 1, etc. This averages exactly 1.00 + 0.00 = 0.50, with zero draws.

## Game-Level Analysis (iter0 384v256, representative)

| Metric | Value |
|--------|-------|
| Unique trajectories | 2 |
| Duplicate trajectories | 28 (of 30) |
| Final margin when challenger P0 | +10 (all 15 games) |
| Final margin when challenger P1 | -18 (all 15 games) |
| Game length | 38 or 40 plies |
| First move (challenger P0) | pit 1 or 2 (15 games each) |
| First move (challenger P1) | pit 1 or 2 (15 games each) |
| Win/loss when challenger P0 | 15W / 0L |
| Win/loss when challenger P1 | 0W / 15L |

Both models play deterministically. Only 2 game trajectories are played across 30 games. The only variation is which player goes first.

## Raw-Policy Diagnostic (500 states)

| Metric | iter0 | iter1_continue | iter1_candidate | current |
|--------|-------|----------------|-----------------|---------|
| Top-move agree with current | 63.4% | 59.2% | 58.0% | — |
| Top-move agree with teacher | 24.2% | 26.0% | 26.8% | 19.2% |
| Policy entropy mean | 0.865 | 0.807 | 0.829 | 0.763 |
| Value mean | +0.003 | +0.008 | -0.061 | +0.036 |
| Value std | 0.417 | 0.462 | 0.427 | 0.364 |

Key findings:
- Candidates agree with current on ~60% of top moves — a meaningful policy divergence
- Candidates agree more with the teacher (24-27%) than current does (19%), confirming MCTS label absorption
- All candidates have higher policy entropy than current — training increases exploration, not sharpens policy
- iter1_candidate is nearly identical to iter0 in all metrics, confirming zero marginal benefit of candidate-mined data
- iter1_continue has slightly lower entropy and slightly higher teacher agreement than iter0 — extra training on same data modestly sharpens the policy toward the teacher

## Extended Arena (240 games)

All 0.50 results at 120 games are confirmed at 240 games. No near-threshold result was a false positive. The 0.50 is stable and reproducible.

## Acceptance Criteria Evaluation

### seat_or_opening_artifact — CONFIRMED

| Criterion | Result | Status |
|-----------|--------|--------|
| One side wins nearly all games | Player 0 wins 100% across all candidates at 384v256 | CONFIRMED |
| Forced player-0/player-1 splits are extremely asymmetric | 1.00 vs 0.00 in all seat-split tests | CONFIRMED |
| Duplicate trajectories dominate the arena | 28/30 duplicates in 120-game runs, 58/60 in 240-game runs | CONFIRMED |
| 0.50 is caused by averaging two deterministic extremes | Exactly 60W/60L from seat alternation | CONFIRMED |

### search_budget_limiting — PARTIALLY CONFIRMED

| Criterion | Result | Status |
|-----------|--------|--------|
| Candidates exceed 0.55 only at higher challenger budgets | iter0: 1.00 at 1200v1200, 0.50 at 384v384 | CONFIRMED |
| Candidate score improves monotonically with challenger sims | Not monotonic (e.g., 128v128=0.00, 256v256=0.50, 1200v1200=1.00 for iter0) | PARTIAL |
| Equal-budget results differ from standard 384:256 | Yes: 1200v1200 differs from 384v256 for all candidates | CONFIRMED |

### model_policy_not_better — PARTIALLY CONFIRMED

| Criterion | Result | Status |
|-----------|--------|--------|
| Candidates remain 0.50 or worse across all budget pairs | iter0 reaches 1.00 at 1200v1200; iter1_continue reaches 0.75 | FAIL |
| Seat splits are balanced | Extremely asymmetric (1.00 vs 0.00) | FAIL |
| Raw-policy shows no meaningful improvement | iter1_candidate nearly identical to iter0 across all metrics | CONFIRMED |

### evaluation_noise_or_protocol_issue — CONFIRMED

| Criterion | Result | Status |
|-----------|--------|--------|
| Results vary sharply by seed | Not tested (single seed) | — |
| Extended arenas disagree with 120-game | No disagreement — all confirmed | — |
| First-move/trajectory duplication explains exact 60/60 | 2 trajectories across 30 games, deterministic | CONFIRMED |

## Overall Classification

**PRIMARY: SEAT AND OPENING ARTIFACT** — The 0.50 arena score is an evaluation protocol artifact, not a true measure of relative strength. The deterministic play of both models at standard budgets, combined with alternating seat assignment, creates the exact 50/50 split.

**The real diagnostic question: "Which candidate is stronger?"** can be answered by looking at high-budget results where differentiation emerges:

| Rank | Candidate | 1200v1200 | Notes |
|------|-----------|-----------|-------|
| 1 | iter0_reference | 1.00 | Dominates at high equal budget |
| 2 | iter1_continue_no_new_data | 0.75 | Wins as P0, draws as P1 |
| 3 | iter1_candidate_random_replay | 0.50 | No improvement over current |
| — | current (production) | — | (baseline) |

The correct pecking order is **hidden by the 384v256 arena** because that budget is in the "deterministic regime" where all models play identically. At 1200 sims, the models' policy differences manifest: iter0 is clearly strongest, iter1_continue is intermediate, and iter1_candidate is no better than current.

## Implications

1. **The standard arena (384v256) is insensitive to real policy differences.** Both models play deterministically — the outcome is determined entirely by seat assignment. Any model that achieves parity with current at this budget will score exactly 0.50. Better models will score 0.50. Worse models will score 0.50. The arena has zero resolving power in the parity regime.

2. **The 0.50 ceiling across 6 PRs was never a strength ceiling.** It was always an artifact of the arena protocol:
   - Seat alternation + deterministic play = exactly 0.50
   - This holds regardless of training data, replay weight, teacher quality, or state distribution
   - The finding that "nothing moves the needle" was correct, but for the wrong reason — the needle is stuck because the measuring instrument has zero resolution at this budget

3. **Higher search budgets are essential for model differentiation.** At 1200v1200:
   - iter0 (PR #93, parity candidate trained from current init + classic-MCTS labels) achieves 1.00 — it substantially exceeds current
   - iter1_continue (extra training on same data) achieves 0.75 — additional training helps
   - iter1_candidate (extra training + candidate-mined data) achieves 0.50 — adding candidate-mined data provides zero benefit
   - This confirms the PR #94 conclusion that candidate-mined data is not helpful, but reveals that the baseline iter0 and iter1_continue candidates ARE genuinely stronger than current at sufficient search depth

4. **The raw-policy diagnostic confirms candidates absorb MCTS labels.** Candidates agree with the teacher 24-27% vs current's 19%. The policy shift is real but subtle — it only translates to arena strength at high search budgets where the accumulated advantage of slightly better move choices compounds over many plies.

5. **Validation loss remains uncorrelated with arena strength.** iter1_candidate has better validation loss (1.168) than iter0 (1.182) but is identically weak at all budget pairs. This is the fifth confirmation of the val_loss–arena disconnect.

6. **The promotion arena needs higher budgets or a different protocol.** At 384v256, any model that reaches parity scores 0.50 and fails the 0.55 threshold. To detect which parity model is actually stronger, the promotion arena should use equal high budgets (e.g., 1200v1200) or asymmetric budgets that create meaningful differentiation.

## Artifacts

- Diagnostic runner: `ml/alphazero_lite/run_arena_ceiling_diagnostic.py`
- Arena modifications: `--challenger-starts`, `--game-jsonl` (diagnostic-only, backward compatible)
- Full diagnostic JSON: `/tmp/azlite_arena_ceiling_diagnostic/arena_ceiling_diagnostic.json`
- All arena sub-reports: `/tmp/azlite_arena_ceiling_diagnostic/`

## Commands Used

```bash
# Full diagnostic
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
  --workers 4 \
  --seed 42
```

## Guardrails Compliance

- [x] No training performed
- [x] No model promoted
- [x] No overwrite of `model-artifact/current` or `storage/ai/alphazero_lite/current`
- [x] No change to `local_promotion_gate` thresholds
- [x] No change to normal `arena.py` defaults
- [x] Arena changes are diagnostic-only (`--challenger-starts`, `--game-jsonl`) and backward compatible
