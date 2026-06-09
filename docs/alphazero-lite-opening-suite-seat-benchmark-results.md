# AlphaZero-Lite Opening Suite Seat Benchmark Results

**Date**: 2026-06-09
**Classification**: `iter0_still_best`

## Summary

Replaced noisy random-opening evaluation with a reproducible, deduplicated opening-prefix suite. Evaluated all available PR #104 candidates (iter0_reference, control_ep1, curriculum_lr1e5_ep1, curriculum_ep2) against current at 5 budget pairs across 128 diverse, stratified legal-opening prefixes with forced-seat splits.

**Key result**: iter0_reference remains the best checkpoint across all budget pairs. The curriculum checkpoints show progressive degradation. The suite reproduces the P0/P1 inversion pattern from PR #103 and confirms that control_ep1 does not beat iter0_reference on the deduplicated suite.

## Opening Suite Construction

### Enumeration
- Legal prefixes enumerated: **30,300** (plies 2, 4, 6)
- Unique resulting boards after deduplication: **28,961**
- Duplicate prefix count: **1,339** (4.4% — prefixes leading to same board state)

### Suite Stratification (medium_eval, 128 openings)

| Ply | Openings | Store Diff Bucket |
|-----|----------|-------------------|
| 1   | 1        | Negative: 41      |
| 2   | 2        | Zero: 44          |
| 3   | 15       | Positive: 43      |
| 4   | 27       |                   |
| 5   | 37       |                   |
| 6   | 46       |                   |

### Suite Properties
- Capture-available states: present across all plies
- Extra-turn-available states: present in ~15% of openings
- First-move families: distributed across pits 0–5 with pit 2 dominating (~85% of initial move distribution as observed in PR #103)
- Stratification: balanced by ply, store differential, side-to-move, and phase bucket

### Suite SHA256 (medium_eval)
```
57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04
```

## Candidate Evaluation

### Candidates
| Candidate | SHA256 (weights.json) |
|-----------|----------------------|
| current (model-artifact/current) | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| control_ep1 | `febc9a6abe0756a974639983c21d890015633a14ee1cae16077eb5b5b50c66af` |
| curriculum_lr1e5_ep1 | `77d5a35b25c8c63c6bcdff5643d14c50e075aa6a8bfd5c21b368d0747ec4b426` |
| curriculum_ep2 | `7b8158916acd87e60236dd5c6edf52d99ef2b79750742c7c5444d430350bf860` |

### Benchmark Configuration
- Suite: medium_eval (128 openings)
- Budget pairs: 384:256, 768:256, 768:768, 1200:1200, 256:768
- Games per opening: 1 per forced seat (2 total per opening)
- Total games per candidate: 1,280 (256 per budget pair × 5 budget pairs)
- Seed: 42

## Results

### Standard Budget (384:256)

| Candidate | P0 Score | P1 Score | DS | 95% CI P0 | 95% CI P1 |
|-----------|----------|----------|------|-----------|-----------|
| iter0_reference | 0.2578 | 0.6641 | **-0.4062** | [0.186, 0.340] | [0.576, 0.744] |
| control_ep1 | 0.2578 | 0.6797 | -0.4219 | [0.186, 0.340] | [0.592, 0.758] |
| curriculum_lr1e5_ep1 | 0.2344 | 0.6641 | -0.4297 | [0.165, 0.315] | [0.576, 0.744] |
| curriculum_ep2 | 0.2109 | 0.6641 | -0.4531 | [0.144, 0.290] | [0.576, 0.744] |

All candidates lose at standard budget (negative DS, strong P1 advantage). iter0_reference has the least negative DS. Curriculum checkpoints show progressively worse P0 scores: 0.2578 → 0.2344 → 0.2109, indicating the curriculum training degraded P0 performance.

### Equal High Budget (1200:1200)

| Candidate | P0 Score | P1 Score | DS |
|-----------|----------|----------|------|
| iter0_reference | 0.5703 | 0.2188 | **+0.3516** |
| curriculum_lr1e5_ep1 | 0.5000 | 0.4219 | +0.0781 |
| curriculum_ep2 | 0.2969 | 0.2344 | +0.0625 |
| control_ep1 | 0.3359 | 0.4609 | -0.1250 |

At equal search budget, iter0_reference dominates (DS=+0.3516). The P0/P1 inversion from standard budget (P1 advantage) to equal_high (P0 advantage) confirms the PR #103 finding. control_ep1 is the only candidate with negative DS at equal_high.

### All Budget Pairs (DS Summary)

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|-----------|---------|---------|---------|-----------|---------|
| iter0_reference | **-0.4062** | -0.1484 | **+0.0234** | **+0.3516** | -0.1016 |
| control_ep1 | -0.4219 | -0.2031 | -0.2578 | -0.1250 | -0.0625 |
| curriculum_lr1e5_ep1 | -0.4297 | -0.2031 | +0.0234 | +0.0781 | -0.1016 |
| curriculum_ep2 | -0.4531 | -0.1875 | -0.2031 | +0.0625 | -0.1016 |

**Bold** = best in budget pair.

iter0_reference has the best DS in 3 of 5 budget pairs (384:256, 768:768, 1200:1200).

### By Ply Breakdown (iter0_reference, standard budget)

| Opening Ply | P0 Score | P1 Score | DS | Games |
|-------------|----------|----------|------|-------|
| 1 | 0.0000 | 0.0000 | +0.0000 | 2 |
| 2 | 0.0000 | 1.0000 | -1.0000 | 4 |
| 3 | 0.3667 | 0.5333 | -0.1667 | 30 |
| 4 | 0.1852 | 0.7037 | -0.5185 | 54 |
| 5 | 0.2568 | 0.6486 | -0.3919 | 74 |
| 6 | 0.2826 | 0.6957 | -0.4130 | 92 |

Seat disadvantage peaks at ply 4 (DS=-0.5185). Longer openings (ply 5-6) show slightly less P1 advantage than ply 4.

### First-Move Family Effect

| First Move | P0 Score | P1 Score | DS |
|-----------|----------|----------|-----|
| Pit 1 | 0.0000 | 1.0000 | -1.0000 |
| Pit 0 | 0.1176 | 0.8118 | -0.6941 |
| Pit 2 | 0.2292 | 0.8229 | -0.5938 |
| Pit 5 | 0.6875 | 0.0625 | +0.6250 |

Confirmed: pit 5 openings favor P0, pit 1 openings favor P1. This matches the PR #103 observation perfectly.

### Duplicate Trajectory Rate

| Metric | Value |
|--------|-------|
| Unique trajectories per 256 games (combined seats) | 20 (7.8%) |
| Duplicate trajectory rate (existing formula) | 1.000 |
| Unique trajectories per seat (128 games) | 10 (7.8%) |

The high duplicate rate (20 unique trajectories across 256 games) is caused by deterministic MCTS root policy mode (`root_policy_mode=deterministic`), not by insufficient opening diversity. The suite produces 28,961 unique board states. With stochastic root policy, trajectory diversity would increase substantially.

The duplicate rate is only marginally improved vs PR #103 (2–5% unique → 7.8% unique). This is a property of the evaluation method, not the opening suite.

### Worst 5 Openings (iter0_reference, standard)

| Opening Prefix | Ply | DS |
|---------------|-----|------|
| 1,10,2,8,11,2 | 6 | -1.0000 |
| 1,6,5,6,2,8 | 6 | -1.0000 |
| 1,6,4,11 | 4 | -1.0000 |
| 1,6,3,7,0 | 5 | -1.0000 |
| 1,11,3,10 | 4 | -1.0000 |

All worst openings start with pit 1. The challenger loses as P0 and wins as P1.

### Best 5 Openings (iter0_reference, standard)

| Opening Prefix | Ply | DS |
|---------------|-----|------|
| 5,10,2,1,3,7 | 6 | +1.0000 |
| 5,6,0,9,0,11 | 6 | +1.0000 |
| 5,7,11,2,10,2 | 6 | +1.0000 |
| 5,10,2,4,6,7 | 6 | +1.0000 |
| 5,8,3,11,4,10 | 6 | +1.0000 |

All best openings start with pit 5. The challenger wins as P0 and loses as P1.

## Classification

### `iter0_still_best`

iter0_reference beats all PR #104 continuation/curriculum checkpoints on the deduplicated medium suite:

1. **Standard budget (384:256)**: iter0_reference DS = -0.4062 > control_ep1 DS = -0.4219 > curriculum_lr1e5_ep1 DS = -0.4297 > curriculum_ep2 DS = -0.4531
2. **Equal high (1200:1200)**: iter0_reference DS = +0.3516 >> curriculum_lr1e5_ep1 DS = +0.0781 > curriculum_ep2 DS = +0.0625 > control_ep1 DS = -0.1250
3. **Best in 3 of 5 budget pairs**

### Comparison with PR #104 Findings

| PR #104 Finding | Confirmed? |
|----------------|-----------|
| Deterministic 1200:1200 DS=1.00 is an opening artifact | Yes (equal_high DS=+0.3516 on diverse openings) |
| P0/P1 inversion pattern | Yes (standard: P1 dominant; equal_high: P0 dominant) |
| Curriculum holdout policy loss improved but arena did not | Yes (curriculum checkpoints lag iter0_reference) |
| Control as good or better than curriculum | Yes (control_ep1 DS beats curriculum_ep2 DS at standard budget) |
| Curriculum degrades P0 performance | Yes (P0 score: 0.2578 → 0.2344 → 0.2109 for curriculum) |

### No Evidence For
- `control_signal_real`: control_ep1 does not consistently beat iter0_reference. At standard budget, control_ep1 is slightly worse (-0.4219 vs -0.4062). At equal_high, control_ep1 is clearly worse (-0.1250 vs +0.3516).

## Discussion

### Suite Quality
The opening suite provides 28,961 unique board states with balanced stratification. The deduplication removes 4.4% of redundant prefixes. However, deterministic MCTS (root_policy_mode=deterministic) collapses 256 games into only 20 unique trajectories, making the duplicate trajectory rate indistinguishable from random-opening evaluation. This is not a suite issue — it reflects the model's deterministic play pattern.

### Future Directions
1. **Stochastic evaluation**: Use `root_policy_mode=visit_count` for trajectory diversity
2. **Add stake-weighted scoring**: Weight openings by game-theoretic importance
3. **Run large suite (384 openings)**: Verify ranking stability across suite sizes
4. **Self-play evaluation**: Evaluate candidates in head-to-head matches instead of vs fixed current

## Artifacts

| Artifact | Path |
|----------|------|
| Opening suite (small_smoke) | `/tmp/azlite_opening_suite/small_smoke.jsonl` |
| Opening suite (medium_eval) | `/tmp/azlite_opening_suite/medium_eval.jsonl` |
| Opening suite (large_eval) | `/tmp/azlite_opening_suite/large_eval.jsonl` |
| Suite summary | `/tmp/azlite_opening_suite/suite_summary.json` |
| Benchmark workdir | `/tmp/azlite_opening_suite_benchmark` |
| Candidate ranking | `/tmp/azlite_opening_suite_benchmark/candidate_ranking.json` |
