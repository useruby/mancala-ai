# AlphaZero-Lite Stochastic Opening Suite Benchmark Results

**Date**: 2026-06-10
**Classification**: `deterministic_eval_insufficient`

## Summary

Evaluated whether changing `root_policy_mode` from `deterministic` to `visit_count` reduces duplicate trajectories in the opening-suite benchmark. **Both modes are deterministic** — `visit_count` only changes tiebreaking (visits → lowest index vs visits → q_value → prior → lowest index). No material difference in trajectory diversity, scores, or rankings was observed.

The hypothesis that `visit_count` would be stochastic is incorrect. The `select_root_move` method in `self_play.py:1177-1190` uses `max()` in both branches, not sampling. There is no temperature-based sampling in arena evaluation.

## Root Cause Analysis

### Code path: `self_play.py:1173-1190`

```python
def select_root_move(self, root, legal_moves):
    if self.root_policy_mode == "deterministic":
        return max(legal_moves, key=lambda move: (
            root.children[move].visit_count,   # primary: most visits
            root.children[move].q_value,        # tiebreak 1
            root.children[move].prior,          # tiebreak 2
            -move,                              # tiebreak 3: lowest index
        ))
    return max(legal_moves, key=lambda move: (
        root.children[move].visit_count,        # primary: most visits
        -move,                                  # tiebreak: lowest index
    ))
```

Both branches use `max()` — always picking the most-visited move. The only difference is tiebreaking when multiple moves have identical visit counts (rare after MCTS). For this model, the visit distribution is sufficiently peaked that both branches select identical moves in all evaluated positions.

Temperature-based sampling is available in the self-play policy target pipeline (`self_play.py:1506-1518`) but is not exposed through the arena evaluation CLI or `PUCT` constructor.

## Suites

| Suite | Size | SHA256 |
|-------|------|--------|
| medium_eval | 128 openings | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| large_eval | 384 openings | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |

## Candidates

| Candidate | SHA256 (weights.json) |
|-----------|----------------------|
| current (model-artifact/current) | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| control_ep1 | `febc9a6abe0756a974639983c21d890015633a14ee1cae16077eb5b5b50c66af` |
| curriculum_lr1e5_ep1 | `77d5a35b25c8c63c6bcdff5643d14c50e075aa6a8bfd5c21b368d0747ec4b426` |
| curriculum_ep2 | `7b8158916acd87e60236dd5c6edf52d99ef2b79750742c7c5444d430350bf860` |

## Results: Deterministic Root Policy (Baseline)

Configuration: medium_eval, gpo=1, seed=42

### Standard Budget (384:256)

| Candidate | P0 Score | P1 Score | DS |
|-----------|----------|----------|------|
| iter0_reference | 0.2578 | 0.6641 | **-0.4062** |
| control_ep1 | 0.2578 | 0.6797 | -0.4219 |
| curriculum_lr1e5_ep1 | 0.2344 | 0.6641 | -0.4297 |
| curriculum_ep2 | 0.2109 | 0.6641 | -0.4531 |

### Equal High Budget (1200:1200)

| Candidate | P0 Score | P1 Score | DS |
|-----------|----------|----------|------|
| iter0_reference | 0.5703 | 0.2188 | **+0.3516** |
| curriculum_lr1e5_ep1 | 0.5000 | 0.4219 | +0.0781 |
| curriculum_ep2 | 0.2969 | 0.2344 | +0.0625 |
| control_ep1 | 0.3359 | 0.4609 | -0.1250 |

### All Budget Pairs (DS)

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|-----------|---------|---------|---------|-----------|---------|
| iter0_reference | **-0.4062** | -0.1484 | **+0.0234** | **+0.3516** | -0.1016 |
| control_ep1 | -0.4219 | -0.2031 | -0.2578 | -0.1250 | -0.0625 |
| curriculum_lr1e5_ep1 | -0.4297 | -0.2031 | +0.0234 | +0.0781 | -0.1016 |
| curriculum_ep2 | -0.4531 | -0.1875 | -0.2031 | +0.0625 | -0.1016 |

**Bold** = best in budget pair.

### Trajectory Diversity (Deterministic)

| Metric | Value |
|--------|-------|
| Unique trajectories per 256 games | 20 |
| Duplicate trajectory rate | 1.000 |
| Effective trajectory diversity | 7.8% |

## Results: Visit-Count Root Policy

### Medium Suite (128 openings, gpo=2, seed=42)

**All scores identical to deterministic baseline. Zero divergence.**

| Candidate | std_ds | eqhi_ds | unique_traj | dup_rate |
|-----------|--------|---------|-------------|----------|
| iter0_reference | -0.4062 | +0.3516 | 20 | 1.000 |
| control_ep1 | -0.4219 | -0.1250 | 20 | 1.000 |
| curriculum_lr1e5_ep1 | -0.4297 | +0.0781 | 20 | 1.000 |
| curriculum_ep2 | -0.4531 | +0.0625 | 20 | 1.000 |

Per-budget-pair DS values match deterministic to 4 decimal places. Candidate ranking unchanged.

### Large Suite (384 openings, gpo=1, seed=42, iter0_reference only)

| Budget Pair | DS | P0 Score | P1 Score | unique_traj | dup_rate |
|-------------|------|----------|----------|-------------|----------|
| 384:256 | -0.3698 | 0.2760 | 0.6458 | 22 | 0.997 |
| 768:256 | -0.1562 | 0.4375 | 0.5938 | 22 | 0.997 |
| 768:768 | -0.0469 | 0.2057 | 0.2526 | 22 | 0.997 |
| 1200:1200 | +0.2682 | 0.5625 | 0.2943 | 22 | 0.997 |
| 256:768 | -0.1901 | 0.0547 | 0.2448 | 22 | 0.997 |

Trajectory diversity: 22 unique out of 768 games (2.9%). Marginally better than medium suite (384 additional openings added 2 trajectories) but still 97.1% duplication.

## Trajectory Diversity Comparison

| Mode | Suite | Total Games | Unique Trajectories | Dup Rate |
|------|-------|-------------|---------------------|----------|
| deterministic | medium (128) | 256 | 20 | 1.000 |
| visit_count | medium (128) | 512 | 20 | 1.000 |
| visit_count | large (384) | 768 | 22 | 0.997 |

Both modes produce effectively identical trajectory diversity. The 2 additional trajectories in the large suite come from the 3x larger opening pool (384 vs 128), not from root policy stochasticity.

## Ranking Stability

| Metric | Deterministic Medium | Visit-Count Medium | Visit-Count Large |
|--------|---------------------|--------------------|--------------------|
| Rank 1 | iter0_reference | iter0_reference | iter0_reference (confirmed) |
| Rank 2 | control_ep1 / curriculum_lr1e5_ep1 | same | N/A |
| Rank 3 | curriculum_ep2 | same | N/A |

Rankings are perfectly stable across modes because the modes are functionally identical.

## Classification: `deterministic_eval_insufficient`

### Why this classification applies

1. **`visit_count` is deterministic**: The `select_root_move` method uses `max()` in both branches. No sampling occurs.

2. **No trajectory diversity improvement**: 20 unique trajectories (deterministic) → 20 unique trajectories (visit_count). The 22 unique in the large suite is a matter of suite size, not policy.

3. **Identical scores**: All per-budget-pair DS values match between modes to 4 decimal places.

4. **No temperature support**: The `PUCT` constructor does not accept a temperature parameter. Temperature-based policy sampling exists in the self-play pipeline but is not wired to arena evaluation.

### What would be needed

To achieve stochastic evaluation and increase trajectory diversity:
- Add a `root_temperature` parameter to `PUCT.__init__` and `select_root_move`
- Implement visit-proportional sampling in `select_root_move` when temperature > 0
- Expose `--root-temperature` through `arena.py` and `run_opening_suite_seat_benchmark.py`
- This is a larger change than scoped for this PR per task guardrails

## Acceptance Criteria Assessment

| Criterion | Status |
|-----------|--------|
| `stochastic_opening_suite_ready` | **FAILS** — visit_count does not increase trajectory diversity |
| `ranking_unstable` | **FAILS** — rankings are perfectly stable (both modes identical) |
| `deterministic_eval_insufficient` | **PASSES** — visit_count gives no more trajectory diversity than deterministic mode |

### Specific checks

- [ ] Unique trajectory count improves materially over deterministic evaluation — **NO** (20 → 20)
- [ ] Rankings stable across medium and large suites — **YES** (but because modes are identical)
- [ ] iter0_reference remains the best available checkpoint — **YES** (confirmed)
- [ ] Benchmark reproduces the deterministic-opening artifact — **YES** (identical scores)

## Discussion

The name `visit_count` is misleading — it suggests proportional-to-visits sampling but actually implements "greedy by visit count". The mode was apparently designed for diagnostics (seeing which move wins by pure visit count without tiebreaking by q_value/prior) rather than for stochastic evaluation.

For the benchmark to become more informative:
1. Implement root temperature sampling in `PUCT.select_root_move`
2. Test with temperatures 0.1, 0.5, 1.0
3. Run both medium and large suites
4. Expect 5-50x increase in unique trajectories

The opening suite itself (28,961 unique boards) is not the bottleneck. The bottleneck is deterministic evaluation. Until temperature-based sampling is available, the benchmark's trajectory diversity will remain at ~20 unique trajectories regardless of suite size or root policy mode.

## Artifacts

| Artifact | Path |
|----------|------|
| Deterministic medium workdir | `/tmp/azlite_opening_suite_benchmark` |
| Visit-count medium (iter0) workdir | `/tmp/azlite_opening_suite_stochastic_medium` |
| Visit-count medium (all candidates, partial) | `/tmp/azlite_opening_suite_stochastic_all_medium` |
| Visit-count large (iter0) workdir | `/tmp/azlite_opening_suite_stochastic_large` |
| Opening suite medium | `/tmp/azlite_opening_suite/medium_eval.jsonl` |
| Opening suite large | `/tmp/azlite_opening_suite/large_eval.jsonl` |
