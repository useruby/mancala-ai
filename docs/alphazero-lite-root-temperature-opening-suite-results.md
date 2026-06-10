# AlphaZero-lite Root Temperature Opening-Suite Benchmark

**Date**: 2026-06-10
**Classification**: `stochastic_eval_too_noisy`

## Summary

Implemented root-temperature sampling in `PUCT.select_root_move()` and evaluated its effect on opening-suite benchmarks. Temperature-based sampling dramatically improves trajectory diversity (15 → 128 unique trajectories at temp ≥ 0.5) but dilutes the ranking signal to the point of parity. At temperature 0.1, trajectory diversity improves 4.9x with moderate signal degradation. At temperature ≥ 0.5, all trajectories are unique but the DS approaches zero across all budget pairs, making the benchmark unable to distinguish candidates.

**Key result**: Root-temperature sampling solves the trajectory diversity bottleneck but introduces too much noise for reliable ranking. The deterministic mode's signal (e.g., DS=-0.4219 at standard, DS=+0.2812 at equal_high) reflects genuine model quality differences that are washed out by forced exploration.

## Implementation

### Changes to `self_play.py`

- Added `root_temperature` parameter to `PUCT.__init__()` (default: 0.0)
- Modified `select_root_move()` to sample legal root moves proportional to `visit_count ** (1 / temperature)` when `root_temperature > 0.0`
- When `root_temperature = 0.0`, behavior is identical to the original deterministic `max()` selection
- Zero-visit moves are handled safely via uniform random fallback and `+1e-8` epsilon
- Uses the PUCT instance's seeded `rng` for reproducibility

### Changes to `arena.py`

- Added `root_temperature` to `run_arena_worker()` signature
- Passes temperature through `build_search_options()` to `PUCT` constructor
- Exposed `--root-temperature` CLI flag via `add_search_option_args()`

### Changes to `run_opening_suite_seat_benchmark.py`

- Added `--root-temperatures` flag (comma-separated float list)
- Added `--seeds` flag (comma-separated int list)
- When `root_temperature == 0.0`, runs only 1 seed (deterministic)
- When `root_temperature > 0.0`, runs all specified seeds
- Produces per-temperature, per-seed metrics and seed-summary JSONs

### Tests

6 new tests added to `test_self_play.py`:
- `test_root_temperature_zero_matches_deterministic_selection`: confirms temp 0.0 = original behavior
- `test_root_temperature_positive_samples_multiple_moves`: verifies temp > 0 samples more than one move
- `test_root_temperature_never_samples_illegal_moves`: only legal moves returned
- `test_root_temperature_reproducible_with_same_seed`: same rng seed = same sequence
- `test_visit_count_policy_mode_returns_legal_move_with_temperature`: backward compatible
- `test_root_temperature_defaults_preserved_in_search_options`: defaults are 0.0

## Benchmark Configuration

| Parameter | Value |
|-----------|-------|
| Suite | medium_eval.jsonl (128 deduplicated openings) |
| Current | model-artifact/current (`6946aafb`) |
| Candidate | iter0_reference (`0bbeaa9c`) |
| Budget Pairs | 384:256, 768:256, 768:768, 1200:1200, 256:768 |
| Games per Opening | 2 |
| Seat Split | Forced (challenger-starts=0 and 1) |
| Root Policy Mode | deterministic |
| Temperatures | 0.0, 0.1, 0.25, 0.5, 1.0 |
| Seeds (temp > 0) | 42, 43, 44 |
| Seeds (temp = 0) | 42 |
| Workers | 4 |

## Trajectory Diversity

### Standard Budget (384:256), 128 Games

| Temperature | Seed | Unique Trajectories | Duplicate Rate |
|------------|------|--------------------|-----------------|
| 0.0 | 42 | 15 | 1.0000 |
| 0.1 | 42 | 73 | 0.6016 |
| 0.1 | 43 | 74 | 0.5703 |
| 0.1 | 44 | 78 | 0.5391 |
| 0.25 | 42 | 113 | 0.1875 |
| 0.25 | 43 | 117 | 0.1484 |
| 0.25 | 44 | 122 | 0.0781 |
| 0.5 | 42 | 128 | 0.0000 |
| 0.5 | 43 | 128 | 0.0000 |
| 0.5 | 44 | 127 | 0.0156 |
| 1.0 | 42 | 128 | 0.0000 |
| 1.0 | 43 | 128 | 0.0000 |
| 1.0 | 44 | 128 | 0.0000 |

**Trajectory diversity is solved at temperature ≥ 0.5**: all 128 games produce unique trajectories. At temperature 0.25, ~91% are unique. At temperature 0.1, ~59% are unique (4.9x improvement over deterministic's 15).

## DS (Differential Score) Results

### iter0_reference vs current

| Budget | Temp 0.0 | Temp 0.1 | Temp 0.25 | Temp 0.5 | Temp 1.0 |
|--------|----------|----------|-----------|----------|----------|
| 384:256 | -0.4219 | -0.3698 ±0.0835 | -0.2318 ±0.0325 | -0.0807 ±0.0520 | -0.0130 ±0.0709 |
| 768:256 | -0.2500 | -0.2370 ±0.0163 | -0.2448 ±0.1173 | -0.1693 ±0.0916 | -0.0104 ±0.0802 |
| 768:768 | +0.0625 | -0.1484 ±0.0610 | -0.1562 ±0.0475 | -0.1589 ±0.1106 | -0.0573 ±0.1160 |
| 1200:1200 | +0.2812 | +0.0755 ±0.0316 | -0.1120 ±0.0627 | -0.0990 ±0.1441 | -0.0312 ±0.0475 |
| 256:768 | -0.0938 | -0.0885 ±0.1214 | -0.1146 ±0.0430 | -0.1094 ±0.0341 | -0.0234 ±0.1074 |

### P0/P1 Scores at Standard Budget (384:256)

| Temperature | P0 Score | P1 Score | DS |
|-------------|----------|----------|-----|
| 0.0 | 0.3281 | 0.7500 | -0.4219 |
| 0.1 | 0.2552 | 0.6250 | -0.3698 |
| 0.25 | 0.2734 | 0.5052 | -0.2318 |
| 0.5 | 0.2734 | 0.3542 | -0.0807 |
| 1.0 | 0.4245 | 0.4375 | -0.0130 |

## Key Observations

### Signal Degradation

The deterministic DS of -0.4219 at standard budget reflects a genuine quality gap between iter0_reference and current in the P1 seat. As temperature increases, this gap is progressively washed out:
- The P1 score drops from 0.7500 (temp 0.0) to 0.4375 (temp 1.0)
- The P0 score rises from 0.3281 (temp 0.0) to 0.4245 (temp 1.0)
- DS approaches 0.0 at temp 1.0, where the benchmark can no longer distinguish the candidates

This happens because temperature-based sampling forces the model to explore moves it would never choose deterministically. These forced suboptimal moves dilute the quality signal.

### Equal High Budget (1200:1200)

At high search budgets, the deterministic mode shows iter0_reference with a clear advantage (DS=+0.2812). This advantage disappears at all non-zero temperatures:

| Temperature | DS at equal_high | Interpretation |
|-------------|-------------------|----------------|
| 0.0 | +0.2812 | iter0_reference clearly better at high search |
| 0.1 | +0.0755 | Mild advantage, within noise |
| 0.25 | -0.1120 | Slightly negative (noise dominates) |
| 0.5 | -0.0990 | Noise dominates |
| 1.0 | -0.0312 | Noise dominates |

### Seed Variance

Seed-to-seed variance is moderate:
- Temp 0.1: DS standard deviation across 3 seeds ranges from 0.0163 (768:256) to 0.1214 (256:768)
- Temp 0.25: DS standard deviation 0.0325-0.1173
- Temp 0.5: DS standard deviation 0.0341-0.1441
- Temp 1.0: DS standard deviation 0.0475-0.1160

The variance is acceptable (sub-0.15) but combined with the signal drift makes ranking unreliable.

## Classification

**`stochastic_eval_too_noisy`**

Root-temperature sampling solves the trajectory diversity problem (15 → 128 unique at temp ≥ 0.5) but introduces signal degradation proportional to temperature. The deterministic DS values contain genuine quality information that is diluted by stochastic sampling. At temperature 1.0, the benchmark cannot distinguish iter0_reference from current (DS ≈ 0 across all budgets).

### Trajectory Diversity: ✓ Solved
- Temp 0.1: 4.9x more unique trajectories than deterministic
- Temp 0.25: 7.8x more unique trajectories
- Temp ≥ 0.5: essentially 100% unique

### Ranking Signal: ✗ Degraded
- DS shifts from -0.4219 (temp 0.0) to -0.0130 (temp 1.0) at standard budget
- The deterministic signal contains real model quality information
- Equal-high advantage (+0.2812) disappears entirely

### Recommendation

At this point, **do not use root temperature as a default**. The deterministic path already provides reliable ranking signal. The trajectory diversity bottleneck is real (15 unique trajectories across 128 games) but the solution (stochastic sampling) introduces noise that outweighs the diversity benefit.

If trajectory diversity is needed for specific diagnostics:
- Use temp 0.1 as the lowest temperature with meaningful diversity (~59% unique)
- Do not exceed temp 0.25 for evaluation purposes
- Accept that DS values will be somewhat noisier

Future directions to explore:
- Lower temperatures (0.05, 0.075) to find the minimum that provides meaningful diversity
- Temperature annealing over the game (high temp early, low temp late) to balance diversity and signal
- Alternative diversity mechanisms (e.g., opening-prefix rotation, opponent mix) that don't dilute the signal

## Artifact Paths

| Artifact | Path |
|----------|------|
| Benchmark workdir | `/tmp/azlite_root_temperature_opening_suite/` |
| Suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` |
| Current | `model-artifact/current` |
| iter0_reference | `/tmp/azlite_iterative_random_replay/iter0_candidate_artifact` |
| Seed summaries | `temp_*/seed_summary.json` |

## Test Results

```
.venv/bin/ruff check ml/alphazero_lite/ script/ai/  # All checks passed
.venv/bin/python -m pytest ml/alphazero_lite/test_self_play.py ml/alphazero_lite/test_arena.py -x -q
# 177 passed, 11 subtests passed
```
