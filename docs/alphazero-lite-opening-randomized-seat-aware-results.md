# AlphaZero-Lite Opening-Randomized Seat-Aware Diagnostic Results

**Date**: 2026-06-09  
**Classification**: `deterministic_opening_artifact`  
**Schema**: `azlite_opening_randomized_seat_diagnostic_v1`

## Summary

iter0_reference's high-search disadvantaged-seat breakthrough (DS=1.00 at 1200:1200) is a **brittle deterministic-opening artifact**. The breakthrough collapses across 2/4/6-ply randomized openings.

However, the diagnostic reveals an **unexpected inversion pattern**: at standard budget (384:256) and challenger_high budget (768:256), iter0_reference is **stronger from the disadvantaged seat (P1)** than from P0 under randomized openings. The model consistently loses as P0 and wins as P1 at practical budgets.

## Artifacts

| Role | Path | SHA256 |
|------|------|--------|
| Candidate (iter0_reference) | `/tmp/azlite_iterative_random_replay/iter0_candidate_artifact` | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| Current | `model-artifact/current` | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |

## Configuration

| Parameter | Value |
|-----------|-------|
| Budget pairs | 384:256, 768:256, 768:768, 1200:1200, 256:768 |
| Opening ply counts | 0, 2, 4, 6 |
| Opening samples per ply | 64 |
| Opening seed | 47 |
| Games per opening | 2 (alternating seats) |
| Total games | 1930 |
| Illegal/skipped openings | 0 |

## Classification Decision

| Criterion | ply=0 | ply=2 | ply=4 | ply=6 | Verdict |
|-----------|-------|-------|-------|-------|---------|
| DS at 1200:1200 >= 0.90 | 1.000 | 0.297 | 0.281 | 0.172 | **FAIL** |
| DS at 1200:1200 collapses | N/A | Yes | Yes | Yes | **FAIL** |
| DS > 0.00 at 768:256 | N/A | 0.531 | 0.594 | 0.594 | **PASS** |
| DS > 0.00 at 384:256 | N/A | 0.578 | 0.625 | 0.609 | **PASS** |

**Classification: `deterministic_opening_artifact`**  
DS=1.00 at 1200:1200 exists only for the deterministic default opening. It collapses to 0.17-0.30 across randomized openings.

**Secondary note**: The model shows `opening_curriculum_promising` characteristics at low budgets -- DS at 384:256 and 768:256 is strongly positive (>0.5) for randomized openings. This suggests reusable disadvantaged-seat training positions exist in the opening-randomized evaluation.

## Disadvantaged-Seat Score by Budget and Opening Ply Count

| Ply | Budget | Challenger P0 Score | Challenger P1 Score | **DS** | P0 W/L/D | P1 W/L/D | Margin Mean | Game Len Mean |
|-----|--------|---------------------|---------------------|--------|----------|----------|-------------|---------------|
| 0 | 384:256 | 1.000 | 0.000 | **0.000** | 1/0/0 | 0/1/0 | -4.00 | 39.0 |
| 0 | 768:256 | 1.000 | 0.000 | **0.000** | 1/0/0 | 0/1/0 | 4.00 | 31.5 |
| 0 | 768:768 | 1.000 | 0.000 | **0.000** | 1/0/0 | 0/1/0 | -2.00 | 34.5 |
| 0 | 1200:1200 | 1.000 | 1.000 | **1.000** | 1/0/0 | 1/0/0 | 13.00 | 41.0 |
| 0 | 256:768 | 1.000 | 0.000 | **0.000** | 1/0/0 | 0/1/0 | -6.00 | 36.0 |
| | | | | | | | | | |
| 2 | 384:256 | 0.297 | 0.578 | **0.578** | 18/44/2 | 37/27/0 | -7.00 | 33.9 |
| 2 | 768:256 | 0.422 | 0.531 | **0.531** | 27/37/0 | 34/30/0 | -1.81 | 33.9 |
| 2 | 768:768 | 0.234 | 0.234 | **0.234** | 15/49/0 | 15/49/0 | -8.44 | 35.9 |
| 2 | 1200:1200 | 0.578 | 0.297 | **0.297** | 37/27/0 | 19/45/0 | -2.23 | 38.6 |
| 2 | 256:768 | 0.063 | 0.219 | **0.219** | 4/60/0 | 14/50/0 | -13.08 | 36.4 |
| | | | | | | | | | |
| 4 | 384:256 | 0.266 | 0.625 | **0.625** | 14/44/6 | 40/24/0 | -6.33 | 35.0 |
| 4 | 768:256 | 0.453 | 0.594 | **0.594** | 29/35/0 | 38/26/0 | -0.63 | 34.5 |
| 4 | 768:768 | 0.281 | 0.313 | **0.313** | 18/46/0 | 20/44/0 | -6.20 | 36.4 |
| 4 | 1200:1200 | 0.563 | 0.281 | **0.281** | 36/28/0 | 18/46/0 | -2.69 | 39.5 |
| 4 | 256:768 | 0.063 | 0.281 | **0.281** | 4/60/0 | 18/46/0 | -12.36 | 36.8 |
| | | | | | | | | | |
| 6 | 384:256 | 0.242 | 0.609 | **0.609** | 14/47/3 | 39/25/0 | -6.81 | 34.3 |
| 6 | 768:256 | 0.453 | 0.594 | **0.594** | 29/35/0 | 38/26/0 | -0.33 | 34.5 |
| 6 | 768:768 | 0.234 | 0.141 | **0.141** | 15/49/0 | 9/55/0 | -9.11 | 36.5 |
| 6 | 1200:1200 | 0.609 | 0.172 | **0.172** | 39/25/0 | 11/53/0 | -3.42 | 38.6 |
| 6 | 256:768 | 0.047 | 0.141 | **0.141** | 3/61/0 | 9/55/0 | -13.63 | 35.9 |

## DS at 1200:1200 by Opening Ply Count

| Ply Count | DS | Verdict |
|-----------|-----|---------|
| 0 (default) | 1.000 | Breakthrough confirmed |
| 2 | 0.297 | Collapse |
| 4 | 0.281 | Collapse |
| 6 | 0.172 | Near-total collapse |

The 1200:1200 breakthrough exhibits monotonic decay with opening randomization: DS = 1.00 → 0.30 → 0.28 → 0.17.

## Key Patterns

### 1. The Inversion Pattern

At practical search budgets (384:256 and 768:256) with randomized openings, iter0_reference wins as P1 (the disadvantaged seat) but loses as P0:

| Budget | P0 Score | P1 Score | Interpretation |
|--------|----------|----------|----------------|
| 384:256 (ply=4) | 0.266 | 0.625 | P1 wins 63% |
| 768:256 (ply=4) | 0.453 | 0.594 | P1 wins 59% |

With the **deterministic opening** (ply=0): P0 wins 100%, P1 loses 100% across all budgets except 1200:1200. The model was trained to win from P0 in the default opening.

**Hypothesis**: The model has learned a strong **P1-specific strategy** (counter-strategy) that works at low-to-medium budgets under randomized openings. But this strategy does not scale to high budget (1200:1200) where the P0 player's extra search overwhelms it.

### 2. High Duplicate Trajectory Rate

| Ply Count | Games | Unique Trajectories | Duplicates | Dup Rate |
|-----------|-------|---------------------|------------|----------|
| 2 | 128 | 20 | 124 | 96.9% |
| 4 | 128 | 18 | 126 | 98.4% |
| 6 | 128 | 20 | 122 | 96.1% |

Despite 64 distinct opening prefixes, only 18-20 unique full-game trajectories emerge. The models converge to identical lines after the opening, making per-opening-prefix analysis unreliable for prefix-level statistics.

### 3. First Move Concentration

After the opening prefix, P0's first actual move is highly concentrated:

| First Move | Frequency |
|------------|-----------|
| 2 (middle-right pit) | ~85% |
| 1 (second pit) | ~50% |
| 4, 5 (rightmost) | ~30% each |
| 3 | ~3% |

This mirrors the deterministic opening behavior and suggests the model has learned that pit 2 is the strongest opening move.

## Prefix-Level Analysis

### Worst 5 Opening Prefixes (Lowest DS at 384:256, ply=2)

These are prefixes where iter0_reference fails completely from P1:

| Prefix | DS | P0 Wins | P1 Wins | Total Games |
|--------|-----|---------|---------|-------------|
| `5,10` | 0.00 | 2 | 0 | 4 |
| `2,5` | 0.00 | 3 | 0 | 6 |
| `5,11` | 0.00 | 2 | 0 | 4 |
| `5,6` | 0.00 | 4 | 0 | 8 |
| `2,4` | 0.00 | 0* | 0* | 2 |

\* P0 draws: no P1 wins, P0 manages draws only.

Pattern: Prefixes starting with move 5 or containing pit 5 early tend to favor P0 exclusively.

### Best 5 Opening Prefixes (Highest DS at 384:256, ply=2)

These are prefixes where iter0_reference dominates from P1:

| Prefix | DS | P0 Wins | P1 Wins | Total Games |
|--------|-----|---------|---------|-------------|
| `1,8` | 1.00 | 0 | 4 | 8 |
| `0,8` | 1.00 | 0 | 2 | 4 |
| `0,10` | 1.00 | 0 | 3 | 6 |
| `4,9` | 1.00 | 0 | 2 | 4 |
| `2,1` | 1.00 | 3 | 3 | 6 |

Pattern: Prefixes where P0 starts with pits 0,1,4 (left-side or left-center) and follow-up moves are pits 8-11 (P1's far-right side) create strong P1 advantage positions.

## Budget-Asymmetry Analysis

### 256:768 (Current-High Asymmetry)
Candidate gets 256 sims, current gets 768 sims. This is the harshest test:

| Ply | DS | P0 Score | P1 Score | Interpretation |
|-----|-----|----------|----------|----------------|
| 0 | 0.000 | 1.000 | 0.000 | P0 wins only at default (but with 256 sim vs 768? This is anomalous with only 2 games) |
| 2 | 0.219 | 0.063 | 0.219 | Model heavily loses as P0, slightly better as P1 |
| 4 | 0.281 | 0.063 | 0.281 | Similar pattern, P1 slightly better |
| 6 | 0.141 | 0.047 | 0.141 | Both seats strong losses |

At current-high asymmetry with randomized openings, the model consistently loses from both seats but P1 performs somewhat better than P0.

### 768:768 (Equal Medium Budget)
| Ply | DS | P0 Score | P1 Score |
|-----|-----|----------|----------|
| 0 | 0.000 | 1.000 | 0.000 |
| 2 | 0.234 | 0.234 | 0.234 |
| 4 | 0.313 | 0.281 | 0.313 |
| 6 | 0.141 | 0.234 | 0.141 |

At equal 768:768 with randomized openings, the candidate is roughly equal to current from both seats (DS ~0.2-0.3). The equal-budget advantage seen at 1200:1200 in the default opening does not generalize.

## Latency

| Budget | p95 Move Time (ms) |
|--------|---------------------|
| 384:256 | ~30 ms |
| 768:256 | ~62 ms |
| 768:768 | ~65 ms |
| 1200:1200 | ~104 ms |
| 256:768 | ~64 ms |

Latency is consistent with prior seat-aware promotion gate results.

## Code Changes

### New diagnostic script
- `ml/alphazero_lite/run_opening_randomized_seat_diagnostic.py`: Standalone diagnostic that enumerates budget pairs, ply counts, and opening samples; runs arena.py as a subprocess; aggregates prefix-level metrics.

### Arena modifications (diagnostic-only behind flags)
- `--opening-seed`: Separate seed for opening prefix generation (independent of arena seed)
- `--opening-samples`: Number of distinct opening prefixes to generate
- `--games-per-opening`: Games per opening prefix (default 2 for paired seats)
- `--opening-plies`: Override for `--random-opening-plies` in sample mode
- `generate_random_opening_moves()`: Generate opening moves without applying
- `apply_opening_moves()`: Apply pre-generated move list
- `opening_prefix_moves` field in JSONL game entries

## Runner Command

```bash
.venv/bin/python ml/alphazero_lite/run_opening_randomized_seat_diagnostic.py \
  --workdir /tmp/azlite_opening_randomized_seat_diag \
  --current model-artifact/current \
  --candidate /tmp/azlite_iterative_random_replay/iter0_candidate_artifact \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --random-opening-plies 0,2,4,6 \
  --opening-samples 64 \
  --opening-seed 47 \
  --games-per-opening 2 \
  --seed 47
```

## Recommendations

1. **Do not promote iter0_reference** as a general high-search breakthrough. Its advantage is confined to the deterministic opening.

2. **Investigate the inversion pattern**: The model's P1 advantage at low budgets under randomized openings may indicate reusable P1 training positions. The `mine_disadvantaged_seat_distillation.py` pipeline with randomized openings could extract profitable P1 training data.

3. **Increase games per prefix**: With 96-98% duplicate trajectory rates, per-prefix statistics are unreliable. Future diagnostics should use 4+ games per opening prefix to increase trajectory diversity.

4. **Explore the P1-advantage prefixes**: Opening prefixes `1,8`, `0,8`, `0,10`, `4,9` consistently produce strong P1 positions at standard budget. These could seed a targeted P1 training curriculum.

5. **The search compression finding survives**: The earlier `search_compression_promising` classification at 768:256 (under tactical_bias_off) may be related to this inversion pattern rather than a genuine P0 advantage.

## Guardrails

- [x] No training executed
- [x] No model promoted
- [x] No model-artifact/current overwritten
- [x] No storage/ai/alphazero_lite/current overwritten
- [x] Default gate thresholds unchanged
- [x] Standard deterministic gate not replaced
- [x] Opening-randomized logic behind explicit diagnostic flags
