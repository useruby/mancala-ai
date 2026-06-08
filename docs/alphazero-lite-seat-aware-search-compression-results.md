# AlphaZero-Lite Seat-Aware Search-Compression Sweep Results

**Date:** 2026-06-08

## Summary

The seat-aware search-compression sweep evaluates whether the high-search
breakthrough from `iter0_reference` (which wins from both seats at 1200:1200)
can be surfaced at practical or near-practical search budgets by adjusting only
MCTS/evaluation search parameters (c_puct, root_policy_mode, tactical_root_bias,
root_prior_transform).

**Result: `tactical_bias_off` (tactical_root_bias=0.0) partially compresses the
breakthrough to 768:256 (disadvantaged-seat score = 0.50, all games drawn from
the disadvantaged seat). No setting achieves displacement-seat breakthrough at
the standard 384:256 budget.**

No training, no model promotion, no architecture changes, no replay generation.

## Artifacts

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |

## Budget Pairs

| Label | Challenger Sims | Current Sims | Purpose |
|-------|----------------|--------------|---------|
| standard_384_256 | 384 | 256 | Standard practical budget |
| moderate_challenger_768_256 | 768 | 256 | Moderate challenger search increase |
| moderate_equal_768_768 | 768 | 768 | Moderate equal budget |
| equal_high_1200_1200 | 1200 | 1200 | Positive-control breakthrough budget |
| current_high_asymmetry_256_768 | 256 | 768 | Regression/asymmetry sentinel |

## Search Setting Lanes

| Lane | c_puct | Root Policy Mode | Tactical Root Bias | Root Prior Transform |
|------|--------|-----------------|-------------------|---------------------|
| default_eval | 1.25 | deterministic | 0.1 | — |
| c_puct_low | 0.75 | deterministic | 0.1 | — |
| c_puct_mid_low | 1.0 | deterministic | 0.1 | — |
| c_puct_default | 1.25 | deterministic | 0.1 | — |
| c_puct_high | 1.5 | deterministic | 0.1 | — |
| c_puct_very_high | 2.0 | deterministic | 0.1 | — |
| root_visit_count | 1.25 | visit_count | 0.1 | — |
| tactical_bias_off | 1.25 | deterministic | 0.0 | — |
| tactical_bias_high | 1.25 | deterministic | 0.2 | — |
| root_prior_transform_damp_010 | 1.25 | deterministic | 0.1 | seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5 |

## Classification Categories

| Classification | Criteria |
|---------------|----------|
| `search_compression_promising` | DS > 0.00 at 384:256, or DS >= 0.50 at 768:256/768:768, without 256:768 collapse |
| `high_search_only` | Still breaks through at 1200:1200 (DS > 0.1), but every lower budget has DS = 0.00 |
| `search_setting_regression` | Lost 1200:1200 breakthrough (DS <= 0.0), with no compensating lower-budget gain |
| `unclassified` | Does not match any category |

## Results

### Consolidated Ranking Table

| Rank | Lane | c_puct | Policy Mode | Bias | DS 384:256 | DS 768:256 | DS 768:768 | DS 1200:1200 | DS 256:768 | Alt Score 1200:1200 | Latency p95 ms | Classification |
|------|------|--------|-------------|------|------------|------------|------------|-------------|-----------|--------------------|----------------|----------------|
| 1 | **tactical_bias_off** | 1.25 | deterministic | 0.00 | 0.00 | **0.50** | 0.00 | 0.50 | 0.00 | 0.75 | 109 | search_compression_promising |
| 2 | default_eval | 1.25 | deterministic | 0.10 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 109 | high_search_only |
| 3 | c_puct_default | 1.25 | deterministic | 0.10 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 109 | high_search_only |
| 4 | root_visit_count | 1.25 | visit_count | 0.10 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 109 | high_search_only |
| 5 | root_prior_transform_damp_010 | 1.25 | deterministic | 0.10 | 0.00 | 0.00 | 0.00 | 1.00 | 0.00 | 1.00 | 109 | high_search_only |
| 6 | c_puct_low | 0.75 | deterministic | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 111 | search_setting_regression |
| 7 | c_puct_mid_low | 1.00 | deterministic | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 109 | search_setting_regression |
| 8 | c_puct_high | 1.50 | deterministic | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 109 | search_setting_regression |
| 9 | c_puct_very_high | 2.00 | deterministic | 0.10 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 107 | search_setting_regression |
| 10 | tactical_bias_high | 1.25 | deterministic | 0.20 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.50 | 109 | search_setting_regression |

### Primary Decision Metric

**Disadvantaged-seat score at 384:256: 0.00 for all lanes.**
No search setting achieves a breakthrough at the standard practical budget.

### Key Finding: tactical_bias_off @ 768:256

Turning off tactical_root_bias (0.0 instead of 0.1) is the only change that
produces a non-zero disadvantaged-seat score at a moderate budget:

| Budget | P0 (challenger starts first) | P1 (challenger starts second) | DS Score |
|--------|------------------------------|-------------------------------|----------|
| 384:256 | W=60 L=0 D=0 (1.00) | W=0 L=60 D=0 (0.00) | 0.00 |
| **768:256** | **W=60 L=0 D=0 (1.00)** | **W=0 L=0 D=60 (0.50)** | **0.50** |
| 768:768 | W=60 L=0 D=0 (1.00) | W=0 L=60 D=0 (0.00) | 0.00 |
| 1200:1200 | W=60 L=0 D=0 (1.00) | W=0 L=0 D=60 (0.50) | 0.50 |
| 256:768 | W=60 L=0 D=0 (1.00) | W=0 L=60 D=0 (0.00) | 0.00 |

At 768:256, the challenger draws all 60 games from the disadvantaged seat
(previously all losses). At 1200:1200, the pattern changes from wins-from-both-
seats (under default tactical_root_bias=0.1) to draws-from-disadvantaged-seat.

### What tactical_root_bias Does

`tactical_root_bias` adds a bonus to the root node's policy for moves that
produce captures (tactical moves). Default is 0.1 (10% boost). Turning it off
reduces the model's preference for early tactical/capture moves, which allows
the learned policy to explore more broadly. This broadened exploration improves
disadvantaged-seat performance at moderate budgets (768:256) at the cost of
reducing the absolute ceiling at 1200:1200 (from Wins to Draws).

### c_puct Sensitivity

c_puct variation has a dramatic negative effect:
- c_puct **away from 1.25 in either direction** causes the 1200:1200
  breakthrough to collapse entirely (DS goes from 1.00 to 0.00)
- c_puct=0.75 is especially damaging: at 768:768 it causes a total collapse
  (alternating score = 0.00, losing all games including from seat 0)
- c_puct=2.0 collapses from seat 0 at 256:768 asymmetry (alternating score =
  0.00, losing from both seats)
- The default c_puct=1.25 is optimal for this model architecture

### root_policy_mode and root_prior_transform

- `visit_count` mode (vs default `deterministic`): No difference — same DS
  scores across all budget pairs
- `root_prior_transform_damp_010`: No difference — same results as default

### Latency

Latency p95 (at 1200:1200) is consistent across all lanes at ~109 ms. No search
setting meaningfully changes move-time latency.

### Per-Lane Detailed Breakdown: tactical_bias_off

| Budget | Alt Score | W/L/D | P0 W/L/D | P1 W/L/D | DS | Margin Mean | Game Len Mean | Dup Traj | Move Time Mean | Move Time p95 |
|--------|-----------|-------|----------|----------|----|-------------|---------------|----------|----------------|---------------|
| 384:256 | 0.50 | 60/60/0 | 60/0/0 | 0/60/0 | 0.00 | -4.0 | 40.5 | 120 | 18.9 | 30.3 |
| 768:256 | 0.75 | 60/0/60 | 60/0/0 | 0/0/60 | 0.50 | 3.0 | 42.5 | 120 | 29.2 | 62.0 |
| 768:768 | 0.50 | 60/60/0 | 60/0/0 | 0/60/0 | 0.00 | -9.0 | 38.5 | 120 | 45.7 | 68.6 |
| 1200:1200 | 0.75 | 60/0/60 | 60/0/0 | 0/0/60 | 0.50 | 10.0 | 37.0 | 120 | 72.2 | 109.1 |
| 256:768 | 0.50 | 60/60/0 | 60/0/0 | 0/60/0 | 0.00 | -5.0 | 36.5 | 120 | 34.6 | 67.8 |

Note: `tactical_bias_off` at 768:256 has unique_trajectories=2 and
duplicate_trajectory_count=120 — the 60 P1 draws follow identical trajectories
to the 60 P0 wins, flipped by seat. This is expected for deterministic play
with two predictable outcomes (win-from-P0 or draw-from-P1 with identical openings).

### Hotspot Analysis

1. **tactical_root_bias is the dominant search setting.** All other parameters
   have either no effect (root_policy_mode, root_prior_transform) or negative
   effects (c_puct deviations from 1.25).

2. **The 1200:1200 breakthrough survives only at c_puct=1.25.** Any deviation
   from the default c_puct causes the model to lose its ability to win from the
   disadvantaged seat even at the highest budget.

3. **256:768 asymmetry sentinel is safe across all settings except extreme
   c_puct values.** At c_puct=0.75 and c_puct=2.0, the candidate collapses from
   both seats (alternating score drops below 0.50). All other settings survive.

4. **Standard 384:256 budget remains deterministic seat-locked.** DS = 0.00 for
   all settings. No single search parameter change can compress the breakthrough
   to the standard budget.

## Classification

**Overall: `search_compression_promising` at 768:256 only.**

The sweep classifies:
- 1 lane as `search_compression_promising` (tactical_bias_off)
- 4 lanes as `high_search_only` (retain 1200:1200 breakthrough)
- 5 lanes as `search_setting_regression` (lose 1200:1200 breakthrough)

### Recommendation

Do not train a new model yet. The `tactical_bias_off` result (DS=0.50 at
768:256) suggests that the iter0_reference policy has mechanically useful
improvements that surface with modest search budget increases when the default
tactical root bias (0.1) is removed. However:

1. The standard 384:256 budget still shows DS=0.00 for all settings.
2. A phase-2 combination (e.g., tactical_bias_off + c_puct swarm) could be
   explored but is unlikely to help given c_puct sensitivity.
3. The most productive next step may be to train iteration 1 continuing from
   iter0_reference with the iter0 teacher data (matching the iter1_continue_
   no_new_data path) using tactical_root_bias=0.0 in training — the candidate
   model may learn to avoid over-indexing on tactical moves.

## Runner Command

```bash
.venv/bin/python ml/alphazero_lite/run_seat_aware_search_compression_sweep.py \
  --workdir /tmp/azlite_search_compression_sweep \
  --current model-artifact/current \
  --candidate /tmp/azlite_iterative_random_replay/iter0_candidate_artifact \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --games 120 \
  --seed 42
```

## Verification

```bash
.venv/bin/python -m unittest discover ml/alphazero_lite  # all passed
.venv/bin/ruff check ml/alphazero_lite/run_seat_aware_search_compression_sweep.py  # clean
```

## Guardrails

- No training performed.
- No model promoted.
- `model-artifact/current` and `storage/ai/alphazero_lite/current` not overwritten.
- Default promotion thresholds unchanged.
- Default `local_promotion_gate` behavior unchanged.
- No search setting made the new default.
