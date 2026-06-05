# AlphaZero-Lite Tablebase Value Overlay — 1600-Game Results

**Date:** 2026-06-06
**Experiment:** 1600-game tablebase value overlay strength evaluation at production data scale
**Seed:** 42 | **Workers:** 1 | **Games:** 1600 | **Sims:** 1200

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Games | 1600 |
| Engine | Classic MCTS |
| Simulations | 1200 |
| Seed | 42 |
| Workers | 1 |
| Max positions per game | 24 |
| Input encoding | `kalah_v3` |
| Policy target mode | `sharpened` |
| Value target mode | `sharpened` |
| Model type | `residual_v3` |
| Hidden sizes | 96,3 |
| Epochs | 10 |
| Batch size | 512 |
| Value loss | huber, huber_delta=1.0 |
| Value loss weight | 0.3 |
| Val split | 0.1 |
| Grad clip | 1.0 |
| Arena games | 120 |
| MCTS games | 40 |

## 2. Lanes

| Lane | Tablebase value overlay |
|------|------------------------|
| baseline | off |
| overwrite | overwrite |

## 3. Dataset Metrics

| Metric | baseline | overwrite |
|--------|----------|-----------|
| rows_written | 37647 | 37647 |
| positions_visited | 52191 | 52191 |
| tablebase_rows | 0 | 9790 |
| tablebase_coverage_rate | 0% | **26.00%** |
| mean_abs_value_delta | N/A | 0.233220 |
| max_abs_value_delta | N/A | 2.000 |
| coverage_early | 0 | 0 |
| coverage_mid | 0 | 3012 |
| coverage_late | 0 | **6778** (69.2%) |

### Row-Level Comparison (baseline vs overwrite)

| Metric | Value |
|--------|-------|
| total_rows | 37647 |
| policy_diff_count | **0** |
| state_diff_count | **0** |
| player_diff_count | 0 |
| move_index_diff_count | 0 |
| value_diff_rows (tablebase-affected) | 4551 |
| value_match_rows (unchanged) | 33096 |

**Policy target verification:** All 37647 policy vectors are identical between baseline and overwrite (0 diffs). State encodings are also identical. Only value targets differ, exclusively on the 4551 tablebase-affected rows (12.1% of rows have value changes vs 12.3% in the 600-game run).

**Coverage observations:**
- 26.00% of training rows have solved tablebase values (consistent with 26.03% in the 600-game run)
- Zero coverage in early-game (move_index <= 8)
- 69.2% of tablebase coverage is in late-game (move_index >= 25), up slightly from 68.7% at 600 games
- Mean abs value delta of 0.233 confirms substantive MCTS vs tablebase value differences
- Max delta of 2.0: some MCTS search values are at the opposite extreme from tablebase truth

## 4. Training Metrics

| Metric | baseline | overwrite |
|--------|----------|-----------|
| policy_loss | 1.461247 | 1.467404 |
| value_loss | 0.258423 | 0.259072 |
| best_val_loss | 1.547088 | 1.553418 |
| top-k checkpoints | 3 | 3 |

**Value loss analysis:** Overwrite value loss is slightly higher (+0.000649), consistent with the PR #82 and PR #83 findings. Tablebase values are harder targets (tending toward -1/0/+1) while MCTS values are smoother. Policy loss delta is +0.006157 (0.4%), which is slightly higher than the 600-game run (+0.000082).

**Value improvement from scale:** Both lanes show improved training metrics vs the 600-game run. Baseline policy_loss dropped from 1.616 to 1.461 (-9.6%), value_loss from 0.274 to 0.258 (-5.8%), and best_val_loss from 1.708 to 1.547 (-9.4%). This confirms the expected benefit of larger datasets.

### Checkpoint SHA256 Hashes

| Checkpoint | baseline | overwrite |
|-----------|----------|-----------|
| checkpoint.npz | `2c01fed7d356ab69e57226d70bdb262748b8a12fe0a9e14b81cfc88cb0b0924c` | `e544e648349396784cdd8b5387c1cc4216c1c9604445c9f022d0ef39a99793ed` |
| checkpoint.top1.npz | `2c01fed7d356ab69e57226d70bdb262748b8a12fe0a9e14b81cfc88cb0b0924c` | `e544e648349396784cdd8b5387c1cc4216c1c9604445c9f022d0ef39a99793ed` |
| checkpoint.top2.npz | `8fd1b4d1a3b57e2c75379598e2bb2d923389ad6f76bd1e3d1bfeceb626b6c5e3` | `a5596ae2510639b0d169331c0795607093b0a459962628ece184f0dec3640887` |
| checkpoint.top3.npz | `3f039bf4747a1632524a66ddaea8825849ca2ff9ef21358c6c2a3b1e7e9821a8` | `c1315133576ca0b517765d8c1fd7b42e4bd042fa8ea50fa1a47240e1c56518e6` |

Note: baseline checkpoint.npz and top1.npz have the same hash (top1 was the best epoch, which happened to be the final epoch's checkpoint). Overwrite shows the same pattern.

## 5. Strength Metrics

| Metric | baseline | overwrite | current |
|--------|----------|-----------|---------|
| arena_score vs current | 0.5 | 0.5 | N/A |
| arena_wins | 60 | 60 | 60 |
| arena_losses | 60 | 60 | 60 |
| arena_draws | 0 | 0 | 0 |
| hard_score | N/A (not run) | N/A (not run) | N/A |
| candidate_mcts_score | N/A | N/A | N/A |
| current_mcts_score | N/A | N/A | N/A |
| gate_passed | False | False | N/A |
| failure_reasons | arena_score_below_threshold | arena_score_below_threshold | N/A |

**Notable:** Both lanes achieved exactly 0.5 (60W/60L/0D) against the current production model. This is a substantial improvement over the 600-game results (0.0, 0W/120L/0D) and confirms the expected benefit of 2.7x more training data. However, 0.5 is still below the 0.55 arena prefilter threshold, so MCTS1200 evaluations were not run.

**Move timing:** The arena report does not include move time metrics. Both models use the same `residual_v3` architecture with identical sizes and inference paths, so move times are expected to be identical (~19.5ms mean, ~31ms p95 per the 600-game run on the same hardware).

## 6. Computation Cost

| Lane | Dataset generation (s) | Training (s) | Export (s) | Gate (s) | Total (s) |
|------|------------------------|-------------|-----------|---------|-----------|
| baseline | 2083.78 | 7.74 | 0.18 | 63.54 | 2155.24 |
| overwrite | 27895.26 | 6.61 | 0.17 | 60.80 | 27962.84 |

**Dataset generation overhead:** Tablebase solving adds ~13.4x overhead (27895s vs 2084s). The cache-clearing per-game strategy (added to prevent memory exhaustion at 1600 games) means the tablebase solver re-solves positions for each game rather than sharing a global cache. This is consistent with the 13x overhead observed in the 100-game run (PR #82). For practical iteration, a shared persistent tablebase cache would be needed.

## 7. Acceptance Criteria Assessment

| Criterion | Status | Detail |
|-----------|--------|--------|
| policy_diff_count is 0 | **PASS** | 0 policy diffs across 37647 rows |
| state_diff_count is 0 | **PASS** | 0 state diffs across 37647 rows |
| overwrite arena_score >= baseline | **PASS** | Both 0.5 (equal, not worse) |
| overwrite candidate_mcts_score >= baseline | **INCONCLUSIVE** | MCTS1200 not run (arena prefilter failed) |
| No regression vs current | **INCONCLUSIVE** | Both 0.5 vs current; no regression vs each other |
| overwrite passes local_promotion_gate | **FAIL** | Both lanes fail arena_score threshold (0.5 < 0.55) |
| overwrite shows clear strength improvement | **FAIL** | Overwrite = baseline at 0.5; no improvement |
| No exact-position/control regression | **PASS** | Policy vectors identical; no PR #81-style mechanism |
| Tablebase coverage non-trivial | **PASS** | 26.00% coverage, 69.2% late-game |

## 8. Interpretation

1. **Policy preservation confirmed at 1600 games.** All 37647 policy vectors are identical between baseline and overwrite. State encodings are also identical. Only value labels differ on tablebase-affected rows. The mechanism works correctly at production scale.

2. **Tablebase coverage is stable at 26%.** Coverage rate is independent of dataset size (26.03% at 600 games, 26.00% at 1600 games). The proportion of training rows that reach endgame positions (<16 seeds) is a function of game length, not dataset size. Coverage remains concentrated in late-game (69.2% at move_index >= 25).

3. **Value loss impact is negligible at 1600 games.** The overwrite value loss increase (+0.000649, 0.25%) is smaller than at 600 games (+0.005874, 2.1%). With more training data, the harder tablebase targets are absorbed with less relative penalty.

4. **1600 games is sufficient for non-trivial strength but insufficient for gate passage.** Both lanes improved from 0.0 to 0.5 vs the current model when scaled from 600 to 1600 games. This is a meaningful improvement but still 0.05 below the 0.55 gate threshold. The current model was trained on 1600+ games and has the benefit of being the product of iterative self-play refinement, not single-phase bootstrap.

5. **No evidence of control regression.** Unlike PR #81 (which patched production moves and caused 8.7%+ control regression), the value-only overlay preserves policy and state identically. No degradation mechanism observed.

6. **No strength difference between lanes at 1600 games.** Both baseline and overwrite scored exactly 0.5 (60W/60L/0D). The tablebase value overlay does not improve (or harm) model strength at this data scale when measured against the current production model. The value label improvements may be too small a signal relative to the 74% of training rows that are unaffected.

7. **Generation cost is prohibitive for iterative development.** At 7.75 hours for dataset generation alone, the tablebase overlay adds ~13x cost to each bootstrap iteration. Without a shared persistent cache architecture, this is not practical for rapid experimentation.

## 9. Decision: NOT COMPETITIVE

**Classification:** `tablebase_value_overlay_not_competitive_as_single_phase_bootstrap`

**Rationale:**
- The mechanism works correctly (policy preservation, coverage, metadata).
- Both lanes produce functionally equivalent models at 1600 games / 1200 sims.
- Neither lane meets the arena prefilter threshold (0.5 < 0.55).
- Overwrite shows zero strength advantage over baseline.
- Generation cost is 13x baseline for no measurable gain.

**Why this differs from "inconclusive" (600-game classification):**
The 600-game run was inconclusive because both lanes scored 0.0, making it impossible to distinguish noise from signal. At 1600 games, both lanes score 0.5 — a meaningful measurement — but the overwrite lane shows no advantage over baseline. The overlay neither helps nor hurts at this scale, which is a stronger signal than the 600-game result.

**Implications for the overlay direction:**
- Tablebase value overlay as a standalone bootstrap improvement is not viable.
- The value label improvements (on 26% of rows, concentrated in late-game) are too small a training signal to shift model strength at this architecture and data scale.
- The overlay might still have value in other contexts (e.g., combined with policy improvements, used in multi-phase training, or applied in an exact-solve search context), but as a single-phase bootstrap mechanism it does not deliver.

## 10. Artifacts

| Artifact | Path |
|----------|------|
| Baseline dataset | `/tmp/azlite_tb_value_production_1600/datasets/baseline.jsonl` |
| Overwrite dataset | `/tmp/azlite_tb_value_production_1600/datasets/overwrite.jsonl` |
| Baseline checkpoint | `/tmp/azlite_tb_value_production_1600/models/baseline_checkpoint.npz` |
| Baseline top-k | `/tmp/azlite_tb_value_production_1600/models/baseline_checkpoint.top[1-3].npz` |
| Overwrite checkpoint | `/tmp/azlite_tb_value_production_1600/models/overwrite_checkpoint.npz` |
| Overwrite top-k | `/tmp/azlite_tb_value_production_1600/models/overwrite_checkpoint.top[1-3].npz` |
| Baseline artifact | `/tmp/azlite_tb_value_production_1600/models/baseline/` |
| Overwrite artifact | `/tmp/azlite_tb_value_production_1600/models/overwrite/` |
| Experiment summary | `/tmp/azlite_tb_value_production_1600/experiment_summary.json` |
| Gate baseline | `/tmp/azlite_tb_value_production_1600/evaluations/gate_baseline.json` |
| Gate overwrite | `/tmp/azlite_tb_value_production_1600/evaluations/gate_overwrite.json` |
| Arena report | `/tmp/azlite_tb_value_production_1600/evaluations/candidate_vs_current_arena.json` |
