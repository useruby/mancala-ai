# AlphaZero-Lite Tablebase Value Overlay — Production Results

**Date:** 2026-06-05
**Experiment:** Production-scale tablebase value overlay strength evaluation
**Seed:** 42 | **Workers:** 1 | **Games:** 600 | **Sims:** 1200

## 1. Configuration

| Parameter | Value |
|-----------|-------|
| Games | 600 |
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
| rows_written | 14108 | 14108 |
| positions_visited | 19506 | 19506 |
| tablebase_rows | 0 | 3672 |
| tablebase_coverage_rate | 0% | **26.03%** |
| mean_abs_value_delta | N/A | 0.247430 |
| max_abs_value_delta | N/A | 2.000 |
| coverage_early | 0 | 0 |
| coverage_mid | 0 | 1149 |
| coverage_late | 0 | **2523** (68.7%) |

### Row-Level Comparison (baseline vs overwrite)

| Metric | Value |
|--------|-------|
| total_rows | 14108 |
| policy_diff_count | **0** |
| state_diff_count | **0** |
| player_diff_count | 0 |
| move_index_diff_count | 0 |
| value_diff_rows (tablebase-affected) | 1736 |
| value_match_rows (unchanged) | 12372 |

**Policy target verification:** All 14108 policy vectors are identical between baseline and overwrite (0 diffs). State encodings are also identical. Only value targets differ, exclusively on the 1736 tablebase-affected rows.

**Coverage observations:**
- 26.03% of training rows have solved tablebase values (up from 24.44% in PR #82 test run)
- Zero coverage in early-game (move_index <= 8)
- 68.7% of tablebase coverage is in late-game (move_index >= 25)
- Mean abs value delta of 0.247 confirms substantive MCTS vs tablebase value differences
- Max delta of 2.0: some MCTS search values are at the opposite extreme from tablebase truth

## 4. Training Metrics

| Metric | baseline | overwrite |
|--------|----------|-----------|
| policy_loss | 1.616085 | 1.616167 |
| value_loss | **0.273895** | 0.279769 |
| best_val_loss | **1.708404** | 1.709668 |
| top-k checkpoints | 3 | 3 |

**Value loss analysis:** Overwrite value loss is slightly higher (+0.005874), consistent with PR #82 small-scale results. Tablebase values are harder targets (tending toward -1/0/+1) while MCTS values are smoother. Policy loss is nearly identical (delta < 0.0001).

### Checkpoint SHA256 Hashes

| Checkpoint | baseline | overwrite |
|-----------|----------|-----------|
| baseline_checkpoint.npz | `e73183bed1e80d8f...` | `450273e6d3bdbc0b...` |
| baseline_checkpoint.top2.npz | `bb72bb3e3f60e7cb...` | `b1738f520d204b74...` |
| baseline_checkpoint.top3.npz | `19b36afa9f9cd44c...` | `c5862a246bc89873...` |

## 5. Strength Metrics

Both lanes scored **0.0** (0W/120L/0D) against the current production model. This is expected: 600-game / 1200-sim models cannot compete with the current model trained on 1600+ games. The arena prefilter (0.55 threshold) was not met, so MCTS1200 evaluations were not run.

| Metric | baseline | overwrite | current |
|--------|----------|-----------|---------|
| arena_score vs current | 0.0 | 0.0 | N/A |
| arena_losses | 120 | 120 | N/A |
| candidate_mcts_score | N/A | N/A | N/A |
| move_time_mean_ms | 19.47 | 19.47 | N/A |
| move_time_p95_ms | 31.1 | 31.1 | N/A |
| gate_passed | False | False | N/A |
| failure_reasons | arena_score_below_threshold | arena_score_below_threshold | N/A |

**Note:** The arena prefilter failing means MCTS1200 baseline and relative strength checks could not run. Both lanes produced functionally equivalent models at this data scale. To compare overwrite vs baseline MCTS1200 strength, a larger dataset (matching current production scale of 1600+ games) would be needed.

## 6. Acceptance Criteria Assessment

| Criterion | Status | Detail |
|-----------|--------|--------|
| policy_diff_count is 0 | **PASS** | 0 policy diffs across 14108 rows |
| state_diff_count is 0 | **PASS** | 0 state diffs across 14108 rows |
| Tablebase coverage non-trivial | **PASS** | 26.03% coverage, 68.7% late-game |
| overwrite arena_score >= baseline | **PASS** | Both 0.0 (equal) |
| overwrite candidate_mcts_score >= baseline | **INCONCLUSIVE** | MCTS1200 not run (arena prefilter failed) |
| No regression vs current | **INCONCLUSIVE** | Both 0.0 vs current; need larger dataset for meaningful comparison |
| local_promotion_gate passes | **FAIL** | Both lanes fail arena_score threshold (0.0 < 0.55) |
| Overwrite does not regress value loss excessively | **PASS** | Value loss +0.006 (2.1% increase), consistent with harder targets |

## 7. Interpretation

1. **Policy preservation confirmed at production scale.** All 14108 policy vectors are identical between baseline and overwrite. State encodings are also identical. Only value labels differ on tablebase-affected rows.

2. **Tablebase coverage scales with data volume.** 26.03% coverage on 600 games vs 24.44% on 100 games in PR #82. Coverage pattern remains concentrated in late-game (68.7% at move_index >= 25).

3. **Value loss impact is small and consistent.** +0.005874 increase in value loss is consistent with the PR #82 small-scale result (+0.0096). The tablebase values are exact ground truth, creating a harder but potentially better calibration target.

4. **600 games is insufficient for production competitiveness.** Both baseline and overwrite scored 0.0 vs the current model (trained on 1600+ games). The arena prefilter prevents MCTS1200 comparison. A 1600+ game experiment would be needed to assess whether overwrite improves over baseline at production-relevant scale.

5. **No evidence of control regression.** Unlike PR #81 (which patched production moves and caused 8.7%+ control regression), the value-only overlay preserves policy and state identically. No degradation mechanism observed.

## 8. Decision: NEED LARGER DATASET

**Classification:** `tablebase_value_overlay_production_inconclusive`

**Rationale:** The mechanism works correctly at production scale (policy preservation proved, coverage substantial, value deltas meaningful). However, 600 games is insufficient for MCTS1200 strength comparison against the production baseline. The experiment needs to run at the same data scale as the current production model (1600+ games, 1200+ sims) to determine whether tablebase value overlay improves or regresses MCTS1200 strength.

**Next steps:**
1. Run experiment at 1600 games (matching current model bootstrap scale):
   ```
   .venv/bin/python ml/alphazero_lite/run_tablebase_value_overlay_experiment.py \
     --workdir /tmp/azlite_tb_value_production_1600 \
     --lanes baseline,overwrite --games 1600 --simulations 1200 \
     --workers 1 --epochs 10 --arena-games 120 --mcts-games 40
   ```
2. If arena_score passes prefilter at 1600 games, MCTS1200 comparison will run automatically.
3. Do NOT promote based on 600-game results alone.

## 9. Artifacts

| Artifact | Path |
|----------|------|
| Baseline dataset | `/tmp/azlite_tb_value_production/datasets/baseline.jsonl` |
| Overwrite dataset | `/tmp/azlite_tb_value_production/datasets/overwrite.jsonl` |
| Baseline checkpoint | `/tmp/azlite_tb_value_production/models/baseline_checkpoint.npz` |
| Overwrite checkpoint | `/tmp/azlite_tb_value_production/models/overwrite_checkpoint.npz` |
| Baseline top-k | `/tmp/azlite_tb_value_production/models/baseline_checkpoint.top[1-3].npz` |
| Overwrite top-k | `/tmp/azlite_tb_value_production/models/overwrite_checkpoint.top[1-3].npz` |
| Experiment summary | `/tmp/azlite_tb_value_production/experiment_summary.json` |
| Baseline artifact | `/tmp/azlite_tb_value_production/models/baseline/` |
| Overwrite artifact | `/tmp/azlite_tb_value_production/models/overwrite/` |
| Gate baseline | `/tmp/azlite_tb_value_production/evaluations/gate_baseline.json` |
| Gate overwrite | `/tmp/azlite_tb_value_production/evaluations/gate_overwrite.json` |
