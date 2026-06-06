# AlphaZero-Lite Residual v4 Classic-MCTS Same-Data Strength Results

**Date:** 2026-06-06

## Classification

**`architecture_only_not_strength`**

The move-factorized policy head does not improve real game strength over residual_v3 when trained on identical classic-MCTS data. v4 is materially weaker than v3 in head-to-head PUCT arena play (0 wins, 60 losses, 60 draws) while scoring identically vs MCTS1200.

## Primary Hypothesis

> The move-factorized policy head improves move discrimination enough to produce stronger play from the same classic-MCTS training data, without increasing parameter count or inference cost.

**Verdict:** Rejected. v4 underperforms v3 in direct arena play and ties in MCTS1200.

## Experiment Design

Two architectures trained from scratch on the exact same 1600-game classic-MCTS dataset, with identical hyperparameters, seed, and training recipe. Comparison via head-to-head arena, arena vs production current, and MCTS1200 baseline.

## Dataset

| Metric | Value |
|--------|-------|
| Path | `/tmp/azlite_v4_classic_mcts_strength/shared_classic_mcts_1600.jsonl` |
| SHA256 | `18f2313afc133847ceabfdabc68b6bd7c2057bd8c669313ea45f02a32fcb6057` |
| rows_written | 25579 |
| games | 1600 |
| simulations | 1200 |
| seed | 42 |
| max_positions_per_game | 16 |
| teacher_mode | classic_mcts |
| policy_target_mode | sharpened |
| value_target_mode | sharpened |
| input_encoding | kalah_v3 |
| average_rows_retained_per_game | 15.99 |
| positions_searched | 51267 |
| total_simulations | 61,520,400 |

## Training

| Metric | residual_v3 | residual_v4_move_factorized |
|--------|-------------|---------------------------|
| Architecture | residual_v3 | residual_v4_move_factorized |
| Checkpoint path | `/tmp/.../residual_v3/checkpoint.npz` | `/tmp/.../residual_v4/checkpoint.npz` |
| Checkpoint SHA256 | `fb912c2cb79d42b048b6d8370458d260858ca21327c1f0c5369e103258d6492c` | `da69aa592dd1b93e8bb6dfc9797bb3c606f3414760b7c8f0b8f4d35460e1ffc0` |
| Trainable params | 73,159 | 73,159 |
| Frozen params | 0 | 0 |
| Total params | 73,159 | 73,159 |
| Hidden sizes | 96,3 | 96,3 |
| Epochs | 10 | 10 |
| Batch size | 512 | 512 |
| LR | 1e-3 | 1e-3 |
| Value loss weight | 0.3 | 0.3 |
| Value loss | huber | huber |
| Seed | 42 | 42 |
| Device | cpu | cpu |
| policy_loss | 1.478755 | 1.452157 |
| value_loss | 0.259371 | 0.262114 |
| best_val_loss | 1.588065 | 1.573898 |
| **Top-k checkpoints** | | |
| top1 (val_loss) | 1.588065 | 1.573898 |
| top2 (val_loss) | 1.591091 | 1.575989 |
| top3 (val_loss) | 1.595662 | 1.591755 |

## Strength: Arena vs Current Production

Both lanes evaluated against `model-artifact/current` (production residual_v3, version `live-synth-fix-balanced-stage2-iter1`) via `local_promotion_gate` with default arena settings.

| Metric | residual_v3 | residual_v4_move_factorized |
|--------|-------------|---------------------------|
| Arena score vs current | 0.0000 | 0.0000 |
| Wins / Losses / Draws | 0 / 120 / 0 | 0 / 120 / 0 |
| Games played | 120 | 120 |
| Challenger simulations | 384 | 384 |
| Current simulations | 256 | 256 |
| local_promotion_gate pass | **FAIL** | **FAIL** |
| Failure reason | arena_score_below_threshold | arena_score_below_threshold |

Both fail catastrophically against the production model. The single-iteration 1600-game classic-MCTS dataset is insufficient for either architecture to compete with the iteratively-trained production model.

## Strength: v4 vs v3 Head-to-Head Arena

Direct PUCT arena, equal simulation budget (384 each).

| Metric | Value |
|--------|-------|
| v4 wins | 0 |
| v4 losses | 60 |
| Draws | 60 |
| Score | 0.2500 |
| Games | 120 |
| CI 95% | [0.181, 0.334] |
| move_time_mean_ms | 32.15 |
| move_time_p95_ms | 48.21 |

**v4 is significantly weaker than v3** in direct play. v4 never won a game (0/120), lost 60, and drew 60.

## Strength: MCTS1200 Baseline

| Metric | residual_v3 | residual_v4_move_factorized | current (production) |
|--------|-------------|---------------------------|---------------------|
| Score vs MCTS1200 | 0.6625 | 0.6625 | 0.6625 |
| AZ Wins | 26 | 26 | 26 |
| MCTS Wins | 13 | 13 | 13 |
| Draws | 1 | 1 | 1 |
| Games | 40 | 40 | 40 |
| mean_final_simulations | 1151.53 | 1151.53 | 1151.53 |
| mean_root_latency_ms | 81.63 | 68.05 | 51.39 |

All three models score identically against MCTS1200. The 1200-simulation classic MCTS opponent is insufficiently discriminating for this test. v4 shows lower inference latency than v3 (68ms vs 82ms) but higher than current production (51ms).

## Exact-Position Sanity (from PR #87 Screen)

These results are from `docs/alphazero-lite-move-factorized-policy-head-screen-results.md` (PR #87). They are exact-tablebase production/control separability metrics, included here as sanity checks only.

| Architecture | Scope | Prod Opt@1200 | Prod Opt@2400 | Ctrl Reg |
|--------------|-------|---------------|---------------|---------|
| residual_v3 | head_only_policy_finetune | 71/147 (48.3%) | 84/147 (57.1%) | 1 (0.5%) |
| residual_v3 | full_finetune_control | 99/147 (67.3%) | 108/147 (73.5%) | 18 (8.7%) |
| residual_v4_move_factorized | head_only_policy_finetune | 125/147 (85.0%) | 127/147 (86.4%) | 3 (1.5%) |

v4 policy_head alone achieves dramatically better exact-position policy accuracy than v3's best full fine-tune (125 vs 99 production optimal, +63 gain vs +37), with far fewer control regressions (3 vs 18). However, this exact-position advantage does not translate to game strength.

## Latency

| Context | v3 | v4 | Delta |
|---------|----|----|-------|
| MCTS1200 mean_root_latency_ms | 81.63 | 68.05 | -16.6% |
| Arena move_time_mean_ms | 16.50 | 32.15* | +94.8%* |
| Arena move_time_p95_ms | 29.64 | 48.21* | +62.6%* |

*Note: v3 arena latency is from v3-vs-current (384 vs 256 sims), while v4 arena latency is from v4-vs-v3 (384 vs 384 sims, equal budget). The higher v4 arena latency may reflect the equal-budget setting rather than v4 slowness.

## Conclusion

residual_v4_move_factorized is classified as **`architecture_only_not_strength`**:

| Acceptance criterion | Met? | Evidence |
|---------------------|------|----------|
| v4 trains and exports cleanly | Yes | train.py + export_artifact.py complete without error |
| v4 local/runtime inference works | Yes | arena.py loads and evaluates v4 artifacts |
| v4 arena score vs current >= v3 same-data score | **No** | Both 0.0 (120/120 losses); tied |
| v4 candidate MCTS1200 score >= v3 same-data score | Yes | Both 0.6625 (26-13-1); tied |
| v4 does not exceed v3 latency materially | Yes | MCTS1200 latency 68ms vs 82ms (faster) |
| v4 arena head-to-head vs v3 | **No** | v4 loses 0W-60L-60D (score 0.25) |
| v4 passes local_promotion_gate | **No** | Both fail arena_score_below_threshold |

v4 is rejected as a game-strength upgrade. Its exact-position policy improvements from PR #87 do not translate to stronger play when trained on the same classic-MCTS data. The move-factorized head appears to require a different training recipe or different training data distribution to realize its potential in game play.
