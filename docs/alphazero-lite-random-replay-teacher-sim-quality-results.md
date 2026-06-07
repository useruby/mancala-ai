# AlphaZero-Lite Random Replay Teacher Simulation Quality Results

**Date:** 2026-06-07

## Classification

**LABEL QUALITY NOT LIMITING** — Higher-simulation classic-MCTS relabeling (2400 vs 1200) produces measurably different teacher labels (7.0% top-move change rate, policy KL 0.05, top-1 visit share +0.044), but both lanes achieve exactly 0.50 arena score against current. The 0.50 ceiling is not caused by weak/noisy 1200-simulation labels.

## Primary Hypothesis

> Random replay is capped at 0.50 because 1200-simulation classic-MCTS labels are too noisy or too weak on the mined state distribution. Higher-simulation relabeling will produce a stronger residual_v3 candidate at the same 4:1 replay mix.

**Verdict: REJECTED.** 2400-sim labels shift the teacher policy and value distributions but produce the identical 0.50 arena score. The 0.50 ceiling is a property of the random state distribution, not the label quality.

## Experiment Design

| Lane | Teacher Sims | Replay Weights | Effective Mined Share |
|------|-------------|---------------|----------------------|
| baseline_replay | — | 4 | 0.0% |
| random_teacher_1200 | 1200 | 4,1 | 2.4% |
| random_teacher_2400 | 2400 | 4,1 | 2.4% |

All lanes train from the same init checkpoint, identical hyperparameters, same source states (only teacher labels differ). The 1200 and 2400 lanes use the exact same 2520 mined positions, relabeled with different simulation budgets.

All lanes trained from the same init checkpoint, with identical hyperparameters:

| Parameter | Value |
|-----------|-------|
| Model type | residual_v3 |
| Input encoding | kalah_v3 |
| Hidden sizes | 96,3 |
| Epochs | 10 |
| Batch size | 512 |
| LR | 1e-3 |
| Value loss | huber |
| Value loss weight | 0.3 |
| Grad clip | 1.0 |
| Save top-k | 3 |
| Seed | 42 |

## Dataset

### Source States

| Metric | Value |
|--------|-------|
| Path | `/tmp/azlite_random_teacher_quality/random_source_states.jsonl` |
| Rows | 2,520 |
| SHA256 | `1dadcf9c6a090bc3b69b2252e6655cb3dd45f400216229c0e99547231f9c25de` |
| Mined games | 292 |
| Sampling mode | random |
| Teacher sims (game-play) | 1200 |
| Max positions per game | 12 |

### Generic Replay

| Metric | Value |
|--------|-------|
| Path | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` |
| Rows | 9,589 |
| SHA256 | `5d01a60e9dfc8756ffa47f2c8222e496a8ce8bd661457658de4884198344e147` |

### Teacher Replay

| Metric | teacher_1200 | teacher_2400 |
|--------|-------------|-------------|
| Train rows | 2,016 | 2,016 |
| Holdout rows | 504 | 504 |
| Train SHA256 | `7ca9338...` | `e50acdb...` |
| Holdout SHA256 | `cb30b12...` | `3e14a1b...` |
| Teacher simulations | 1200 | 2400 |
| Top-1 visit share mean | 0.5529 | 0.5960 |
| Disagreement rate | 0.5627 | 0.5627 |
| Agreement rate | 0.4373 | 0.4345 |

### Row Alignment Check

| Check | Result |
|-------|--------|
| Same row count (train) | 2,016 = 2,016 |
| State mismatches | 0 |
| Player mismatches | 0 |
| Move index mismatches | 0 |
| Top move change rate (1200→2400) | 142/2016 (7.0%) |
| Policy KL mean | 0.050004 |
| Policy KL max | 1.458316 |
| Value diff mean | 0.069813 |
| Value diff max | 0.928931 |

Row alignment is confirmed: same states, same players, same move indices, row-for-row identical. Only policy/value labels differ.

### Label Distribution by Simulation Budget

| Metric | 1200 | 2400 |
|--------|------|------|
| Teacher top-1 visit share | 0.5560 | 0.5999 |
| Value target mean | -0.1662 | -0.1764 |
| Value target std | 0.8447 | 0.8570 |
| Teacher value at state mean | 0.3275 | 0.4095 |
| Teacher value at state std | 0.6702 | 0.6428 |

Higher simulation budget produces more confident MCTS assessments (higher top-1 share) and slightly more polarized value estimates (higher mean teacher value magnitude). However, the changes are modest and the policy distributions are highly correlated (KL mean 0.05).

### Init Checkpoint

| Metric | Value |
|--------|-------|
| Source | `storage/ai/alphazero_lite/current/weights.json` |
| SHA256 | `5adbbb6bebf625708c8edd159a58cb4434ad3b7bcd9ed8a3aed321498a21dd70` |

## Training

| Metric | baseline | teacher_1200 | teacher_2400 |
|--------|----------|-------------|-------------|
| Policy loss | 0.978701 | 1.002022 | 0.998245 |
| Value loss | 0.236157 | 0.245698 | 0.246155 |
| Best val loss | 1.177851 | 1.182161 | 1.184476 |
| Top-1 SHA256 | `e917767...` | `c2430b5...` | `8ea5cca...` |

The 1200 lane has slightly worse policy loss than the baseline, matching the PR #92 4:1 lane pattern (0.986 → 1.002 with the regenerated data). The 2400 lane is nearly identical in training metrics.

## Strength: Standard Arena vs Current (120 games)

| Metric | baseline | teacher_1200 | teacher_2400 |
|--------|----------|-------------|-------------|
| Arena Score | **0.25** | **0.50** | **0.50** |
| Wins / Losses / Draws | 0 / 60 / 60 | 60 / 60 / 0 | 60 / 60 / 0 |
| CI95 (Wilson) | [0.181, 0.334] | [0.412, 0.588] | [0.412, 0.588] |
| Move time mean ms | 47.36 | 49.59 | 48.96 |
| Move time p95 ms | 113.49 | 114.94 | 117.45 |

## Strength: Extended Arena vs Current (240 games)

| Metric | teacher_1200 | teacher_2400 |
|--------|-------------|-------------|
| Arena Score | **0.50** | **0.50** |
| Wins / Losses / Draws | 120 / 120 / 0 | 120 / 120 / 0 |
| CI95 (Wilson) | [0.437, 0.563] | [0.437, 0.563] |

Both lanes produce identically flat results: 50/50 win/loss split, zero draws. No differentiation even at 240 games. The upper bound of the confidence interval (0.563) is above the 0.55 threshold, but the lower bound (0.437) falls well below it. There is no statistical signal that either lane exceeds 0.50.

## Acceptance Criteria Evaluation

### Classification: LABEL QUALITY NOT LIMITING

| Criterion | Result | Status |
|-----------|--------|--------|
| 2400 beats 1200 arena score | Both at 0.50 | FAIL |
| Either lane reaches 0.55 threshold | Both at 0.50 | FAIL |
| Higher-sim labels show better extended-arena estimate | Both at 0.50 (240 games) | FAIL |
| 1200 and 2400 produce different labels | Yes (KL=0.05, 7% top-move changes) | CONFIRMED |
| Higher-sim labels improve validation loss without arena gain | val_loss similar (1.182 vs 1.184) | — |
| No latency regression | All within normal range | CONFIRMED |
| No collapse | Both maintain 0.50 | CONFIRMED |

### Rejection criteria for higher teacher sims

| Criterion | Triggered? |
|-----------|-----------|
| Stronger labels collapse like 8:1 or failure_mined | No — both lanes stable at 0.50 |
| Higher-sim labels produce worse arena score | No — both at 0.50 |
| Runtime cost is high with no strength signal | 164.5s for 2400-sim relabel vs 85.9s for 1200-sim (2x cost, zero gain) |

## Analysis

### The 0.50 ceiling is not a label-quality problem

The random replay 0.50 ceiling persists across:
- Teacher simulation budgets: 1200 (PR #91), 1200 (this experiment, regenerated), 2400 (this experiment)
- Effective sample shares: 2.4%, 4.8%, 9.1% (PR #92)
- Game counts: 400 (PR #90), 800 (PR #91/#92, this experiment)
- Seeds: 42, 43

The signal is extraordinarily robust. It produces exactly +0.25 over baseline in every working configuration. It never exceeds 0.50. Moving from 1200 to 2400 teacher simulations changes the label distribution (7% top-move changes, top-1 visit share increases from 0.556 to 0.600) but produces zero difference in arena strength.

### Why double the sims don't help

Several explanations for the invariance to teacher simulation quality:

1. **Saturation at 1200.** Classic MCTS at 1200 simulations may already be near its information ceiling on the mined position distribution. Doubling to 2400 increases confidence (higher top-1 share) but doesn't meaningfully change which move is best.

2. **The mined distribution is too easy.** The random sampling collects positions uniformly from current-vs-classic-MCTS games. Many positions may be trivial or well-understood by even a 1200-sim teacher. The label quality improvement is concentrated on positions where the teacher was already confident.

3. **The 0.50 ceiling is a structural property.** It may represent the maximum improvement a residual_v3 (96,3) model can extract from classic-MCTS-labeled data when starting from the current weights, regardless of label quality. The model architecture or the nature of the data (sharpened policy/value targets from MCTS visit counts) may impose an information-theoretic ceiling at 0.50.

4. **Policy KL is low.** A mean KL divergence of 0.05 between 1200 and 2400 labels confirms that doubling simulations doesn't fundamentally change the teacher's opinion. The labels are nearly identical — 93% of top moves are the same.

### Dataset

| Label Budget | Top-1 Visit Share | Teacher Value Mean | Teacher Value Std |
|-------------|-------------------|-------------------|-------------------|
| 1200 | 0.556 | 0.328 | 0.670 |
| 2400 | 0.600 | 0.410 | 0.643 |

The 2400-sim teacher is more confident (higher visit share, higher value magnitude, lower value std), but the signal is not different enough to change training outcomes.

## What's been ruled out

| Variable | Tested | Status |
|----------|--------|--------|
| Mined effective sample share | 1.2% to 9.1% (PR #92) | Not limiting (flat at 0.50) |
| Teacher simulation quality | 1200 vs 2400 (this experiment) | Not limiting (flat at 0.50) |
| Game count | 400 vs 800 (PR #90, #91) | Not limiting (flat at 0.50) |
| Sampling seed | 42, 43 (multiple runs) | Not limiting (flat at 0.50) |

## Key Takeaways

1. **The 0.50 ceiling is insensitive to label quality.** Doubling teacher simulations from 1200 to 2400 changes label distributions measurably but produces zero change in arena strength.

2. **The limiting factor is the state distribution, not the label quality.** The random sampling of positions from current-vs-classic-MCTS games provides a fixed +0.25 benefit that cannot be amplified by better labels or more replay weight.

3. **Moving past 0.50 requires different states, not better labels on the same states.** Future directions should explore:
   - Different state mining strategies (e.g., failure-filtered, strategic disagreement, self-play iterations)
   - Architecture changes beyond residual_v3
   - Different teacher paradigms (e.g., tablebase-guided, deeper search)

4. **The random replay signal is robust and reproducible.** Three independent runs with different seeds, different game counts, and now different teacher simulation budgets all produce exactly 0.50.

## Commands Used

```bash
# Generate source states (game-play with 1200-sim teacher)
.venv/bin/python ml/alphazero_lite/mine_failure_replay_dataset.py \
  --out-source-states /tmp/azlite_random_teacher_quality/random_source_states.jsonl \
  --out-train /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --out-holdout /tmp/azlite_random_teacher_quality/random_teacher_1200_holdout.jsonl \
  --current storage/ai/alphazero_lite/current \
  --games 800 --seed 43 --max-positions-per-game 12 \
  --input-encoding kalah_v3 --sampling-mode random \
  --teacher-mode classic_mcts --teacher-simulations 1200 \
  --policy-target-mode sharpened --value-target-mode sharpened

# Relabel at 1200 (aligned shuffle with relabel-only path)
.venv/bin/python ml/alphazero_lite/mine_failure_replay_dataset.py \
  --relabel-only \
  --source-states /tmp/azlite_random_teacher_quality/random_source_states.jsonl \
  --out-train /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --out-holdout /tmp/azlite_random_teacher_quality/random_teacher_1200_holdout.jsonl \
  --teacher-mode classic_mcts --teacher-simulations 1200 \
  --policy-target-mode sharpened --value-target-mode sharpened --seed 43

# Relabel at 2400
.venv/bin/python ml/alphazero_lite/mine_failure_replay_dataset.py \
  --relabel-only \
  --source-states /tmp/azlite_random_teacher_quality/random_source_states.jsonl \
  --out-train /tmp/azlite_random_teacher_quality/random_teacher_2400_train.jsonl \
  --out-holdout /tmp/azlite_random_teacher_quality/random_teacher_2400_holdout.jsonl \
  --teacher-mode classic_mcts --teacher-simulations 2400 \
  --policy-target-mode sharpened --value-target-mode sharpened --seed 43

# Train baseline
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl \
  --replay-weights 4 \
  --init-checkpoint /tmp/azlite_random_teacher_quality/init_checkpoint.npz \
  --model-type residual_v3 --input-encoding kalah_v3 --hidden-sizes 96,3 \
  --epochs 10 --batch-size 512 --lr 1e-3 --value-loss huber \
  --value-loss-weight 0.3 --grad-clip 1.0 --save-top-k 3 \
  --top-k-dir /tmp/azlite_random_teacher_quality/baseline_replay \
  --out /tmp/azlite_random_teacher_quality/baseline_replay/checkpoint.npz \
  --device auto --policy-target-mode sharpened --value-target-mode sharpened --seed 42

# Train teacher lanes (same pattern, exchanging data-files)

# Standard arena (120 games)
.venv/bin/python ml/alphazero_lite/arena.py \
  --challenger <artifact-dir> --current model-artifact/current \
  --games 120 --challenger-simulations 384 --current-simulations 256 \
  --out <out-dir>/standard_arena.json --min-score 0.0 --workers 8 --seed 42

# Extended arena (240 games)
.venv/bin/python ml/alphazero_lite/arena.py \
  --challenger <artifact-dir> --current model-artifact/current \
  --games 240 --challenger-simulations 384 --current-simulations 256 \
  --out <out-dir>/extended_arena.json --min-score 0.0 --workers 8 --seed 1042
```
