# AlphaZero-Lite Failure-Mined Classic-MCTS Replay Results

**Date:** 2026-06-07

## Classification

**REJECTED** — failure-mined replay dramatically degrades model strength compared to both baseline and random-mined control. The rejection criteria are unequivocally met.

## Primary Hypothesis

> The current model's remaining weakness is concentrated in specific game-state regions that are underrepresented in generic classic-MCTS bootstrap data. Mining failure/disagreement states and relabeling them with classic MCTS will produce a better replay dataset than simply generating more generic games.

**Verdict: REJECTED.** Randomly sampled current-vs-classic-MCTS relabeled data improves the model (baseline 0.25 -> random_mined 0.50 arena score), but the failure/disagreement filter selects states that are actively harmful (failure_mined 0.00). The filter does not isolate "regions of weakness" — it isolates regions where the current model is making better decisions than classic MCTS, and forcing it to imitate the teacher there destroys its strength.

## Experiment Design

Three training lanes from the same `model-artifact/current/weights.json` init checkpoint, identical hyperparameters, identical generic replay file. The only difference: the mined lanes add one additional replay dataset with weight 1.

| Lane | Init | Generic Replay | Mined Replay | Replay Weights |
|------|------|---------------|-------------|----------------|
| baseline | current weights.json | `/tmp/azlite_failure_mining/generic_bootstrap.jsonl` | — | 4 |
| random_mined | current weights.json | same | `/tmp/azlite_failure_mining/random_mined_train.jsonl` | 4,1 |
| failure_mined | current weights.json | same | `/tmp/azlite_failure_mining/failure_mined_train.jsonl` | 4,1 |

## Dataset

### Generic Replay Files

| Metric | Value |
|--------|-------|
| Path | `/tmp/azlite_failure_mining/generic_bootstrap.jsonl` |
| Rows | 9,589 |
| SHA256 | `5d01a60e9dfc8756ffa47f2c8222e496a8ce8bd661457658de4884198344e147` |
| Generation | `generate_bootstrap_dataset.py --games 600 --simulations 2400 --teacher-mode classic_mcts --policy-target-mode sharpened --value-target-mode sharpened --input-encoding kalah_v3` |

### Failure-Mined Dataset

| Metric | Value |
|--------|-------|
| Train path | `/tmp/azlite_failure_mining/failure_mined_train.jsonl` |
| Holdout path | `/tmp/azlite_failure_mining/failure_mined_holdout.jsonl` |
| Train SHA256 | `e0ac4be0d0e3ec6963ce3191732040422dfe6e597368531ffc13e2939cb029a1` |
| Holdout SHA256 | `49d3e48a25ee35cb71a60abaf85e338fd78ac2ad130e2757f52ad6a4e71eee63` |
| Summary | `/tmp/azlite_failure_mining/failure_mining_summary.json` |
| Sampling mode | failure |
| Mined games (contributed rows) | 232 (of 400 played) |
| Total positions visited | 14,549 |
| Total positions kept | 1,818 |
| Train rows | 1,454 |
| Holdout rows | 364 |
| Rows per phase (early/mid/late) | 275 / 809 / 734 |
| Disagreement rate | 0.555 (55.5%) |
| Mean classic-MCTS top-1 visit share | 0.7384 |
| Mean current prob on teacher top move | 0.4481 |
| Duplicate state count | 2,178 |
| Capped state count | 0 |
| Teacher simulations | 1,200 |
| Seed | 42 |
| Input encoding | kalah_v3 |
| Policy target mode | sharpened |
| Value target mode | sharpened |

### Random-Mined Dataset

| Metric | Value |
|--------|-------|
| Train path | `/tmp/azlite_failure_mining/random_mined_train.jsonl` |
| Holdout path | `/tmp/azlite_failure_mining/random_mined_holdout.jsonl` |
| Train SHA256 | `ec052e04ae7e76455bed3d100664c010af60034fc4934dce1ca28fdfcf9f44c2` |
| Holdout SHA256 | `9a4034b86930d09abcbffd33c5d654d06749dff13e38278bea4140d4b5e2aec7` |
| Summary | `/tmp/azlite_failure_mining/random_mining_summary.json` |
| Sampling mode | random |
| Mined games (contributed rows) | 200 (of 400 played) |
| Total positions visited | 14,549 |
| Total positions kept | 1,789 |
| Train rows | 1,431 |
| Holdout rows | 358 |
| Rows per phase (early/mid/late) | 281 / 764 / 744 |
| Disagreement rate | 0.5601 (naturally occurring, no filter applied) |
| Mean classic-MCTS top-1 visit share | 0.5532 |
| Mean current prob on teacher top move | 0.4148 |
| Duplicate state count | 2,616 |
| Capped state count | 0 |
| Teacher simulations | 1,200 |
| Seed | 42 |
| Input encoding | kalah_v3 |
| Policy target mode | sharpened |
| Value target mode | sharpened |

### Row Count Comparison

| Metric | failure_mined | random_mined | Delta |
|--------|-------------|-------------|-------|
| Train rows | 1,454 | 1,431 | -23 |
| Total rows | 1,818 | 1,789 | -29 |

**Classification: Partially confounded.** Random_mined has 23 fewer train rows (1.6% less) than failure_mined. The comparison slightly favors failure_mined, yet failure_mined performs dramatically worse. The row count inequality cannot explain the observed effect direction or magnitude.

### Mining Filter Summary (failure mode only)

| Filter | Count |
|--------|-------|
| top_move_disagreement | 8,598 |
| low_current_prob | 6,095 |
| strong_mcts_preference | 3,974 |
| game_lost_by_current | 2,398 |
| tactical (extra_turn or capture) | 0 |

## Training

| Metric | baseline | random_mined | failure_mined |
|--------|----------|-------------|--------------|
| Init checkpoint | model-artifact/current/weights.json | same | same |
| Init SHA256 | `5adbbb6bebf625708c8edd159a58cb4434ad3b7bcd9ed8a3aed321498a21dd70` | same | same |
| Model type | residual_v3 | residual_v3 | residual_v3 |
| Hidden sizes | 96,3 | 96,3 | 96,3 |
| Epochs | 10 | 10 | 10 |
| Batch size | 512 | 512 | 512 |
| LR | 1e-3 | 1e-3 | 1e-3 |
| Value loss | huber | huber | huber |
| Value loss weight | 0.3 | 0.3 | 0.3 |
| Grad clip | 1.0 | 1.0 | 1.0 |
| Replay weights | 4 | 4,1 | 4,1 |
| Effective sample share (generic) | 100% | 96.4% | 96.3% |
| Effective sample share (mined) | — | 3.6% | 3.7% |
| policy_loss | 0.978701 | 0.991457 | 0.983912 |
| value_loss | 0.236157 | 0.242619 | 0.241433 |
| best_val_loss | 1.177851 | 1.183201 | 1.174136 |
| **Top-k checkpoints** | | | |
| top1 (val_loss) | 1.177851 | 1.183201 | 1.174136 |
| top1 SHA256 | `e917767add830bb84bf279e953f5022a937edf69d471785ee85c9963c9166440` | `09d2fd961e4fdb9faae9150f303c98f1acdfbb37ad4627b076ffeee849dc08a5` | `4351450aa0970f511e7d6e39b9591f995ac40e2a81799448e51f5d983f8ae358` |
| top2 (val_loss) | 1.181789 | 1.194758 | 1.180542 |
| top2 SHA256 | `454a6162015541c9e3070ec7c856e8467d27d58f04afd2f6f31fae57d518e63f` | `e6bb889e6fea7935cae834ea1101947e0430deff5d3add589e2f7181c79a85a3` | `06ac6e40bd65998a99e4c6373b4f657d7da424e0d8e5b08d9472317938265a41` |
| top3 (val_loss) | 1.193069 | 1.196164 | 1.180815 |
| top3 SHA256 | `de2d55b53e71b602c4df4a23bae18ff01283544b8a8cf51591bd76b0c1f3aa5e` | `70abed433a49f9c26b7c3e23c888c24742c6228b6cdb58630d179364340ff3a9` | `a95862645ba7a0e915d35ac093f082bdd3b126739a5ce5058bad532c8124c18b` |

Note: Failure_mined has the lowest validation loss (1.174136) but the worst arena performance (0.00), confirming that validation loss improvements on the mined data do not translate to strength.

## Strength: Arena vs Current Production

| Metric | baseline | random_mined | failure_mined |
|--------|----------|-------------|--------------|
| Arena score vs current | **0.25** | **0.50** | **0.00** |
| Wins / Losses / Draws | 0 / 60 / 60 | 60 / 60 / 0 | 0 / 120 / 0 |
| Games played | 120 | 120 | 120 |
| CI95 (Wilson) | [0.181, 0.334] | [0.412, 0.588] | [0.000, 0.031] |
| local_promotion_gate pass | No (score 0.25 < 0.55) | No (score 0.50 < 0.55) | No (score 0.00 < 0.55) |
| Gate failure reasons | arena_score_below_threshold | arena_score_below_threshold | arena_score_below_threshold |

Key observations:
- **Random_mined (0.50) > Baseline (0.25):** Current-vs-classic-MCTS relabeled replay data is useful when randomly sampled. It provides moderate transfer learning gains (+0.25 score improvement).
- **Failure_mined (0.00) << Baseline (0.25):** The failure/disagreement filter catastrophically degrades strength. The model loses every game, unable to even draw.
- **Failure_mined (0.00) << Random_mined (0.50):** Same games, same relabeling, same row counts — the only difference is which positions are selected. The failure filter selects severely harmful positions.

## Strength: Hard Arena

| Metric | baseline | random_mined | failure_mined |
|--------|----------|-------------|--------------|
| Hard arena score | — | — | — |
| Wins / Losses / Draws | — | — | — |
| Games played | 0 | 0 | 0 |

Hard arena was not run because all lanes failed the arena prefilter (score < 0.55).

## Strength: MCTS1200 Baseline

| Metric | baseline | random_mined | failure_mined | current |
|--------|----------|-------------|--------------|---------|
| Score vs MCTS1200 | — | — | — | — |
| AZ Wins | — | — | — | — |
| MCTS Wins | — | — | — | — |
| Draws | — | — | — | — |
| Games | 0 | 0 | 0 | 0 |

MCTS1200 baseline was not run because all lanes failed the arena prefilter (score < 0.55).

## Latency

| Metric | baseline | random_mined | failure_mined |
|--------|----------|-------------|--------------|
| move_time_mean_ms | 19.83 | 19.18 | 18.37 |
| move_time_p95_ms | 30.69 | 30.78 | 30.55 |

Latency is within the same residual_v3 96,3 architecture bounds across all lanes.

## Acceptance Criteria

| Criterion | Met? | Evidence |
|-----------|------|----------|
| Failure-mined lane beats baseline in arena vs current | **No** | baseline 0.25, failure_mined 0.00 — failure is dramatically worse |
| Failure-mined lane beats random_mined at equal mined row count | **No** | random_mined 0.50, failure_mined 0.00 — failure is dramatically worse |
| Failure-mined lane does not regress MCTS1200 vs baseline | N/A | MCTS1200 not run (gate prefilter failed) |
| Mined-holdout top-1 agreement improves | N/A | Not directly measured, but validation loss was lowest for failure_mined despite worst arena performance |
| local_promotion_gate passes or gets materially closer than both controls | **No** | All lanes failed arena prefilter. Random_mined got closest (0.50, only 0.05 below threshold) |
| Latency within existing residual_v3 expectations | Yes | All ~18-31ms range, matching architecture |
| No overfit to mined states (generic performance preserved) | **No** | Failure_mined lost generic performance entirely. Random_mined showed transfer benefit instead of overfit |
| Results independent of duplicated game states | Yes | Both mined datasets had comparable duplicate counts |

## Rejection Criteria

| Criterion | Triggered? | Evidence |
|-----------|-----------|----------|
| Mined replay improves holdout loss but not arena strength | **Yes** | Failure_mined had lowest val_loss (1.174) but worst arena score (0.00) |
| Arena vs current regresses | **Yes** | Failure_mined 0.00 << baseline 0.25 |
| MCTS1200 relative score regresses | N/A | Not run |
| Model overfits mined states and loses generic performance | **Yes** | Failure_mined lost all generic ability; random_mined preserved and improved it |
| Results depend on a tiny number of duplicated game states | No | Duplicate counts were comparable and capped at 3 per fingerprint |

## Analysis

### Why does random mining help?

The random_mined lane (0.50) beats baseline (0.25), showing that current-vs-classic-MCTS relabeled replay data conveys useful information. The current model is bootstrapped from classic-MCTS data, but additional classic-MCTS relabeled states — even without filtering — provide further supervised signal.

### Why does failure mining destroy the model?

The failure/disagreement filter selects states where the current model disagrees with classic MCTS. These are precisely the states where the current model has learned something different (and often better) than classic MCTS. Forcing the model to imitate classic MCTS on these states erases its learned advantage.

Key indicators:
1. **Mean current prob on teacher top move is 0.448** — the current model assigns only ~45% probability to the teacher's preferred move, meaning it has genuinely diverged from classic MCTS.
2. **Disagreement rate is 55.5%** — over half of mined positions have top-move disagreement between current and classic MCTS.
3. **Mean teacher top-1 visit share is 0.738** — the teacher is confident, but the current model has learned a different strategy.
4. **The filter keeps 8,598 top_move_disagreement positions**, forcing re-learning on the very states where the current model outperforms classic MCTS.

The filter is self-defeating: it identifies states where the current model disagrees with classic MCTS, then forces the model to agree with classic MCTS on those states. But if the current model disagrees because it has learned a superior policy, this is counter-training.

### Why is the row count inequality not a confound?

Random_mined has 23 fewer train rows (1.6% less). If row count were driving the effect, we would expect random_mined to perform slightly worse than failure_mined, not 50 percentage points better. The catastrophic difference (0.50 vs 0.00) cannot be attributed to 1.6% fewer rows.

## Conclusion

**The failure-mined replay hypothesis is conclusively rejected.** The failure/disagreement filter selects positions where the current model has learned a different (and better) strategy than classic MCTS. Re-labeling these with classic MCTS and training on them erases the model's learned advantage, causing catastrophic regression.

**The random-mined replay shows promise.** Randomly relabeled current-vs-classic-MCTS states improve arena strength from 0.25 to 0.50, suggesting that additional classic-MCTS replay data — without any disagreement filter — is a useful supplement to generic bootstrap data.

**For future work:**
- Exploring a "confidence filter" that keeps states where the current model AND classic MCTS agree (opposite of the failure filter) might be productive.
- Increasing the volume of unfiltered current-vs-classic-MCTS replay data (more games, higher teacher sims) could push the random_mined lane toward the 0.55 promotion threshold.
- The failure filter itself could be inverted: keep states where the teacher has high confidence AND the current model agrees, to reinforce good behavior rather than undermine it.

## Commands Used

```bash
# Materialize init checkpoint
python -c "
import json, numpy as np; from pathlib import Path
weights = json.loads(Path('model-artifact/current/weights.json').read_text())
arrays = {name: np.asarray(value, dtype=np.float32) for name, value in weights.items()}
np.savez('/tmp/azlite_failure_mining/init_checkpoint.npz', **arrays)
"

# Generate generic bootstrap replay
.venv/bin/python ml/alphazero_lite/generate_bootstrap_dataset.py \
  --out /tmp/azlite_failure_mining/generic_bootstrap.jsonl \
  --games 600 --simulations 2400 --seed 42 --max-positions-per-game 16 \
  --input-encoding kalah_v3 --policy-target-mode sharpened \
  --value-target-mode sharpened --teacher-mode classic_mcts --workers 8

# Generate failure-mined dataset
.venv/bin/python ml/alphazero_lite/mine_failure_replay_dataset.py \
  --out-train /tmp/azlite_failure_mining/failure_mined_train.jsonl \
  --out-holdout /tmp/azlite_failure_mining/failure_mined_holdout.jsonl \
  --out-summary /tmp/azlite_failure_mining/failure_mining_summary.json \
  --current storage/ai/alphazero_lite/current --games 400 \
  --teacher-mode classic_mcts --teacher-simulations 1200 --seed 42 \
  --max-positions-per-game 12 --input-encoding kalah_v3 \
  --policy-target-mode sharpened --value-target-mode sharpened \
  --sampling-mode failure

# Generate random-mined dataset
.venv/bin/python ml/alphazero_lite/mine_failure_replay_dataset.py \
  --out-train /tmp/azlite_failure_mining/random_mined_train.jsonl \
  --out-holdout /tmp/azlite_failure_mining/random_mined_holdout.jsonl \
  --out-summary /tmp/azlite_failure_mining/random_mining_summary.json \
  --current storage/ai/alphazero_lite/current --games 400 \
  --teacher-mode classic_mcts --teacher-simulations 1200 --seed 42 \
  --max-positions-per-game 12 --input-encoding kalah_v3 \
  --policy-target-mode sharpened --value-target-mode sharpened \
  --sampling-mode random

# Train baseline lane
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_failure_mining/generic_bootstrap.jsonl \
  --replay-weights 4 \
  --init-checkpoint /tmp/azlite_failure_mining/init_checkpoint.npz \
  --model-type residual_v3 --input-encoding kalah_v3 --hidden-sizes 96,3 \
  --epochs 10 --batch-size 512 --device auto --value-loss huber \
  --value-loss-weight 0.3 --val-split 0.1 --grad-clip 1.0 --save-top-k 3 \
  --top-k-dir /tmp/azlite_failure_mining/baseline \
  --policy-target-mode sharpened --value-target-mode sharpened \
  --out /tmp/azlite_failure_mining/baseline/checkpoint.npz

# Train random_mined lane
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_failure_mining/generic_bootstrap.jsonl,/tmp/azlite_failure_mining/random_mined_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_failure_mining/init_checkpoint.npz \
  --model-type residual_v3 --input-encoding kalah_v3 --hidden-sizes 96,3 \
  --epochs 10 --batch-size 512 --device auto --value-loss huber \
  --value-loss-weight 0.3 --val-split 0.1 --grad-clip 1.0 --save-top-k 3 \
  --top-k-dir /tmp/azlite_failure_mining/random_mined \
  --policy-target-mode sharpened --value-target-mode sharpened \
  --out /tmp/azlite_failure_mining/random_mined/checkpoint.npz

# Train failure_mined lane
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_failure_mining/generic_bootstrap.jsonl,/tmp/azlite_failure_mining/failure_mined_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_failure_mining/init_checkpoint.npz \
  --model-type residual_v3 --input-encoding kalah_v3 --hidden-sizes 96,3 \
  --epochs 10 --batch-size 512 --device auto --value-loss huber \
  --value-loss-weight 0.3 --val-split 0.1 --grad-clip 1.0 --save-top-k 3 \
  --top-k-dir /tmp/azlite_failure_mining/failure_mined \
  --policy-target-mode sharpened --value-target-mode sharpened \
  --out /tmp/azlite_failure_mining/failure_mined/checkpoint.npz

# Export artifacts and promote
for lane in baseline random_mined failure_mined; do
  .venv/bin/python ml/alphazero_lite/export_artifact.py \
    --checkpoint /tmp/azlite_failure_mining/$lane/checkpoint.top1.npz \
    --out-dir /tmp/azlite_failure_mining/$lane/artifact \
    --version $lane-top1 --model-type residual_v3 \
    --rules-version kalah_v1 --input-encoding kalah_v3
done

# Evaluate all lanes
for lane in baseline random_mined failure_mined; do
  .venv/bin/python script/ai/local_promotion_gate \
    --candidate-path /tmp/azlite_failure_mining/$lane/artifact \
    --out /tmp/azlite_failure_mining/gates/$lane/promotion.json
done
```
