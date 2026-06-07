# AlphaZero-Lite Failure-Mined Classic-MCTS Replay Results

**Date:** TBD (run date)

## Classification

TBD — run the experiment to determine:
- `promising` — failure-mined lane beats baseline lane in arena vs current
- `rejected` — mined replay improves mined-holdout loss but not arena strength

## Primary Hypothesis

> The current model's remaining weakness is concentrated in specific game-state regions that are underrepresented in generic classic-MCTS bootstrap data. Mining failure/disagreement states and relabeling them with classic MCTS will produce a better replay dataset than simply generating more generic games.

**Verdict:** TBD

## Experiment Design

Two training lanes from the same `storage/ai/alphazero_lite/current` init checkpoint, identical hyperparameters, identical generic replay files. The only difference: the failure-mined lane adds one targeted replay dataset with weight 1.

| Lane | Init | Generic Replay | Failure-Mined Replay | Replay Weights |
|------|------|---------------|---------------------|----------------|
| baseline | current model.npz | `<generic replay files>` | — | `<generic weights>` |
| failure_mined | current model.npz | `<generic replay files>` | `/tmp/azlite_failure_mining/failure_mined_train.jsonl` | `<generic weights>,1` |

## Dataset

### Generic Replay Files

| Metric | Value |
|--------|-------|
| Paths | TBD |
| SHA256 (each) | TBD |

### Failure-Mined Dataset

| Metric | Value |
|--------|-------|
| Train path | `/tmp/azlite_failure_mining/failure_mined_train.jsonl` |
| Holdout path | `/tmp/azlite_failure_mining/failure_mined_holdout.jsonl` |
| Train SHA256 | TBD |
| Holdout SHA256 | TBD |
| Summary | `/tmp/azlite_failure_mining/failure_mining_summary.json` |
| Mined games | TBD |
| Mined rows (total) | TBD |
| Train rows | TBD |
| Holdout rows | TBD |
| Rows per phase (early/mid/late) | TBD |
| Disagreement rate | TBD |
| Mean classic-MCTS top-1 visit share | TBD |
| Mean current prob on teacher top move | TBD |
| Duplicate state count | TBD |
| Capped state count | TBD |
| Teacher simulations | 1200 |
| Seed | 42 |
| Input encoding | kalah_v3 |
| Policy target mode | default |
| Value target mode | default |

### Mining Filter Summary

| Filter | Count |
|--------|-------|
| top_move_disagreement | TBD |
| low_current_prob | TBD |
| strong_mcts_preference | TBD |
| game_lost_by_current | TBD |
| tactical (extra_turn or capture) | TBD |

## Training

| Metric | baseline | failure_mined |
|--------|----------|--------------|
| Init checkpoint | storage/ai/alphazero_lite/current/model.npz | same |
| Model type | residual_v3 | residual_v3 |
| Hidden sizes | TBD | TBD |
| Epochs | TBD | TBD |
| Batch size | TBD | TBD |
| LR | TBD | TBD |
| Value loss | TBD | TBD |
| Value loss weight | TBD | TBD |
| Replay weights | TBD | TBD |
| Effective sample share (generic) | TBD | TBD |
| Effective sample share (failure-mined) | — | TBD |
| policy_loss | TBD | TBD |
| value_loss | TBD | TBD |
| best_val_loss | TBD | TBD |
| mined-holdout policy loss | — | TBD |
| mined-holdout top-1 agreement with classic MCTS | — | TBD |
| **Top-k checkpoints** | | |
| top1 (val_loss) | TBD | TBD |
| top1 SHA256 | TBD | TBD |
| top2 (val_loss) | TBD | TBD |
| top2 SHA256 | TBD | TBD |
| top3 (val_loss) | TBD | TBD |
| top3 SHA256 | TBD | TBD |

## Strength: Arena vs Current Production

| Metric | baseline | failure_mined |
|--------|----------|--------------|
| Arena score vs current | TBD | TBD |
| Wins / Losses / Draws | TBD | TBD |
| Games played | 120 | 120 |
| local_promotion_gate pass | TBD | TBD |

## Strength: Hard Arena

| Metric | baseline | failure_mined |
|--------|----------|--------------|
| Hard arena score | TBD | TBD |
| Wins / Losses / Draws | TBD | TBD |
| Games played | 120 | 120 |

## Strength: MCTS1200 Baseline

| Metric | baseline | failure_mined | current |
|--------|----------|--------------|---------|
| Score vs MCTS1200 | TBD | TBD | TBD |
| AZ Wins | TBD | TBD | TBD |
| MCTS Wins | TBD | TBD | TBD |
| Draws | TBD | TBD | TBD |
| Games | 40 | 40 | 40 |

## Latency

| Metric | baseline | failure_mined |
|--------|----------|--------------|
| move_time_mean_ms | TBD | TBD |
| move_time_p95_ms | TBD | TBD |

## Acceptance Criteria

| Criterion | Met? | Evidence |
|-----------|------|----------|
| Failure-mined lane beats baseline in arena vs current | TBD | |
| Failure-mined lane does not regress MCTS1200 vs baseline | TBD | |
| Mined-holdout top-1 agreement improves | TBD | |
| local_promotion_gate passes or gets closer than baseline | TBD | |
| Latency within existing residual_v3 expectations | TBD | |
| No overfit to mined states (generic performance preserved) | TBD | |
| Results independent of duplicated game states | TBD | |

## Rejection Criteria

| Criterion | Triggered? | Evidence |
|-----------|-----------|----------|
| Mined replay improves holdout loss but not arena strength | TBD | |
| Arena vs current regresses | TBD | |
| MCTS1200 relative score regresses | TBD | |
| Model overfits mined states and loses generic performance | TBD | |
| Results depend on a tiny number of duplicated game states | TBD | |

## Conclusion

TBD

## Commands Used

```bash
# Generate failure-mined dataset
.venv/bin/python ml/alphazero_lite/mine_failure_replay_dataset.py \
  --out-train /tmp/azlite_failure_mining/failure_mined_train.jsonl \
  --out-holdout /tmp/azlite_failure_mining/failure_mined_holdout.jsonl \
  --current storage/ai/alphazero_lite/current \
  --games 400 \
  --teacher-mode classic_mcts \
  --teacher-simulations 1200 \
  --seed 42 \
  --max-positions-per-game 12 \
  --input-encoding kalah_v3

# Train baseline lane
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files <generic_replay_files> \
  --replay-weights <generic_weights> \
  --init-checkpoint <materialized_npz> \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 10 \
  --batch-size 512 \
  --device auto \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --val-split 0.1 \
  --grad-clip 1.0 \
  --save-top-k 3 \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --out /tmp/azlite_failure_mining/baseline/checkpoint.npz

# Train failure-mined lane
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files <generic_replay_files>,/tmp/azlite_failure_mining/failure_mined_train.jsonl \
  --replay-weights <generic_weights>,1 \
  --init-checkpoint <materialized_npz> \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 10 \
  --batch-size 512 \
  --device auto \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --val-split 0.1 \
  --grad-clip 1.0 \
  --save-top-k 3 \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --out /tmp/azlite_failure_mining/failure_mined/checkpoint.npz

# Evaluate best baseline
script/ai/local_promotion_gate \
  --candidate-path <best_baseline_artifact_dir> \
  --out /tmp/azlite_failure_mining/baseline_gate.json

# Evaluate best failure-mined
script/ai/local_promotion_gate \
  --candidate-path <best_failure_mined_artifact_dir> \
  --out /tmp/azlite_failure_mining/failure_mined_gate.json
```
