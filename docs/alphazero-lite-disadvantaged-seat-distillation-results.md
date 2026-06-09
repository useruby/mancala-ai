# AlphaZero-Lite Disadvantaged-Seat Distillation Results

**Date:** 2026-06-09

## Summary

Trained disadvantaged-seat high-search distillation curriculum: player-1 states
from iter0_reference labeled by high-search (1200-sim) classic MCTS, mixed into
the original iter0 training data. Evaluated whether the distillation shifts raw
policy enough that lower-budget MCTS finds the same disadvantaged-seat lines
that iter0_reference only finds at 1200:1200.

**Overall Classification: `high_search_not_distillable`**

Distillation improves holdout imitation loss (1.621 vs 1.629 iter0_reference)
but does not improve disadvantaged-seat score at 384:256 or 768:256 under
default search. The 1200:1200 breakthrough (DS=1.00) is preserved across all
checkpoints. No collapse at 256:768.

## Artifact Lineage

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference checkpoint | /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| iter0_reference artifact | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |

## Distillation Dataset

| Metric | Value |
|--------|-------|
| Train rows | 1024 |
| Holdout rows | 256 |
| Total kept positions | 3323 |
| P1 positions visited | 4116 |
| Games with rows | 240 / 480 |
| Train SHA256 | `44f32b1e8101c90815346a0425408d5fd03734ef1e413511cc744f8c5c062dad` |
| Holdout SHA256 | `a61e55b5db5b174037f313ae2c5c86d0321af455d15a863a6b68effa20bd8e42` |
| Random opening plies | 4 |

### Distillation Data Quality

| Metric | Value |
|--------|-------|
| Disagreement rate (low-budget top != high-search top) | 68.46% |
| Mean teacher top-1 visit share | 0.633 |
| Mean low-budget prob on teacher top move | 0.3503 |
| Mean low-budget policy entropy | 0.9548 |
| First-divergence ply mean | 1.35 |
| First-divergence ply median | 1.0 |
| Duplicate state count | 458 |
| Capped state count | 0 |

### Ply Distribution

| Phase | Rows |
|-------|------|
| Early (<=8) | 768 |
| Mid (9-24) | 1410 |
| Late (>=25) | 1145 |

### Filter Breakdown

| Filter | Positions |
|--------|-----------|
| Top-move disagreement | 2275 |
| Strong high-search preference (visit share >= 0.55) | 1949 |
| Low low-budget prob on teacher top (<= 0.30) | 1982 |
| No divergence (positions that failed all filters) | 710 |

## Training Configuration

All lanes use: residual_v3 (96,3), kalah_v3, batch_size=512, huber value
loss (weight=0.3), grad_clip=1.0, lr_scheduler=none, val_split=0.1, seed=42,
policy_target_mode=sharpened, value_target_mode=sharpened.

### Data Inputs

| Dataset | Path | Weight |
|---------|------|--------|
| generic bootstrap | /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl | 4 |
| old current-mined random replay | /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl | 1 |
| p1 distillation (distill lane only) | /tmp/azlite_p1_distill/p1_distill_train.jsonl | 1 |

### Experiment Lanes

| Lane | Init | LR | Data | Epochs |
|------|------|----|------|--------|
| control_ep1 | iter0_reference | 3e-5 | generic + old replay | 1 |
| control_ep2 | iter0_reference | 3e-5 | generic + old replay | 2 |
| distill_ep1 | iter0_reference | 3e-5 | generic + old replay + p1 distill | 1 |
| distill_ep2 | iter0_reference | 3e-5 | generic + old replay + p1 distill | 2 |

## Training Results

| Lane | Epoch | Policy Loss | Value Loss | Best Val Loss | Checkpoint SHA256 | Delta Norm vs iter0_ref | Rel Delta |
|------|-------|-------------|------------|---------------|--------------------|------------------------|-----------|
| iter0_reference | — | — | — | — | `c2430b56...` | 0.0 | 0.0% |
| control_ep1 | 1 | 1.000793 | 0.245539 | 1.182145 | `099c6e6a...` | 0.048892 | 0.18% |
| control_ep2 | 2 | 0.998653 | 0.245369 | 1.179735 | `619376db...` | 0.090479 | 0.34% |
| distill_ep1 | 1 | 1.015310 | 0.244694 | 1.188630 | `65d4c54a...` | 0.049207 | 0.18% |
| distill_ep2 | 2 | 1.012673 | 0.244478 | 1.188337 | `7c2b9c05...` | 0.091408 | 0.34% |

Note: Distill lane has HIGHER validation loss than control lane because the
distillation data adds harder P1 positions that increase overall loss.

### Distillation Holdout Policy Loss

Imitation quality on the held-out distillation positions:

| Checkpoint | Holdout Policy Loss |
|------------|-------------------|
| iter0_reference | 1.629488 |
| control_ep1 | 1.625373 |
| control_ep2 | 1.630873 |
| distill_ep1 | **1.620831** |
| distill_ep2 | **1.620808** |

Distillation improves holdout policy imitation by ~0.009 (0.5% relative).
However, this improvement does not translate to arena strength at lower budgets.

### Generic Validation Loss

| Checkpoint | Val Policy Loss | Val Value Loss | Val Total |
|------------|-----------------|----------------|-----------|
| iter0_reference | 1.146195 | 0.272072 | 1.227817 |
| control_ep1 | 1.144730 | 0.272085 | 1.226356 |
| control_ep2 | 1.143385 | 0.271975 | 1.224977 |
| distill_ep1 | 1.144899 | 0.272180 | 1.226553 |
| distill_ep2 | 1.144657 | 0.272069 | 1.226278 |

Generic validation loss improves monotonically while seat-aware strength does
not — confirming validation loss is not a reliable proxy per prior findings.

## Seat-Aware Strength

All evaluations at 120 games, seed=42, c_puct=1.25, tactical_root_bias=0.1
(default), root_policy_mode=deterministic, challenger vs model-artifact/current.

### control_ep1 (epoch 1, no distillation)

| Budget | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|--------|-----|----------|----------|--------|-----|-----|------|
| 384:256 | 0.00 | 0/0/0 | 0/30/0 | -18.0 | 38.0 | 30 | (0, 0.11) |
| 768:256 | 0.00 | 0/0/0 | 0/30/0 | -16.0 | 53.0 | 30 | (0, 0.11) |
| 768:768 | 0.00 | 0/0/0 | 0/30/0 | -18.0 | 44.0 | 30 | (0, 0.11) |
| **1200:1200** | **1.00** | **0/0/0** | **30/0/0** | **6.0** | **51.0** | **30** | **(0.89, 1.00)** |
| 256:768 | 0.00 | 0/0/0 | 0/30/0 | -16.0 | 36.0 | 30 | (0, 0.11) |
| TBO 768:256 | 0.00 | — | — | — | — | — | alt=0.50 (60W/60L) |

### control_ep2 (epoch 2, no distillation)

| Budget | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|--------|-----|----------|----------|--------|-----|-----|------|
| 384:256 | 0.00 | 0/0/0 | 0/30/0 | -18.0 | 38.0 | 30 | (0, 0.11) |
| 768:256 | 0.00 | 0/0/0 | 0/30/0 | -12.0 | 47.0 | 30 | (0, 0.11) |
| 768:768 | 0.00 | 0/0/0 | 0/30/0 | -18.0 | 44.0 | 30 | (0, 0.11) |
| **1200:1200** | **1.00** | **0/0/0** | **30/0/0** | **6.0** | **51.0** | **30** | **(0.89, 1.00)** |
| 256:768 | 0.00 | 0/0/0 | 0/30/0 | -16.0 | 36.0 | 30 | (0, 0.11) |
| TBO 768:256 | 0.00 | — | — | — | — | — | alt=0.50 (60W/60L) |

### distill_ep1 (epoch 1, with distillation)

| Budget | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|--------|-----|----------|----------|--------|-----|-----|------|
| 384:256 | 0.00 | 0/0/0 | 0/30/0 | -18.0 | 38.0 | 30 | (0, 0.11) |
| 768:256 | 0.00 | 0/0/0 | 0/30/0 | -8.0 | 37.0 | 30 | (0, 0.11) |
| 768:768 | 0.00 | 0/0/0 | 0/30/0 | -22.0 | 39.0 | 30 | (0, 0.11) |
| **1200:1200** | **1.00** | **0/0/0** | **30/0/0** | **6.0** | **51.0** | **30** | **(0.89, 1.00)** |
| 256:768 | 0.00 | 0/0/0 | 0/30/0 | -16.0 | 36.0 | 30 | (0, 0.11) |
| TBO 768:256 | 0.00 | — | — | — | — | — | alt=0.50 (60W/60L) |

### distill_ep2 (epoch 2, with distillation)

| Budget | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|--------|-----|----------|----------|--------|-----|-----|------|
| 384:256 | 0.00 | 0/0/0 | 0/30/0 | -18.0 | 38.0 | 30 | (0, 0.11) |
| 768:256 | 0.00 | 0/0/0 | 0/30/0 | -8.0 | 37.0 | 30 | (0, 0.11) |
| 768:768 | 0.00 | 0/0/0 | 0/30/0 | -18.0 | 44.0 | 30 | (0, 0.11) |
| **1200:1200** | **1.00** | **0/0/0** | **30/0/0** | **6.0** | **51.0** | **30** | **(0.89, 1.00)** |
| 256:768 | 0.00 | 0/0/0 | 0/30/0 | -16.0 | 36.0 | 30 | (0, 0.11) |
| TBO 768:256 | 0.00 | — | — | — | — | — | alt=0.50 (60W/60L) |

### Move Time (p95 ms)

| Checkpoint | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|------------|---------|---------|---------|-----------|---------|
| control_ep1 | 78 | 216 | 210 | 347 | 217 |
| control_ep2 | 71 | 112 | 225 | 358 | 108 |
| distill_ep1 | 106 | 208 | 218 | 353 | 110 |
| distill_ep2 | 107 | 216 | 222 | 330 | 195 |

## Consolidated Ranking Table

| Rank | Lane | DS 384:256 | DS 768:256 | DS 768:768 | DS 1200:1200 | DS 256:768 | TBO DS 768:256 | Classification |
|------|------|-----------|-----------|-----------|-------------|-----------|---------------|----------------|
| 1 | iter0_reference | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | **0.50** | high_search_breakthrough |
| 2 | control_ep1 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | high_search_breakthrough |
| 3 | control_ep2 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | high_search_breakthrough |
| 4 | distill_ep1 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | high_search_breakthrough |
| 5 | distill_ep2 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | high_search_breakthrough |

All fine-tuned checkpoints preserve DS=1.00 at 1200:1200. None achieve DS > 0.00
at any lower budget. The TBO secondary advantage (DS=0.50 at 768:256) is lost in
all fine-tuned checkpoints, consistent with the preservation finetune findings.

## Classification

**Overall: `high_search_not_distillable`**

### Why `high_search_not_distillable`

| Criterion | Evidence |
|-----------|----------|
| Distillation improves holdout imitation | Holdout policy loss: 1.621 (distill_ep2) vs 1.629 (iter0_reference) |
| But DS unchanged at lower budgets | All checkpoints: DS=0.00 at 384:256, 768:256, 768:768 |
| 1200:1200 breakthrough preserved | All checkpoints: DS=1.00 at 1200:1200 |
| No collapse at 256:768 | All checkpoints: DS=0.00 at 256:768 (no regression) |
| Validation loss is misleading | Generic val loss improves (1.228 -> 1.225) while DS does not |

### Why NOT `disadvantaged_seat_distillation_promising`

No checkpoint achieves DS > 0.00 at 384:256 or DS >= 0.50 at 768:256. The
disadvantaged-seat scores are identical to the control lane (no distillation
data) at all budgets.

### Why NOT `distillation_too_destructive`

The 1200:1200 breakthrough is preserved (DS=1.00). 256:768 does not collapse.
However, TBO secondary advantage at 768:256 is lost (consistent with prior
preservation finetune findings), and validation loss improves while seat-aware
strength shows no improvement.

## Primary Findings

1. **High-search P1 policy exists but doesn't distill at LR=3e-5.**
   The distillation holdout imitation improves (1.621 vs 1.629), meaning the
   model shifts toward high-search P1 policy. But the shift magnitude at
   LR=3e-5 is too small to change MCTS outcomes at lower budgets (384 or 768
   simulations). The MCTS search process dominates the raw policy at practical
   budgets.

2. **Parameter delta is very small.** Relative delta norm vs iter0_reference
   is 0.18%–0.34% across all checkpoints. The learning signal is diluted by
   the 95% generic/old-replay data share. Even the targeted P1 distillation
   data represents only ~14% of training positions by weight (1024 rows x
   weight 1 vs 9589 x 4 + 2016 x 1 = ~40372 effective).

3. **TBO secondary advantage is lost.** All fine-tuned checkpoints lose the
   tactical_bias_off DS=0.50 at 768:256 that iter0_reference had. This is
   consistent with the preservation finetune sweep, where even LR=3e-5 caused
   this secondary advantage to vanish.

4. **Kalah state space limits distillation dataset size.** From 480 games
   with 4 random opening plies, only 3323 unique P1 positions were collected.
   The early Kalah state space is small; many positions are immediate
   duplicates. Random openings are essential for dataset diversity.

5. **Distillation mining is feasible.** The mining script
   `ml/alphazero_lite/mine_disadvantaged_seat_distillation.py` generates
   paired games from iter0_reference vs current, collects P1 positions,
   relabels with high-search MCTS, and filters by policy divergence criteria.
   The infrastructure is reusable for future distillation experiments.

## Runner Commands

### Mining

```bash
.venv/bin/python ml/alphazero_lite/mine_disadvantaged_seat_distillation.py \
  --out-train /tmp/azlite_p1_distill/p1_distill_train.jsonl \
  --out-holdout /tmp/azlite_p1_distill/p1_distill_holdout.jsonl \
  --out-summary /tmp/azlite_p1_distill/p1_distill_summary.json \
  --candidate /tmp/azlite_iterative_random_replay/iter0_candidate_artifact \
  --current model-artifact/current \
  --low-budget 384:256 \
  --teacher-budget 1200:1200 \
  --teacher-simulations 1200 \
  --games 480 \
  --seed 46 \
  --max-positions-per-game 20 \
  --target-train-rows 1024 \
  --target-holdout-rows 256 \
  --random-opening-plies 4 \
  --input-encoding kalah_v3 \
  --policy-target-mode sharpened \
  --value-target-mode sharpened
```

### Training (control lane)

```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 2 \
  --batch-size 512 \
  --lr 3e-5 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 3 \
  --top-k-dir /tmp/azlite_p1_distill/checkpoints/control \
  --out /tmp/azlite_p1_distill/checkpoints/control/checkpoint_ep2.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Training (distill lane)

```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl,/tmp/azlite_p1_distill/p1_distill_train.jsonl \
  --replay-weights 4,1,1 \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 2 \
  --batch-size 512 \
  --lr 3e-5 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 3 \
  --top-k-dir /tmp/azlite_p1_distill/checkpoints/distill \
  --out /tmp/azlite_p1_distill/checkpoints/distill/checkpoint_ep2.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Evaluation

```bash
.venv/bin/python script/ai/seat_aware_promotion_gate \
  --candidate-path /tmp/azlite_p1_distill/artifacts/<artifact> \
  --current-path model-artifact/current \
  --games 120 --seed 42 --workers 4 \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --out /tmp/azlite_p1_distill/eval/<name>_seat_aware.json
```

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS: no model promoted |
| No overwrite model-artifact/current | PASS: current unchanged |
| No overwrite storage/ai/alphazero_lite/current | PASS: unchanged |
| No LR above 3e-5 | PASS: all lanes use LR=3e-5 |
| No generic random replay generation | PASS: no new data generated |
| No failure/disagreement mining outside P1 curriculum | PASS: only P1 distillation used |
| No architecture change | PASS: residual_v3 only |
| No default search settings changed | PASS: all default eval settings |
| No promotion by validation loss alone | PASS: seat-aware eval used |
| No residual_v4 | PASS: residual_v3 only |
| No tablebase overlay | PASS: not used |
| No agreement filtering | PASS: not used |
| No tactical-bias-off replay | PASS: not generated |

## Verification

```bash
.venv/bin/ruff check ml/alphazero_lite/mine_disadvantaged_seat_distillation.py  # clean
.venv/bin/ruff check ml/alphazero_lite script/ai  # clean
```
