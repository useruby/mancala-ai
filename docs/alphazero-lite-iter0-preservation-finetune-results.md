# AlphaZero-Lite iter0 Preservation Finetune Sweep Results

**Date:** 2026-06-08

## Summary

Conservative fine-tuning from iter0_reference with much smaller learning
rates and shorter schedules to test whether the 1200:1200 disadvantaged-seat
breakthrough survives continuation training at reduced intensity.

**Overall Classification: `high_lr_was_problem`**

The 1200:1200 breakthrough (DS=1.00) is preserved at LR <= 3e-5 across all
epoch levels (1, 2, 4). At LR=1e-4 the breakthrough is destroyed even after
just 1 epoch. Policy-head-only training at LR=1e-4 is also destructive.
PR #100's 1e-3 / 10-epoch regression is a learning rate & schedule intensity
problem, not primarily a data problem.

## Artifact Lineage

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference checkpoint | /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| iter0_reference artifact | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |

### Fine-Tuned Checkpoints

| Lane | Checkpoint SHA256 | Artifact SHA256 |
|------|------------------|-----------------|
| LR=1e-4 ep=1 | `2dadb66d0e8fe524dedad0972b4c5cc0b1e20c495ad361e013cd6cd09d5e93bb` | `e7f52eef033dd13137a433dacca612f61ef06a537e12898a58f58fda91d578b1` |
| LR=1e-4 ep=2 | `012ac85206646d7eaa46417b02e5c00f8a9331e8fcecb82b00fa22a1dba4c7e1` | `7290e5a37df94d31a254aecf712dfa6e7d5b8bf7c7e6d5e94bab1233698c7b5f` |
| LR=1e-4 ep=4 | `012ac85206646d7eaa46417b02e5c00f8a9331e8fcecb82b00fa22a1dba4c7e1` | `7290e5a37df94d31a254aecf712dfa6e7d5b8bf7c7e6d5e94bab1233698c7b5f` |
| LR=3e-5 ep=1 | `dc2014242356003089bd51824994d9325c02492c51617192a3663ca385f0cc06` | `0eea536ffc85c9bc204c5c0cf5a1ca97916fb573e57845fce59894888fde066f` |
| LR=3e-5 ep=2 | `011570f3f9e5f152e4cc3354c656640aedf1849bbab3e3eabc7954717e4c87fb` | `13237078efc63169285dce4f620ffc646f626c256b6fa1f66ab3235065a9bca6` |
| LR=3e-5 ep=4 | `011570f3f9e5f152e4cc3354c656640aedf1849bbab3e3eabc7954717e4c87fb` | `13237078efc63169285dce4f620ffc646f626c256b6fa1f66ab3235065a9bca6` |
| LR=1e-5 ep=1 | `e500eab78ba16d2a06b08037e102b48ec239e39544925a4860c700751af5f17e` | `1be97a5154d198d9e9320b767e2ad4530e5c044fead264e408624b7c3f0efc63` |
| LR=1e-5 ep=2 | `a5d878bf51fd7e4f61ad0f1d152e82eddf5c6802a14dca19ac8b79442313256b` | `ff3e3a2803da9277e922c15e2da74e0e55f64456a891391552f0fc1140dfd371` |
| LR=1e-5 ep=4 | `a5d878bf51fd7e4f61ad0f1d152e82eddf5c6802a14dca19ac8b79442313256b` | `ff3e3a2803da9277e922c15e2da74e0e55f64456a891391552f0fc1140dfd371` |
| policy_head LR=1e-4 ep=1 | `bd97af41648257fd8f0759b1400cfc4028b36b491741532bb1928d252443c5b5` | `2c4a13dc6a9251731b9a3e88f281136316cae2cbb6f78cb94e1f81ee5efc35f5` |
| policy_head LR=1e-4 ep=2 | `bd97af41648257fd8f0759b1400cfc4028b36b491741532bb1928d252443c5b5` | `2c4a13dc6a9251731b9a3e88f281136316cae2cbb6f78cb94e1f81ee5efc35f5` |
| policy_head LR=1e-4 ep=4 | `bd97af41648257fd8f0759b1400cfc4028b36b491741532bb1928d252443c5b5` | `2c4a13dc6a9251731b9a3e88f281136316cae2cbb6f78cb94e1f81ee5efc35f5` |

Note: epochs 2 and 4 produce identical checkpoints at LR=1e-4, 3e-5, and 1e-5
(best-validation-loss selection picks the same early optimum). Policy-head-only
lanes produce identical checkpoints across all epochs (the training does not
meaningfully change any trainable weights).

## Dataset

| Dataset | Path | SHA256 | Rows |
|---------|------|--------|------|
| generic bootstrap | /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl | `5d01a60e...` | 9589 |
| old current-mined random replay | /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl | `7ca93389...` | 2016 |

### Replay Weights

- Generic bootstrap: 4x
- Old current-mined random replay: 1x
- Effective sample share: ~95% generic, ~5% old replay

### Data Configuration

- policy_target_mode: sharpened
- value_target_mode: sharpened
- No new data generation

## Training

All lanes use: residual_v3 (96,3), kalah_v3, batch_size=512, huber value loss
(weight=0.3), grad_clip=1.0, lr_scheduler=none, val_split=0.1, seed=42.

| Lane | LR | Epochs | Scope | Policy Loss | Value Loss | Val Loss |
|------|-----|-------|-------|-------------|------------|----------|
| iter0_reference | — | — | — | — | — | — |
| LR=1e-4 ep=1 | 1e-4 | 1 | all | 1.0052 | 0.2459 | 1.1857 |
| LR=1e-4 ep=2 | 1e-4 | 2 | all | 0.9982 | 0.2452 | 1.1800 |
| LR=1e-4 ep=4 | 1e-4 | 4 | all | 0.9890 | 0.2446 | 1.1800 |
| LR=3e-5 ep=1 | 3e-5 | 1 | all | 1.0008 | 0.2455 | 1.1821 |
| LR=3e-5 ep=2 | 3e-5 | 2 | all | 0.9987 | 0.2454 | 1.1797 |
| LR=3e-5 ep=4 | 3e-5 | 4 | all | 0.9952 | 0.2451 | 1.1797 |
| LR=1e-5 ep=1 | 1e-5 | 1 | all | 0.9995 | 0.2454 | 1.1810 |
| LR=1e-5 ep=2 | 1e-5 | 2 | all | 0.9986 | 0.2454 | 1.1805 |
| LR=1e-5 ep=4 | 1e-5 | 4 | all | 0.9974 | 0.2452 | 1.1805 |
| policy_head LR=1e-4 ep=1 | 1e-4 | 1 | policy_head | 0.9995 | 0.2454 | 1.1828 |
| policy_head LR=1e-4 ep=2 | 1e-4 | 2 | policy_head | 0.9986 | 0.2454 | 1.1828 |
| policy_head LR=1e-4 ep=4 | 1e-4 | 4 | policy_head | 0.9977 | 0.2454 | 1.1828 |

Key observation: validation loss improves monotonically across all lanes (from
~1.186 to as low as 1.180) even when seat-aware strength regresses. Validation
loss is not a reliable proxy for seat-aware arena strength.

## Seat-Aware Strength

All evaluations at 120 games, seed=42, c_puct=1.25, tactical_root_bias=0.1
(default), root_policy_mode=deterministic, challenger vs current
(model-artifact/current).

### iter0_reference (eval only, baseline)

Matches PR #99 / PR #100 results:

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 4 | 32 | 118 | (0, 0.06) |
| default | **1200:1200** | **1.00** | **1.00** | **60/0/0** | **60/0/0** | **13** | **41** | **118** | **(0.94, 1.00)** |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -6 | 36 | 118 | (0, 0.06) |
| tbo | **768:256** | **0.75** | **0.50** | **60/0/0** | **0/0/60** | **3** | **42** | **118** | **(0.38, 0.62)** |

Classification: `high_search_only` (default), `search_compression_promising` (tbo at 768:256)

### LR=1e-4 epoch=1 (full network)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | **384:256** | **0.00** | **0.00** | **0/60/0** | **0/60/0** | **-16** | **44** | **118** | **(0, 0.06)** |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 6 | 39 | 118 | (0, 0.06) |
| default | 1200:1200 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 6 | 36 | 118 | (0, 0.06) |
| default | **256:768** | **0.00** | **0.00** | **0/60/0** | **0/60/0** | **-15** | **36** | **118** | **(0, 0.06)** |

Classification: **`regression`**. Only 1 epoch at LR=1e-4 causes total collapse
at 384:256 and 256:768 (challenger loses all 60 games from both seats).

### LR=1e-4 epoch=2 (full network)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 1 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 4 | 32 | 118 | (0, 0.06) |
| default | 1200:1200 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 3 | 36 | 118 | (0, 0.06) |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -8 | 34 | 118 | (0, 0.06) |

Classification: **`regression`**. 1200:1200 breakthrough lost (DS=1.00 -> 0.00).
Recovers seat dominance at 384:256 (no longer collapsed) but shows no
disadvantaged-seat strength at any budget.

### LR=1e-4 epoch=4 (full network)

Identical to epoch=2 (same checkpoint selected by val loss).

### LR=3e-5 epoch=1 (full network)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 1 | 41 | 118 | (0, 0.06) |
| default | **1200:1200** | **1.00** | **1.00** | **60/0/0** | **60/0/0** | **11** | **37** | **118** | **(0.94, 1.00)** |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -6 | 36 | 118 | (0, 0.06) |
| tbo | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 3 | 38 | 118 | (0, 0.06) |

Classification: **`preserves_1200_breakthrough`**. 1200:1200 DS=1.00 preserved.
No improvement at lower budgets. TBO secondary lost (DS=0.00 vs iter0_ref's 0.50).

### LR=3e-5 epoch=2 (full network)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 3 | 38 | 118 | (0, 0.06) |
| default | **1200:1200** | **1.00** | **1.00** | **60/0/0** | **60/0/0** | **13** | **41** | **118** | **(0.94, 1.00)** |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -6 | 38 | 118 | (0, 0.06) |
| tbo | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 1 | 34 | 118 | (0, 0.06) |

### LR=3e-5 epoch=4 (full network)

Identical to epoch=2 (same checkpoint selected by val loss).

### LR=1e-5 epoch=1 (full network)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 1 | 41 | 118 | (0, 0.06) |
| default | **1200:1200** | **1.00** | **1.00** | **60/0/0** | **60/0/0** | **11** | **37** | **118** | **(0.94, 1.00)** |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -6 | 38 | 118 | (0, 0.06) |
| tbo | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 0 | 38 | 118 | (0, 0.06) |

Classification: **`preserves_1200_breakthrough`**. 1200:1200 DS=1.00 preserved.

### LR=1e-5 epoch=2 (full network)

Identical to epoch=4 (same checkpoint selected by val loss).

### LR=1e-5 epoch=4 (full network)

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 1 | 42 | 118 | (0, 0.06) |
| default | **1200:1200** | **1.00** | **1.00** | **60/0/0** | **60/0/0** | **13** | **41** | **118** | **(0.94, 1.00)** |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -6 | 38 | 118 | (0, 0.06) |
| tbo | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 1 | 34 | 118 | (0, 0.06) |

### Policy-Head-Only LR=1e-4 epochs 1, 2, 4

All three produce identical results (checkpoints are byte-identical):

| Eval | Budget | Alt | DS | P0 W/L/D | P1 W/L/D | Margin | Len | Dup | CI95 |
|------|--------|-----|----|----------|----------|--------|-----|-----|------|
| default | 384:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -4 | 39 | 118 | (0, 0.06) |
| default | 768:256 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 1 | 41 | 118 | (0, 0.06) |
| default | 1200:1200 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | 6 | 30 | 118 | (0, 0.06) |
| default | 256:768 | 0.50 | 0.00 | 60/0/0 | 0/60/0 | -6 | 38 | 118 | (0, 0.06) |

Classification: **`regression`**. 1200:1200 breakthrough lost. Policy-head-only
training at LR=1e-4 destroys the breakthrough as thoroughly as full-network
training at the same LR.

## Consolidated Ranking Table

| Lane | LR | Epoch | Scope | DS 384:256 | DS 768:256 | DS 1200:1200 | DS 256:768 | TBO DS 768:256 | Classification |
|------|-----|-------|-------|------------|------------|-------------|-----------|---------------|----------------|
| iter0_reference | — | — | — | 0.00 | 0.00 | **1.00** | 0.00 | **0.50** | preserves_1200_breakthrough |
| LR=3e-5 ep=1 | 3e-5 | 1 | all | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | preserves_1200_breakthrough |
| LR=3e-5 ep=2,4 | 3e-5 | 2 | all | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | preserves_1200_breakthrough |
| LR=1e-5 ep=1 | 1e-5 | 1 | all | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | preserves_1200_breakthrough |
| LR=1e-5 ep=2,4 | 1e-5 | 2 | all | 0.00 | 0.00 | **1.00** | 0.00 | 0.00 | preserves_1200_breakthrough |
| LR=1e-4 ep=2,4 | 1e-4 | 2 | all | 0.00 | 0.00 | 0.00 | 0.00 | — | regression |
| policy_head LR=1e-4 | 1e-4 | 1 | policy_head | 0.00 | 0.00 | 0.00 | 0.00 | — | regression |
| **LR=1e-4 ep=1** | **1e-4** | **1** | **all** | **0.00** | **0.00** | **0.00** | **0.00** | **—** | **regression (collapsed)** |

The LR=1e-4 epoch=1 row is uniquely destructive — it collapses to alt=0.00 at
both 384:256 and 256:768 (loses all 120 games from both seats), worse than the
epoch=2,4 checkpoints which at least maintain seat dominance.

## Classification

**Overall: `high_lr_was_problem`**

### Why `high_lr_was_problem`

| Condition | Evidence |
|-----------|----------|
| Small LR / short epochs preserve 1200:1200 DS | LR <= 3e-5 at all epoch levels (1,2,4) maintains DS=1.00 at 1200:1200 |
| PR #100's 1e-3 / 10-epoch control regressed | iter1_control regressed to DS=0.50 at 1200:1200 and collapsed at 256:768 |
| Training intensity is the primary cause | Same data, same architecture — only LR and schedule differ |

### Why NOT `preservation_finetune_promising`

No fine-tuned checkpoint improves DS at 768:256 or 384:256 beyond iter0_reference
baseline. While 1200:1200 DS is preserved, lower-budget behavior does not improve.

### Why NOT `iter0_frozen_best`

The 1200:1200 breakthrough does survive fine-tuning at LR <= 3e-5 — it is not
fragile to ALL continuation training, only to training at too-high intensity.

## Primary Findings

1. **LR threshold is around 3e-5.** The 1200:1200 breakthrough survives at
   LR=3e-5 and LR=1e-5 but is destroyed at LR=1e-4 (a 2.5x difference from 3e-5
   to 1e-4). PR #100's LR=1e-3 was 33x above the survivable threshold.

2. **Even 1 epoch at LR=1e-4 is destructive.** The epoch=1 checkpoint at
   LR=1e-4 is the most damaged — it collapses to alt=0.00 at both 384:256 and
   256:768, losing all games from both seats. Checkpoints at epochs 2 and 4
   partially recover seat dominance but still lose the breakthrough.

3. **Policy-head-only training is not safer.** Training only the policy head
   at LR=1e-4 destroys the breakthrough identically to full-network training
   at the same LR. The fragility is not localized to the value head.

4. **Validation loss is a misleading signal.** Validation loss improves
   monotonically with more training across all LR levels (from ~1.186 to as
   low as 1.180), while seat-aware strength shows dramatic divergence:
   - LR=1e-4 ep=1: val_loss=1.186 but collapsed arena performance
   - LR=3e-5 ep=4: val_loss=1.180 and preserved breakthrough
   - Delta in val_loss does not predict arena outcome.

5. **TBO secondary advantage is lost even when breakthrough is preserved.**
   iter0_reference's tactical-bias-off advantage at 768:256 (DS=0.50) is lost
   in all fine-tuned checkpoints (DS=0.00), including those that preserve the
   default 1200:1200 breakthrough. The fine-tuning process shifts weights in a
   way that eliminates the search-compression benefit while maintaining the
   high-search ceiling.

6. **Epoch 2 checkpoints are identical to epoch 4.** Best-validation-loss
   selection picks the epoch-2 state for the epoch=4 runs at LR=1e-4, 3e-5,
   and 1e-5, meaning the model overshoots a local optimum after epoch 2 and
   the val-loss-based selection reverts to the earlier point.

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS: no model promoted |
| No overwrite model-artifact/current | PASS: current unchanged |
| No overwrite storage/ai/alphazero_lite/current | PASS: unchanged |
| No new replay generation | PASS: no new data generated |
| No tactical-bias-off replay | PASS: only original iter0 data used |
| No architecture change | PASS: residual_v3 only |
| No c_puct changes | PASS: all at 1.25 default |
| No tablebase overlay | PASS: not used |
| No root-prior transforms | PASS: not used |
| No tablebase overlay in eval | PASS: not configured |
| No promotion gate thresholds changed | PASS: not run |
| No local_promotion_gate run | PASS: not invoked |

## Runner Command

```bash
.venv/bin/python ml/alphazero_lite/run_iter0_preservation_finetune_sweep.py \
  --workdir /tmp/azlite_iter0_preservation_finetune \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --current model-artifact/current \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --learning-rates 1e-4,3e-5,1e-5 \
  --epochs 1,2,4 \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --batch-size 512 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --seed 42 \
  --budget-pairs 384:256,768:256,1200:1200,256:768 \
  --games 120 \
  --workers 4 \
  --policy-head-only
```

## Verification

```bash
.venv/bin/ruff check ml/alphazero_lite/run_iter0_preservation_finetune_sweep.py  # clean
```

Note: `ml/alphazero_lite/test_train.py` has a pre-existing failure
(`TypeError: train() missing 1 required keyword-only argument: 'lr_scheduler'`)
in `test_train_replay_indexes_match_duplicated_sgd_behavior_for_small_batches`
— unrelated to this PR.
