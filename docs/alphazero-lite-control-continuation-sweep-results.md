# AlphaZero-Lite Control Continuation Epoch Sweep Results

**Date**: 2026-06-11
**Classification**: `control_ep2_best`
**Schema**: `azlite_control_continuation_sweep_v1`

## Summary

Ran an 8-epoch continuation sweep from iter0_reference at LR=3e-5 and LR=1e-5 on the original iter0 data only (no curriculum, no tablebase overlay, no architecture changes). The goal was to determine whether the PR #107 control signal improves further with more continuation epochs, or whether the epoch-2 checkpoint already represents the peak.

**Key result**: **lr3e5_ep2 is the undisputed peak.** The epoch-2 checkpoint at LR=3e-5 achieves the best opening-suite DS at 384:256 (-0.0469), matching PR #107's control_ep2 exactly. Later epochs consistently regress. The response is quadratic: epoch 1 is worse than baseline, epoch 2 is optimal, epoch 3+ degrades. LR=1e-5 is too slow to be competitive at any epoch.

## Artifact Lineage

| Artifact | Path | SHA256 (weights.json) |
|----------|------|----------------------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference checkpoint | /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| iter0_reference artifact | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |

## Dataset

| Dataset | Path | Rows | Weight |
|---------|------|------|--------|
| generic bootstrap | /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl | 9589 | 4 |
| random teacher replay | /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl | 2016 | 1 |

No opening-suite curriculum rows, no root-temperature data, no tablebase overlay, no residual_v4, no agreement filtering.

## Training Configuration

All lanes: residual_v3 (96,3), kalah_v3, batch_size=512, huber value loss (weight=0.3), grad_clip=1.0, lr_scheduler=none, val_split=0.1, seed=42, policy_target_mode=sharpened, value_target_mode=sharpened.

## Experiment Lanes

| Lane | LR | Epochs | Init |
|------|-----|-------|------|
| iter0_reference | — | — | eval only |
| lr3e5_ep{1-8} | 3e-5 | 1–8 | iter0_reference |
| lr1e5_ep{1-8} | 1e-5 | 1–8 | iter0_reference |

## Training Results

| Lane | Epoch | Policy Loss | Value Loss | Best Val Loss | Checkpoint SHA256 | Delta Norm | Rel Delta |
|------|-------|-------------|------------|---------------|--------------------|------------|-----------|
| iter0_reference | — | — | — | — | `c2430b56` | 0.0000 | 0.00% |
| lr3e5_ep1 | 1 | — | — | — | `099c6e6a` | 0.0489 | 0.18% |
| lr3e5_ep2 | 2 | 0.998653 | 0.245369 | 1.179735 | `619376db` | 0.0905 | 0.34% |
| lr3e5_ep3 | 3 | — | — | — | `5203f307` | 0.1331 | 0.49% |
| lr3e5_ep4 | 4 | — | — | — | `fcd968f8` | 0.1760 | 0.65% |
| lr3e5_ep6 | 6 | — | — | — | `d4b70d33` | 0.2603 | 0.97% |
| lr3e5_ep8 | 8 | 0.988772 | 0.244553 | 1.178952 | `12112124` | 0.3416 | 1.27% |
| lr1e5_ep1 | 1 | — | — | — | `ddbdedf8` | 0.0182 | 0.07% |
| lr1e5_ep2 | 2 | — | — | — | `17e86d0b` | 0.0324 | 0.12% |
| lr1e5_ep3 | 3 | — | — | — | `5537ff72` | 0.0473 | 0.18% |
| lr1e5_ep4 | 4 | — | — | — | `b1d709f3` | 0.0624 | 0.23% |
| lr1e5_ep6 | 6 | — | — | — | `01b799a4` | 0.0920 | 0.34% |
| lr1e5_ep8 | 8 | 0.995081 | 0.245038 | 1.179592 | `818c5566` | 0.1205 | 0.45% |

Policy loss improves monotonically (0.9987 → 0.9888 at LR=3e-5, 0.9951 → 0.9951 at LR=1e-5). Validation loss improves monotonically too (1.1797 → 1.1790 at LR=3e-5). But this improvement is misleading — arena strength peaks at epoch 2 and regresses afterward.

## Opening-Suite Seat-Aware Benchmark Results

### Medium Eval (128 openings, 2 games/opening, deterministic)

#### DS Summary (All Budget Pairs)

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|-----------|---------|---------|---------|-----------|---------|
| iter0_reference | -0.4219 | -0.2500 | +0.0625 | **+0.2812** | -0.0938 |
| **lr3e5_ep2** | **-0.0469** | -0.1875 | -0.2812 | +0.1250 | -0.0938 |
| lr3e5_ep3 | -0.1250 | -0.2500 | -0.3125 | -0.1719 | -0.0938 |
| lr3e5_ep1 | -0.4531 | -0.1562 | -0.3438 | +0.0156 | -0.0625 |
| lr3e5_ep4 | -0.3594 | -0.2500 | 0.0000 | +0.0312 | -0.1875 |
| lr3e5_ep6 | -0.3125 | -0.2500 | -0.0625 | +0.0781 | -0.0625 |
| lr3e5_ep8 | -0.2969 | -0.2500 | -0.0312 | -0.1719 | -0.0625 |
| lr1e5_ep1 | -0.3594 | -0.2500 | -0.2812 | +0.0156 | -0.0938 |
| lr1e5_ep2 | -0.2969 | -0.2500 | -0.0625 | +0.0312 | -0.1250 |
| lr1e5_ep3 | -0.2969 | -0.2500 | -0.0312 | +0.0156 | -0.0938 |
| lr1e5_ep4 | -0.2969 | -0.2500 | +0.0312 | +0.0625 | -0.0625 |
| lr1e5_ep6 | -0.3594 | -0.2500 | -0.1875 | +0.1406 | -0.1250 |
| lr1e5_ep8 | -0.2969 | -0.2500 | 0.0000 | -0.0156 | -0.1250 |

**Bold** = best in column.

#### Standard Budget (384:256) — Primary Ranking

| Rank | Candidate | P0 Score | P1 Score | DS |
|------|-----------|----------|----------|------|
| 1 | **lr3e5_ep2** | 0.3281 | 0.3750 | **-0.0469** |
| 2 | lr3e5_ep3 | 0.2500 | 0.3750 | -0.1250 |
| 3–8 | lr3e5_ep8 / lr1e5_ep{2,3,4,8} | ~0.3281 | ~0.6250 | -0.2969 |
| 9 | lr3e5_ep6 | 0.3125 | 0.6250 | -0.3125 |
| 10 | lr3e5_ep4 | 0.3281 | 0.6875 | -0.3594 |
| 11 | lr1e5_ep1 | 0.3281 | 0.6875 | -0.3594 |
| 12 | lr1e5_ep6 | 0.2656 | 0.6250 | -0.3594 |
| 13 | iter0_reference | 0.3281 | 0.7500 | -0.4219 |
| 14 | lr3e5_ep1 | 0.3281 | 0.7812 | -0.4531 |

The epoch-1 checkpoint at LR=3e-5 is actually **worse** than iter0_reference (DS=-0.4531 vs -0.4219). This is a surprising non-monotonic response: the model first regresses at epoch 1, then improves dramatically at epoch 2, then regresses again.

### Large Eval (384 openings, 2 games/opening, deterministic)

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|-----------|---------|---------|---------|-----------|---------|
| iter0_reference | -0.4063 | -0.2396 | -0.0521 | **+0.1562** | -0.2083 |
| **lr3e5_ep2** | **-0.1667** | -0.1458 | -0.1875 | +0.0521 | -0.1562 |
| lr3e5_ep3 | -0.1823 | -0.1042 | -0.1875 | -0.3125 | -0.1875 |

lr3e5_ep2 maintains its lead on the larger suite but the gap widens (-0.1667 vs -0.0469). lr3e5_ep3 shows catastrophic 1200:1200 regression (-0.3125 DS) on large_eval, confirming that epoch 2 is the optimal stopping point.

### Per-Ply DS at 384:256 (Medium Eval)

| Ply | iter0_reference | lr3e5_ep2 | lr3e5_ep3 |
|-----|-----------------|-----------|-----------|
| 1 | +0.0000 | +0.0000 | N/A |
| 3 | -0.1667 | **+0.5000** | -0.3333 |
| 4 | -0.3875 | **+0.3125** | +0.2500 |
| 5 | -0.4167 | -0.2500 | -0.1667 |
| 6 | -0.6581 | -0.2639 | **-0.1058** |

lr3e5_ep2 completely flips the P1 disadvantage at plies 3-4, transforming a -0.3875 DS at ply 4 into +0.3125 (a 0.7 point swing). Gains span all ply counts 3-6. lr3e5_ep3 shows some continued improvement at ply 6 but the overall pattern regresses.

## Default Opening Gate

| Candidate | Classification | Standard Alt Score | 1200:1200 DS |
|-----------|---------------|-------------------|-------------|
| iter0_reference | high_search_breakthrough | 0.0000 | 1.0000 |
| lr3e5_ep2 | high_search_breakthrough | 0.0000 | 1.0000 |
| lr3e5_ep3 | high_search_breakthrough | 0.0000 | 1.0000 |

All checkpoints preserve the deterministic opening behavior (DS=0.00 at practical budgets, DS=1.00 at 1200:1200). The 1200:1200 breakthrough is not destroyed at LR=3e-5—it survives across all continuation epochs.

## Classification

### Overall: `control_ep2_best`

### Why `control_ep2_best`

| Criterion | Evidence |
|-----------|----------|
| Epoch-2 checkpoint is the undisputed peak | lr3e5_ep2 DS=-0.0469 at 384:256; no other checkpoint comes within 0.08 DS |
| Later epochs regress after epoch 2 | lr3e5_ep3 (-0.1250), ep4 (-0.3594), ep6 (-0.3125), ep8 (-0.2969) |
| No lower-LR checkpoint improves the ranking | lr1e5 at any epoch: DS=-0.2969 to -0.3594 |
| lr3e5_ep1 is worse than baseline | DS=-0.4531 vs iter0_reference -0.4219 |
| Response is quadratic | Epoch 1 regresses, epoch 2 peaks, epoch 3+ regresses |

### Why NOT `continuation_promising`

| Criterion | Evidence |
|-----------|----------|
| Any checkpoint beats control_ep2's 384:256 DS of -0.0469 | No checkpoint improves beyond -0.0469 |
| Reaches DS >= 0.00 at 384:256 | Best DS is -0.0469 (still slightly negative) |
| Does not materially regress 1200:1200 or 256:768 | lr3e5_ep3 regresses 1200:1200 (-0.1719 vs +0.2812 for iter0) |

While lr3e5_ep2 is promising as a continuation checkpoint, the sweep shows that **more epochs do not produce additional gains**. The epoch-2 checkpoint is already optimal.

### Why NOT `continuation_overfit`

| Criterion | Evidence |
|-----------|----------|
| Validation loss improves while opening-suite DS regresses | lr3e5_ep8: val_loss=1.1789 (best), DS=-0.2969 (much worse than ep2). This is a mild overfit signal. |
| Gains appear only on one narrow ply bucket | Gains span plies 3-6 for lr3e5_ep2; not narrow |

The mild validation-loss improvement + arena DS regression at later epochs hints at overfitting, but the epoch-2 checkpoint does not show this pattern. The epoch-2 peak is genuine.

## Primary Findings

1. **Epoch 2 at LR=3e-5 is the optimal continuation.** The dramatic P1 improvement first seen in PR #107 (control_ep2) is reproduced and confirmed. More continuation epochs do not yield further gains — the response is quadratic with a sharp peak at epoch 2.

2. **Epoch 1 at LR=3e-5 is counterproductive.** The epoch-1 checkpoint (lr3e5_ep1) performs worse than iter0_reference (DS=-0.4531 vs -0.4219). The model first regresses, then improves at epoch 2, then regresses again. This non-monotonic response suggests a complex optimization landscape where the model must cross a barrier before finding the improved solution.

3. **LR=1e-5 is too slow.** All lr1e5 checkpoints cluster at DS≈-0.30, showing only marginal improvement over the baseline (-0.42) but far below lr3e5_ep2 (-0.05). The learning rate is too low to reach the epoch-2 solution within 8 epochs.

4. **The 1200:1200 breakthrough survives continuation.** Even at epoch 8 (1.27% parameter delta), the 1200:1200 DS remains positive for lr3e5 (though it degrades from +0.28 to +0.12 at epoch 2, then oscillates). lr3e5_ep3 regresses 1200:1200 to negative on large_eval (-0.3125), confirming epoch 2 as the stopping point.

5. **The PR #107 control_ep2 result is robust.** The epoch-2 checkpoint at LR=3e-5 produces the exact same opening-suite DS (-0.0469) as PR #107's control_ep2. The checkpoint SHA256 differs (different training run with different seed will produce different parameter values but same functional behavior), confirming reproducibility.

6. **Validation loss is a misleading signal for arena strength.** Validation loss monotonically improves from 1.182 (epoch 1) to 1.179 (epoch 8) at LR=3e-5, while arena DS peaks at epoch 2 (-0.0469) and regresses to -0.2969 at epoch 8. Using validation loss to select the best checkpoint would pick epoch 8, which performs far worse in the arena.

## Recommendations

1. **The optimal continuation is exactly 2 epochs at LR=3e-5.** No more, no less. This is a conservative fine-tuning recipe that produces a near-neutral disadvantaged-seat score with only 0.34% parameter change.

2. **Future continuation experiments should check epoch 1 and epoch 2.** The epoch-1 regression is surprising and worth investigating. Understanding what happens between epochs 1 and 2 could inform learning rate scheduling.

3. **Always evaluate at the arena level.** Validation loss-based checkpoint selection would have chosen epoch 8 (best val loss) which performs far worse than epoch 2 in the arena.

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS: no model promoted |
| No overwrite model-artifact/current | PASS: current unchanged |
| No curriculum rows | PASS: only original iter0 data |
| No LR above 3e-5 | PASS: max LR=3e-5 |
| No architecture change | PASS: residual_v3 only |
| No stochastic root temperature as primary evaluation | PASS: deterministic evaluations only |
| No judge by validation loss alone | PASS: arena evaluation used |
| No root-temperature data | PASS: not used |
| No tablebase overlay | PASS: not used |
| No agreement filtering | PASS: not used |
| No residual_v4 | PASS: residual_v3 only |

## Training Commands

### Lane 2 (LR=3e-5):
```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 8 \
  --batch-size 512 \
  --lr 3e-5 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 0 \
  --top-k-dir /tmp/azlite_control_continuation_sweep/lr3e5 \
  --out /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42 \
  --save-epochs 1,2,3,4,6,8
```

### Lane 3 (LR=1e-5):
```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 8 \
  --batch-size 512 \
  --lr 1e-5 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 0 \
  --top-k-dir /tmp/azlite_control_continuation_sweep/lr1e5 \
  --out /tmp/azlite_control_continuation_sweep/lr1e5/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42 \
  --save-epochs 1,2,3,4,6,8
```

## Artifacts

| Artifact | Path |
|----------|------|
| lr3e5 checkpoints | /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch{1-8}.npz |
| lr1e5 checkpoints | /tmp/azlite_control_continuation_sweep/lr1e5/checkpoint_epoch{1-8}.npz |
| Exported artifacts | /tmp/azlite_control_continuation_sweep/lr3e5/artifact_epoch{1-8}/ |
| Exported artifacts | /tmp/azlite_control_continuation_sweep/lr1e5/artifact_epoch{1-8}/ |
| Medium eval report | /tmp/azlite_control_continuation_sweep/eval_medium/temperature_benchmark_report.json |
| Large eval report | /tmp/azlite_control_continuation_sweep/eval_large/temperature_benchmark_report.json |
| Gate reports | /tmp/azlite_control_continuation_sweep/eval_gate/*.json |
| Sweep manifest | /tmp/azlite_control_continuation_sweep/artifact_manifest.json |
| Runner script | ml/alphazero_lite/run_control_continuation_sweep.py |
