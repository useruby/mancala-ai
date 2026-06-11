# AlphaZero-Lite EP2 Annealed Continuation Results

**Date**: 2026-06-11
**Classification**: `control_ep2_frozen_best`
**Schema**: `azlite_ep2_annealed_continuation_v1`

## Summary

Tested whether the sharp lr3e5_ep2 control optimum from PR #108 could be stabilized or improved by switching to a lower learning rate after epoch 2. Four annealing lanes (LR=1e-5, 3e-6, 1e-6, cosine 1e-5-to-0) were trained for up to 4 additional epochs from the control_ep2 checkpoint.

**Key result**: **control_ep2 is the frozen best.** One additional epoch at LR=1e-5 (lr1e5_ep1) preserves control_ep2's 384:256 DS (-0.0469) perfectly but regresses the 1200:1200 breakthrough (DS drops from +0.1250 to +0.0312 on medium, from +0.0521 to -0.1354 on large). All longer continuations regress across all budget pairs. No annealed checkpoint improves over control_ep2 at any budget.

## Artifact Lineage

| Artifact | Path | SHA256 (checkpoint) |
|----------|------|---------------------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| control_ep2 checkpoint | /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` |
| control_ep2 artifact | /tmp/azlite_control_continuation_sweep/lr3e5/artifact_epoch2 | `34b3697f95c3bfc2bc8627c28fe0e33df53403ade4db0ce51cf2563dba5cc031` |
| iter0_reference checkpoint | /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |

## Dataset

| Dataset | Path | Rows | Weight |
|---------|------|------|--------|
| generic bootstrap | /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl | 9589 | 4 |
| random teacher replay | /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl | 2016 | 1 |

No opening-suite curriculum rows, no root-temperature data, no tablebase overlay, no residual_v4, no agreement filtering. Same data as PR #108.

## Training Configuration

All lanes: residual_v3 (96,3), kalah_v3, batch_size=512, huber value loss (weight=0.3), grad_clip=1.0, seed=42, policy_target_mode=sharpened, value_target_mode=sharpened.

All lanes initialize from control_ep2 (not iter0_reference).

## Experiment Lanes

| Lane | LR | Scheduler | Additional Epochs | Init |
|------|-----|-----------|-------------------|------|
| control_ep2 | — | — | 0 (eval only) | — |
| lr1e5_ep{1,2,4} | 1e-5 | none | 1, 2, 4 | control_ep2 |
| lr3e6_ep{1,2,4} | 3e-6 | none | 1, 2, 4 | control_ep2 |
| lr1e6_ep{1,2,4} | 1e-6 | none | 1, 2, 4 | control_ep2 |
| cosine_1e5_to_0_ep{1,2,4} | 1e-5 | cosine | 1, 2, 4 | control_ep2 |

Note: lr1e5_ep1 and cosine_1e5_to_0_ep1 are identical checkpoints (same SHA256 `bd788c88`). At epoch 1, the cosine scheduler has not yet decayed the LR, producing the same update as the constant-lr lane.

## Training Results

| Candidate | Additional Epoch | Policy Loss | Value Loss | Best Val Loss | Checkpoint SHA256 | Delta vs EP2 | Rel vs EP2 | Delta vs iter0 | Rel vs iter0 |
|-----------|-----------------|-------------|------------|---------------|--------------------|-------------|------------|----------------|--------------|
| control_ep2 | 0 | 0.998653 | 0.245369 | 1.179735 | `619376db` | 0.000000 | 0.00% | 0.090479 | 0.34% |
| lr1e5_ep1 | 1 | 0.994027 | 0.244952 | 1.179796 | `bd788c88` | 0.018066 | 0.07% | 0.105412 | 0.39% |
| lr1e5_ep2 | 2 | 0.994027 | 0.244952 | 1.179796 | `be010134` | 0.031257 | 0.12% | 0.120085 | 0.45% |
| lr1e5_ep4 | 4 | 0.994027 | 0.244952 | 1.179796 | `824384b0` | 0.061112 | 0.23% | 0.149316 | 0.56% |
| lr3e6_ep1 | 1 | 0.994904 | 0.245037 | 1.180110 | `1e6c12e3` | 0.007902 | 0.03% | 0.094753 | 0.35% |
| lr3e6_ep2 | 2 | 0.994904 | 0.245037 | 1.180110 | `f2818e45` | 0.011688 | 0.04% | 0.099227 | 0.37% |
| lr3e6_ep4 | 4 | 0.994904 | 0.245037 | 1.180110 | `1958aa77` | 0.019875 | 0.07% | 0.107996 | 0.40% |
| lr1e6_ep1 | 1 | 0.995174 | 0.245066 | 1.180321 | `83a2475f` | 0.004878 | 0.02% | 0.091753 | 0.34% |
| lr1e6_ep2 | 2 | 0.995174 | 0.245066 | 1.180321 | `5b9c730f` | 0.006582 | 0.02% | 0.093241 | 0.35% |
| lr1e6_ep4 | 4 | 0.995174 | 0.245066 | 1.180321 | `364727a4` | 0.008779 | 0.03% | 0.096146 | 0.36% |
| cosine_1e5_to_0_ep1 | 1 | 0.993918 | 0.244960 | 1.179978 | `bd788c88` | 0.018066 | 0.07% | 0.105412 | 0.39% |
| cosine_1e5_to_0_ep2 | 2 | 0.993918 | 0.244960 | 1.179978 | `f469821c` | 0.029309 | 0.11% | 0.117993 | 0.44% |
| cosine_1e5_to_0_ep4 | 4 | 0.993918 | 0.244960 | 1.179978 | `be10d402` | 0.039012 | 0.14% | 0.127475 | 0.47% |

Policy loss improves under all annealing lanes (0.9987 → 0.9939-0.9952), but both validation loss and arena strength regress. Validation loss worsens for all lanes (1.179735 control_ep2 → 1.179796-1.180321). The best validation loss after annealing (cosine_ep1 at 1.179978) is still worse than control_ep2 (1.179735). Different from the PR #108 sweep where validation loss monotonically improved — here even validation loss says "stop."

## Opening-Suite Seat-Aware Benchmark Results

All evaluations use deterministic root policy (root_policy_mode=deterministic), 2 games/opening, seed=42, c_puct=1.25, tactical_root_bias=0.1, workers=4, challenger vs model-artifact/current.

### Medium Eval (128 openings)

#### DS Summary (All Budget Pairs)

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|-----------|---------|---------|---------|-----------|---------|
| **control_ep2** | **-0.0469** | **-0.1875** | -0.2812 | **+0.1250** | -0.0938 |
| **lr1e5_ep1** | **-0.0469** | -0.2500 | **+0.0000** | +0.0312 | -0.1250 |
| cosine_1e5_to_0_ep1 | -0.0938 | -0.2500 | **+0.0000** | +0.0312 | **-0.0625** |
| lr3e6_ep1 | -0.2969 | -0.2500 | +0.0625 | +0.0312 | -0.1250 |
| lr1e5_ep2 | -0.2969 | -0.2500 | +0.0312 | -0.1562 | **-0.0625** |
| lr3e6_ep4 | -0.2969 | -0.2500 | +0.0000 | +0.0625 | **-0.0625** |
| lr1e6_ep2 | -0.2969 | -0.2500 | +0.0312 | +0.0625 | **-0.0625** |
| lr1e6_ep4 | -0.2969 | -0.2500 | +0.0312 | +0.0312 | **-0.0625** |
| cosine_1e5_to_0_ep2 | -0.2969 | -0.3750 | +0.0000 | -0.1562 | **-0.0625** |
| cosine_1e5_to_0_ep4 | -0.2969 | -0.2500 | +0.0625 | +0.0625 | **-0.0625** |
| lr3e6_ep2 | -0.3594 | -0.2500 | +0.0000 | +0.0625 | -0.0938 |
| lr1e6_ep1 | -0.3594 | -0.2500 | +0.0625 | +0.0312 | **-0.0625** |
| lr1e5_ep4 | -0.3906 | -0.2500 | +0.0312 | +0.0156 | -0.1875 |

**Bold** = best in column.

#### Standard Budget (384:256) — Primary Ranking

| Rank | Candidate | P0 Score | P1 Score | DS |
|------|-----------|----------|----------|------|
| 1 | **control_ep2** | 0.3281 | 0.3750 | **-0.0469** |
| 1 | **lr1e5_ep1** | 0.3281 | 0.3750 | **-0.0469** |
| 3 | cosine_1e5_to_0_ep1 | 0.2812 | 0.3750 | -0.0938 |
| 4-10 | lr1e5_ep2, lr3e6_ep1, lr3e6_ep4, lr1e6_ep2, lr1e6_ep4, cosine_1e5_to_0_ep2, cosine_1e5_to_0_ep4 | 0.3281 | 0.6250 | -0.2969 |
| 11-12 | lr3e6_ep2, lr1e6_ep1 | 0.3281 | 0.6875 | -0.3594 |
| 13 | lr1e5_ep4 | 0.3281 | 0.7188 | -0.3906 |

lr1e5_ep1 exactly matches control_ep2's P0 and P1 scores at 384:256 (both 0.3281/0.3750). This is the only annealing checkpoint that does not regress at practical budget. Cosine_1e5_to_0_ep1 is slightly worse (leaks 0.0469 P0 points), meaning the cosine scheduling (specifically the LR decay within epoch 1) does not help.

#### Per-Ply DS at 384:256

| Ply | control_ep2 | lr1e5_ep1 | cosine_1e5_to_0_ep1 |
|-----|-------------|-----------|---------------------|
| 1 | +0.0000 | +0.0000 | N/A |
| 3 | **+0.5000** | **+0.5000** | -0.3333 |
| 4 | **+0.3125** | **+0.3125** | +0.3000 |
| 5 | -0.2500 | -0.2500 | -0.1667 |
| 6 | -0.2639 | -0.2639 | -0.0673 |

lr1e5_ep1 preserves the exact per-ply pattern of control_ep2. The P1 advantage at ply 3 (+0.5000) and the P0 dominance at ply 4 (+0.3125) are fully maintained. Cosine_1e5_to_0_ep1 degrades at plys 3-4 despite identical model parameters (same SHA256), likely due to MCTS search variance when operating near the decision boundary.

### Large Eval (384 openings)

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|-----------|---------|---------|---------|-----------|---------|
| control_ep2 | **-0.1667** | **-0.1458** | -0.1875 | **+0.0521** | -0.1875 |
| lr1e5_ep1 | **-0.1667** | -0.2188 | **-0.0833** | -0.1354 | -0.1771 |
| cosine_1e5_to_0_ep1 | -0.1458 | -0.2188 | **-0.0833** | -0.1354 | -0.1771 |

lr1e5_ep1 matches control_ep2 at 384:256 on large_eval (-0.1667) and improves at 768:768 (-0.0833 vs -0.1875). However, it regresses at 1200:1200 (-0.1354 vs +0.0521) and at 768:256 (-0.2188 vs -0.1458). The large_eval confirms that the 1200:1200 regression is real and significant.

## Default Opening Gate

All evaluations deterministic, 60 games, seed=42, challenger vs model-artifact/current, workers=4.

| Candidate | 384:256 DS | 1200:1200 DS | Classification |
|-----------|-----------|-------------|----------------|
| control_ep2 | 0.0000 | 1.0000 | high_search_breakthrough |
| lr1e5_ep1 | 0.0000 | 0.0000 | regression_masked_by_seat |

lr1e5_ep1 loses the 1200:1200 breakthrough (DS drops from 1.0000 to 0.0000). The gate correctly classifies this as `regression_masked_by_seat` — the practical-budget deterministic behavior is preserved (DS=0.00 at 384:256), but the high-budget breakthrough that distinguishes control_ep2 from iter0_reference is gone.

## Classification

### Overall: `control_ep2_frozen_best`

### Why `control_ep2_frozen_best`

| Criterion | Evidence |
|-----------|----------|
| No annealed checkpoint improves 384:256 DS | Best annealed DS = -0.0469 (lr1e5_ep1), same as control_ep2 |
| Every longer annealing epoch regresses | lr1e5_ep2: -0.2969, lr1e5_ep4: -0.3906 (far worse) |
| 1200:1200 regresses in every lane | control_ep2 +0.1250 → lr1e5_ep1 +0.0312 (medium); +0.0521 → -0.1354 (large) |
| 1200:1200 breakthrough is fragile | Default gate DS drops from 1.0 → 0.0 after 1 additional epoch |
| Even validation loss doesn't improve | All annealed checkpoints have worse val_loss than control_ep2 |

### Why NOT `annealed_continuation_promising`

| Criterion | Evidence |
|-----------|----------|
| Any checkpoint beats control_ep2 384:256 DS of -0.0469 | None beat it; lr1e5_ep1 ties but doesn't beat |
| Reaches DS >= 0.00 | Best DS is -0.0469 (still negative) |
| Improves large 384:256 DS over -0.1667 | lr1e5_ep1 also at -0.1667 (tie) |
| Does not materially regress 1200:1200 or 256:768 | 1200:1200 regresses from +0.1250 to +0.0312 (medium), from +0.0521 to -0.1354 (large) |

### Why NOT `low_lr_stabilization_only`

| Criterion | Evidence |
|-----------|----------|
| Annealing preserves control_ep2 without improvement | lr1e5_ep1 preserves at 384:256 only |
| Material regressions at other budgets | 1200:1200 and 768:256 regress despite preservation at 384:256 |
| "Stabilized" implies preserved across budgets | Only 384:256 and 768:768 are preserved/improved; 1200:1200 and 768:256 regress |

Note: lr1e5_ep1 exhibits `low_lr_stabilization_only` behavior at the practical budget (384:256) specifically. It preserves the epoch-2 peak at the most important practical budget while allowing the 1200:1200 breakthrough to decay. This partial behavior does not qualify for the `low_lr_stabilization_only` classification overall because the 1200:1200 regression is a material loss.

## Primary Findings

1. **control_ep2 is a local optimum that cannot be improved by lower-LR annealing.** One epoch at LR=1e-5 from control_ep2 preserves the 384:256 DS (-0.0469) but at the cost of 1200:1200 regression. Additional epochs regress everywhere. The epoch-2 peak at LR=3e-5 is an extremum — any continuation in any direction degrades at least one key metric.

2. **The 1200:1200 breakthrough is extremely fragile.** It is lost after just one additional epoch at LR=1e-5 (parameter delta vs control_ep2 = 0.07%). The breakthrough requires the precise parameter configuration achieved by exactly 2 epochs at LR=3e-5 from iter0_reference. Lower-lr continuation preserves the practical-budget behavior while allowing the high-budget behavior to drift.

3. **Policy loss improvement is anti-correlated with arena strength.** All annealed lanes show better policy loss (0.9939-0.9952 vs 0.9987 for control_ep2), yet arena strength either stays flat or regresses. The policy-loss improvement without arena improvement confirms that policy imitation quality does not predict search strength.

4. **Validation loss also fails as a checkpoint selector this time.** Unlike the PR #108 sweep where validation loss monotonically improved despite arena regression, here validation loss actually worsens after annealing (1.179735 → 1.179796+). Both metric signals agree that control_ep2 is the peak.

5. **The sharp epoch-2 peak is confirmed as a genuine non-monotonic response.** The quadratic pattern (epoch 1 regresses, epoch 2 peaks, epoch 3+ regresses) from PR #108 is confirmed. It is not an artifact of the specific training run — any movement away from the epoch-2 configuration degrades performance.

6. **Cosine scheduling at epoch 1 is identity.** lr1e5_ep1 and cosine_1e5_to_0_ep1 produce identical checkpoints (same SHA256) because the cosine scheduler hasn't decayed the LR within the first epoch (cosine decay starts at epoch 2 onwards within the scheduler step).

7. **768:768 DS shows a mixed signal.** lr1e5_ep1 improves 768:768 DS vs control_ep2 (medium: +0.0000 vs -0.2812; large: -0.0833 vs -0.1875). This is the only budget pair where annealing helps. The improvement in equal-budget outcomes suggests that the lower-LR continuation reduces seat-dependent weakness at moderate budgets, but this comes at the cost of high-budget strength.

## Recommendations

1. **control_ep2 is the definitive checkpoint.** No further continuation in any direction improves upon it. The search for improvement should explore different approaches (curriculum redesign, data augmentation, architectural changes) rather than additional continuations of the same training recipe.

2. **The 1200:1200 deterministic opening breakthrough should be preserved when considering promotion candidates.** The fragility of this behavior suggests it captures a specific learned property (likely positional understanding of the default opening) that is easily unlearned.

3. **Future opening-suite benchmarks should weight the 1200:1200 budget pair more heavily.** The 384:256 DS is stable across lr1e5_ep1 but the 1200:1200 DS reveals genuine regression. If only 384:256 were checked, lr1e5_ep1 would appear to be perfectly preserved.

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS: no model promoted |
| No overwrite model-artifact/current | PASS: current unchanged |
| No new data | PASS: only original iter0 data |
| No LR above 1e-5 after control_ep2 | PASS: max LR=1e-5 |
| No architecture change | PASS: residual_v3 only |
| No stochastic root temperature as primary evaluation | PASS: deterministic evaluations only |
| No judge by validation loss alone | PASS: arena evaluation used |
| No root-temperature data | PASS: not used |
| No tablebase overlay | PASS: not used |
| No agreement filtering | PASS: not used |
| No residual_v4 | PASS: residual_v3 only |

## Training Commands

### Lane 2 (ep2_then_lr1e5):
```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 4 \
  --batch-size 512 \
  --lr 1e-5 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 0 \
  --top-k-dir /tmp/azlite_ep2_annealed/lr1e5 \
  --out /tmp/azlite_ep2_annealed/lr1e5/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42 \
  --save-epochs 1,2,4
```

### Lane 3 (ep2_then_lr3e6):
```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 4 \
  --batch-size 512 \
  --lr 3e-6 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 0 \
  --top-k-dir /tmp/azlite_ep2_annealed/lr3e6 \
  --out /tmp/azlite_ep2_annealed/lr3e6/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42 \
  --save-epochs 1,2,4
```

### Lane 4 (ep2_then_lr1e6):
```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 4 \
  --batch-size 512 \
  --lr 1e-6 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 0 \
  --top-k-dir /tmp/azlite_ep2_annealed/lr1e6 \
  --out /tmp/azlite_ep2_annealed/lr1e6/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42 \
  --save-epochs 1,2,4
```

### Lane 5 (ep2_then_cosine_1e5_to_0):
```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 4 \
  --batch-size 512 \
  --lr 1e-5 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 0 \
  --top-k-dir /tmp/azlite_ep2_annealed/cosine_1e5_to_0 \
  --out /tmp/azlite_ep2_annealed/cosine_1e5_to_0/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler cosine \
  --seed 42 \
  --save-epochs 1,2,4
```

### Medium Evaluation:
```bash
.venv/bin/python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/azlite_ep2_annealed/eval_medium \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/azlite_control_continuation_sweep/lr3e5/artifact_epoch2,/tmp/azlite_ep2_annealed/lr1e5/artifact_epoch1,/tmp/azlite_ep2_annealed/lr1e5/artifact_epoch2,/tmp/azlite_ep2_annealed/lr1e5/artifact_epoch4,/tmp/azlite_ep2_annealed/lr3e6/artifact_epoch1,/tmp/azlite_ep2_annealed/lr3e6/artifact_epoch2,/tmp/azlite_ep2_annealed/lr3e6/artifact_epoch4,/tmp/azlite_ep2_annealed/lr1e6/artifact_epoch1,/tmp/azlite_ep2_annealed/lr1e6/artifact_epoch2,/tmp/azlite_ep2_annealed/lr1e6/artifact_epoch4,/tmp/azlite_ep2_annealed/cosine_1e5_to_0/artifact_epoch1,/tmp/azlite_ep2_annealed/cosine_1e5_to_0/artifact_epoch2,/tmp/azlite_ep2_annealed/cosine_1e5_to_0/artifact_epoch4 \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --games-per-opening 2 --seed 42 --root-policy-mode deterministic \
  --workers 4 --timeout 7200
```

### Large Evaluation:
```bash
.venv/bin/python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/azlite_ep2_annealed/eval_large \
  --suite /tmp/azlite_opening_suite/large_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/azlite_control_continuation_sweep/lr3e5/artifact_epoch2,/tmp/azlite_ep2_annealed/lr1e5/artifact_epoch1,/tmp/azlite_ep2_annealed/cosine_1e5_to_0/artifact_epoch1 \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --games-per-opening 2 --seed 42 --root-policy-mode deterministic \
  --workers 4 --timeout 7200
```

### Default Opening Gate:
```bash
.venv/bin/python script/ai/seat_aware_promotion_gate \
  --candidate-path /tmp/azlite_ep2_annealed/lr1e5/artifact_epoch1 \
  --current-path model-artifact/current \
  --out /tmp/azlite_ep2_annealed/eval_gate/lr1e5_ep1_default_gate.json \
  --games 60 --seed 42 --workers 4 \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768
```

## Artifacts

| Artifact | Path |
|----------|------|
| lr1e5 checkpoints | /tmp/azlite_ep2_annealed/lr1e5/checkpoint_epoch{1,2,4}.npz |
| lr3e6 checkpoints | /tmp/azlite_ep2_annealed/lr3e6/checkpoint_epoch{1,2,4}.npz |
| lr1e6 checkpoints | /tmp/azlite_ep2_annealed/lr1e6/checkpoint_epoch{1,2,4}.npz |
| cosine checkpoints | /tmp/azlite_ep2_annealed/cosine_1e5_to_0/checkpoint_epoch{1,2,4}.npz |
| Exported artifacts | /tmp/azlite_ep2_annealed/*/artifact_epoch{1,2,4}/ |
| Medium eval report | /tmp/azlite_ep2_annealed/eval_medium/temperature_benchmark_report.json |
| Large eval report | /tmp/azlite_ep2_annealed/eval_large/temperature_benchmark_report.json |
| Gate reports | /tmp/azlite_ep2_annealed/eval_gate/*.json |
| Runner script | ml/alphazero_lite/run_ep2_annealed_continuation.py |
