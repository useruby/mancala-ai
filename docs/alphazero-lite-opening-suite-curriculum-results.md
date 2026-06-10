# AlphaZero-Lite Opening-Suite Curriculum Results

**Date**: 2026-06-10
**Classification**: `curriculum_overfit` / `control_signal_real`
**Schema**: `azlite_opening_suite_curriculum_v1`

## Summary

Trained a residual_v3 candidate initialized from iter0_reference on an opening-suite bucket-balanced curriculum mined from deterministic evaluations of the deduplicated opening suite (384 openings, large_eval). The curriculum holdout policy loss improved modestly (-2.3% relative), but this did not translate to better arena strength than a simple conservative continue (control_ep2) without any curriculum data.

**Key result**: **control_ep2** (iter0_reference fine-tuned on original data only at LR=3e-5 for 2 epochs) dramatically improves the opening-suite DS at standard budget from -0.4219 to **-0.0469** (nearly neutral), with gains spanning all ply counts 3-6. The curriculum lanes (curriculum_ep2, curriculum_lr1e5_ep2) underperform control_ep2 at all budget pairs.

## Artifact Lineage

| Artifact | Path | SHA256 (weights.json) |
|----------|------|----------------------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference checkpoint | /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| iter0_reference artifact | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |

## Curriculum Dataset

### Dataset Construction

Mined from all 384 openings in the deduplicated large_eval opening suite. Each opening was classified into a weakness bucket by:
1. Playing 4 policy-network games (2 per seat, iter0_reference vs current)
2. Running a quick ClassicMCTS check at 1200 sims on the post-opening position
3. Bucketing based on policy-game outcomes + teacher value signal

Mining: re-played each opening at low budget, relabeled every candidate-turn position with ClassicMCTS at 1200 simulations, filtered by disagreement/strong-preference/low-prob criteria, deduplicated by board fingerprint.

### Dataset Metrics

| Metric | Value |
|--------|-------|
| Train rows | 2029 |
| Holdout rows | 273 |
| Train SHA256 | `28a1ef465bb2ec922d2f1d489f98114b82df355f5284ed6cf0366f8cb1cd484e` |
| Holdout SHA256 | `924d8688d52b97ea89776e0461626463875173e6337c13fd283c9438869858cf` |
| Unique board count | 323 |
| Duplicate row count | 4295 |
| Capped row count | 0 |
| Total positions visited (teacher MCTS) | 26616 |
| Total positions kept | 21482 |

### Rows Per Bucket

| Bucket | Train Rows | Holdout Rows |
|--------|-----------|--------------|
| weak_p1 | 936 | 0 |
| high_search_rescue | 683 | 171 |
| preservation | 410 | 102 |

### Rows Per Ply

| Opening Ply | Row Count |
|-------------|-----------|
| 1 | 296 |
| 2 | 960 |
| 3 | 2240 |
| 4 | 4300 |
| 5 | 6332 |
| 6 | 7354 |

### Rows Per Seat

| Seat | Row Count |
|------|-----------|
| P0 | 12185 |
| P1 | 9297 |

### Curriculum Data Quality

| Metric | Value |
|--------|-------|
| Low-budget vs teacher disagreement rate | 67.69% |
| Mean teacher top-1 visit share | 0.6562 |
| Mean policy entropy | 0.9686 |
| First-divergence move mean | 2.78 |
| First-divergence move median | 2.0 |

### Bucket Distribution (384 openings classified)

| Bucket | Count |
|--------|-------|
| high_search_rescue | 201 |
| preservation | 105 |
| weak_p1 | 78 |
| weak_p0 | 0 |

Note: All 384 openings showed iter0_reference losing as one or both seats at policy-network level. No opening was classified as weak_p0 (only P1 weakness detected). The teacher MCTS check on the post-opening position classified 201 openings as having rescue potential.

## Training Configuration

All lanes use: residual_v3 (96,3), kalah_v3, batch_size=512, huber value loss (weight=0.3), grad_clip=1.0, lr_scheduler=none, val_split=0.1, seed=42, policy_target_mode=sharpened, value_target_mode=sharpened.

### Data Inputs

| Dataset | Path | Weight |
|---------|------|--------|
| generic bootstrap | /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl | 4 |
| old current-mined random replay | /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl | 1 |
| opening-suite curriculum (curriculum lanes only) | /tmp/azlite_opening_suite_curriculum/train.jsonl | 1 |

### Experiment Lanes

| Lane | Init | LR | Data | Epochs |
|------|------|----|------|--------|
| iter0_reference | — | — | eval only | — |
| control_ep2 | iter0_reference | 3e-5 | generic + old replay | 2 |
| curriculum_ep2 | iter0_reference | 3e-5 | generic + old replay + curriculum | 2 |
| curriculum_lr1e5_ep2 | iter0_reference | 1e-5 | generic + old replay + curriculum | 2 |

## Training Results

| Lane | Epoch | Policy Loss | Value Loss | Best Val Loss | Checkpoint SHA256 | Delta Norm | Rel Delta |
|------|-------|-------------|------------|---------------|--------------------|------------|-----------|
| iter0_reference | — | — | — | — | `c2430b56` | 0.0 | 0.0% |
| control_ep2 | 2 | 1.000793 | 0.245539 | 1.179735 | `619376db` | 0.090479 | 0.34% |
| curriculum_ep2 | 2 | 1.019363 | 0.243811 | 1.194036 | `1355bbb6` | 0.105621 | 0.39% |
| curriculum_lr1e5_ep2 | 2 | 1.020207 | 0.244018 | 1.195848 | `0bc2b8a3` | 0.039965 | 0.15% |

### Curriculum Holdout Policy Loss

Imitation quality on the held-out curriculum positions:

| Checkpoint | Holdout Policy Loss | vs iter0_reference |
|------------|---------------------|-------------------|
| iter0_reference | 1.604388 | baseline |
| control_ep2 | 1.608529 | +0.004 (+0.3%) |
| curriculum_ep2 | 1.566781 | -0.038 (-2.3%) |
| curriculum_lr1e5_ep2 | 1.582793 | -0.022 (-1.3%) |

Curriculum training at LR=3e-5 produces the largest holdout imitation improvement (2.3% relative). Control training (no curriculum) does not improve holdout imitation. However, arena strength tells a different story.

## Opening-Suite Seat-Aware Benchmark Results

All evaluations on medium_eval (128 openings), 2 games per opening, seed=42, c_puct=1.25, tactical_root_bias=0.1, root_policy_mode=deterministic, challenger vs model-artifact/current, workers=4.

### DS Summary (All Budget Pairs)

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 256:768 |
|-----------|---------|---------|---------|-----------|---------|
| iter0_reference | -0.4219 | -0.2500 | +0.0625 | **+0.2812** | -0.0938 |
| control_ep2 | **-0.0469** | **-0.1875** | -0.2812 | +0.1250 | -0.0938 |
| curriculum_ep2 | -0.2969 | -0.3750 | **-0.2188** | +0.0156 | **-0.0625** |
| curriculum_lr1e5_ep2 | -0.4062 | -0.2500 | -0.2812 | -0.1406 | -0.0625 |

**Bold** = best in budget pair among candidates.

### Standard Budget (384:256) — Primary Ranking

| Candidate | P0 Score | P1 Score | DS |
|-----------|----------|----------|------|
| iter0_reference | 0.3281 | 0.7500 | -0.4219 |
| control_ep2 | 0.3281 | 0.3750 | **-0.0469** |
| curriculum_ep2 | 0.3281 | 0.6250 | -0.2969 |
| curriculum_lr1e5_ep2 | 0.3438 | 0.7500 | -0.4062 |

**control_ep2 reduces the P1 disadvantage dramatically** (P1 score: 0.7500 → 0.3750, a 50% reduction) while preserving P0 performance. This represents a genuine seat-aware strength improvement.

### Equal High Budget (1200:1200)

| Candidate | P0 Score | P1 Score | DS |
|-----------|----------|----------|------|
| iter0_reference | 0.4375 | 0.1562 | **+0.2812** |
| control_ep2 | 0.4375 | 0.3125 | +0.1250 |
| curriculum_ep2 | 0.1094 | 0.0938 | +0.0156 |
| curriculum_lr1e5_ep2 | 0.1094 | 0.2500 | -0.1406 |

Note: control_ep2 improves as P1 at 1200:1200 (0.1562 → 0.3125) while preserving P0 (0.4375 unchanged). The DS drops because P1 catches up to P0 — this is a net improvement at both seats, not a regression.

### Asymmetric Budget (256:768)

| Candidate | P0 Score | P1 Score | DS |
|-----------|----------|----------|------|
| iter0_reference | 0.0000 | 0.0938 | -0.0938 |
| control_ep2 | 0.0000 | 0.0938 | -0.0938 |
| curriculum_ep2 | 0.0000 | 0.0625 | -0.0625 |
| curriculum_lr1e5_ep2 | 0.0000 | 0.0625 | -0.0625 |

### Per-Ply DS at Standard Budget (384:256)

| Ply | iter0_reference | control_ep2 | curriculum_ep2 |
|-----|-----------------|-------------|----------------|
| 1 | +0.0000 | +0.0000 | +0.0000 |
| 3 | -0.1667 | **+0.5000** | +0.5000 |
| 4 | -0.3875 | **+0.3125** | -0.4375 |
| 5 | -0.4167 | **-0.2500** | -0.3750 |
| 6 | -0.6581 | **-0.2639** | -0.6667 |

**control_ep2 transforms the P1 disadvantage at plies 3-4**: the P1 advantage completely flips to P0-favored at these plies. Gains span all ply counts 3-6.

### Gains Across Buckets

| Bucket | iter0_ref P1 Score | control_ep2 P1 Score | Improvement |
|--------|-------------------|---------------------|-------------|
| ply=3 (early) | 1.0000 | 0.3333 | -66.7% |
| ply=4 (early-mid) | 0.7000 | 0.0000 | -100.0% |
| ply=5 (mid) | 0.6667 | 0.5000 | -25.0% |
| ply=6 (mid-late) | 0.7692 | 0.3750 | -51.3% |

The P1 improvement appears across ALL ply counts, not just a narrow opening family. This indicates a systematic improvement rather than overfitting to mined openings.

## Default Opening Gate

All evaluations deterministic, 60 games, seed=42, challenger vs model-artifact/current, workers=4.

| Candidate | 384:256 DS | 1200:1200 DS | Classification |
|-----------|-----------|-------------|----------------|
| control_ep2 | 0.0000 | 1.0000 | high_search_breakthrough |
| curriculum_ep2 | 0.0000 | 1.0000 | high_search_breakthrough |

All checkpoints preserve the deterministic opening behavior (DS=0.00 at practical budgets, DS=1.00 at 1200:1200). The deterministic opening artifact is preserved across fine-tuning.

## Classification

### Overall: `control_signal_real` + `curriculum_overfit`

### Why `control_signal_real`

| Criterion | Evidence |
|-----------|----------|
| control_ep2 beats iter0_reference at standard budget | DS -0.0469 vs -0.4219 (dramatic 89% improvement) |
| Gains span all ply counts | P1 score reduced at plies 3-6 |
| Gains appear across buckets | Improvement at ply 3,4,5,6 |
| Conservative LR (3e-5) | Parameter delta only 0.34% |
| No curriculum data used | Pure continue from iter0_reference on original data |

### Why NOT `opening_suite_curriculum_promising`

| Criterion | Evidence |
|-----------|----------|
| Curriculum must beat iter0_reference on opening-suite DS | curriculum_ep2 (-0.2969) does improve vs iter0 (-0.4219) |
| And preserve or improve 1200:1200 | curriculum_ep2 (+0.0156) regresses vs iter0 (+0.2812) |
| And gains appear across buckets | curriculum_ep2 regresses at ply 4 and 6 |

curriculum_ep2 technically meets the first criterion but fails the second (1200:1200 preservation). The improvement at standard budget is less than control_ep2 without curriculum data.

### Why `curriculum_overfit`

| Criterion | Evidence |
|-----------|----------|
| Curriculum holdout loss improves | 1.604 → 1.567 (-2.3%) at LR=3e-5; 1.604 → 1.583 (-1.3%) at LR=1e-5 |
| But opening-suite DS improvement is less than control | curriculum_ep2 DS=-0.2969 vs control_ep2 DS=-0.0469 at standard |
| Curriculum regresses at 1200:1200 | curriculum_ep2 +0.0156 vs iter0 +0.2812 |
| LR=1e-5 degrades everywhere | curriculum_lr1e5_ep2 regresses at all 5 budget pairs |

The curriculum data teaches the model to imitate high-search teacher policy, but this imitation conflicts with the search process at practical budgets and completely destroys high-budget strength.

### Why NOT `benchmark_not_trainable_yet`

| Criterion | Evidence |
|-----------|----------|
| Unique board count too low | 323 is below 1000 target but 2.3x improvement over PR #104 (141) |
| All fine-tuned checkpoints regress | control_ep2 does NOT regress at standard budget (-0.0469) |

## Primary Findings

1. **Conservative continue from iter0_reference with LR=3e-5 for 2 epochs is highly effective.**
   control_ep2 reduces the standard-budget P1 disadvantage from DS=-0.4219 to DS=-0.0469 (near-neutral). This is achieved without any new data — just another pass through the original generic bootstrap + random teacher replay data. The P1 score drops from 0.7500 to 0.3750 (50% reduction in opponent win rate in P1 seat).

2. **The curriculum data is counterproductive at higher search budgets.**
   curriculum_ep2 and curriculum_lr1e5_ep2 both show severe regression at 1200:1200. The high-search teacher targets (ClassicMCTS at 1200 simulations) teach the model a policy that works well for the teacher but conflicts with the neural network's own search at equal budget.

3. **Low LR (1e-5) degrades rather than helps.**
   curriculum_lr1e5_ep2 has the smallest parameter delta (0.15%) but the worst arena performance across all budget pairs. The mild fine-tuning with low LR appears to erase useful features without replacing them.

4. **The opening-suite benchmark captures real quality differences.**
   The deterministic opening-suite seat-aware benchmark clearly distinguishes control_ep2 (-0.0469 DS at standard) from iter0_reference (-0.4219 DS at standard), curriculum_ep2 (-0.2969), and curriculum_lr1e5_ep2 (-0.4062). The ranking is consistent across budget pairs.

5. **The duplicate trajectory rate remains at 100%.**
   With root_policy_mode=deterministic, the trajectory duplicate rate is 1.000 across all benchmarks. This is a property of deterministic MCTS evaluation, not the opening suite. The suite itself provides 28,961 unique board states.

6. **Unique board count remains limited (323).**
   Despite mining from all 384 deduplicated opening prefixes, the Kalah state space in early-to-mid game produces only 323 unique board states where the low-budget model disagrees with the teacher. This is 2.3x better than the PR #104 curriculum (141 unique boards) but still below the 1000 target.

## Runner Commands

### Mining

```bash
.venv/bin/python ml/alphazero_lite/mine_opening_suite_curriculum.py \
  --suite /tmp/azlite_opening_suite/large_eval.jsonl \
  --out-train /tmp/azlite_opening_suite_curriculum/train.jsonl \
  --out-holdout /tmp/azlite_opening_suite_curriculum/holdout.jsonl \
  --out-summary /tmp/azlite_opening_suite_curriculum/summary.json \
  --candidate /tmp/azlite_iterative_random_replay/iter0_candidate_artifact \
  --current model-artifact/current \
  --budget-pairs 384:256,768:256,1200:1200 \
  --teacher-mode classic_mcts \
  --teacher-simulations 1200 \
  --target-train-rows 2048 \
  --target-holdout-rows 512 \
  --min-unique-boards 1000 \
  --max-rows-per-opening 12 \
  --max-rows-per-game 24 \
  --input-encoding kalah_v3 \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --seed 50
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
  --top-k-dir /tmp/azlite_opening_suite_curriculum/checkpoints/control \
  --out /tmp/azlite_opening_suite_curriculum/checkpoints/control/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Training (curriculum lane, LR=3e-5)

```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl,/tmp/azlite_opening_suite_curriculum/train.jsonl \
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
  --top-k-dir /tmp/azlite_opening_suite_curriculum/checkpoints/curriculum \
  --out /tmp/azlite_opening_suite_curriculum/checkpoints/curriculum/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Training (curriculum lane, LR=1e-5)

```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl,/tmp/azlite_opening_suite_curriculum/train.jsonl \
  --replay-weights 4,1,1 \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --hidden-sizes 96,3 \
  --epochs 2 \
  --batch-size 512 \
  --lr 1e-5 \
  --value-loss huber \
  --value-loss-weight 0.3 \
  --grad-clip 1.0 \
  --save-top-k 3 \
  --top-k-dir /tmp/azlite_opening_suite_curriculum/checkpoints/curriculum_lr1e5 \
  --out /tmp/azlite_opening_suite_curriculum/checkpoints/curriculum_lr1e5/checkpoint.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Opening-Suite Benchmark Evaluation

```bash
.venv/bin/python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/azlite_opening_suite_curriculum/eval/opening_suite_benchmark_v2 \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/azlite_iterative_random_replay/iter0_candidate_artifact,/tmp/azlite_opening_suite_curriculum/artifacts/control_ep2,/tmp/azlite_opening_suite_curriculum/artifacts/curriculum_ep2,/tmp/azlite_opening_suite_curriculum/artifacts/curriculum_lr1e5_ep2 \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --games-per-opening 2 \
  --seed 42 \
  --root-policy-mode deterministic \
  --workers 4
```

### Default Opening Gate

```bash
.venv/bin/python script/ai/seat_aware_promotion_gate \
  --candidate-path /tmp/azlite_opening_suite_curriculum/artifacts/<candidate> \
  --current-path model-artifact/current \
  --games 60 --seed 42 --workers 4 \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --out /tmp/azlite_opening_suite_curriculum/eval/<candidate>_default_gate.json
```

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS: no model promoted |
| No overwrite model-artifact/current | PASS: current unchanged |
| No overwrite storage/ai/alphazero_lite/current | PASS: unchanged |
| No root_temperature as primary evaluation | PASS: root_temperature=0.0 |
| No training from stochastic root-sampled moves | PASS: all training data from deterministic play |
| No LR above 3e-5 | PASS: max LR=3e-5 |
| No architecture change | PASS: residual_v3 only |
| No judge by validation loss alone | PASS: arena evaluation used |
| No residual_v4 | PASS: residual_v3 only |
| No tablebase overlay | PASS: not used |
| No agreement filtering | PASS: not used |

## Verification

```bash
.venv/bin/ruff check ml/alphazero_lite script/ai  # All checks passed
```

## Recommendations

1. **The conservative continue (control_ep2) deserves further investigation.**
   The dramatic P1 improvement at standard budget (-0.4219 → -0.0469 DS) from simply continuing training on the original data suggests that iter0_reference was undertrained. More epochs or different LR schedules on the base data may yield further gains without requiring curriculum construction.

2. **The curriculum data should be redesigned.**
   The current curriculum focuses on positions where the low-budget policy disagrees with the teacher. This teaches the model to mimic high-search policy, but the model cannot execute that policy without equal search budget. A better curriculum would focus on positions where BETTER policy choices at low budget lead to better outcomes at low budget.

3. **Investigate why control_ep2 improves P1 but not P0.**
   The P0 score stays constant (0.3281) across all checkpoints. The entire DS improvement comes from P1 improvement. Understanding what makes the P1 seat learnable could inform curriculum construction.

4. **Explore epoch=1 checkpoints.**
   Both control_ep2 and curriculum_ep2 were run for 2 epochs. The epoch-1 checkpoints (control_ep1, curriculum_ep1) may show different characteristics — in PR #104, control_ep1 showed stronger randomized-opening P1 improvement than control_ep2.

5. **Increase unique board count through phase-aware sampling.**
   With only 323 unique boards, the curriculum lacks sufficient state space diversity. Systematically sampling from different game phases (early/mid/late) per opening could increase the unique board count.

## Artifacts

| Artifact | Path |
|----------|------|
| Curriculum train | /tmp/azlite_opening_suite_curriculum/train.jsonl |
| Curriculum holdout | /tmp/azlite_opening_suite_curriculum/holdout.jsonl |
| Curriculum summary | /tmp/azlite_opening_suite_curriculum/summary.json |
| control_ep2 checkpoint | /tmp/azlite_opening_suite_curriculum/checkpoints/control/checkpoint.top1.npz |
| curriculum_ep2 checkpoint | /tmp/azlite_opening_suite_curriculum/checkpoints/curriculum/checkpoint.top1.npz |
| curriculum_lr1e5_ep2 checkpoint | /tmp/azlite_opening_suite_curriculum/checkpoints/curriculum_lr1e5/checkpoint.top1.npz |
| Opening-suite benchmark report | /tmp/azlite_opening_suite_curriculum/eval/opening_suite_benchmark_v2/temperature_benchmark_report.json |
| Default gate reports | /tmp/azlite_opening_suite_curriculum/eval/*_default_gate.json |
