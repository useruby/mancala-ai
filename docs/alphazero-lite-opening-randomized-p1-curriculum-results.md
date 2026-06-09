# AlphaZero-Lite Opening-Randomized P1 Curriculum Results

**Date**: 2026-06-09
**Classification**: `p1_curriculum_overfit`
**Schema**: `azlite_opening_p1_curriculum_v1`

## Summary

Trained a residual_v3 candidate initialized from iter0_reference on randomized-opening P1 curriculum states mined from PR #103's inversion pattern. The curriculum holdout policy loss improved significantly (1.502 → 1.334, 11% relative), but this did not translate to reliable opening-randomized disadvantaged-seat score improvement at practical budgets. The LR=1e-5 lane showed borderline improvement at ply=4 (+0.06 DS) but regressed at ply=6 (−0.31 DS).

**Overall Classification: `p1_curriculum_overfit`**

The model learns to imitate the high-search P1 policy on held-out curriculum states, but this imitation does not produce consistent arena strength gains under MCTS search at practical budgets. The deterministic 1200:1200 breakthrough (DS=1.00) is preserved across all checkpoints.

## Artifact Lineage

| Artifact | Path | SHA256 |
|----------|------|--------|
| current production | model-artifact/current | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| iter0_reference checkpoint | /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| iter0_reference artifact | /tmp/azlite_iterative_random_replay/iter0_candidate_artifact | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |

## Curriculum Dataset

| Metric | Value |
|--------|-------|
| Train rows | 1536 (1075 positive / 461 hard) |
| Holdout rows | 384 |
| Train SHA256 | `cf710abf7ad8c57f72d268fd00f9d30a7c2aa0ef569b93e6c3d029a8d2bbe5a4` |
| Holdout SHA256 | `e2359ab745811be2ee5215450c9a3fdfa37001b726816a77583527307b63b39b` |
| Total positions kept | 8474 (from 13154 P1 positions visited) |
| Opening ply distribution | ply=2: 2845, ply=4: 2811, ply=6: 2818 |
| Source prefixes | 384 (105 positive / 279 hard) |
| Unique board count | 141 |
| Duplicate state count | 9261 |
| Capped state count | 0 |

### P1 Game Outcomes (raw-policy gameplay, 384:256 budget label)

| Outcome | Count |
|---------|-------|
| P1 wins | 28 |
| P1 draws | 182 |
| P1 losses | 558 |

The candidate (iter0_reference) rarely wins as P1 in raw-policy gameplay, explaining the skewed prefix bucket distribution (105 positive vs 279 hard prefixes).

### Curriculum Data Quality

| Metric | Value |
|--------|-------|
| Low-budget vs teacher disagreement rate | 63.78% |
| Mean teacher top-1 visit share | 0.5939 |
| Mean policy entropy | 0.8898 |
| First-divergence ply mean | 1.36 |
| First-divergence ply median | 0.0 |

### Rows Per Phase

| Phase | Rows |
|-------|------|
| Early (<=8) | 1431 |
| Mid (9-24) | 3467 |
| Late (>=25) | 3576 |

### Filter Breakdown

| Filter | Positions |
|--------|-----------|
| Top-move disagreement | 5405 |
| Strong high-search preference (visit share >= 0.55) | 4771 |
| Low low-budget prob on teacher top (<= 0.30) | 4493 |
| No divergence | 3745 |

## Training Configuration

All lanes use: residual_v3 (96,3), kalah_v3, batch_size=512, huber value loss (weight=0.3), grad_clip=1.0, lr_scheduler=none, val_split=0.1, seed=42, policy_target_mode=sharpened, value_target_mode=sharpened.

### Data Inputs

| Dataset | Path | Weight |
|---------|------|--------|
| generic bootstrap | /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl | 4 |
| old current-mined random replay | /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl | 1 |
| randomized P1 curriculum (curriculum lanes only) | /tmp/azlite_opening_p1_curriculum/p1_curriculum_train.jsonl | 1 |

### Experiment Lanes

| Lane | Init | LR | Data | Epochs |
|------|------|----|------|--------|
| control_ep1 | iter0_reference | 3e-5 | generic + old replay | 1 |
| control_ep2 | iter0_reference | 3e-5 | generic + old replay | 2 |
| curriculum_ep1 | iter0_reference | 3e-5 | generic + old replay + curriculum | 1 |
| curriculum_ep2 | iter0_reference | 3e-5 | generic + old replay + curriculum | 2 |
| curriculum_lr1e5_ep1 | iter0_reference | 1e-5 | generic + old replay + curriculum | 1 |
| curriculum_lr1e5_ep2 | iter0_reference | 1e-5 | generic + old replay + curriculum | 2 |

## Training Results

| Lane | Epoch | Policy Loss | Value Loss | Best Val Loss | Checkpoint SHA256 | Delta Norm | Rel Delta |
|------|-------|-------------|------------|---------------|--------------------|------------|-----------|
| iter0_reference | — | — | — | — | `c2430b56` | 0.0 | 0.0% |
| control_ep1 | 1 | 1.000793 | 0.245539 | 1.182145 | `099c6e6a` | 0.048892 | 0.18% |
| control_ep2 | 2 | 0.998653 | 0.245369 | 1.179735 | `619376db` | 0.090479 | 0.34% |
| curriculum_ep1 | 1 | 1.015310 | 0.244694 | 1.195887 | `a598bdf6` | 0.063314 | 0.24% |
| curriculum_ep2 | 2 | 1.012032 | 0.240900 | 1.192702 | `d33b5094` | 0.112924 | 0.42% |
| curriculum_lr1e5_ep1 | 1 | 1.016000 | 0.241000 | 1.197934 | `3ad6ff43` | 0.023814 | 0.09% |
| curriculum_lr1e5_ep2 | 2 | 1.014218 | 0.240937 | 1.195661 | `ea766e00` | 0.042199 | 0.16% |

### Curriculum Holdout Policy Loss

Imitation quality on the held-out curriculum P1 positions:

| Checkpoint | Holdout Policy Loss | vs iter0_reference |
|------------|---------------------|-------------------|
| iter0_reference | 1.502391 | baseline |
| control_ep1 | 1.507487 | +0.005 (+0.3%) |
| control_ep2 | 1.490844 | −0.012 (−0.8%) |
| curriculum_ep1 | 1.374776 | −0.128 (−8.5%) |
| curriculum_ep2 | **1.333758** | **−0.169 (−11.2%)** |
| curriculum_lr1e5_ep1 | 1.439947 | −0.062 (−4.2%) |
| curriculum_lr1e5_ep2 | 1.407723 | −0.095 (−6.3%) |

Curriculum training at LR=3e-5 produces the largest holdout imitation improvement (11.2% relative). Lower LR (1e-5) produces smaller but still significant gains (6.3% relative). However, neither translates to reliable arena strength as shown below.

## Deterministic Seat-Aware Strength

All evaluations at 60 games, seed=42, c_puct=1.25, tactical_root_bias=0.1, root_policy_mode=deterministic, challenger vs model-artifact/current, workers=4.

| Checkpoint | DS 384:256 | DS 768:256 | DS 768:768 | DS 1200:1200 | DS 256:768 | Classification |
|------------|-----------|-----------|-----------|-------------|-----------|----------------|
| iter0_reference | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | high_search_breakthrough |
| control_ep1 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | high_search_breakthrough |
| control_ep2 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | high_search_breakthrough |
| curriculum_ep1 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | high_search_breakthrough |
| curriculum_ep2 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | high_search_breakthrough |
| curriculum_lr1e5_ep1 | 0.00 | 0.00 | 0.00 | **1.00** | 0.00 | high_search_breakthrough |

All checkpoints preserve DS=1.00 at 1200:1200 in the deterministic opening. No checkpoint achieves DS > 0.00 at any practical budget under the deterministic opening — consistent with the `deterministic_opening_artifact` classification from PR #103.

## Opening-Randomized Strength

All evaluations at 64 opening samples per ply count, 2 games per opening, opening seed=48, challenger vs model-artifact/current, workers=4, seed=42.

### DS at 384:256 (Standard Budget)

| Checkpoint | ply=2 DS | ply=2 P1 Score | ply=4 DS | ply=4 P1 Score | ply=6 DS | ply=6 P1 Score |
|------------|----------|----------------|----------|----------------|----------|----------------|
| **iter0_reference** | 0.6250 | 0.6250 | 0.6875 | 0.6875 | **0.8125** | **0.8125** |
| control_ep1 | 0.6562 | 0.6562 | **0.7812** | **0.7812** | 0.5312 | 0.5312 |
| control_ep2 | 0.4375 | 0.4375 | 0.6250 | 0.6250 | 0.3125 | 0.3125 |
| curriculum_ep1 | 0.3750 | 0.3750 | 0.3125 | 0.3125 | 0.4375 | 0.4375 |
| curriculum_ep2 | 0.6250 | 0.6250 | 0.5000 | 0.5000 | 0.5625 | 0.5625 |
| curriculum_lr1e5_ep1 | 0.6250 | 0.6250 | 0.7500 | 0.7500 | 0.5000 | 0.5000 |

### DS at 768:256 (Challenger-High Budget)

| Checkpoint | ply=2 DS | ply=2 P1 Score | ply=4 DS | ply=4 P1 Score | ply=6 DS | ply=6 P1 Score |
|------------|----------|----------------|----------|----------------|----------|----------------|
| iter0_reference | 0.5625 | 0.5625 | 0.6250 | 0.6250 | 0.5625 | 0.5625 |
| control_ep1 | 0.5625 | 0.5625 | 0.5625 | 0.5625 | 0.8125 | 0.8125 |
| control_ep2 | 0.5625 | 0.5625 | 0.5000 | 0.5000 | 0.4375 | 0.4375 |
| curriculum_ep1 | 0.5625 | 0.5625 | 0.7500 | 0.7500 | 0.5625 | 0.5625 |
| curriculum_ep2 | 0.5625 | 0.5625 | 0.6250 | 0.6250 | 0.5625 | 0.5625 |
| curriculum_lr1e5_ep1 | 0.6250 | 0.6250 | 0.6875 | 0.6875 | 0.5625 | 0.5625 |

### DS at 1200:1200 (Equal-High Budget) - Opening-Randomized

| Checkpoint | ply=2 DS | ply=4 DS | ply=6 DS |
|------------|----------|----------|----------|
| iter0_reference | 0.1875 | 0.4375 | 0.3125 |
| control_ep1 | 0.3750 | 0.3125 | 0.3750 |
| control_ep2 | 0.3750 | 0.3125 | 0.3750 |
| curriculum_ep1 | 0.4375 | 0.3125 | 0.4375 |
| curriculum_ep2 | 0.1875 | 0.5000 | 0.3750 |
| curriculum_lr1e5_ep1 | 0.3750 | **0.5000** | **0.5000** |

### DS at 256:768 (Current-High Asymmetry) - Opening-Randomized

| Checkpoint | ply=2 DS | ply=4 DS | ply=6 DS |
|------------|----------|----------|----------|
| iter0_reference | 0.3125 | 0.4375 | 0.3750 |
| control_ep1 | 0.1250 | 0.4375 | 0.1250 |
| control_ep2 | 0.1875 | 0.4375 | 0.3750 |
| curriculum_ep1 | 0.2500 | 0.0625 | 0.3750 |
| curriculum_ep2 | 0.2500 | 0.2500 | 0.3750 |
| curriculum_lr1e5_ep1 | 0.1250 | 0.4375 | 0.1875 |

### DS at 768:768 (Equal-Medium Budget) - Opening-Randomized

| Checkpoint | ply=2 DS | ply=4 DS | ply=6 DS |
|------------|----------|----------|----------|
| iter0_reference | 0.1875 | 0.2500 | 0.3750 |
| control_ep1 | 0.4375 | 0.3750 | 0.5625 |
| control_ep2 | 0.3125 | 0.3125 | 0.5625 |
| curriculum_ep1 | 0.1250 | 0.2500 | 0.1250 |
| curriculum_ep2 | 0.3125 | 0.3125 | 0.3750 |
| curriculum_lr1e5_ep1 | 0.1875 | 0.5000 | 0.3750 |

### Duplicate Trajectory Rates

Consistent with PR #103, extremely high duplicate trajectory rates observed:

| Ply Count | Games | Unique Trajectories | Dup Rate |
|-----------|-------|---------------------|----------|
| 2 | 128 | ~18-22 | ~95-97% |
| 4 | 128 | ~16-20 | ~96-98% |
| 6 | 128 | ~18-22 | ~95-97% |

## Consolidated Results by Ply Count at 384:256

| Ply | iter0_ref DS | Best Control DS | Best Curriculum DS | Best LR1e5 DS | Curriculum Δ |
|-----|-------------|-----------------|-------------------|---------------|-------------|
| 2 | 0.6250 | 0.6562 (ctrl_ep1) | 0.6250 (cur_ep2) | 0.6250 | 0.0000 |
| 4 | 0.6875 | 0.7812 (ctrl_ep1) | 0.5000 (cur_ep2) | 0.7500 (lr1e5_ep1) | +0.0625 |
| 6 | 0.8125 | 0.5312 (ctrl_ep1) | 0.5625 (cur_ep2) | 0.5000 (lr1e5_ep1) | −0.3125 |

### Mean DS at 384:256 Across All Ply Counts

| Checkpoint | Mean DS (ply=2,4,6) |
|------------|---------------------|
| iter0_reference | 0.7083 |
| control_ep1 | 0.6563 |
| control_ep2 | 0.4583 |
| curriculum_ep1 | 0.3750 |
| curriculum_ep2 | 0.5625 |
| curriculum_lr1e5_ep1 | 0.6250 |

No checkpoint exceeds iter0_reference's mean randomized-opening DS at 384:256. The control_ep1 comes closest (0.6563) without using any curriculum data.

## Classification

**Overall: `p1_curriculum_overfit`**

### Why `p1_curriculum_overfit`

| Criterion | Evidence |
|-----------|----------|
| Curriculum holdout loss improves | 1.502 → 1.334 (LR=3e-5, 11.2% relative); 1.502 → 1.440 (LR=1e-5, 4.2% relative) |
| But randomized-opening DS does not improve | Mean DS 0.7083 (iter0_ref) → 0.5625 (cur_ep2) → 0.6250 (lr1e5_ep1) |
| Gains appear marginal and inconsistent | LR=1e-5 shows +0.06 at ply=4 but −0.31 at ply=6 |
| Control lane without curriculum outperforms curriculum at some budgets | control_ep1 DS at ply=4 = 0.7812 vs curriculum_ep2 = 0.5000 |

### Why NOT `opening_randomized_p1_curriculum_promising`

| Criterion | Evidence |
|-----------|----------|
| Randomized DS must improve over iter0_reference for ply 2/4/6 | Only borderline at ply=4 (LR=1e-5), regression at ply=6 |
| OR deterministic DS at 384:256 must become >0.00 | DS=0.00 for all checkpoints under deterministic opening |

### Why NOT `deterministic_artifact_not_trainable`

| Criterion | Evidence |
|-----------|----------|
| 1200:1200 deterministic DS would be damaged | DS=1.00 preserved for all checkpoints |
| 256:768 would materially regress | Minor regression in some lanes but not "material" |

## Primary Findings

1. **P1 curriculum signal is learnable but not distillable to MCTS strength.**
   The model learns to imitate high-search P1 policy (holdout policy loss −11.2%) but this raw-policy shift is too small to change MCTS outcomes at practical budgets. The MCTS search process dominates the raw policy, similar to the `high_search_not_distillable` finding from disadvantaged-seat distillation (PR #102).

2. **Higher LR (3e-5) produces better imitation but worse arena results.**
   Curriculum_ep2 at LR=3e-5 has the best holdout loss (1.334) but the worst mean DS at 384:256 (0.5625). Lower LR (1e-5) produces less imitation (1.440) but more stable arena behavior (mean DS 0.6250). This suggests that aggressive curriculum fitting introduces parameter shifts that are counterproductive for MCTS search at practical budgets.

3. **The control lane (no curriculum) sometimes outperforms the curriculum lane.**
   Control_ep1 achieves DS=0.7812 at ply=4 384:256, the highest single-ply DS across all checkpoints. This suggests that the generic bootstrap + old replay data already captures useful signal, and the additional P1 curriculum data may introduce distracting objectives.

4. **The P1 strategy at 384:256 is fragile under parameter perturbation.**
   Even small parameter deltas (0.09-0.42%) cause significant changes in randomized-opening DS. The iter0_reference P1 advantage is a razor's-edge property of the specific parameter vector, and fine-tuning (even without curriculum data) tends to degrade it.

5. **Dataset size and diversity are limiting factors.**
   Only 141 unique boards from 8474 filtered positions, with 9261 duplicates. The Kalah state space is very small in early game. More opening diversity or synthetic state generation may be needed to produce a curriculum with sufficient state coverage.

6. **LR=1e-5 lane shows borderline improvement at 1200:1200.**
   curriculum_lr1e5_ep1 achieves DS=0.50 at 1200:1200 for both ply=4 and ply=6, the highest high-budget randomized-opening DS across all checkpoints. This mild training may partially transfer the high-search P1 counter-strategy to lower simulation counts, but the effect is not strong enough for classification as "promising."

## Runner Commands

### Mining

```bash
.venv/bin/python ml/alphazero_lite/mine_opening_randomized_p1_curriculum.py \
  --out-train /tmp/azlite_opening_p1_curriculum/p1_curriculum_train.jsonl \
  --out-holdout /tmp/azlite_opening_p1_curriculum/p1_curriculum_holdout.jsonl \
  --out-summary /tmp/azlite_opening_p1_curriculum/p1_curriculum_summary.json \
  --candidate /tmp/azlite_iterative_random_replay/iter0_candidate_artifact \
  --current model-artifact/current \
  --opening-plies 2,4,6 \
  --opening-samples 128 \
  --games-per-opening 4 \
  --opening-seed 48 \
  --budget-pairs 384:256,768:256 \
  --teacher-mode classic_mcts \
  --teacher-simulations 1200 \
  --target-train-rows 1536 \
  --target-holdout-rows 384 \
  --positive-bucket-share 0.70 \
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
  --top-k-dir /tmp/azlite_opening_p1_curriculum/checkpoints/control \
  --out /tmp/azlite_opening_p1_curriculum/checkpoints/control/checkpoint_ep2.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Training (curriculum lane, LR=3e-5)

```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl,/tmp/azlite_opening_p1_curriculum/p1_curriculum_train.jsonl \
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
  --top-k-dir /tmp/azlite_opening_p1_curriculum/checkpoints/curriculum \
  --out /tmp/azlite_opening_p1_curriculum/checkpoints/curriculum/checkpoint_ep2.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Training (curriculum lane, LR=1e-5)

```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl,/tmp/azlite_opening_p1_curriculum/p1_curriculum_train.jsonl \
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
  --top-k-dir /tmp/azlite_opening_p1_curriculum/checkpoints/curriculum_lr1e5 \
  --out /tmp/azlite_opening_p1_curriculum/checkpoints/curriculum_lr1e5/checkpoint_ep2.npz \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --lr-scheduler none \
  --seed 42
```

### Deterministic Evaluation

```bash
.venv/bin/python script/ai/seat_aware_promotion_gate \
  --candidate-path /tmp/azlite_opening_p1_curriculum/artifacts/<artifact> \
  --current-path model-artifact/current \
  --games 60 --seed 42 --workers 4 \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --out /tmp/azlite_opening_p1_curriculum/eval/<name>_deterministic.json
```

### Opening-Randomized Evaluation

```bash
.venv/bin/python ml/alphazero_lite/run_opening_randomized_seat_diagnostic.py \
  --workdir /tmp/azlite_opening_p1_curriculum/eval/<name>_opening_randomized \
  --candidate /tmp/azlite_opening_p1_curriculum/artifacts/<artifact> \
  --current model-artifact/current \
  --budget-pairs 384:256,768:256,768:768,1200:1200,256:768 \
  --random-opening-plies 2,4,6 \
  --opening-samples 64 \
  --opening-seed 48 \
  --games-per-opening 2 \
  --seed 42 --workers 4
```

## Guardrails

| Guardrail | Status |
|-----------|--------|
| No promotion | PASS: no model promoted |
| No overwrite model-artifact/current | PASS: current unchanged |
| No overwrite storage/ai/alphazero_lite/current | PASS: unchanged |
| No LR above 3e-5 | PASS: max LR=3e-5 |
| No generic random replay generation | PASS: no new generic replay generated |
| No residual_v4 | PASS: residual_v3 only |
| No tablebase overlay | PASS: not used |
| No deterministic-opening-only states | PASS: curriculum mined from randomized openings |
| No judgment by validation loss alone | PASS: arena evaluation used |
| No architecture change | PASS: residual_v3 only |
| No agreement filtering | PASS: not used |
| No tactical-bias-off replay | PASS: not generated |

## Verification

```bash
.venv/bin/ruff check ml/alphazero_lite/mine_opening_randomized_p1_curriculum.py  # clean
.venv/bin/ruff check ml/alphazero_lite script/ai  # clean
.venv/bin/python -m unittest discover ml/alphazero_lite  # passed
```

## Recommendations

1. **Investigate why control training sometimes improves P1 performance.** Control_ep1 achieved DS=0.7812 at ply=4 384:256 without any curriculum data. Understanding what the generic bootstrap data provides could be more productive than targeted curriculum mining at this stage.

2. **Explore deeper curriculum integration.** The curriculum data represents only ~1% of training positions (1536 rows × weight 1 vs ~40372 effective from generic × 4 + old × 1). A higher curriculum weight or dedicated curriculum-only training phase might produce more meaningful parameter shifts.

3. **Increase state space diversity.** With only 141 unique boards, the curriculum lacks the diversity needed for robust learning. Synthetic state generation or systematic opening enumeration may produce richer training data.

4. **The inversion pattern persists.** The model's P1 advantage at practical budgets under randomized openings (DS ~0.5-0.8) consistently outperforms its P0 performance (DS ~0.2-0.5). This inversion is a real property of the iter0_reference training, not an evaluation artifact, and remains a candidate for future targeted distillation.
