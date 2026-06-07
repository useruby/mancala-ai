# AlphaZero-Lite Iterative Random Replay Results

**Date:** 2026-06-07

## Classification

**CANDIDATE DISTRIBUTION NOT HELPFUL** — Changing the source-state distribution by generating replay data from a parity candidate (instead of the current production model) produces a meaningfully different dataset (0.8% state overlap with old data) but zero arena-strength improvement. Both iter1 lanes (continue-on-old-data and candidate-mined-replay) score exactly 0.50 against current at 120 and 240 games, identical to the iter0 reference. The 0.50 ceiling is not a state-distribution problem accessible via single-iteration candidate-game mining.

## Primary Hypothesis

> A random-replay candidate that reaches parity against current visits a different state distribution. Mining and relabeling random positions from that candidate's games will produce useful second-iteration replay data that can exceed the 0.50 ceiling.

**Verdict: REJECTED.** The candidate visits a genuinely different state distribution (0.8% overlap, 99.2% new states), but training on that data produces zero arena-strength gain over the iter0 baseline.

## Experiment Design

| Lane | Init | Data | Weights | Purpose |
|------|------|------|---------|---------|
| iter0_reference | current weights.json → PR #93 teacher_1200 | generic + old random | 4,1 | Baseline parity candidate |
| iter1_continue_no_new_data | iter0_candidate | generic + old random | 4,1 | Control: extra training on same data |
| iter1_candidate_random_replay | iter0_candidate | generic + old random + new candidate-mined random | 4,1,1 | Test: candidate-mined distribution |

All lanes use identical hyperparameters:

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

## Artifact Lineage

| Artifact | SHA256 |
|----------|--------|
| current production (model-artifact/current) | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| current init checkpoint (from current weights.json) | `5adbbb6bebf625708c8edd159a58cb4434ad3b7bcd9ed8a3aed321498a21dd70` |
| iter0 candidate artifact weights.json | `0bbeaa9c6954ea147b084cdb3e68355887df1b0ab34c1fbc952b7988fdddc3bd` |
| iter0 candidate checkpoint/model.npz | `c2430b56e7a68386343579aeac3610f4c88e1c35904e39179902e437fef36c18` |
| iter1_continue top1 checkpoint | `d84391672ef5d7866da110224201ecf4926c573434986c5bfab1f315df0593b9` |
| iter1_continue artifact weights.json | `a4a86dceb064a2be63d186785e9e3e30e5e5bed52bd4be2fb878d42edc45a5a9` |
| iter1_candidate top1 checkpoint | `e8f06e984216683e002e704d2ad5cf82544b8cfeaa2e73f12c86553c8d8fd06a` |
| iter1_candidate artifact weights.json | `777b25d4c5a601ff4b3cdc3a750550f2fee2756da296ad15c300b0d749b61b44` |

**iter0 status:** Reused from PR #93 teacher_1200 lane. Not regenerated.

## Dataset

### Data Files

| Dataset | Path | Rows | SHA256 |
|---------|------|------|--------|
| Generic bootstrap | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | 9,589 | `5d01a60e9dfc8756ffa47f2c8222e496a8ce8bd661457658de4884198344e147` |
| Old current-mined random replay | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | 2,016 | `7ca93389d1be93bd1cf09d23ddfb9f040bb402a718cd991ac49e082bd7e2f69a` |
| New candidate-mined random replay | `/tmp/azlite_iterative_random_replay/iter1_candidate_random_train.jsonl` | 2,840 | `0c07ac756fa4941af75b558095e6714fb0dffb3f246b2d63c79a100774fab231` |

### Source State Comparison

| Metric | Old (current-mined) | New (candidate-mined) |
|--------|---------------------|----------------------|
| Path | `/tmp/azlite_random_teacher_quality/random_source_states.jsonl` | `/tmp/azlite_iterative_random_replay/iter1_candidate_source_states.jsonl` |
| SHA256 | `1dadcf9c6a090bc3b69b2252e6655cb3dd45f400216229c0e99547231f9c25de` | `6a3be99694938939117b008576cfac4989c0b934525ba675cbf7288294156cbb` |
| Source rows | 2,520 | 3,550 |
| Unique state fingerprints | 1,552 | 2,101 |
| State overlap | — | 16 (0.8%) |
| Player model | current production | iter0_candidate |
| Seed | 43 | 44 |
| Games contributed | 292 | 427 |
| Positions visited | 28,460 | 26,679 |
| Duplicate state count | 3,716 | 4,551 |
| Disagreement rate | 0.5627 | 0.5231 |
| Agreement rate | 0.4373 | 0.4769 |
| Teacher simulations | 1,200 | 1,200 |
| Top-1 visit share mean | 0.5529 | 0.5817 |
| Current prob on teacher top mean | 0.4064 | 0.4372 |

### Source State Overlap

```
Old source states:     1,552 unique fingerprints
New source states:     2,101 unique fingerprints
Overlap:                  16 (0.8%)
New unique states:     2,085 (99.2%)
```

The candidate visits a fundamentally different state distribution. Only 0.8% of positions overlap with the old dataset.

### Phase Distribution

| Phase | Old (current-mined) | New (candidate-mined) |
|-------|---------------------|----------------------|
| Early | 319 (12.7%) | 275 (7.7%) |
| Mid | 1,109 (44.0%) | 1,713 (48.3%) |
| Late | 1,092 (43.3%) | 1,562 (44.0%) |

The candidate-mined distribution has fewer early-phase positions and shifts slightly toward mid-game. The candidate model may play different openings, reducing early-game coverage.

### Effective Sample Shares

| Lane | Generic | Old Mined | New Mined |
|------|---------|-----------|-----------|
| iter0_reference (4:1) | 95.0% | 5.0% | — |
| iter1_continue (4:1) | 95.0% | 5.0% | — |
| iter1_candidate (4:1:1) | 88.8% | 4.7% | 6.6% |

Total mined share in the candidate lane: 11.2%. This is above the PR #92 weight sweep ceiling (2.4-9.1% all at 0.50), but the PR #92 sweep tested higher absolute mined share on the same data, not additional data from a different distribution.

## Training

| Metric | iter0_reference (PR #93) | iter1_continue_no_new_data | iter1_candidate_random_replay |
|--------|--------------------------|---------------------------|------------------------------|
| Init checkpoint | current weights.json | iter0_candidate | iter0_candidate |
| policy_loss | 1.002022 | 0.856604 | 0.884349 |
| value_loss | 0.245698 | 0.233250 | 0.246227 |
| best_val_loss | 1.182161 | 1.168476 | 1.167614 |

Both iter1 lanes achieve lower policy loss and better validation loss than the iter0 baseline. This is expected since both start from a better initialization (the iter0 candidate, already at 0.50 strength, vs the base current model at 0.25).

The candidate-mined lane has slightly worse policy loss (0.884 vs 0.857) but nearly identical best_val_loss (1.168 vs 1.167). Adding more mined data of a different distribution degrades the training signal slightly but does not improve validation loss — consistent with the PR #92 observation that higher mined share increases loss without arena benefit.

### Top-k Checkpoints

**iter1_continue_no_new_data:**
| Rank | val_loss | SHA256 |
|------|----------|--------|
| top1 | 1.168476 | `d84391672ef5d7866da110224201ecf4926c573434986c5bfab1f315df0593b9` |
| top2 | 1.172281 | `66d1be6e3dcc382b968ab692015799c318babe8c31fb2d923122338f622d5a89` |
| top3 | 1.174029 | `9c4e4df67e39d42c1cd9ddec8872505fd1f002eeda5786ff544842428e6e92da` |

**iter1_candidate_random_replay:**
| Rank | val_loss | SHA256 |
|------|----------|--------|
| top1 | 1.167614 | `e8f06e984216683e002e704d2ad5cf82544b8cfeaa2e73f12c86553c8d8fd06a` |
| top2 | 1.172850 | `c22d7b5b47d89cda03c221fbe26a37df737227a6053c9e9e91196e7a7e81ee06` |
| top3 | 1.176751 | `cc8d826023dbf4ba74f8fa73b413320a843221c0cf54f7ab39157a4c0be3d45c` |

## Strength: Arena vs Current

### Standard Arena (120 games)

| Metric | iter0_reference | iter1_continue_no_new_data | iter1_candidate_random_replay |
|--------|----------------|---------------------------|------------------------------|
| Arena Score | **0.50** | **0.50** | **0.50** |
| Wins / Losses / Draws | 60 / 60 / 0 | 60 / 60 / 0 | 60 / 60 / 0 |
| CI95 (Wilson) | [0.412, 0.588] | [0.412, 0.588] | [0.412, 0.588] |
| Move time mean ms | 49.59 | 32.27 | 38.86 |
| Move time p95 ms | 114.94 | 73.89 | 103.15 |

### Extended Arena (240 games)

| Metric | iter1_continue_no_new_data | iter1_candidate_random_replay |
|--------|---------------------------|------------------------------|
| Arena Score | **0.50** | **0.50** |
| Wins / Losses / Draws | 120 / 120 / 0 | 120 / 120 / 0 |
| CI95 (Wilson) | [0.437, 0.563] | [0.437, 0.563] |

All three lanes produce identically flat results. No differentiation even at 240 games. The upper bound of the CI95 (0.563) is above the 0.55 threshold, but the lower bound (0.437) falls well below it. There is no statistical evidence that any lane exceeds 0.50.

### Latency

No latency regression. All move times are within normal residual_v3 (96,3) bounds.

## Acceptance Criteria Evaluation

### Classification: CANDIDATE DISTRIBUTION NOT HELPFUL

| Criterion | Result | Status |
|-----------|--------|--------|
| iter1_candidate beats iter0_reference | Both at 0.50 | FAIL |
| iter1_candidate beats iter1_continue | Both at 0.50 | FAIL |
| Either lane reaches 0.55 threshold | Both at 0.50 | FAIL |
| Either lane shows better 240-game estimate | Both at 0.50 (240 games) | FAIL |
| State distribution genuinely changed | 0.8% overlap | CONFIRMED |
| No latency regression | All normal | CONFIRMED |
| No collapse | Both maintain 0.50 | CONFIRMED |

### Classification: extra_training_not_data

| Criterion | Result | Status |
|-----------|--------|--------|
| iter1_continue improves as much as candidate-mined | Identical (0.50 = 0.50) | — |
| Extra training alone matches or exceeds candidate benefit | No benefit observed | — |

`extra_training_not_data` is not supported: continuing training doesn't change anything, and adding new data doesn't change anything. Both are flat.

### Rejection criteria

| Criterion | Triggered? |
|-----------|-----------|
| iter1_candidate falls below 0.50 | No — at 0.50 |
| Candidate-mined data improves validation loss but not arena | Slightly worse val_loss (1.168 vs 1.167), no arena gain |
| Generated states heavily overlap old states | No — 0.8% overlap |
| local_promotion_gate fails badly | Gate not run (< 0.55 arena) |

## Analysis

### The state distribution is different but doesn't help

The candidate-mined dataset is genuinely different from the old current-mined dataset:
- 0.8% state overlap (essentially zero)
- Different phase distribution (fewer early, more mid)
- Slightly higher teacher confidence (top-1 share 0.582 vs 0.553)
- Slightly higher agreement rate (0.477 vs 0.437)

Yet training on this data produces zero change in arena strength. This means the 0.50 ceiling is not a property of the specific state distribution — changing it doesn't move the needle.

### The 0.50 ceiling is deeper than state distribution

The 0.50 arena ceiling persists across:
- **Sampling strategy:** disagreement-filtered (PR #90, collapses), random (PR #90/#91/#92/#93, 0.50), agreement-filtered (PR #91, 0.25)
- **Replay weight:** 1.2% to 11.2% effective mined share (PR #92, this experiment)
- **Teacher simulation budget:** 1200 to 2400 (PR #93)
- **Game count:** 400 to 800 (PR #90/#91)
- **Seed:** 42, 43, 44 (multiple runs)
- **State distribution:** current-mined vs candidate-mined (this experiment)
- **Training epochs:** single round vs continued training (this experiment)
- **Init model:** current weights vs iter0 candidate (this experiment)

Every variation converges to exactly 0.50.

### Possible explanations

1. **Architectural ceiling.** residual_v3 (96,3) may not have enough capacity to internalize more than +0.25 over baseline from classic-MCTS-labeled data, regardless of the data's origin.

2. **Search-budget ceiling.** The arena uses 384 challenger simulations vs 256 current simulations. If the learned policy improvement saturates at a level the search budget cannot amplify, the arena score caps regardless.

3. **Teacher ceiling.** Classic MCTS at 1200 simulations may be fundamentally insufficient as a teacher for producing models stronger than 0.50. The teacher itself is matched or beaten by 0.50 models.

4. **Zero-sum game dynamics.** The zero-draw result (60W/60L/0D at 120 games) suggests the model changes playing style (from defensive/draw-prone to aggressive) without improving net strength. The learned transfer is "style transfer" not "strength transfer."

5. **Single-iteration limit.** One round of training on candidate-mined data may not be enough to create a self-reinforcing improvement loop. The candidate was already trained to mimic classic-MCTS labels — mining from it and training again may just reinforce the same MCTS pattern with no new information.

### What's been ruled out (cumulative)

| Variable | Tested | Status |
|----------|--------|--------|
| Mined effective sample share | 1.2% to 11.2% | Not limiting |
| Teacher simulation quality | 1200 vs 2400 | Not limiting |
| Game count | 400 vs 800 | Not limiting |
| Sampling seed | 42, 43, 44 | Not limiting |
| State distribution | current-mined vs candidate-mined | Not limiting |
| Training duration | single vs continued | Not limiting |
| Sampling filter | failure, random, agreement | random is best at 0.50 |

## Key Takeaways

1. **Changing the state distribution by mining from a parity candidate does not exceed 0.50.** Despite 99.2% novel states and different phase profiles, the benefit is zero.

2. **The 0.50 ceiling is extraordinarily robust.** It survives every variable tested across 6 PRs: replay weight, label quality, state distribution, training duration, initialization, seed.

3. **Neither more training nor more data helps.** Both iter1 lanes (continue on old data, add new data) produce exactly 0.50, identical to the single-round baseline.

4. **The limiting factor may be architecture or teacher quality.** The 0.50 ceiling now appears to be a fundamental limit of what residual_v3 (96,3) models can extract from classic-MCTS-labeled data at these simulation budgets. Future directions should explore:
   - Architecture changes (residual_v4, move-factorized heads, wider models)
   - Different teacher paradigms (deeper search, tablebase guidance, self-play iterations)
   - Search improvements that better exploit the learned policy

5. **Validation loss remains uncorrelated with arena strength for mined replay.** Both iter1 lanes have better val_loss (1.168) than iter0 (1.182) but identical arena scores. This is the fourth independent confirmation of the val_loss-arena disconnect (PR #90 failure_mined, PR #91 agreement_mined, PR #92 weight sweep, this experiment).

## Commands Used

```bash
# Set up experiment directory
mkdir -p /tmp/azlite_iterative_random_replay
cp -r /tmp/azlite_random_teacher_quality/random_teacher_1200/artifact \
  /tmp/azlite_iterative_random_replay/iter0_candidate_artifact
cp /tmp/azlite_random_teacher_quality/random_teacher_1200/checkpoint.top1.npz \
  /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz

# Generate candidate-mined source states and replay data
.venv/bin/python ml/alphazero_lite/mine_failure_replay_dataset.py \
  --out-source-states /tmp/azlite_iterative_random_replay/iter1_candidate_source_states.jsonl \
  --out-train /tmp/azlite_iterative_random_replay/iter1_candidate_random_train.jsonl \
  --out-holdout /tmp/azlite_iterative_random_replay/iter1_candidate_random_holdout.jsonl \
  --out-summary /tmp/azlite_iterative_random_replay/iter1_candidate_random_summary.json \
  --current /tmp/azlite_iterative_random_replay/iter0_candidate_artifact \
  --games 800 --seed 44 --max-positions-per-game 12 \
  --input-encoding kalah_v3 --sampling-mode random \
  --teacher-mode classic_mcts --teacher-simulations 1200 \
  --policy-target-mode sharpened --value-target-mode sharpened

# Train iter1_continue_no_new_data (control)
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --replay-weights 4,1 \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --model-type residual_v3 --input-encoding kalah_v3 --hidden-sizes 96,3 \
  --epochs 10 --batch-size 512 --lr 1e-3 --value-loss huber \
  --value-loss-weight 0.3 --grad-clip 1.0 --save-top-k 3 \
  --top-k-dir /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data \
  --out /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data/checkpoint.npz \
  --policy-target-mode sharpened --value-target-mode sharpened --seed 42

# Train iter1_candidate_random_replay (experiment)
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl,/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl,/tmp/azlite_iterative_random_replay/iter1_candidate_random_train.jsonl \
  --replay-weights 4,1,1 \
  --init-checkpoint /tmp/azlite_iterative_random_replay/iter0_candidate_checkpoint.npz \
  --model-type residual_v3 --input-encoding kalah_v3 --hidden-sizes 96,3 \
  --epochs 10 --batch-size 512 --lr 1e-3 --value-loss huber \
  --value-loss-weight 0.3 --grad-clip 1.0 --save-top-k 3 \
  --top-k-dir /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay \
  --out /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay/checkpoint.npz \
  --policy-target-mode sharpened --value-target-mode sharpened --seed 42

# Export artifacts
.venv/bin/python ml/alphazero_lite/export_artifact.py \
  --checkpoint /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data/checkpoint.top1.npz \
  --out-dir /tmp/azlite_iterative_random_replay/iter1_continue_no_new_data/artifact \
  --version iter1-continue-no-new-data --model-type residual_v3 --input-encoding kalah_v3

.venv/bin/python ml/alphazero_lite/export_artifact.py \
  --checkpoint /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay/checkpoint.top1.npz \
  --out-dir /tmp/azlite_iterative_random_replay/iter1_candidate_random_replay/artifact \
  --version iter1-candidate-random-replay --model-type residual_v3 --input-encoding kalah_v3

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
