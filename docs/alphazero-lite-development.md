# AlphaZero-Lite Development Reference

This document consolidates the knowledge accumulated during the AlphaZero-lite
development effort (Feb 28 -- Mar 14, 2026). It replaces the ~80 individual
plan, design, and implementation documents that lived in `docs/plans/` and
`docs/superpowers/`.

---

## 1. Project Overview

**Goal:** Replace the remote LLM-based bot (Scaleway) with a local
AlphaZero-lite neural network for the `hard` difficulty level,
while keeping `easy`/`medium` on pure MCTS.

**Outcome:** After ~20 experiment iterations exploring model architectures,
training targets, teacher data strategies, and search improvements, the root
cause of persistent failure against MCTS1200 was identified as a
teacher-student mismatch. The solution was to use classic MCTS (the same engine
the model must beat) as both bootstrap teacher and self-play player. A
multi-iteration hybrid pipeline then produced models that play well.

---

## 2. Architecture Evolution

### 2.1 State Encoding

| Version     | Features | Description |
|-------------|----------|-------------|
| `kalah_v1`  | 15       | Baseline: 6 own pits + 6 opponent pits + own store + opponent store + current player flag |
| `kalah_v3`  | 27       | Extends `kalah_v1` with tactical signals: extra-turn/capture outcomes, empty-pit structure, endgame/starvation pressure, score-pressure context, mobility, vulnerable opposite pits |

> `kalah_v2` was an internal intermediate encoding that was superseded by
> `kalah_v3` before any production use.

`kalah_v3` groups features into four categories:
1. **Immediate tactical outcomes** -- extra turns, captures, empty-landings, reply threats
2. **Pit structure signals** -- empty own pits, high-seed pits, vulnerable opposite pits, one-sow-away tactical landings
3. **Endgame pressure** -- stones remaining per side, mobility, starvation proximity
4. **Score-pressure context** -- normalized score delta, ahead/behind flag, tactical urgency

### 2.2 Model Families

| Family         | Description | Key Properties |
|----------------|-------------|----------------|
| Basic MLP      | Two-layer MLP with shared trunk, separate policy + value output layers | Original architecture; canonical weight keys: `w1`, `b1`, `w2`, `b2`, `w_policy`, `b_policy`, `w_value`, `b_value` |
| `residual_v2`  | Residual MLP with configurable block count | Added residual skip connections for deeper training |
| `residual_v3`  | Specialized policy/value heads split from compact shared trunk | Policy head gets an extra hidden transformation (`w_policy_hidden`, `b_policy_hidden`) for sharper move discrimination; value head gets its own (`w_value_hidden`, `b_value_hidden`) |

Capacity experiments tested hidden sizes: `64,2` -> `96,3` -> `128,3`. Wider
models improved arena (self-play) scores but did not move MCTS1200 win rate on
their own.

### 2.3 Training Stack

- **Initial:** Lightweight NumPy weight updates
- **Final:** Full-backprop PyTorch training with Adam optimizer, CUDA
  auto-selection, mini-batch training, cosine LR scheduling, gradient clipping,
  validation split, and best-validation checkpoint tracking
- **Constraint:** Exported checkpoint key names must be preserved so that
  `export_artifact.py` and the Ruby inference loader continue to work unchanged

---

## 3. Contracts

### 3.1 Kalah Rules Contract (`kalah_v1`)

**Canonical state schema:**
```
{
  "player_pits": [int, int, int, int, int, int],
  "opponent_pits": [int, int, int, int, int, int],
  "player_store": int,
  "opponent_store": int,
  "current_player": 0 or 1
}
```

**Move semantics:**
- `relative_move` in `[0..5]` from the current player's perspective
- `absolute_move = relative_move + current_player * 6`
- Legal only when the selected pit belongs to `current_player` and has seeds

**Rule semantics:**
- Sowing: clockwise through all 12 pits
- Store crossing: deposits one seed in mover's store (not opponent pit); may
  grant extra-turn only when that is the last seed
- Capture: final seed lands in empty own pit AND opposite pit has seeds
- Game over: current player has no playable seeds after move resolution
- On game over: remaining seeds swept to opposite player's store
- Winner: larger store; equal stores = draw (`winner = null`)

**Parity requirement:** Ruby and Python must match on per-ply state
transitions, legal move lists, `over` flag, and `winner` value.

**Test sources:**
- Golden vectors: `test/fixtures/ai/kalah_rule_vectors.json`
- Ruby parity: `test/models/games/kalah_rules_parity_test.rb`
- Python parity: `ml/alphazero_lite/tests/test_kalah_rules_parity.py`
- Differential fuzz: `ml/alphazero_lite/parity_fuzz.py`

### 3.2 Benchmark Contract

**Protocol:**
- 60-game primary sample, 120-game confirmation
- Strict seat alternation: `game_index.even?` starts as player 0
- Fixed random seed per run, recorded in report

**Score formula:**
```
score = (wins + 0.5 * draws) / games_played
```

**Mandatory gates:**

| Gate | Threshold | Description |
|------|-----------|-------------|
| Identity | score in `[0.40, 0.60]` | AZ vs same AZ artifact |
| Monotonic | score `>= 0.50` | MCTS1800 vs MCTS1200 |
| Runtime parity | delta `<= 0.10` | Chooser vs preloaded-evaluator |
| Promotion | score `>= 0.55` | Challenger vs current |

**Report schema (all required):** `schema`, `seed`, `games_played`,
`challenger_path`, `current_path`, `challenger_simulations`,
`current_simulations`, `wins`, `losses`, `draws`, `score`,
`move_time_mean_ms`, `move_time_p95_ms`

**Baseline-relative MCTS gate:** After discovering that the fixed 0.45
threshold was meaningless (current model scored 0.0 vs MCTS1200), the gate was
changed to require challengers to score at least as well as the currently
promoted model against MCTS1200. This turns it into a true non-regression
check.

### 3.3 Artifact Format

- Ruby runtime expects `weights.json` (not `.npz`)
- Canonical weight keys for basic MLP: `w1`, `b1`, `w2`, `b2`, `w_policy`,
  `b_policy`, `w_value`, `b_value`
- Additional keys for `residual_v3`: `w_policy_hidden`, `b_policy_hidden`,
  `w_value_hidden`, `b_value_hidden`, plus residual block weights

### 3.4 Local Promotion Gate

**Script:** `script/ai/local_promotion_gate`

**Three evaluations:** candidate vs current arena, candidate vs MCTS1200,
current vs MCTS1200

**Gate criteria (all must pass):**
1. Arena report uses >= configured game count (default: 30)
2. MCTS1200 reports use >= configured game count (default: 30)
3. Arena score >= minimum (default: 0.55)
4. Candidate MCTS1200 score >= current MCTS1200 score

**Failure codes:** `arena_below_min`, `arena_games_below_minimum`,
`candidate_mcts_games_below_minimum`, `current_mcts_games_below_minimum`,
`candidate_not_above_current_mcts`, `missing_report`

---

## 4. Training Pipeline

### 4.1 Pipeline Orchestration

- **Orchestrator:** `pipeline.py` with YAML config files (e.g.,
  `aggressive_v1.yaml`)
- **Stages:** bootstrap dataset generation -> self-play -> training ->
  evaluation -> promotion gates
- **Multi-iteration:** `start_iteration` param allows numbering to begin at
  2+; `skip_before_final_iteration` step flag controls which stages run in
  intermediate iterations
- **Phase chaining:** `--phase2-config` flag on `runpod_training_experiment`
  runs two pipeline phases on a single pod

### 4.2 Data Generation

**MCTS bootstrap dataset:**
- Ruby MCTS engine generates training data via `ai:generate_mcts_dataset` rake
  task
- Outputs JSONL rows (final schema): `state`, `policy`, `player`,
  `move_index`, `value`
- Configurable: game count, simulations, seed, curriculum sampling
  (early/mid/late plies), max positions per game
- Curriculum-aware preselection: determine which positions survive filtering
  before running expensive MCTS search

**Self-play:**
- Python-side PUCT self-play or classic MCTS self-play (`--player-mode
  classic_mcts`)
- Parallel via `--workers` + `concurrent.futures`
- Deterministic game partitioning by worker ID with seed offsets
- Shard JSONL output, parent-level merge in worker-ID order

**Replay window:**
- `--data-files file1,file2 --replay-weights 1,2` for multi-file weighted
  training across iterations
- Backward compatible: single `--data` runs unchanged

### 4.3 Teacher Modes

| Mode | Engine | Policy Source | Value Source |
|------|--------|---------------|-------------|
| `puct` (default) | PUCT + HeuristicEvaluator | Visit counts | Search Q-value |
| `classic_mcts` | Classic MCTS with heuristic playouts | Visit counts (`child.visits / total_visits`) | Win rate (`2*(root.wins/root.visits) - 1`, from current player's perspective) |

The `classic_mcts` teacher was the breakthrough -- it generates data from the
exact engine the model must beat.

### 4.4 Training Hyperparameters Explored

- Policy-target shaping: temperature `tau`, top-k truncation, Dirichlet noise,
  sharpened targets
- Value-target modes: terminal outcome, search Q-value, hybrid
  (outcome+search), phase-aware blending
- Optimizer: Adam with cosine LR schedule, gradient clipping
- Loss: configurable `--value-loss-weight`, weighted loss
- Multi-seed candidate selection: 3-5 seeds with arena gating before promotion
- Staged training: bootstrap-heavy stage 1 -> self-play fine-tune stage 2
  (`--init-checkpoint`)

### 4.5 RunPod Automation

- Pod profiles: `cpu3c` (8 vCPU, default), `cpu3c-16-32` (16 vCPU, large)
- Named `--pod-profile` CLI option
- Experiment runner handles pod lifecycle, config upload, execution, artifact
  retrieval

---

## 5. Experiment Timeline & Key Findings

### Phase 1: Foundation (Feb 28 -- Mar 2)

- Built Ruby-side PUCT search, state encoder, model loader, chooser with MCTS
  fallback
- Replaced Scaleway runtime with local inference path
- Created MCTS-supervised dataset generator
- Upgraded to PyTorch trainer
- Added training tuning knobs (tau, top-k, Dirichlet, LR schedule, gradient
  clipping, multi-seed selection)
- Added value curriculum (terminal outcome + normalized score-delta blend)
- Built parallel self-play and arena evaluation
- Froze Kalah rules contract and benchmark protocol
- Built inference budget tuning with phase-aware scaling

### Phase 2: Critical Discovery (Mar 3)

**Arena parity mismatch report:** Promotion arena measured AZ-vs-AZ only
(120W/0L/0D), but post-promotion testing against MCTS1200 showed 0W/30L/0D.
Passing promotion only meant "better than current AZ artifact," not
"competitive against MCTS1200."

**Resolution:** Added required post-promotion MCTS1200 benchmark gate,
implemented baseline-relative comparison.

### Phase 3: Throughput Optimization (Mar 7)

- MCTS bootstrap dataset: curriculum-aware preselection reduced wasted search
  (3019s -> 2077s baseline)
- MCTS engine hot-path: reduced object churn, cached repeated values, avoided
  temp arrays in tight loops
- MCTS1200 baseline: moved from serial Ruby runner to parallel Python benchmark
  driver (2844s -> parallelized)

### Phase 4: Training Ablations (Mar 8 -- 9)

All hit a ceiling -- improved arena scores but could not move MCTS1200 win
rate above 0.0:

| Ablation | Result |
|----------|--------|
| Data mix (light/medium/heavy bootstrap share) | Ceiling |
| Staged training (bootstrap -> self-play fine-tune) | MCTS1200 reached 0.1667 but did not survive strict validation |
| Stage balance (blend_light/medium/selfplay_bias) | Arena 0.25, could not improve both metrics |
| Wider student (larger hidden sizes in staged path) | No improvement |
| Duration/LR tuning (short_ft, low_lr presets) | No improvement |

### Phase 5: Architecture & Teacher Experiments (Mar 9)

V2 architecture with richer encoding and residual MLP:

| Experiment | Arena | MCTS1200 |
|------------|-------|----------|
| V2 local tuning (budget_up, search_up, small_widen) | Varied | 0.0 |
| Tactical teacher (MCTS on extra-turn/capture positions) | ~0.5 | 0.0 |
| Hybrid teacher (tactical + disagreement positions) | ~0.5 | 0.0 |

### Phase 6: Encoding & Model Refinement (Mar 10 -- 11)

| Experiment | Prefilter | MCTS1200 |
|------------|-----------|----------|
| `kalah_v3` tactical encoding | 0W 15L 15D (improvement) | 0.0 |
| `residual_v3` specialized heads | 15W 15L 0D (best yet) | 0.0 |
| Wider `residual_v3` | 0W 15L 15D (regressed) | 0.0 |
| Sharpened policy targets | Improved direct play | 0.0 |
| Sharpened value targets | No improvement | 0.0 |
| Phase-aware value targets | Regressed, failed prefilter | 0.0 |
| Hybrid value targets (outcome+search blend) | No improvement | 0.0 |
| Value-target semantic alignment (fix bootstrap/self-play mismatch) | Baseline | 0.0 |
| Search quality improvements (FPU, subtree reuse, value normalization, root policy, tactical bias) | No improvement | 0.0 |

### Phase 7: Capacity & Teacher Strength (Mar 12)

| Experiment | Arena | MCTS1200 |
|------------|-------|----------|
| Capacity 96,3 | Improved | 0.0 |
| Capacity 128,3 | Improved | 0.0 |
| Stronger bootstrap (2400 sims) | No change | 0.0 |

### Phase 8: Root Cause & Breakthrough (Mar 13)

**Root cause identified: teacher-student mismatch.**

The bootstrap teacher used `PUCT + HeuristicEvaluator`:
- Policy priors: proportional to `pit_seeds + 1` (crude heuristic)
- Value: `tanh(store_delta / 12.0)` (simple store-difference)

The evaluation opponent (`classic_mcts.MCTS`) uses:
- Heuristic playouts with analytical capture-gain: `capture_gain * 10 +
  extra_turn * 3`
- Phase-adaptive depth: 6/10/14 plies based on seeds remaining
- Game-ending sweep awareness

The student was never trained on data from the kind of player it needed to
beat.

**Solution:** `--teacher-mode classic_mcts` and `--player-mode classic_mcts` --
generate training data from classic MCTS, extracting policy from visit counts
and value from win rate.

### Phase 9: Multi-Iteration Pipelines (Mar 14)

With classic MCTS teacher working, multi-iteration hybrid pipelines were built:

1. **MCTS-seeded hybrid:** Phase 1 generates classic MCTS self-play data,
   Phase 2 runs neural self-play iterations 2-3 seeded from Phase 1.
   Achieved ~97.5% vs MCTS1200 but only ~50% vs the incumbent model.
2. **Bootstrap-anchored hybrid:** Adds heuristic PUCT bootstrap anchor
   (600 games, 2400 sims, weighted 4x) to the multi-iteration pipeline.
   Still ~50% vs incumbent -- the bootstrap anchor alone was not enough.
3. **Clone-extend:** Exactly replicates the incumbent model's recipe
   (1600 games, seed 42, dirichlet-epsilon 0.3), then adds one refinement
   iteration. Revealed that **data volume** (1600 vs 600 games) was a
   significant factor in the incumbent's strength.

Key infrastructure additions:
- `start_iteration` param for pipeline
- `skip_before_final_iteration` step flag
- `--phase2-config` flag for chaining pipeline phases on a single pod

### Ongoing: Inference & Search Micro-Optimizations

Applied throughout the development cycle to reduce per-move latency:

- Affine weight pre-transposition at load time (~1.8x inference speedup)
- `stones_in_pits` caching on MCTS Node to avoid repeated `game.to_state`
- Eliminated `children.to_a` allocation in `select_child` hot path

---

## 6. Bug Fixes

### PUCT Backpropagation Sign Error
`backpropagate` unconditionally negated the value at every step. On extra-turn
edges (parent and child have the same `to_play`), a positive leaf value should
be preserved, not flipped. Fix: only negate when the next node has a different
`to_play`.

### `residual_v3` Checkpoint Loading
`CheckpointEvaluator` in `self_play.py` could not load `residual_v3`
checkpoints due to missing specialized-head weight handling
(`w_policy_hidden`, `b_policy_hidden`, `w_value_hidden`, `b_value_hidden`).
Fix: detect and load the four hidden-layer weights when present.

### Ruby Model Loader `.npz` Acceptance
`ModelLoader.load!` accepted `model.npz` but the Ruby runtime only supports
`weights.json`. Fix: `resolve_weights_path!` only returns `weights.json`;
raises `ModelLoader::Error` if only `.npz` exists.

### Bootstrap Playout Regression
MCTS engine optimizations broke bootstrap data quality (arena prefilter dropped
to 0.25). Fix: added a `legacy_playout: true` option to `AI::MCTS` that
restores the original random depth-5 playout behavior. Only the bootstrap
dataset generator opts into this mode; all other callers use the optimized
path.

---

## 7. Key Lessons Learned

1. **Self-play promotion gates are not sufficient.** AZ-vs-AZ promotion only
   measures relative improvement within the model family. An absolute benchmark
   against the target opponent (MCTS1200) is essential.

2. **Teacher-student alignment matters more than model capacity.** ~20
   experiments adjusting architecture, encoding, targets, search, and capacity
   all failed because the training data came from a fundamentally different
   engine than the evaluation opponent.

3. **Train from the engine you must beat.** Using classic MCTS as both
   bootstrap teacher and self-play player (matching the evaluation opponent's
   search strategy) was the breakthrough that unlocked progress.

4. **Ablations must be strict.** Small local signals (e.g., 0.1667 MCTS1200
   score from 30 games) frequently did not survive strict 120-game validation
   or RunPod reproduction. Do not promote based on noisy small-sample results.

5. **Optimize the right bottleneck.** Throughput optimizations were necessary
   for iteration speed, but quality improvements required understanding what
   data the model needed, not just generating more of the same data faster.

6. **Multi-iteration is required for self-improvement.** Single-shot supervised
   learning from fixed data cannot discover its own weaknesses; the model must
   iterate with self-play to improve beyond its teacher.

7. **Semantic consistency across data producers is critical.** When bootstrap
   (Ruby) and self-play (Python) use different value-target derivations under
   the same `value_target_mode` label, results are untrustworthy. Align
   implementations before drawing conclusions.
