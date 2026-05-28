# Post-Merge AlphaZero-lite Batch

This runbook is the reproducible first batch to run after the AlphaZero-lite experiment PRs merge.

## Why This Order

- Re-run the baseline from the same commit and environment before judging variants.
- Do not compare new variants only against historical results. Historical runs may differ by code, data generation behavior, incumbent strength, or local environment.
- Screen saved top-k checkpoints before trusting `checkpoint.npz`. Validation loss is only a triage signal; arena strength is the promotion signal.
- Do not promote on training or validation loss alone. A lower `best_val_loss` without stronger arena or MCTS1200 play is not a win.
- Use strict confidence-interval validation only for final confirmation. It is intentionally too harsh for early exploratory filtering at small game counts.
- Always pass explicit incumbent paths to `local_promotion_gate`. The configs and `evaluate_top_k_checkpoints.py` use `storage/ai/alphazero_lite/current`, but `script/ai/local_promotion_gate` defaults to `model-artifact/current`. Leaving that implicit can silently compare against the wrong artifact.

## Explicit Incumbent Paths

Use these explicit paths in every promotion-gate command:

```bash
--current-path storage/ai/alphazero_lite/current \
--hard-path storage/ai/alphazero_lite/current
```

## First Batch Only

Run only this batch first:

1. `ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase1.json`
2. `ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w1.json`
3. `ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w2.json`
4. `ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w8.json`
5. `ml/alphazero_lite/configs/exp_v3_selfplay_sims384.json`
6. `ml/alphazero_lite/configs/exp_v3_value_w060.json`
7. `ml/alphazero_lite/configs/exp_v3_temp_sharp_late.json`

Do not run yet:

- `exp_v3_selfplay_sims768_small`
- `exp_v3_value_w015`
- `exp_v3_value_w100`
- `exp_v3_temp_long_explore`
- `exp_v3_temp_deterministic_late`
- hard-state replay variants
- expensive search-factorial retraining

## One-Command Helper

Dry run:

```bash
script/ai/run_alpha_zero_post_merge_batch --dry-run
```

Actual batch:

```bash
script/ai/run_alpha_zero_post_merge_batch
```

The helper validates each JSON config, runs the seven configs in the fixed order above, stops on failure, and prints the follow-up top-k and promotion-gate commands. It does not run promotion automatically.

## Expected Iteration Directories

These paths match the current inspected `run_id` and `versions_dir` values in the configs:

```text
/tmp/azlite_v3_clone_extend_versions/aggressive-v3-clone-extend-iter1
/tmp/azlite_exp_v3_replay_bootstrap_w1_versions/exp-v3-replay-bootstrap-w1-iter1
/tmp/azlite_exp_v3_replay_bootstrap_w2_versions/exp-v3-replay-bootstrap-w2-iter1
/tmp/azlite_exp_v3_replay_bootstrap_w8_versions/exp-v3-replay-bootstrap-w8-iter1
/tmp/azlite_exp_v3_selfplay_sims384_versions/exp-v3-selfplay-sims384-iter1
/tmp/azlite_exp_v3_value_w060_versions/exp-v3-value-w060-iter1
/tmp/azlite_exp_v3_temp_sharp_late_versions/exp-v3-temp-sharp-late-iter1
```

## Manual Commands

### 1. Fresh baseline

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase1.json
```

Why rerun it:

- It freezes a fresh baseline from the same merged code and runtime environment.
- It prevents over-interpreting improvements against stale historical artifacts.

### 2. Variant batch

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w1.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w2.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w8.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_selfplay_sims384.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_value_w060.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_temp_sharp_late.json
```

### 3. Top-k screening before promotion

Run this for every produced iteration directory before trusting `checkpoint.npz`:

```bash
.venv/bin/python ml/alphazero_lite/evaluate_top_k_checkpoints.py \
  --iter-dir ITER_DIR \
  --current-path storage/ai/alphazero_lite/current \
  --games 60
```

Examples:

```bash
.venv/bin/python ml/alphazero_lite/evaluate_top_k_checkpoints.py \
  --iter-dir /tmp/azlite_v3_clone_extend_versions/aggressive-v3-clone-extend-iter1 \
  --current-path storage/ai/alphazero_lite/current \
  --games 60

.venv/bin/python ml/alphazero_lite/evaluate_top_k_checkpoints.py \
  --iter-dir /tmp/azlite_exp_v3_replay_bootstrap_w2_versions/exp-v3-replay-bootstrap-w2-iter1 \
  --current-path storage/ai/alphazero_lite/current \
  --games 60
```

Pick the strongest `checkpoint.npz` or `checkpoint.topN.npz` export from the top-k summary before any promotion-gate run.

### 4. Local promotion gate examples

Always pass explicit incumbent paths:

```bash
script/ai/local_promotion_gate \
  --candidate-path /path/to/top_k_exports/checkpoint_or_checkpoint.topN \
  --current-path storage/ai/alphazero_lite/current \
  --hard-path storage/ai/alphazero_lite/current \
  --config-path ml/alphazero_lite/configs/THE_WINNING_CONFIG.json \
  --out /tmp/azlite/THE_RUN_ID-local-promotion.json
```

Concrete example:

```bash
script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_exp_v3_replay_bootstrap_w2_versions/exp-v3-replay-bootstrap-w2-iter1/top_k_exports/checkpoint.top1 \
  --current-path storage/ai/alphazero_lite/current \
  --hard-path storage/ai/alphazero_lite/current \
  --config-path ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w2.json \
  --out /tmp/azlite/exp-v3-replay-bootstrap-w2-local-promotion.json
```

### 5. Final confirmation only for the best one or two

Use larger games only after a candidate already looks good on the smaller screen:

```bash
script/ai/local_promotion_gate \
  --candidate-path /path/to/top_k_exports/checkpoint_or_checkpoint.topN \
  --current-path storage/ai/alphazero_lite/current \
  --hard-path storage/ai/alphazero_lite/current \
  --config-path ml/alphazero_lite/configs/THE_WINNING_CONFIG.json \
  --arena-games 240 \
  --mcts-games 80 \
  --out /tmp/azlite/THE_WINNING_RUN-confirm-promotion.json
```

Then validate the emitted arena report with the strict CI gate:

```bash
.venv/bin/python ml/alphazero_lite/validate_arena_report.py \
  --report /path/to/gate-arena-report.json \
  --min-score 0.55 \
  --require-pass \
  --require-ci-above-threshold
```

Strict CI is a final-confirmation tool, not an exploratory batch filter.

## Result Classification Rules

- Fails score threshold: reject.
- Passes threshold but `unstable_decision` is `true`: do not promote; rerun with more games.
- Passes threshold, CI is clearly above threshold, and MCTS1200 passes: candidate for final confirmation.
- Arena improves but MCTS1200 fails: diagnose search/value mismatch; do not promote.
- Training or validation loss improves but arena does not: reject.

## Results Table Template

```text
| run_id | config | iter_dir | best_top_k_checkpoint | top_k_score | local_gate_score | local_gate_ci_low | local_gate_ci_high | unstable_decision | mcts1200_result | promotion_decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| aggressive-v3-clone-extend | ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase1.json | /tmp/azlite_v3_clone_extend_versions/aggressive-v3-clone-extend-iter1 | checkpoint.topN |  |  |  |  |  |  |  |  |
| exp-v3-replay-bootstrap-w1 | ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w1.json | /tmp/azlite_exp_v3_replay_bootstrap_w1_versions/exp-v3-replay-bootstrap-w1-iter1 | checkpoint.topN |  |  |  |  |  |  |  |  |
| exp-v3-replay-bootstrap-w2 | ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w2.json | /tmp/azlite_exp_v3_replay_bootstrap_w2_versions/exp-v3-replay-bootstrap-w2-iter1 | checkpoint.topN |  |  |  |  |  |  |  |  |
| exp-v3-replay-bootstrap-w8 | ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w8.json | /tmp/azlite_exp_v3_replay_bootstrap_w8_versions/exp-v3-replay-bootstrap-w8-iter1 | checkpoint.topN |  |  |  |  |  |  |  |  |
| exp-v3-selfplay-sims384 | ml/alphazero_lite/configs/exp_v3_selfplay_sims384.json | /tmp/azlite_exp_v3_selfplay_sims384_versions/exp-v3-selfplay-sims384-iter1 | checkpoint.topN |  |  |  |  |  |  |  |  |
| exp-v3-value-w060 | ml/alphazero_lite/configs/exp_v3_value_w060.json | /tmp/azlite_exp_v3_value_w060_versions/exp-v3-value-w060-iter1 | checkpoint.topN |  |  |  |  |  |  |  |  |
| exp-v3-temp-sharp-late | ml/alphazero_lite/configs/exp_v3_temp_sharp_late.json | /tmp/azlite_exp_v3_temp_sharp_late_versions/exp-v3-temp-sharp-late-iter1 | checkpoint.topN |  |  |  |  |  |  |  |  |
```

## Guardrails

- Do not modify training code while running this batch.
- Do not change experiment parameters during the run.
- Do not auto-promote any model.
- Do not run all experiment families before reading the first batch.
- Do not use validation loss as a promotion criterion.
