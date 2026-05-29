# AlphaZero-lite Post-Merge Batch 1 Results

Batch helper: `script/ai/run_alpha_zero_post_merge_batch`

Run scope:

- fresh baseline: `aggressive_v3_clone_extend_phase1`
- replay sweep: `w1`, `w2`, `w8`
- secondary checks: `selfplay_sims384`, `value_w060`, `temp_sharp_late`

Guardrails followed:

- ran the fresh baseline from the same commit and environment as the variants
- ran top-k screening before every local promotion gate
- used explicit incumbent paths for every gate run:
  - `--current-path storage/ai/alphazero_lite/current`
  - `--hard-path storage/ai/alphazero_lite/current`
- did not promote any model

Local environment note:

- `script/ai/run_alpha_zero_post_merge_batch` was invoked via `.venv/bin/python` because the helper file was not executable in this worktree.
- `storage/ai/alphazero_lite/current` did not exist locally, so a compatibility symlink was created to the existing `model-artifact/current` artifact to run the batch against the intended incumbent path.

## Outcome

No variant beat the fresh same-commit baseline.

- Replay-weight result: `w2` was the best replay variant, but it only tied the fresh baseline and still failed the arena threshold.
- Search-depth result: `sims384` tied the fresh baseline and did not show a strength gain.
- Value-loss result: `value_w060` did not hold up at the 120-game gate.
- Temperature result: `temp_sharp_late` regressed.

Because every gate run failed arena prefilter, no candidate advanced to `MCTS1200` or strict-CI final confirmation.

## Results Table

| run_id | config | iter_dir | best_top_k_checkpoint | top_k_score | local_gate_score | local_gate_ci_low | local_gate_ci_high | unstable_decision | mcts1200_result | promotion_decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `aggressive-v3-clone-extend` | `ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase1.json` | `/tmp/azlite_v3_clone_extend_versions/aggressive-v3-clone-extend-iter1` | `checkpoint.top2` | `0.50` | `0.50` | `0.4119` | `0.5881` | `true` | `not_run` | `reject` | `checkpoint.top2` and `checkpoint.top3` tied best in top-k; gate failed arena threshold |
| `exp-v3-replay-bootstrap-w1` | `ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w1.json` | `/tmp/azlite_exp_v3_replay_bootstrap_w1_versions/exp-v3-replay-bootstrap-w1-iter1` | `checkpoint` | `0.25` | `0.25` | `0.1811` | `0.3344` | `false` | `not_run` | `reject` | worse than fresh baseline and `w2` |
| `exp-v3-replay-bootstrap-w2` | `ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w2.json` | `/tmp/azlite_exp_v3_replay_bootstrap_w2_versions/exp-v3-replay-bootstrap-w2-iter1` | `checkpoint` | `0.50` | `0.50` | `0.4119` | `0.5881` | `true` | `not_run` | `reject` | best replay variant, but only tied baseline and still below threshold |
| `exp-v3-replay-bootstrap-w8` | `ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w8.json` | `/tmp/azlite_exp_v3_replay_bootstrap_w8_versions/exp-v3-replay-bootstrap-w8-iter1` | `checkpoint` | `0.00` | `0.00` | `0.0000` | `0.0310` | `false` | `not_run` | `reject` | clear regression |
| `exp-v3-selfplay-sims384` | `ml/alphazero_lite/configs/exp_v3_selfplay_sims384.json` | `/tmp/azlite_exp_v3_selfplay_sims384_versions/exp-v3-selfplay-sims384-iter1` | `checkpoint` | `0.50` | `0.50` | `0.4119` | `0.5881` | `true` | `not_run` | `reject` | tied baseline, no evidence of strength gain |
| `exp-v3-value-w060` | `ml/alphazero_lite/configs/exp_v3_value_w060.json` | `/tmp/azlite_exp_v3_value_w060_versions/exp-v3-value-w060-iter1` | `checkpoint.top3` | `0.50` | `0.00` | `0.0000` | `0.0310` | `false` | `not_run` | `reject` | top-k looked mixed, but the 120-game gate collapsed to `0.00` |
| `exp-v3-temp-sharp-late` | `ml/alphazero_lite/configs/exp_v3_temp_sharp_late.json` | `/tmp/azlite_exp_v3_temp_sharp_late_versions/exp-v3-temp-sharp-late-iter1` | `checkpoint` | `0.00` | `0.00` | `0.0000` | `0.0310` | `false` | `not_run` | `reject` | no support for sharper late temperature |

## Recommendation

Recommended next branch: **E**

- Mine fresh failures and run the hard-state replay experiment next.

## Evidence Paths

- top-k summaries: each `iter_dir/top_k_evaluation_summary.json`
- per-run gate reports: `/tmp/azlite/gates/*/local_promotion.json`
