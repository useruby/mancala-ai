# Clean-Root Production-Style Lane Results

## Context

- The trainer-path drift was fixed by sharing the epoch loop and disabling the scheduler for this lane.
- A new clean-root production-style lane was then run so iteration 1 had no inherited lane self-play in the replay window.
- This preserved the intended mixture:
  - current self-play
  - selected replay artifact at weight 1
  - guard-controls artifact at weight 2

## Config

- config: `ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root.json`
- versions dir: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root_versions`
- run id: `exp-v3-denoised-opening-min384-selected-w1-guard-w2-clean-root`

## Verified Train Mixture

The train step used:

- `--data-files clean_root_iter1_self_play,selected_artifact,guard_controls_artifact`
- `--replay-weights 1,1,2`
- `--lr-scheduler none`

So this run matched the intended single-iteration production-style data mixture.

## Rerun Artifact

- iteration dir: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-clean-root-iter1`
- top-k summary: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_clean_root_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-clean-root-iter1/top_k_evaluation_summary.json`

## Training Result

- `policy_loss=1.120810`
- `value_loss=0.062842`
- `best_val_loss=1.142279`

## Top-k Corrected Guard Gate Results

| checkpoint | `002` | `003` | `006` | `007` | `008` | guard_gate_pass | arena_skipped_due_to_guard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `checkpoint.npz` | `2 / 2` | `1 / 1` | `2 / 2` | `2 / 2` | `1 / 1` | false | true |
| `checkpoint.top1.npz` | `2 / 2` | `1 / 1` | `2 / 2` | `2 / 2` | `1 / 1` | false | true |
| `checkpoint.top2.npz` | `1 / 2` | `1 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | false | true |
| `checkpoint.top3.npz` | `1 / 2` | `1 / 2` | `2 / 2` | `2 / 2` | `1 / 1` | false | true |

## Interpretation

- Even with the corrected trainer path and the intended clean single-iteration replay mixture, no exported top-k candidate passed the corrected guard gate.
- The clean-root run recovered `002`, `006`, `007`, and `008` in `checkpoint.npz` and `checkpoint.top1.npz`, but `capture_available-003` regressed all the way to `1 / 1`.
- This means the earlier frozen-data isolate pass is still not enough to guarantee a production-style lane pass once fresh self-play is regenerated.

## Classification

- `production_lane_guard_regression`

## Exactly One Recommended Next Action

Recommendation: **treat fresh self-play generation as the remaining instability source and run a targeted reproducibility investigation on the clean-root iter1 self-play corpus versus the frozen passing corpus before any further production-lane attempts.**
