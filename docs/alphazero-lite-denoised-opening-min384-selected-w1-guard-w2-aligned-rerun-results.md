# Aligned Production Rerun Results For Denoised Opening-Min384 Selected-W1 Guard-W2

## Context

- `train.py` was aligned to the shared epoch loop used by the mirrored PR #40 trace.
- A lane-specific `--lr-scheduler none` override was added to `ml/alphazero_lite/configs/exp_v3_denoised_opening_min384_selected_w1_guard_w2.json`.
- One full production rerun was then executed as iteration 2 of the same lane.

## Important Production Detail

- This was a real production-style rerun, not a frozen-data isolate.
- Because the config keeps `replay_window: 3`, iteration 2 trained on:
  - iteration 2 self-play
  - prior iteration 1 self-play
  - selected replay artifact at weight 1
  - guard-controls artifact at weight 2
- The effective train command therefore used:
  - `--data-files iter2_self_play,iter1_self_play,selected_artifact,guard_controls_artifact`
  - `--replay-weights 2,1,1,2`

## Rerun Artifacts

- iteration dir: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter2`
- top-k summary: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter2/top_k_evaluation_summary.json`

## Training Result

- `--lr-scheduler none` was present in the train command.
- logged metrics:
  - `policy_loss=0.996352`
  - `value_loss=0.059728`
  - `best_val_loss=1.019613`

## Top-k Corrected Guard Gate Results

| checkpoint | `002` | `003` | `006` | `007` | `008` | guard_gate_pass | arena_skipped_due_to_guard |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `checkpoint.npz` | `1 / 1` | `2 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | false | true |
| `checkpoint.top1.npz` | `1 / 1` | `2 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | false | true |
| `checkpoint.top2.npz` | `1 / 1` | `1 / 2` | `2 / 2` | `1 / 2` | `1 / 1` | false | true |
| `checkpoint.top3.npz` | `2 / 2` | `1 / 2` | `2 / 2` | `2 / 2` | `1 / 1` | false | true |

- No exported candidate passed the corrected guard gate.
- Arena was skipped for every candidate.

## Interpretation

- The shared-loop and no-scheduler alignment fixed the trainer-path drift in the controlled frozen-data isolate.
- The full production rerun still failed because the real lane dynamics changed again once iteration 2 included normal replay-window carryover from iteration 1 self-play.
- The failure pattern split across candidates:
  - `checkpoint.npz` and `checkpoint.top1.npz` recovered `003` but failed `002` and `007`
  - `checkpoint.top3.npz` recovered `002` and `007` but failed `003`
- This means the lane is still unstable under the real multi-iteration replay mixture even after trainer alignment.

## Classification

- `production_lane_guard_regression`

## Exactly One Recommended Next Action

Recommendation: **run one more controlled production-style lane from a clean versions root with the aligned shared loop, `--lr-scheduler none`, and no prior lane self-play in the replay window so the production run matches the proven passing single-iteration data mixture before any promotion work.**
