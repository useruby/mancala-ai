# Frozen-Data ValSplit0 Isolate For Denoised Opening-Min384 Selected-W1 Guard-W2

## Context

- The prior comparison showed two production-only differences relative to the PR #40 diagnostic pass:
  - a different self-play corpus
  - the `train.py` production path with `val_split=0.1` and best-validation checkpoint restoration
- To isolate the second factor, this run retrained once on the already-generated production `self_play.jsonl` with the same replay mix and `val_split=0`, then evaluated the explicit epoch-4 artifact with the corrected guard gate.

## Commands Run

Training:

```bash
.venv/bin/python ml/alphazero_lite/train.py \
  --data-files /tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter1/self_play.jsonl,/tmp/azlite_guard_safe_opening_replay/family_leave_one_out_without_opening_extra_turn_overbias.jsonl,/tmp/azlite_guard_safe_opening_replay/guard_safe_controls_only.jsonl \
  --replay-weights 1,1,2 \
  --init-checkpoint /tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter1/parent_init_checkpoint.npz \
  --out /tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain/checkpoint.epoch4.npz \
  --epochs 4 \
  --batch-size 512 \
  --device auto \
  --hidden-sizes 96,3 \
  --model-type residual_v3 \
  --input-encoding kalah_v3 \
  --value-loss huber \
  --huber-delta 1.0 \
  --value-loss-weight 0.3 \
  --val-split 0 \
  --grad-clip 1.0 \
  --policy-target-mode sharpened \
  --value-target-mode sharpened
```

Export:

```bash
.venv/bin/python ml/alphazero_lite/export_artifact.py \
  --checkpoint /tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain/checkpoint.epoch4.npz \
  --out-dir /tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain/exported_artifact \
  --version exp-v3-denoised-opening-min384-selected-w1-guard-w2-frozen-data-valsplit0-epoch4 \
  --model-type residual_v3 \
  --rules-version kalah_v1 \
  --input-encoding kalah_v3
```

Corrected guard gate:

- output: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain/corrected_guard_kill_gate.json`

## Training Output

- checkpoint: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain/checkpoint.epoch4.npz`
- exported artifact: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain/exported_artifact`
- logged metrics:
  - `policy_loss=1.117742`
  - `value_loss=0.062496`
  - `best_val_loss=1.136491`

## Corrected Guard Gate Result

| row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | reference_visit_share_384 | reference_visit_share_1200 | row_pass | notes |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| `capture_available-002` | 2 | 1 | 2 | 0.2734 | 0.4550 | false | `selected_move_not_reference` |
| `capture_available-003` | 2 | 1 | 2 | 0.3672 | 0.4392 | false | `selected_move_not_reference` |
| `capture_available-006` | 2 | 2 | 2 | 0.4635 | 0.6150 | true | `pass_reference_selected` |
| `capture_available-007` | 2 | 2 | 2 | 0.4297 | 0.5600 | true | `pass_reference_selected` |
| `capture_available-008` | 1 | 1 | 1 | 0.7630 | 0.8017 | true | `pass_reference_selected` |

- Guard-gate decision: `reject_guard_regression`
- Overall pass: `false`

## Interpretation

- Removing the validation holdout and best-validation checkpoint restoration was not enough to recover the diagnostic pass.
- The frozen-data isolate still failed on `capture_available-002` and `capture_available-003` at `384` sims.
- `capture_available-007` did recover relative to the original production `checkpoint.npz`, which suggests validation/checkpoint selection affected some rows, but not the decisive `002` and `003` failures.
- The remaining regression is therefore more strongly tied to the difference between the PR #40 diagnostic training path and the production self-play corpus/training dynamics than to `val_split` alone.

## Exactly One Recommended Next Action

Recommendation: **run one production-faithful four-epoch trace on the frozen production self-play corpus that mirrors the PR #40 diagnostic trainer step-for-step, then compare its epoch checkpoints against this `train.py` val-split-0 result to isolate whether the remaining regression comes from trainer dynamics or from the regenerated self-play corpus itself.**
