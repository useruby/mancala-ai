# Shared-Loop Alignment Result

## Change

Code changes made:

- `ml/alphazero_lite/train.py`
  - added shared `compute_policy_cross_entropy`
  - added shared `compute_value_loss_vector`
  - added shared `train_one_epoch`
  - added optional `--lr-scheduler none|cosine` with default `cosine`
  - switched `train()` to use the shared epoch loop
- `ml/alphazero_lite/run_guard_safe_opening_low_epoch_drift_trace.py`
  - switched its local `train_one_epoch` wrapper to call the shared `ml.alphazero_lite.train.train_one_epoch`

Default production behavior is unchanged because `train.py` still defaults to `--lr-scheduler cosine`.

## Confirmation Run

One controlled isolate was rerun with:

- frozen production self-play corpus
- replay weights `[1, 1, 2]`
- `val_split=0`
- `seed=442`
- `--lr-scheduler none`

Paths:

- checkpoint: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain_seed442_no_scheduler_sharedloop/checkpoint.epoch4.npz`
- exported artifact: `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/frozen_data_valsplit0_retrain_seed442_no_scheduler_sharedloop/exported_artifact`

## Hash Result

| artifact | sha256 |
| --- | --- |
| shared-loop `train.py` isolate | `472a3536945148636cca8b5b1a131b17eb8239f2838731fe9180ef28f8bbdbc1` |
| mirrored PR #40 passing checkpoint | `472a3536945148636cca8b5b1a131b17eb8239f2838731fe9180ef28f8bbdbc1` |

The checkpoints match exactly.

## Corrected Guard Gate Result

| row_id | corrected_reference_move | selected_move_384 | selected_move_1200 | row_pass |
| --- | ---: | ---: | ---: | --- |
| `capture_available-002` | 2 | 2 | 2 | true |
| `capture_available-003` | 2 | 2 | 2 | true |
| `capture_available-006` | 2 | 2 | 2 | true |
| `capture_available-007` | 2 | 2 | 2 | true |
| `capture_available-008` | 1 | 1 | 1 | true |

- overall guard result: `pass`
- reason: `all_corrected_reference_rows_selected`

## Interpretation

- The divergence between `train.py` and the mirrored PR #40 path is resolved for the controlled isolate when:
  - both paths share the same epoch-training loop
  - scheduler is disabled to match the mirrored trace
- This confirms the earlier production regression was caused by trainer-path drift, not by the self-play corpus.

## Exactly One Recommended Next Action

Recommendation: **add an explicit config-level switch for this lane to disable the LR scheduler in the production training entrypoint, then rerun exactly one full production lane with the aligned shared loop before making any promotion decision.**
