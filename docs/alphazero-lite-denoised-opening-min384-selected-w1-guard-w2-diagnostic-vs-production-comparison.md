# Diagnostic vs Production Comparison For Denoised Opening-Min384 Selected-W1 Guard-W2

## Context

- PR #40's diagnostic trace `denoised_opening_min384_plus_selected_w1_guard_controls_w2` passed the corrected guard gate at epoch 4.
- The later production lane `exp-v3-denoised-opening-min384-selected-w1-guard-w2` failed the corrected guard gate for every exported top-k candidate.
- This comparison checks the exact artifacts used by both runs rather than inferring from the reports alone.

## Confirmed Differences

### 1. The self-play artifact was not the same file

| artifact | path | sha256 |
| --- | --- | --- |
| diagnostic self-play | `/tmp/azlite_exp_v3_guard_safe_opening_selected_w1_denoised_targets_opening_min_384_versions/exp-v3-guard-safe-opening-selected-w1-denoised-targets-opening-min-384-iter1/self_play.jsonl` | `1e2569c4b2612af8392f732781aba89aec6956d11be3cff9d70be60b23cbd571` |
| production self-play | `/tmp/azlite_exp_v3_denoised_opening_min384_selected_w1_guard_w2_versions/exp-v3-denoised-opening-min384-selected-w1-guard-w2-iter1/self_play.jsonl` | `9341e344e813d8bc5103b83b152f684b638b4650c716f4acf43681f93e3836e4` |

- The files have the same row count, `63959`, but different full-file hashes.
- They also have different order-insensitive multiset digests, so this is not just output ordering:
  - diagnostic multiset digest: `f35d4d4171c200e6c01701d892b412850b7e06056b4c0fc399e7563e4c52d839`
  - production multiset digest: `3f01a73e7aac5e717c55ad50aafc9c4a146b59f3bd82b24acf668ec598b9372b`

### 2. The exact guard-row self-play hit counts stayed the same

| row_id | diagnostic self-play hits | production self-play hits | notes |
| --- | ---: | ---: | --- |
| `capture_available-002` | 4 | 4 | all four targets still top-move `2` |
| `capture_available-003` | 0 | 0 | no exact self-play anchor in either run |
| `capture_available-006` | 0 | 0 | no exact self-play anchor in either run |
| `capture_available-007` | 3 | 3 | all three targets still top-move `2` |
| `capture_available-008` | 3 | 3 | all three targets still top-move `1` |
| `opening_plies_1_8-017` | 16 | 16 | all sixteen targets still top-move `2` |

- The regression is therefore not explained by losing the exact self-play anchors for `002`, `007`, `008`, or `017`.
- The broader self-play distribution still changed, because the whole-file digests differ.

### 3. The diagnostic trainer was not production-faithful

The passing PR #40 trace came from `ml/alphazero_lite/run_denoised_opening_min384_guard_tradeoff_audit.py`, which:

- loads the weighted replay directly with `load_jsonl_replay`
- trains on the full weighted dataset each epoch
- does not apply the `train.py` validation split
- does not restore a best-validation checkpoint
- exports the exact epoch-4 checkpoint that passed the guard gate

The production lane used `ml/alphazero_lite/train.py`, which:

- applies `--val-split 0.1` by source row
- uses a cosine annealing scheduler
- restores `best_state` into `checkpoint.npz`
- saves `checkpoint.top1-3.npz` from validation-ranked snapshots instead of preserving every epoch artifact

### 4. The production validation split held out the strongest `003` replay anchors entirely

Production replay exposure under the deterministic `seed=42` validation split:

| row_id | selected_train_copies | guard_train_copies | selected_held_out | guard_held_out |
| --- | ---: | ---: | --- | --- |
| `capture_available-002` | 1 | 2 | false | false |
| `capture_available-003` | 0 | 0 | true | true |
| `capture_available-006` | 0 | 2 | true | false |
| `capture_available-007` | 1 | 2 | false | false |
| `capture_available-008` | 1 | 2 | false | false |

- `capture_available-003` lost both the selected replay copy and the weight-2 guard-controls copy to validation.
- Because self-play had `0` exact `003` hits in both runs, the production trainer had no replay training exposure for `003` at all.
- This exactly matches the observed production failure on `capture_available-003`.

### 5. The passing diagnostic checkpoint does not match any production export

| checkpoint | sha256 |
| --- | --- |
| diagnostic passing epoch-4 checkpoint | `472a3536945148636cca8b5b1a131b17eb8239f2838731fe9180ef28f8bbdbc1` |
| production `checkpoint.npz` | `e8aa62f166fce14501d1da4e03020789ae1e611e43b02238145fb6653cf25d00` |
| production `checkpoint.top1.npz` | `e8aa62f166fce14501d1da4e03020789ae1e611e43b02238145fb6653cf25d00` |
| production `checkpoint.top2.npz` | `ed2777d92e9635ee6c58ac3a63ec20133e76372b4a4e06be0898f08d58598aed` |
| production `checkpoint.top3.npz` | `a734b9acb52f39df4af03f91db344500a221ff2d1de75d7c584b2a7045d0fe21` |

- The production exports are all distinct from the checkpoint that passed in PR #40.
- `checkpoint.npz` and `checkpoint.top1.npz` are identical, confirming the production lane restored the best-validation state.

## Most Likely Root Cause

The production failure is best explained by a combination of two concrete differences:

1. The diagnostic pass did not use the production training path.
2. The production training path's validation split removed all replay supervision for `capture_available-003` and changed checkpoint selection behavior.

`capture_available-003` is explained directly by the held-out replay anchors.

`capture_available-002` is not explained by replay holdout alone, because its selected and guard-control copies remained in training. That remaining regression is most consistent with the other production-only differences: different full self-play content, the source-row validation split over the whole dataset, cosine scheduling, and best-validation checkpoint restoration.

## Conclusion

- PR #40's passing trace was diagnostic evidence, not a production-faithful rehearsal.
- The strongest production-specific regression lever is the validation split, because it eliminated all replay training exposure for `capture_available-003`.
- The remaining `capture_available-002` miss points to additional sensitivity to the production training path and/or changed self-play corpus.

## Exactly One Recommended Next Action

Recommendation: **run one frozen-data retrain on the already-generated production `self_play.jsonl` with the same replay mix but `val_split=0` and an explicit epoch-4 export, so we can isolate whether the regression is caused by production checkpoint selection/validation holdout rather than the regenerated self-play corpus.**
