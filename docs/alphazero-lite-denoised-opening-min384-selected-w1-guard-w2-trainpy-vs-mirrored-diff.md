# train.py vs Mirrored PR #40 Trainer Diff

## Question

What is the smallest production-path behavior difference that still explains why `train.py` fails this lane while the mirrored PR #40 trainer passes on the exact same frozen production self-play corpus?

## Controls Already Matched

These are already controlled and therefore are not the remaining cause:

- same frozen self-play corpus
- same selected replay artifact
- same guard-controls artifact
- same replay weights `[1, 1, 2]`
- same initializer checkpoint
- same epoch count `4`
- same batch size `512`
- same optimizer family `Adam`
- same model type `residual_v3`
- same input encoding `kalah_v3`
- same loss settings
- same `val_split=0`
- same seed `442`

Evidence:

- mirrored PR #40 trainer on frozen production self-play: full corrected guard pass at epoch 4
- `train.py` on the same frozen production self-play with `val_split=0` and `--seed 442`: still fails `capture_available-003` at `384` sims

## Hash Evidence

| artifact | sha256 |
| --- | --- |
| mirrored PR #40 epoch-4 checkpoint | `472a3536945148636cca8b5b1a131b17eb8239f2838731fe9180ef28f8bbdbc1` |
| `train.py` `val_split=0` seed `442` epoch-4 checkpoint | `3191dc0a4f0c00726af03cd953f810819dc143945fbe14c10af7e82bc156807a` |

The checkpoints are still different after matching data, seed, and validation behavior.

## Remaining Live Behavior Difference

After matching `val_split=0`, the main functional training-path difference left is:

- `train.py` always applies `torch.optim.lr_scheduler.CosineAnnealingLR` and calls `scheduler.step()` every epoch.
- the mirrored PR #40 `run_trace` path does not use a scheduler at all; it keeps a constant learning rate across all four epochs.

## Why This Is The Smallest Plausible Difference

With `val_split=0`:

- `best_state` restoration is inactive in `train.py`
- validation-ranked top-k selection is irrelevant to the explicit epoch-4 isolate
- both paths use seeded `torch.randperm` minibatch shuffling
- both paths use the same replay index construction
- both paths optimize the same objective with the same optimizer family

That leaves the learning-rate schedule as the smallest remaining behavior difference that is still active and capable of changing the epoch-4 checkpoint.

## Supporting Outcome

`train.py` `val_split=0` with seed `442` guard result:

- `capture_available-002`: `2 / 2`
- `capture_available-003`: `1 / 2`
- `capture_available-006`: `2 / 2`
- `capture_available-007`: `2 / 2`
- `capture_available-008`: `1 / 1`

This is much closer to the mirrored pass than the original production lane, which is consistent with removing the validation split helping, but not enough while the scheduler remains active.

## Conclusion

The smallest remaining production-path change most strongly supported by the evidence is:

- disable the cosine annealing scheduler for this training path so `train.py` matches the constant-learning-rate behavior of the passing PR #40 mirrored trainer.

## Exactly One Recommended Next Action

Recommendation: **patch `ml/alphazero_lite/train.py` to make the LR scheduler optional and rerun one frozen-data `train.py` isolate with the scheduler disabled, `val_split=0`, and seed `442` to confirm it reproduces the passing mirrored checkpoint before changing the production lane config.**
