# Final Report: Denoised Opening-Min384 Selected-W1 Guard-W2

## Outcome

- Exactly one controlled production-style lane was run to completion with the intended replay mix.
- The lane was eventually made corrected-guard-safe under the production entrypoint by aligning the trainer path and using:
  - `lr_scheduler=none`
  - `val_split=0`
  - training seed `442`
- The best guard-passing candidates then lost all 60 arena games against `storage/ai/alphazero_lite/current`.

## Final Classification

- `guard_safe_no_strength_gain`

## Final Evidence

- Guard-gate pass:
  - `checkpoint.npz`: pass
  - `checkpoint.top1.npz`: pass
- Arena result for both guard-passing candidates:
  - score `0.0000`
  - `0W-60L-0D`

## What Was Ruled Out

- corrected guard gate implementation bug
- stale corrected reference artifact issue
- missing replay artifacts
- self-play corpus drift between the frozen passing corpus and the clean-root corpus
- trainer-path drift from the diagnostic loop, after shared-loop alignment

## Final Interpretation

- The opening replay intervention can make the lane guard-safe.
- But the resulting model is not competitive with the current artifact.
- This lane is therefore not a promotion candidate and should not receive more opening-replay tuning.

## Exactly One Recommended Next Action

Recommendation: **stop opening replay work on this lane and mine corrected non-opening failure families instead.**
