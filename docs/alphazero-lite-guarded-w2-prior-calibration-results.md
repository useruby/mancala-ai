# AlphaZero-lite Guarded w2 Prior-Calibration Results

## Outcome

Ran the exact guarded `w2` prior-calibration branch defined in `docs/alphazero-lite-guarded-w2-prior-calibration-spec.md`.

Result: reject the branch.

- local gate decision: `reject_local_gate`
- gate reason: `row_002_local_rule_collision_persists`
- arena score: `0.0` (`0-120-0`)
- MCTS1200 score: `0.6167`
- benchmark recommendation: `drop`

Artifacts:

- setup summary: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/setup_summary.json`
- runtime config: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/runtime_config.json`
- candidate dir: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-prior-calibration-iter1`
- gate: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-prior-calibration-iter1/capture_002_003_rule_conditioned_gate.json`
- arena: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-prior-calibration-iter1/arena_report.json`

## What Improved

Relative to the original guarded `w2` candidate, the prior-calibration lane did move row `002` in the intended prior-support direction:

- `capture_available-002` policy reference probability: `0.0668 -> 0.0794`
- `capture_available-002` reference visit share: `0.0547 -> 0.0651`
- `capture_available-002` wrong extra-turn visit share: `0.6693 -> 0.6406`
- `capture_available-003` reference visit share: `0.6875 -> 0.7005`

So the added replay pressure was not ignored.

## Why It Still Failed

The branch did not clear the actual acceptance contract.

- row `002` still search-selected move `2` instead of reference move `4`
- row `002` selected-minus-reference Q margin got worse versus guarded `w2`: `0.0496 -> 0.0594`
- row `003` stayed preserved on selected move, but still failed the repo gate because its reference visit share remained materially below incumbent
- downstream arena collapsed completely to `0-120`

This means the branch improved prior support without producing a stable full-policy fix.

## Interpretation

This run is still useful because it narrows the mechanism further.

- the failure is not that row `002` is unresponsive to training-side prior pressure
- the failure is that a small replay-side prior calibration alone does not convert that pressure into a robust search-selection correction
- the catastrophic arena result says this kind of local replay weighting is too brittle as a general training intervention on the guarded `w2` base

The train logs support that interpretation:

- guarded `w2` `best_val_loss=1.253427`
- prior-calibration lane `best_val_loss=1.252191`

So this was not a generic optimization collapse during fitting. It fit about as well as guarded `w2`, but landed in a worse behavioral tradeoff.

## Updated Recommendation

Do not launch another training-side prior-calibration retry from guarded `w2`.

Best next step: return to diagnostics and write a non-training branch that isolates why root-only prior equalization can flip row `002` while replay-side prior pressure cannot hold the flip through the full learned policy/search stack.

Practical implication:

- stop replay reweighting as the next intervention class for this row pair
- use the root-prior result as a diagnostic clue, not as a direct training recipe
- the next branch should compare root-only prior overrides against learned-policy outputs and visit/Q evolution on the guarded `w2` candidate rather than launching another local lane
