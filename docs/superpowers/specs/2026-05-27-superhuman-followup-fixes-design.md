# Superhuman Follow-up Fixes Design

## Goal

Correct the post-recovery sweep and wrapper behavior so the recorded analysis matches the actual executed settings and user-supplied baseline paths still fail fast.

## Scope

This follow-up change will:

- make `script/ai/superhuman_budget_sweep` apply the requested challenger budget per lane
- restore local validation for explicit non-default `--promotion-current-path` values in `script/ai/runpod_training_experiment`
- align the progress probe spec with the implemented status model

This follow-up change will not:

- change the local recovery pipeline behavior
- change the progress probe implementation
- add broader refactors around RunPod bundling or promotion-gate configuration

## Design

### Budget Sweep

Each sweep lane should evaluate a distinct challenger budget while keeping the incumbent side fixed.

Use the existing phase-2 config as the source config, then write a per-lane config in the lane output directory. Rewrite only these challenger-side values:

- `arena_confirm_report --challenger-simulations`
- `mcts1200_baseline_report --az-base-simulations`

Then invoke `script/ai/local_promotion_gate` with `--config-path` pointing to that lane config.

### RunPod Baseline Path Validation

Bundle filtering should continue to ignore optional repo paths that are absent in this workspace. But when a user explicitly provides a non-default `--promotion-current-path`, the wrapper should validate that the path exists before building the dry-run or remote plan. A typo should fail locally with a clear error instead of being silently omitted from the bundle.

### Progress Spec Alignment

Keep the current progress script behavior:

- `train` once bootstrap output exists
- `completed` once the gate report exists

Update the progress spec text and tests section so it no longer mentions an emitted `gate` state or expects `bootstrap` after both self-play and bootstrap outputs exist.

## Testing

- add a failing sweep test that asserts lane-specific config rewriting happens before running the gate
- add a failing wrapper test that asserts a missing explicit `--promotion-current-path` exits with a local validation error
- rerun the targeted wrapper and promotion test modules after the fixes
