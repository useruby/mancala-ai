# Local Superhuman Recovery Workers Design

## Goal

Allow `script/ai/run_local_superhuman_recovery` to override worker-count settings in its generated runtime config so the Intel 13900 server can run the recovery flow with higher CPU parallelism.

## Problem

`run_local_superhuman_recovery` currently copies `ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json` into a runtime config and executes it unchanged except for path and Python rewrites. The base config hardcodes `--workers 6` across self-play, bootstrap, and arena steps, which underuses the server.

## Recommended Approach

Add a single optional CLI flag:

- `--workers N`

When provided, rewrite any `--workers` flag in the generated runtime config to `N` before the pipeline starts.

## Scope

This change will:

- update `script/ai/run_local_superhuman_recovery`
- add targeted regression coverage

This change will not:

- modify the base phase2 config
- change non-worker search budgets like simulations or game counts
- alter the active in-flight run without restart

## Behavior

- Default behavior remains unchanged when `--workers` is omitted.
- If `--workers 24` is passed, generated runtime config steps that already contain `--workers` will be rewritten from `6` to `24`.
- Steps without a `--workers` flag remain unchanged.

## Testing

Add one focused test that verifies a dry-run with `--workers 24` produces a runtime config whose worker-based steps use `24`.

## Success Criteria

- `run_local_superhuman_recovery --workers 24` rewrites worker flags in the generated runtime config.
- Existing behavior is unchanged when no override is provided.
- Targeted tests pass.
