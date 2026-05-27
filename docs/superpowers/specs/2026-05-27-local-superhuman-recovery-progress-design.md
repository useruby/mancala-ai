# Local Superhuman Recovery Progress Design

## Goal

Add a lightweight progress probe for `script/ai/run_local_superhuman_recovery` so we can quickly inspect the active stage and milestone artifacts during long server runs.

## Problem

The recovery flow currently requires manual inspection of running processes and generated files to answer simple questions like:

- which stage is active now
- which milestones are already complete
- whether key outputs exist yet

This is slow and easy to misread during long-running CPU-heavy jobs.

## Recommended Approach

Add one minimal Python wrapper script:

- `script/ai/local_superhuman_recovery_progress`

The script will summarize recovery progress from the generated runtime config plus the current output tree.

## Scope

This change will:

- add a small progress-reporting script
- add targeted regression coverage

This change will not:

- change the recovery pipeline itself
- attempt to control or restart jobs
- provide speculative finish-time estimates

## Behavior

The script will:

- default to `tmp/local_superhuman_recovery`
- read `pipeline_config.json` if present
- derive the candidate dir name from the config
- report whether milestone outputs exist, including:
  - iter2 self-play data
  - iter1 bootstrap dataset
  - iter2 checkpoint
  - iter2 exported model/artifact
  - iter2 arena report
  - local gate report
- include size and mtime metadata for those outputs
- emit a compact JSON summary

The script will also compute a simple status label from existing outputs:

- `not_started`
- `self_play`
- `bootstrap`
- `train`
- `arena`
- `completed`

## Testing

Add focused tests that create a temporary recovery output tree and verify:

- default status when only config exists
- train status when iter2 self-play and iter1 bootstrap exist
- completed status when gate report exists

## Success Criteria

- Running `script/ai/local_superhuman_recovery_progress` prints a valid JSON summary
- The status label matches existing milestone files
- Tests pass
