# AlphaZero-lite Corrected-Reference Targeted Hard-State Replay Spec

## Goal

Define one exact next branch that mines and replays only corrected-reference forensic failures now that the tracked forensic reference artifact has been rebuilt and validated.

This branch should repair genuine model/search gaps, not reference integrity problems.

## Trigger

The repaired-baseline forensic rebaseline now concludes:

- classification: `genuine_model_search_gap`
- corrected failures remaining: `132`
- reference integrity errors remaining: `0`
- recommendation: `run a new targeted hard-state mining/replay experiment, but only from corrected references.`

Supporting artifacts:

- `/tmp/azlite_forensic_reference_rebaseline/forensic_reference_rebaseline_summary.json`
- `/tmp/azlite_forensic_reference_rebaseline/corrected_failure_inventory.json`
- `docs/alphazero-lite-forensic-reference-rebaseline-results.md`

## Why This Branch

The previous hard-state replay branch was dominated by `large_value_error` and did not improve arena strength.

That failure mode is now avoidable because the corrected-reference inventory cleanly separates true residual failures from the earlier poisoned reference artifact.

The highest-value remaining target families are:

- `capture_available`: `14` failures, all `high` severity
- `opening_plies_1_8`: `33` failures, all `medium` severity
- `incumbent_proxy_disagreement`: `23` failures, all `medium` severity
- `high_value_swing`: `17` failures, all `medium` severity
- `high_imbalance`: `16` failures, all `medium` severity

Key implication:

- the next replay lane must not let `large_value_error` dominate candidate selection again
- it must explicitly preserve family diversity and overweight the high-severity corrected `capture_available` rows

## Exact Next Branch

Run one new branch only:

- branch name: `corrected_reference_targeted_hard_state_replay`
- base runner: `ml/alphazero_lite/run_hard_state_replay_experiment.py`
- base config: `ml/alphazero_lite/configs/aggressive_v3_targeted_hard_state_replay.json`
- mining source of truth: corrected-reference forensic-suite style reports only

## Required Inputs

Use only artifacts derived from the corrected tracked forensic reference artifact:

- `ml/alphazero_lite/fixtures/incumbent_forensic_references_v1.json`
- `ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json`
- `/tmp/azlite_forensic_reference_rebaseline/corrected_failure_inventory.json`
- a fresh forensic-suite style report generated from the current artifact against the corrected tracked references

Do not mine from:

- old train-only forensic references
- pre-rebuild forensic reports
- arena loss reports unrelated to the corrected-reference rerun
- any dataset that still contains `reference_integrity_error` rows

## Mining Scope

Mine only rows whose corrected inventory status is:

- `fail_corrected_reference`

Exclude rows whose corrected inventory status is:

- `pass_corrected_reference`
- `unstable_reference`
- `reference_integrity_error`

This means:

- `capture_available-008` stays excluded as unstable
- `capture_available-002` stays excluded as already fixed under corrected references
- all replay candidates must be traceable to corrected-reference failures only

## Required Family Balancing

The replay branch must use explicit family quotas before teacher labeling.

Required mined candidate composition:

1. `capture_available`
2. `opening_plies_1_8`
3. `incumbent_proxy_disagreement`
4. `high_value_swing`
5. `high_imbalance`

Initial quota rule:

- reserve `40%` of selected rows for `capture_available`
- reserve `15%` each for:
  - `opening_plies_1_8`
  - `incumbent_proxy_disagreement`
  - `high_value_swing`
  - `high_imbalance`

If a family cannot fill its quota with unique corrected-failure rows, spill the remainder in this order:

1. `starvation_pressure`
2. `sparse_endgame`
3. `early_extra_turn`

This branch must not select rows purely by global priority score if that would reduce `capture_available` below quota.

## Required Mining Signals

Use the existing hard-state mining scoring machinery, but constrain accepted rows to corrected-reference failures and preserve reason diversity.

Required source classes to keep visible in the mined summary:

- `student_teacher_disagreement`
- `large_value_error`
- `high_search_entropy`
- `large_best_second_gap`

Additional constraint:

- no single selection reason may account for more than `50%` of the final labeled replay rows

The purpose is to avoid recreating the earlier `large_value_error` monoculture.

## Replay Dataset Shape

Build one replay dataset only.

Requirements:

- all rows must preserve corrected teacher-selected moves and teacher child stats from the rebuilt tracked forensic reference artifact
- all rows must keep source provenance visible in `source_runs`
- row ids and family labels must be preserved into the replay-side summary artifact
- duplicate sampling is allowed only after family quotas are satisfied

Multiplicity rule for the first attempt:

- `capture_available` rows: `2x`
- all other selected rows: `1x`

This is the smallest replay-side weighting change that reflects the corrected high-severity family distribution.

## Acceptance Criteria

Treat the branch as viable only if all of the following hold:

1. corrected-reference `capture_available` improves
2. at least one additional top family improves
3. arena does not regress below the current baseline

Concrete gate requirements:

- `capture_available` family:
  - lower average regret than incumbent on corrected-reference validation
  - lower corrected-failure count than the current baseline inventory
- one of:
  - `opening_plies_1_8`
  - `incumbent_proxy_disagreement`
  - `high_value_swing`
  - `high_imbalance`
  must also show lower average regret than incumbent
- overall corrected-reference validation:
  - must not increase corrected-failure count above `132`
- arena:
  - must beat or tie the current baseline; reject immediately on catastrophic collapse

## Explicit Non-Goals

Do not do any of the following in this branch:

- do not rebuild references again
- do not mine from old poisoned reference artifacts
- do not run a broad replay sweep with multiple unrelated weights
- do not change architecture
- do not change search defaults globally
- do not promote any model

## Practical Implementation

Smallest correct path:

1. generate one fresh corrected-reference forensic-suite report for the current artifact
2. convert the corrected-failure inventory plus that report into a mining-compatible corrected-only artifact
3. run `ml/alphazero_lite/mine_hard_states.py` on that corrected-only artifact
4. enforce family quotas before teacher labeling
5. build one replay dataset with `capture_available` rows duplicated `2x`
6. run one hard-state replay experiment from `ml/alphazero_lite/configs/aggressive_v3_targeted_hard_state_replay.json`
7. evaluate corrected-reference forensic metrics first, then arena

## Expected Deliverables

Produce:

- one corrected-only mining summary JSON
- one family-quota summary JSON
- one replay dataset summary JSON with per-family counts
- one corrected-reference forensic comparison report
- one final results note naming whether this exact branch should be kept or rejected

## Exit Criteria

This spec is complete when one exact next branch is defined tightly enough to answer one question:

Can a corrected-reference-only, family-balanced hard-state replay lane reduce the high-severity `capture_available` failures without collapsing overall strength?
