# AlphaZero-lite Opening-Family Rule-Conditioned Redesign Spec

## Goal

Redesign the opening-family replay artifact construction so tracked opening rows are separated by extra-turn polarity instead of being mixed into one local family bucket.

This is still a non-training branch.

## Trigger

The row-pair rule-collision diagnostic concluded:

- best explanation: `extra_turn_overvaluation`
- recommendation: `rule_conditioned_policy_artifact_redesign`

Supporting artifacts:

- `/tmp/azlite_failure_family_diag/capture_002_003_rule_collision_diagnostic.json`
- `docs/alphazero-lite-capture-002-003-rule-collision-note.md`

## Problem Statement

The current opening-family artifact builders only target row identity and guard-role composition.

They do not distinguish between:

- opening rows whose correct move grants an extra turn
- opening rows whose correct move does not grant an extra turn

That matters because the current failure is structurally asymmetric:

- `capture_available-002` reference move `4` is a no-extra-turn move, but both finetuned variants still search-select extra-turn move `2`
- `capture_available-003` stays stable because its reference move `1` is already an extra-turn move

So the current opening-family grouping is too coarse. It mixes rows that require opposite tactical preferences under the extra-turn rule.

## Redesign Hypothesis

The next artifact should not teach one undifferentiated opening-capture family.

Instead, it should explicitly encode two opening subfamilies:

1. `opening_capture_extra_turn_reference`
2. `opening_capture_no_extra_turn_reference`

Operational expectation:

- extra-turn-reference rows should reinforce the preserved branch represented by `capture_available-003`
- no-extra-turn-reference rows should protect rows like `capture_available-002` from being washed out by extra-turn-biased local preferences

## Required Deliverables

Produce one new artifact-construction spec and one inspection artifact.

### 1. Artifact-Construction Change

Add a new builder or extend the tracked opening-family builder so it:

- reads the tracked opening rows from the shared reference artifact
- computes whether each row's `reference_move` grants an extra turn
- splits rows into explicit replay roles keyed by extra-turn polarity
- preserves exact teacher child stats and teacher-selected moves
- keeps existing guard rows, but does not relabel them as opening-family rows

Required replay-role names:

- `opening_capture_extra_turn_reference`
- `opening_capture_no_extra_turn_reference`

If additional tracked opening rows beyond `capture_available-005..008` are used, each added row must be labeled with one of those two replay roles using the teacher reference move, not the candidate selected move.

### 2. Inspection Artifact

Produce one summary artifact that reports:

- row count per replay role
- row ids per replay role
- teacher-selected-move distribution per replay role
- reference extra-turn polarity per row
- guard-row counts left unchanged

The purpose is to verify that no-extra-turn and extra-turn opening motifs are actually disentangled before any future lane is considered.

## Concrete Requirements

The redesign must:

- reuse the shared reference artifact at `/tmp/azlite_failure_family_diag/train_only_forensic_references.json`
- use the teacher `reference_move` to determine extra-turn polarity
- use the real rules engine, not an approximate heuristic
- keep row provenance visible in `source_runs`
- keep `teacher_child_stats` unchanged from the reference artifact
- avoid changing broad search/value/self-play settings
- avoid launching a new training lane in this branch

The redesign must not:

- merge extra-turn and no-extra-turn tracked opening rows back into one replay role
- infer polarity from the candidate searched move
- rebalance by changing architecture or search defaults
- promote any model

## Suggested Implementation Shape

Smallest correct path:

1. factor a move-polarity helper from `ml/alphazero_lite/capture_002_003_rule_collision_diagnostic.py` or reuse its rules-based move simulation
2. extend the tracked opening artifact builder so each tracked row receives the correct rule-conditioned replay role
3. extend the plus-guards builder so it preserves those replay roles instead of collapsing all tracked rows into `opening_capture_family`
4. emit a summary JSON artifact that proves the split
5. verify whether `capture_available-002` and `capture_available-003` are explicitly present in the resulting guard composition; if not, build them as a dedicated rule-collision guard artifact before any future lane

## Additional Follow-Up

That final coverage check is required because the rule-collision pair can still be missing even after the tracked opening family is split correctly.

If the pair is absent, the next artifact must include explicit guard roles:

- `rule_collision_extra_turn_reference_guard`
- `rule_collision_no_extra_turn_reference_guard`

using the exact rows:

- `capture_available-002`
- `capture_available-003`

## Exit Criteria

This branch is complete when:

1. a rule-conditioned opening-family artifact can be built deterministically from the shared references,
2. the resulting summary proves that extra-turn and no-extra-turn opening rows are separated into different replay roles, and
3. the report explicitly names that artifact redesign as the only justified precursor to any future training retry.
