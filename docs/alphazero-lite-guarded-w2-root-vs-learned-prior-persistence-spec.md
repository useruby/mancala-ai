# AlphaZero-lite Guarded w2 Root-vs-Learned Prior Persistence Spec

## Goal

Diagnose why root-only prior correction can flip guarded `w2` row `002`, but the learned guarded-`w2` policy/search stack does not retain that correction through final selection.

This is a non-training diagnostic branch.

## Trigger

The guarded `w2` root-vs-learned persistence review concluded:

- classification: `root_override_effective_but_training_nonpersistent`
- decision: `write_guarded_w2_root_vs_learned_prior_persistence_spec`

Supporting artifact:

- `/tmp/azlite_capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review/capture-002-003-guarded-w2-root-vs-learned-prior-persistence-review/capture_002_003_guarded_w2_root_vs_learned_prior_persistence_review.json`

## Problem Statement

The guarded `w2` base is responsive to root-only prior correction, but the corresponding learned prior-calibration retry only lifts row `002` prior support slightly and still search-selects move `2`.

That means the current gap is no longer just "does prior matter".

It is now:

why does the root-only corrected prior produce the desired selection, while the learned policy/search stack does not preserve that correction end to end?

## Exact Next Branch

- branch name: `guarded_w2_root_vs_learned_prior_persistence_capture`
- branch type: `non_training_diagnostic`
- base artifact: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- tracked rows: `capture_available-002, capture_available-003`
- required root interventions: `zero_wrong_extra_turn_prior, swap_reference_and_wrong`

## Required Deliverables

- one side-by-side artifact that compares original learned priors, root-overridden priors, and final visit/Q evolution on row 002
- one paired preservation readout showing why row 003 remains stable under root overrides
- one markdown note stating whether the persistence gap is best explained by backup dynamics, non-root policy mismatch, or another search-stage effect

## Concrete Questions

1. On row `002`, where does the root-only corrected trajectory diverge from the learned-policy trajectory: pre-expansion prior, early visit accumulation, child Q ranking, or later backup/selection score evolution?
2. Does the learned guarded-`w2` policy understate only the reference move, or does it also overstate the wrong extra-turn move in a way that root correction temporarily masks?
3. Why does row `003` remain preserved under the persistent root interventions while row `002` does not persist under learned prior pressure?

## Constraints

- Do not launch another training lane in this branch
- Do not change architecture, replay composition, or target modes
- Keep the guarded w2 artifact explicit in every evaluation command
- Compare at least the original prior and the persistent root interventions that held through budgets 384 and 1200

## Exit Criteria

This branch is complete when one diagnostic artifact and note explain where the persistence gap appears between root-only corrected search and the learned guarded-`w2` policy/search stack, or explicitly state what missing telemetry is still required.
