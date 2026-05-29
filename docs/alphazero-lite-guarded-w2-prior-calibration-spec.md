# AlphaZero-lite Guarded w2 Prior-Calibration Spec

## Goal

Define one exact next training branch that starts from the guarded `w2` runtime config and only targets the residual prior-pressure failure on `capture_available-002` while preserving `capture_available-003`.

This is a narrow follow-up spec, not a broad target-shaping retry.

## Trigger

The guarded-`w2` config derivation concluded:

- classification: `guarded_w2_is_best_supported_base`
- decision: `write_guarded_w2_prior_calibration_spec`

Supporting artifacts:

- `/tmp/azlite_capture_002_003_guarded_w2_config_derivation/capture-002-003-guarded-w2-config-derivation/capture_002_003_guarded_w2_config_derivation.json`
- `docs/alphazero-lite-capture-002-003-guarded-w2-config-derivation.md`
- `/tmp/azlite_capture_002_root_prior_intervention/capture-002-root-prior-intervention/root_prior_intervention_summary.json`

## Why This Branch

Current evidence says three things at once:

- guarded `w2` is the best tested base on the `002/003` row pair
- the mechanism is `policy_prior_sensitive`
- broader sharpened-target training changes made the tracked failure worse instead of better

Relevant evidence:

- guarded `w2` row `002`: `policy_reference_probability=0.0668`, `reference_move_visit_share=0.0547`, `searched_selected_move=2`, `selected_minus_reference_q_margin=0.0496`
- policy-target local row `002`: `policy_reference_probability=0.0002`, `reference_move_visit_share=0.0`, `selected_minus_reference_q_margin=0.7022`
- value-target-aligned local row `002`: `policy_reference_probability=0.0003`, `reference_move_visit_share=0.0`, `selected_minus_reference_q_margin=0.4342`
- value-target-aligned local row `003` regressed to searched selected move `2`

The root-prior intervention then isolated the actionable pattern on guarded `w2`:

- `uniform_legal_prior` flipped row `002` through `384` simulations while preserving row `003`
- `equalize_reference_and_wrong` also flipped row `002` through `384` simulations while preserving row `003`
- `swap_reference_and_wrong` was the most stable positive signal on row `002`, including at `1200` simulations, while row `003` remained preserved
- `force_reference_prior_advantage` also flipped row `002` at lower budgets, confirming that stronger root reference support is sufficient to change selection on this base

That means the next branch should try to calibrate the model's prior on the guarded `w2` base, not globally reshape value or policy targets again.

## Exact Next Branch

Run one new branch only:

- branch name: `guarded_w2_prior_calibration`
- base runtime config: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/runtime_config.json`
- branch type: one-iteration local training/eval lane

## Required Config Shape

Start from the guarded `w2` runtime config unchanged except for the minimum additions needed to bias training toward the row-`002` reference prior without broadening the lane.

Retain exactly:

- guarded fixed replay source composition
- replay weight `2`
- `policy-target-mode=sharpened`
- `value-target-mode=sharpened`
- self-play, arena, and MCTS1200 budgets
- deterministic evaluation/search-control bundle already used in diagnostics

Add exactly one new replay source class:

- a tiny rule-collision calibration replay artifact containing only:
  - `capture_available-002`
  - `capture_available-003`

Replay-source requirements:

- use `/tmp/azlite_failure_family_diag/train_only_forensic_references.json` as the source of truth
- preserve teacher-selected move and `teacher_child_stats` exactly
- repeat row `002` more heavily than row `003`
- keep row `003` present as an explicit preservation guard, not as the optimization target

Required weighting rule:

- existing guarded opening replay artifact remains at weight `2`
- new rule-collision calibration artifact weight must be `1`
- inside the new calibration artifact, row multiplicity must be:
  - row `002`: `3x`
  - row `003`: `1x`

This is the smallest supported approximation of the successful root-only interventions:

- strengthen row `002` reference prior support
- do not drop row `003`
- do not globally flatten all legal priors

## Acceptance Criteria

The branch is only considered viable if all of the following hold before any arena result is treated as meaningful:

1. `capture_available-002` must improve relative to guarded `w2`
2. `capture_available-003` must remain preserved

Concrete gate requirements:

- row `002`:
  - `policy_reference_probability > 0.0668`
  - `reference_move_visit_share > 0.0547`
  - `selected_minus_reference_q_margin <= 0.0496`
  - preferred success condition: `searched_selected_move=4`
- row `003`:
  - `searched_selected_move=1`
  - `reference_move_visit_share >= 0.6875`

If row `002` still selects move `2`, but the prior and visit-share metrics improve materially while row `003` holds, the branch may still be kept for one follow-up diagnostic comparison. Otherwise reject it.

## Explicit Non-Goals

Do not do any of the following in this branch:

- do not reuse the policy-target-local config as the base
- do not reuse the value-target-aligned-local config as the base
- do not change architecture
- do not change broad self-play temperature, value-loss weighting, or arena thresholds
- do not launch multiple new branches in parallel
- do not introduce root-only inference overrides into production evaluation

## Practical Implementation

Smallest correct path:

1. build a tiny calibration replay artifact from the shared forensic references with duplicated row `002` and single-copy row `003`
2. derive a new runtime config from guarded `w2` that adds that artifact as an extra fixed replay source at weight `1`
3. run one local iteration
4. evaluate the existing `002/003` local gate first
5. only then inspect arena/MCTS1200 results

Current generated setup artifacts:

- setup summary: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/setup_summary.json`
- runtime config: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/runtime_config.json`
- calibration artifact: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/capture_002_003_guarded_w2_prior_calibration_artifact.jsonl`
- calibration artifact summary: `/tmp/azlite_guarded_w2_prior_calibration_setup/guarded-w2-prior-calibration/capture_002_003_guarded_w2_prior_calibration_artifact_summary.json`

## Exit Criteria

This spec is complete when one exact next branch is defined and constrained enough that a future run can answer a single question:

Can a small guarded replay-side prior calibration lift row `002` above the guarded `w2` baseline without breaking row `003`?

## Execution Status

This exact branch has now been run.

- results note: `docs/alphazero-lite-guarded-w2-prior-calibration-results.md`
- outcome: rejected

Observed result:

- row `002` prior support improved but did not flip the searched selected move
- row `002` selected-minus-reference Q margin worsened versus guarded `w2`
- row `003` remained selected correctly but still failed the visit-share preservation gate
- arena collapsed to `0-120`

So this spec should not be rerun as-is. The next step should return to diagnostics, not another replay-side reweighting retry.
