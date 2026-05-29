# AlphaZero-lite Learned Policy vs Root-Corrected Prior Review Spec

## Goal

Diagnose exactly how the learned guarded-`w2` policy on row `002` differs from the root-corrected prior distributions that flip the row while preserving row `003`.

This is a non-training diagnostic branch.

## Trigger

The focused learned-policy capture concluded:

- classification: `learned_prior_underweights_reference_and_overweights_wrong_extra_turn`
- decision: `write_learned_policy_vs_root_corrected_prior_review_spec`

Supporting artifact:

- `/tmp/azlite_learned_policy_vs_root_corrected_prior_capture/learned-policy-vs-root-corrected-prior-capture/learned_policy_vs_root_corrected_prior_capture.json`

## Exact Next Branch

- branch name: `learned_policy_vs_root_corrected_prior_mismatch_capture`
- branch type: `non_training_diagnostic`
- artifact path: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- focus row: `capture_available-002`
- preservation row: `capture_available-003`
- required interventions: `original_prior, zero_wrong_extra_turn_prior, swap_reference_and_wrong`

## Required Deliverables

- one artifact that compares learned raw policy probabilities against root-corrected prior distributions for every legal move on row 002
- one ranking comparison showing how move order changes between learned policy, root-corrected prior, early selection score, and final visit share on row 002
- one preservation comparison confirming the same corrections do not destabilize row 003
- one markdown note stating whether the learned mismatch is best described as reference underweighting, wrong extra-turn overweighting, or both

## Constraints

- Do not launch another training lane in this branch
- Do not change architecture, replay composition, target modes, or search defaults
- Keep the guarded w2 artifact explicit in every diagnostic command
- Use the exact root-correction interventions that already flipped row 002 under aligned search-control settings

## Exit Criteria

This branch is complete when one diagnostic artifact and note make the learned-policy mismatch explicit enough to explain why row `002` needs root correction while row `003` does not, or explicitly state what missing telemetry remains.
