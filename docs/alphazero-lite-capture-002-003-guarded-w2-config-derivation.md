# Guarded w2 Prior-Calibration Config Derivation

## Outcome

- classification: `guarded_w2_is_best_supported_base`
- decision: `write_guarded_w2_prior_calibration_spec`
- recommended action: keep the guarded `w2` runtime config as the base and run the exact next branch defined in `docs/alphazero-lite-guarded-w2-prior-calibration-spec.md` instead of launching another broad sharpened-target lane

## Inputs

- `root_prior_summary`: `/tmp/azlite_capture_002_root_prior_intervention/capture-002-root-prior-intervention/root_prior_intervention_summary.json`
- `guarded_w2_gate`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1/capture_002_003_rule_conditioned_gate.json`
- `policy_target_gate`: `/tmp/azlite_v3_policy_target_local_versions/aggressive-v3-policy-target-local-iter1/capture_002_003_local_gate.json`
- `value_target_aligned_gate`: `/tmp/azlite_v3_value_target_aligned_local_versions/aggressive-v3-value-target-aligned-local-iter1/capture_002_003_local_gate.json`
- `policy_target_arena`: `/tmp/azlite_v3_policy_target_local_versions/aggressive-v3-policy-target-local-iter1/arena_report.json`
- `value_target_aligned_arena`: `/tmp/azlite_v3_value_target_aligned_local_versions/aggressive-v3-value-target-aligned-local-iter1/arena_report.json`
- `guarded_w2_runtime_config`: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/runtime_config.json`

## Comparative Evidence

- guarded `w2` stayed the best tracked-row base among the compared candidates
- policy-target local arena score: `0.5`
- value-target-aligned local arena score: `0.0`
- guarded `w2` row `002` reference visit share: `0.0547`
- policy-target local row `002` reference visit share: `0.0`
- value-target-aligned local row `002` reference visit share: `0.0`
- guarded `w2` row `003` selected move: `1`
- policy-target local row `003` selected move: `1`
- value-target-aligned local row `003` selected move: `2`

## Recommended Base

- base runtime config: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/runtime_config.json`
- exact next branch spec: `docs/alphazero-lite-guarded-w2-prior-calibration-spec.md`
- retain:
  - rule-conditioned opening full guarded replay artifact
  - replay_weight=2
  - sharpened policy-target mode
  - sharpened value-target mode
  - current deterministic search-control diagnostics bundle

## Constraints

- Any training change must be validated against capture_available-002 and capture_available-003 before arena
- Reject broad sharpened-target changes that lower row 002 reference prior support below guarded w2
- Reject any branch that changes row 003 searched selected move away from reference move 1
- Prefer changes that raise row 002 reference prior/probability without increasing row 002 selected-minus-reference Q margin

## Avoid

- Do not reuse the policy-target-local lane as the next base
- Do not reuse the value-target-aligned-local lane as the next base
- Do not broaden into generic value-target sharpening before guarded row-pair preservation is specified
