# Remove Rails Parity Path Design

## Goal

Remove the remaining Rails-backed parity path from the AlphaZero-lite codebase, including training pipeline hooks, configs, standalone parity tooling, and Rails-dependent tests/fixtures.

## Problem

The repository still contains a Rails parity integration even after the local ablation launcher scripts stopped carrying the `rules_parity_report` gate:

- `ml/alphazero_lite/parity_fuzz.py` shells out to `bin/rails runner`
- `ml/alphazero_lite/pipeline.py` still understands the `rules_parity_report` gate and the `rules_parity_fuzz` step relationship
- many AlphaZero-lite configs still define a `rules_parity_fuzz` step and `rules_parity_report` gate
- `ml/alphazero_lite/tests/test_kalah_rules_parity.py` still includes Rails-backed encoder parity tests
- `test/fixtures/ai/kalah_v3_parity_states.json` exists only to support Rails parity coverage
- docs still describe the Rails parity tooling as active

If the direction is to remove all remaining Rails connection from this area, leaving these pieces behind creates dead or misleading behavior.

## Recommended Approach

Make one coherent cleanup pass that removes the full Rails parity path instead of partially deprecating it.

Specifically:

- delete `ml/alphazero_lite/parity_fuzz.py`
- remove `rules_parity_report` handling from `ml/alphazero_lite/pipeline.py`
- remove `rules_parity_fuzz` steps and `rules_parity_report` gates from AlphaZero-lite configs
- delete Rails-dependent parity tests and fixtures
- keep Python-only rules coverage that does not depend on Rails
- update docs and tests so they describe the post-Rails state accurately

This keeps the repo internally consistent: either Rails parity exists and is supported end-to-end, or it is fully removed.

## Scope

This change will:

- delete `ml/alphazero_lite/parity_fuzz.py`
- remove pipeline support for `rules_parity_report`
- remove `rules_parity_fuzz` step definitions from AlphaZero-lite configs
- remove `rules_parity_report` gate entries from AlphaZero-lite configs
- remove Rails-backed tests from `ml/alphazero_lite/tests/test_kalah_rules_parity.py`
- delete `test/fixtures/ai/kalah_v3_parity_states.json` if it is no longer referenced
- update pipeline/config tests that currently expect `rules_parity_fuzz` or `rules_parity_report`
- update docs that still describe Rails parity tooling as part of the workflow

This change will not:

- remove Python-only Kalah rule/vector coverage
- change unrelated gates such as `perspective_audit_report` or `max_runtime_parity_delta`
- change search, training, arena, or bootstrap behavior outside the removal of the Rails parity step/gate
- introduce a replacement parity framework in the same change

## Behavior

- AlphaZero-lite configs should no longer contain `rules_parity_fuzz` steps.
- AlphaZero-lite configs should no longer contain `rules_parity_report` gates.
- Pipeline gate validation should no longer look for `rules_parity_report` artifacts or treat `rules_parity_fuzz` as a special skip case.
- Rails-dependent parity tests should be removed.
- Python-only rules tests should continue to cover rule correctness without invoking Rails.
- Documentation should no longer direct users to Rails parity tooling for AlphaZero-lite validation.

## Files Affected

Expected primary code/test/doc touch points:

- `ml/alphazero_lite/parity_fuzz.py`
- `ml/alphazero_lite/pipeline.py`
- `ml/alphazero_lite/test_pipeline.py`
- `ml/alphazero_lite/tests/test_kalah_rules_parity.py`
- `test/fixtures/ai/kalah_v3_parity_states.json`
- `docs/alphazero-lite-development.md`
- AlphaZero-lite config files under `ml/alphazero_lite/configs/` that still contain `rules_parity_fuzz` or `rules_parity_report`

## Testing

Update and run focused checks that prove:

- pipeline tests no longer expect the parity step/gate behavior
- config-oriented tests still pass after the parity step/gate removal
- standalone rules tests still pass without Rails-backed parity cases
- repo search in the targeted AlphaZero-lite area no longer finds active references to:
  - `rules_parity_fuzz`
  - `rules_parity_report`
  - `ml/alphazero_lite/parity_fuzz.py`
  - `bin/rails` in the removed parity tests/tooling path

## Success Criteria

- no AlphaZero-lite pipeline/config path still depends on Rails parity tooling
- `ml/alphazero_lite/parity_fuzz.py` is removed
- Rails-dependent parity tests/fixtures are removed
- Python-only rule coverage remains intact
- targeted tests pass
- docs reflect the new non-Rails state accurately
