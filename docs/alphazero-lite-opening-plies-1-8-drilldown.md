# AlphaZero-lite Opening Plies 1-8 Drilldown

## Scope

This drilldown uses the corrected-reference hard validation rows from:

- `/tmp/azlite_forensic_reference_rebaseline/forensic_suite_validation.json`

It does not rely on the mixed-signal inventory probe fields in isolation, because the inventory combines:

- fail/pass status from hard validation
- displayed selected move from a separate 384-simulation probe

For the opening-family diagnosis below, the source of truth is the hard validation artifact.

## Family Summary

- opening family rows: `48`
- corrected-reference failures: `33`
- bucket average regret: `0.0703`
- bucket value calibration MAE: `0.3503`

Primary output artifacts used for this summary:

- `/tmp/azlite_opening_plies_1_8_hard_validation_summary.json`
- `/tmp/azlite_opening_plies_1_8_move_feature_clusters.json`
- `/tmp/azlite_forensic_reference_rebaseline/opening_plies_subfamily_diagnostic.json`

Executable diagnostic:

- `".venv/bin/python" -m ml.alphazero_lite.diagnose_opening_plies_subfamilies --forensics /tmp/azlite_forensic_reference_rebaseline/forensic_suite_validation.json --out /tmp/azlite_forensic_reference_rebaseline/opening_plies_subfamily_diagnostic.json`

## Dominant Failure Clusters

Largest move-pair clusters from hard validation:

- reference `4` -> selected `2`: `8` rows
- reference `4` -> selected `5`: `6` rows
- reference `1` -> selected `2`: `4` rows
- reference `1` -> selected `5`: `3` rows
- reference `2` -> selected `5`: `2` rows
- reference `3` -> selected `2`: `2` rows
- reference `3` -> selected `5`: `2` rows
- reference `5` -> selected `2`: `2` rows

Reference-move concentration:

- corrected reference `4`: `15` failing rows
- corrected reference `1`: `8` failing rows
- corrected reference `3`: `4` failing rows
- corrected reference `2`: `3` failing rows
- corrected reference `5`: `3` failing rows

## Mechanism Readout

The family is not one monolithic failure mode. It is dominated by three recurring patterns.

Pattern A: non-extra-turn reference replaced by extra-turn move `2`

- count: `7` rows in the cleanest cluster
- representative rows:
  - `opening_plies_1_8-004`
  - `opening_plies_1_8-007`
  - `opening_plies_1_8-013`
  - `opening_plies_1_8-020`
  - `opening_plies_1_8-021`
  - `opening_plies_1_8-022`
  - `opening_plies_1_8-050`
- shape:
  - corrected reference move is usually `4`
  - selected move is `2`
  - corrected move does not grant an extra turn
  - selected move does grant an extra turn
- average regret: `0.0306`
- average value error: `0.4233`

Pattern B: corrected extra-turn continuations are missed

- representative rows:
  - `opening_plies_1_8-009`
  - `opening_plies_1_8-010`
  - `opening_plies_1_8-015`
  - `opening_plies_1_8-071`
  - `opening_plies_1_8-072`
- shape:
  - corrected reference move is usually `1`
  - selected move is usually `2` or `5`
  - corrected move grants an extra turn
  - selected move does not
- average regret is materially higher than Pattern A:
  - `1 -> 2`: `0.1246`
  - `1 -> 5`: `0.1499`

Pattern C: long-edge preference toward move `5`

- representative rows:
  - `opening_plies_1_8-014`
  - `opening_plies_1_8-024`
  - `opening_plies_1_8-025`
  - `opening_plies_1_8-054`
  - `opening_plies_1_8-056`
  - `opening_plies_1_8-070`
- shape:
  - corrected reference move is usually `4`
  - selected move is `5`
  - neither move grants an extra turn in this cluster
- average regret: `0.1346`
- average value error: `0.2910`

## Tactical Bias Summary

Across all `33` opening failures, extra-turn behavior splits as follows:

- corrected move non-extra-turn, selected move non-extra-turn: `14`
- corrected move non-extra-turn, selected move extra-turn: `12`
- corrected move extra-turn, selected move non-extra-turn: `6`
- corrected move extra-turn, selected move extra-turn: `1`

Interpretation:

- there is a real bias toward some extra-turn continuations in early opening states
- but the family is not explained by "always over-prefers extra turns"
- a second independent bias prefers edge move `5` in several early symmetric or near-symmetric opening shapes
- the value head is also implicated: many of the largest clusters carry substantial value error even when regret is modest

## Key Implication

The `opening_plies_1_8` family should not be treated as one generic opening bucket anymore.

The current evidence supports splitting it into at least these diagnostic subfamilies:

- opening extra-turn overbias
- opening missed extra-turn continuation
- opening edge-move-5 preference

The executable subfamily report currently splits the `33` failures into:

- `opening_extra_turn_overbias`: `12`
- `opening_edge_move_5_preference`: `10`
- `opening_missed_extra_turn_continuation`: `6`
- `opening_other_mismatch`: `5`

## Executed Subfamily Run

The first subfamily branch was executed on a corrected `opening_plies_1_8` artifact instead of the stale opening-capture artifact.

New artifact build and merge outputs:

- `/tmp/azlite_failure_family_diag/opening_plies_policy_artifact.jsonl`: `33` opening-family rows
- `/tmp/azlite_failure_family_diag/opening_plies_policy_artifact_summary.json`
- `/tmp/azlite_failure_family_diag/opening_plies_family_full_guarded_artifact.jsonl`: `35` rows total
- `/tmp/azlite_failure_family_diag/opening_plies_family_full_guarded_artifact_summary.json`

Executed corrected run:

- `".venv/bin/python" -m ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment --run-id opening-extra-turn-overbias-corrected --artifact-path /tmp/azlite_failure_family_diag/opening_plies_family_full_guarded_artifact.jsonl --opening-subfamily-diagnostic /tmp/azlite_forensic_reference_rebaseline/opening_plies_subfamily_diagnostic.json --opening-subfamily opening_extra_turn_overbias --output-root /tmp/azlite_rule_conditioned_opening_full_guarded --current-path storage/ai/alphazero_lite/current`

Observed filtered artifact contents:

- `12` `opening_extra_turn_overbias` rows
- `2` preserved guard rows: `capture_available-002`, `capture_available-003`
- `14` rows total after filtering

Result:

- both `w1` and `w2` were `reject_local_gate`
- both variants retained the `002/003` guard rows
- the blocking reason remained `row_002_local_rule_collision_persists`
- no arena or `MCTS1200` follow-through was reached because the local gate failed first

Key implication:

- the earlier empty-run result was an artifact mismatch only
- the corrected rerun confirms a real branch result: `opening_extra_turn_overbias` alone does not clear the `capture_available-002/003` gate

Primary run artifacts:

- `/tmp/azlite_rule_conditioned_opening_full_guarded/opening-extra-turn-overbias-corrected/experiment_summary.json`
- `/tmp/azlite_rule_conditioned_opening_full_guarded/opening-extra-turn-overbias-corrected/filtered_opening_subfamily/opening_extra_turn_overbias_summary.json`

## Recommended Next Action

Continue opening-family diagnostics with the same corrected artifact path, but treat `capture_available-002/003` as the active blocker on any guarded branch until search-prior control or guard-specific changes stop the local collision.

Integration point now available:

- `ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment` accepts `--opening-subfamily-diagnostic` and records the grouped row ids in its manifest so follow-on opening-family runs can be anchored to the executable split instead of the handwritten note.
- `ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment` also accepts `--opening-subfamily` and will materialize a filtered opening-family artifact for a single subfamily before running variants.

Corrected filtered run shape:

- `".venv/bin/python" -m ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment --run-id opening-extra-turn-overbias-corrected --artifact-path /tmp/azlite_failure_family_diag/opening_plies_family_full_guarded_artifact.jsonl --opening-subfamily-diagnostic /tmp/azlite_forensic_reference_rebaseline/opening_plies_subfamily_diagnostic.json --opening-subfamily opening_extra_turn_overbias --output-root /tmp/azlite_rule_conditioned_opening_full_guarded --current-path storage/ai/alphazero_lite/current`
