# AlphaZero-lite Capture 002/003 Search-Prior Control Results

## Outcome

Ran the real search-prior control evaluation with deterministic search-control settings:

- `fpu_mode=parent_q`
- `reuse_subtree=true`
- `normalize_values=true`
- `root_policy_mode=deterministic`
- `tactical_root_bias=0.1`

Artifacts:

- summary: `/tmp/azlite_capture_002_003_search_prior_control/capture-002-003-search-prior-control/search_prior_control_summary.json`
- `w1` gate: `/tmp/azlite_capture_002_003_search_prior_control/capture-002-003-search-prior-control/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w1-iter1/capture_002_003_search_prior_control_gate.json`
- `w2` gate: `/tmp/azlite_capture_002_003_search_prior_control/capture-002-003-search-prior-control/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1/capture_002_003_search_prior_control_gate.json`

## Result

The search-prior control did not clear the `002/003` local gate.

- both candidates still search-selected move `2` on `capture_available-002`
- both candidates increased `002` reference-move visit share versus incumbent
- but the wrong extra-turn move still dominated visits on `002`
- `w1` also regressed `003`
- `w2` preserved `003`, but still failed overall because `002` was not fixed

## Results Table

| variant | row_002_reference_move | row_002_searched_move | row_002_reference_visit_share | row_002_wrong_extra_turn_visit_share | row_002_gate_pass | row_003_reference_move | row_003_searched_move | row_003_reference_visit_share | row_003_gate_pass | decision | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `w1` | `4` | `2` | `0.1432` | `0.8047` | `false` | `1` | `1` | `0.9115` | `false` | `reject_local_gate` | `002` reference support improved over incumbent `0.0026`, but wrong extra-turn move still dominated and `003` degraded from `0.9766` to `0.9115` |
| `w2` | `4` | `2` | `0.1458` | `0.7318` | `false` | `1` | `1` | `0.9401` | `true` | `reject_local_gate` | `002` reference support improved over incumbent `0.0026`, `003` stayed stable, but `002` still did not flip away from move `2` |

## Interpretation

- deterministic search control helps local visit-share support for the `002` reference move
- that improvement is not enough to change the final searched move on `002`
- the unresolved failure now looks narrower than generic search-prior drift
- the remaining problem is consistent with `002`-specific selection-score / prior-pressure dominance that survives these search-control changes

## Recommendation

Next action: **run the existing `capture_002` selection-score trace and prior-pressure component audit branch**.

Why this is now the best next step:

- replay redesign did not fix `002`
- deterministic search-prior control improved `002` reference support but still did not flip search selection
- that is exactly the kind of residual failure the repo’s `capture_002_selection_score_trace.py` and `capture_002_prior_pressure_component_audit.py` tooling is designed to isolate
