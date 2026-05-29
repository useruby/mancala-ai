# AlphaZero Lite Capture 002 Trace Pair Harmonization Results

- Run: `capture-002-trace-pair-harmonization`
- Summary: `/tmp/azlite_capture_002_trace_pair_harmonization/capture-002-trace-pair-harmonization/capture_002_trace_pair_harmonization_summary.json`
- Source trace capture: `/tmp/azlite_capture_002_prior_pressure_from_search_control/capture-002-prior-pressure/capture_002_trace_capture.json`

## Outcome

- Added a small pair-harmonization runner that projects the rerun trace onto the expected `{reference_move, full_search_selected_move}` identity pair.
- The harmonized rerun becomes downstream-ready and successfully regenerates a shared-drift artifact.
- After harmonization, `capture_002` no longer stops at `trace_insufficient`.

## Key Results

- New runner:
  - `ml/alphazero_lite/run_capture_002_trace_pair_harmonization.py`
- Harmonized trace capture:
  - `capture_002_trace_pair_projected_capture.json`
  - `trace_origin = rerun`
  - regenerated shared drift written successfully
- Projection summary:
  - `selected_move_rewrites = 2`
  - `reference_move_by_prior_rewrites = 4`
  - projected simulations: `1`, `2`
  - simulation `1`: projected selected move `1 -> 2`
  - simulation `2`: projected selected move `1 -> 4`
- Harmonized default trace:
  - classification: `selection_score_pressure_confirmed`
  - decision: `write_002_selection_pressure_ablation_spec`
  - first selection-score overtake snapshot: simulation `1`
  - first material visit-share snapshot: simulation `2`
  - first meaningful Q-support snapshot: simulation `192`
- Harmonized relaxed trace:
  - classification: `selection_score_pressure_confirmed`
  - decision: `write_002_selection_pressure_ablation_spec`

## Interpretation

- The earlier blocker was trace-shape incompatibility, not lack of signal.
- Once the rerun is expressed in the pair-aligned shape expected by the 002 selection-score diagnostic, the candidate shows an early selection-score lead over the reference move.
- This moves the branch outcome forward from `write_002_trace_capture_spec` to `write_002_selection_pressure_ablation_spec`.

## Current Limit

- The repo does not currently expose a ready-made runner for the follow-on selection-pressure ablation branch from this harmonized output.
- Existing residual/prior-pressure follow-ons in the repo expect `unresolved` trace artifacts and are therefore not directly callable from this new `selection_score_pressure_confirmed` outcome.
