# AlphaZero Lite Capture 002 Selection-Pressure Ablation Results

- Run: `capture-002-selection-pressure-ablation`
- Summary: `/tmp/azlite_capture_002_selection_pressure_ablation/capture-002-selection-pressure-ablation/capture_002_selection_pressure_ablation_summary.json`
- Source trace capture: `/tmp/azlite_capture_002_prior_pressure_from_search_control/capture-002-prior-pressure/capture_002_trace_capture.json`

## Outcome

- Added a focused runner for the missing `write_002_selection_pressure_ablation_spec` branch:
  - `ml/alphazero_lite/run_capture_002_selection_pressure_ablation.py`
- The tested search variants did not remove the harmonized early selection-score lead.
- Top-level result:
  - classification: `selection_pressure_persists`
  - decision: `stop_002_selection_pressure_ablation_inconclusive`

## Variants Tested

- `baseline_full`
- `value_only`
- `policy_only`
- `full_fpu_zero`
- `full_root_visit_count`
- `full_quality_off`

## Key Results

- Every tested variant still classified as:
  - `selection_score_pressure_confirmed`
  - decision: `write_002_selection_pressure_ablation_spec`
- Baseline harmonized trace:
  - first selection-score overtake at simulation `1`
  - first material visit-share support at simulation `2`
  - first meaningful Q-support at simulation `192`
- Strongest pressure-reducing variant tested (`full_quality_off`) still preserved the same ordering:
  - first selection-score overtake at simulation `1`
  - first material visit-share support at simulation `2`
  - first meaningful Q-support at simulation `384`
  - final selected-minus-reference selection-score margin shrank, but remained positive

## Interpretation

- The pair-harmonized `capture_002` signal is not explained away by disabling the main search-pressure knobs tested here.
- This suggests the selected-move advantage is robust under these search-option ablations and is not just a fragile artifact of one deterministic search setting bundle.
- The branch therefore stops as inconclusive for search-pressure relief rather than yielding a clean pressure-sensitive mechanism split.
