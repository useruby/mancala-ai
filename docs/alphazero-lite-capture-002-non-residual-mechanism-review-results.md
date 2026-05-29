# AlphaZero Lite Capture 002 Non-Residual Mechanism Review Results

- Run: `capture-002-non-residual-mechanism-review`
- Summary: `/tmp/azlite_capture_002_non_residual_mechanism_review/capture-002-non-residual-mechanism-review/capture_002_non_residual_mechanism_review_summary.json`
- Source selection-pressure ablation summary: `/tmp/azlite_capture_002_selection_pressure_ablation/capture-002-selection-pressure-ablation/capture_002_selection_pressure_ablation_summary.json`

## Outcome

- Added the missing follow-on review branch for the `selection_pressure_persists` state:
  - `ml/alphazero_lite/capture_002_non_residual_mechanism_review.py`
  - `ml/alphazero_lite/run_capture_002_non_residual_mechanism_review.py`
- Top-level result:
  - classification: `stable_non_residual_selection_advantage`
  - decision: `write_002_non_residual_mechanism_review_spec`

## Key Results

- The review is grounded on the real selection-pressure ablation summary, not on a fresh replay or new search sweep.
- `6` variants were tested in the source ablation run.
- `4` variants were downstream-valid for mechanism review:
  - `baseline_full`
  - `full_fpu_zero`
  - `full_root_visit_count`
  - `full_quality_off`
- `0` downstream-valid variants relieved the early selection-score lead.
- The branch therefore records that the observed `capture_available-002` pressure signal is stable across the tested non-residual search-option changes.

## Interpretation

- This branch closes the repo gap that previously existed after `selection_pressure_persists`.
- The strongest supported statement from current evidence is not that every attempted variant behaved identically, but that every variant which still produced a downstream-valid paired trace preserved the same mechanism-level pressure outcome.
- From the current frontier, the signal is best described as a stable non-residual selection advantage rather than a search-option-sensitive artifact.
