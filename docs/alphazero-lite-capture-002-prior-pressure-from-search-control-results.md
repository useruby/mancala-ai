# AlphaZero Lite Capture 002 Prior-Pressure From Search-Control Results

- Run: `capture-002-prior-pressure`
- Candidate: `/tmp/azlite_rule_conditioned_opening_full_guarded/rule-conditioned-opening-full-guarded/w2/versions/aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1`
- Summary: `/tmp/azlite_capture_002_prior_pressure_from_search_control/capture-002-prior-pressure/capture_002_prior_pressure_summary.json`

## Outcome

- The orchestration run completed after fixing subprocess import handling and adding decision-aware stop logic.
- The branch stopped early at the selection-score stage; it did not reach cadence, decomposition, metric, or prior-pressure component audits.

## Key Results

- `shared_full_search_drift.json`
  - classification: `shared_mechanism_disproved`
  - decision: `write_row_split_followup_spec`
- `capture_002_selection_score_trace.json`
  - classification: `trace_insufficient`
  - decision: `write_002_trace_capture_spec`
  - insufficiency reason: `trace_points_pair_mismatch`
- `capture_002_selection_score_trace_relaxed.json`
  - classification: `trace_insufficient`
  - decision: `write_002_trace_capture_spec`
  - insufficiency reason: `trace_points_pair_mismatch`
- `capture_002_trace_capture.json`
  - `artifact_write_summary.regenerated_shared_drift_skip_reason = trace_capture_not_downstream_ready`

## Interpretation

- This candidate does not yet provide a machine-checkable 002 selection-score trace under the current extracted trace shape.
- Relaxing the visit-share threshold from `0.05` to `0.04` does not change the result because the blocker is trace structure, not threshold strictness.
- The correct next artifact-producing branch from this outcome is still the trace-capture lane, not the downstream prior-pressure component audit.
