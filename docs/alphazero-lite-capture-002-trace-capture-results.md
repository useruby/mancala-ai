# AlphaZero Lite Capture 002 Trace-Capture Results

- Source run: `capture-002-prior-pressure`
- Trace artifact: `/tmp/azlite_capture_002_prior_pressure_from_search_control/capture-002-prior-pressure/capture_002_trace_capture.json`

## Outcome

- The trace-capture lane was already exercised by the prior-pressure orchestration run.
- No additional repo runner exists for a deeper automated follow-on from this exact artifact state.

## Key Results

- Top-level trace-capture artifact remains not downstream-ready.
  - `trace_origin = insufficient`
  - `insufficiency_reasons = ["trace_points_pair_mismatch"]`
  - `artifact_write_summary.regenerated_shared_drift_written = false`
  - `artifact_write_summary.regenerated_shared_drift_skip_reason = trace_capture_not_downstream_ready`
- The embedded rerun does recover a final full-search trace to the selected move.
  - `rerun_trace.trace_origin = rerun`
  - `rerun_trace.insufficiency_reasons = ["trace_points_pair_mismatch"]`
  - `trace_diff_summary.final_trace_matches_full_search_selected_move = true`
- The rerun still does not preserve the extracted trace shape closely enough for downstream regeneration.
  - `trace_diff_summary.extracted_trace_point_count = 5`
  - `trace_diff_summary.final_trace_point_count = 4`
  - `trace_diff_summary.trace_origin_changed = true`
  - `trace_diff_summary.simulation_sequence_changed = true`

## Interpretation

- The candidate can be rerun deterministically to the same final selected move `2`, but the rerun trace does not match the extracted trace structure expected by downstream 002 diagnostic consumers.
- Because of that shape mismatch, the repo correctly refuses to regenerate a downstream-ready shared-drift artifact, and the branch cannot advance into cadence or prior-pressure component audit from this state.
