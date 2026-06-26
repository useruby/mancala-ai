# AlphaZero-Lite Balanced Current Runtime PUCT Override Results

**Date**: 2026-06-26

**Classification**: `search_already_in_eval`

## Artifact Hash

- Current artifact: `model-artifact/current`
- Current artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Expected SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Benchmark Semantics Check

- Result: `search_already_in_eval`
- Benchmark file: `ml/alphazero_lite/run_opening_suite_seat_benchmark.py`
- Arena file: `ml/alphazero_lite/arena.py`
- Benchmark invocation: benchmark invokes arena with --root-policy-mode deterministic and challenger/current simulation budgets
- Arena behavior: arena move selection always constructs PUCT, runs search, and chooses the move via select_root_move(root, legal_moves) when legal moves exist
- Equivalence reason: The requested runtime override substitutes a PUCT move for a raw policy move at decision time, but deterministic opening-suite eval already makes decisions from runtime PUCT search rather than raw policy top-1.

Deterministic opening-suite evaluation already uses runtime PUCT root search for move choice. The proposed override would therefore be behaviorally identical to existing evaluation, so the experiment stops in Phase A by design.

## Override Controller Description

- Not implemented beyond semantics preflight.
- Reason: deterministic eval already selects moves from runtime PUCT, so a raw-policy-to-PUCT decision override does not create a distinct controller lane.

## Inputs

- Medium suite: `/tmp/azlite_opening_suite/medium_eval.jsonl`
- Fixed large suite: `/tmp/azlite_opening_suite/large_eval.jsonl`
- Held-out suites:
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl`
- Override budgets: `384,768,1200`
- Visit-share thresholds: `0.55,0.70`
- Budget pairs: `384:256,768:256,768:768,1200:1200,1200:256,256:768`
- Games per opening: `2`
- Workers: `24`
- Seed: `42`

## Override Audit Table

Not run because the semantics check classified the proposal as `search_already_in_eval`.

## Fixed Large DS Table

Not run because the semantics check classified the proposal as `search_already_in_eval`.

## Held-Out Mean And Worst-Suite DS Table

Not run because the semantics check classified the proposal as `search_already_in_eval`.

## Bootstrap CI

Not run because the semantics check classified the proposal as `search_already_in_eval`.

## P0/P1 Split For 384:256

Not run because the semantics check classified the proposal as `search_already_in_eval`.

## Duplicate Trajectory Counts

Not run because the semantics check classified the proposal as `search_already_in_eval`.

## Gate Classification

- `not_run` because the semantics check classified the proposal as `search_already_in_eval`.

## Runtime Cost Estimate

Not estimated because no distinct override controller was run. Existing deterministic benchmark cost already includes runtime PUCT search per move.
