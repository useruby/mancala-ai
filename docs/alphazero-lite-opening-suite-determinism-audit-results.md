# AlphaZero-Lite Opening-Suite Determinism Audit Results

**Date**: 2026-06-13
**Classification**: `evaluator_duplicate_deterministic`
**Schema**: `azlite_opening_suite_determinism_audit_v1`

## Summary

Audited duplicate aliases of a single candidate artifact against `model-artifact/current` under deterministic root-policy opening-suite evaluation.
 Source artifact weights SHA256: `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad`.

## Conditions

| Condition | Workers | Candidate order | Seed | Alias match |
|-----------|---------|-----------------|------|-------------|
| `workers_1_normal_seed_42` | 1 | normal | 42 | PASS |
| `workers_24_normal_seed_42` | 24 | normal | 42 | PASS |
| `workers_24_reversed_seed_42` | 24 | reversed | 42 | PASS |
| `workers_24_normal_seed_43` | 24 | normal | 43 | PASS |

## Cross-Condition Checks

| Check | Result |
|------|--------|
| `workers_1_vs_24_normal_seed42` | PASS |
| `workers_24_normal_vs_reversed_seed42` | PASS |
| `workers_24_seed42_vs_seed43` | PASS |

## Before Fixes

| Issue | Symptom |
|------|---------|
| `worker_shared_games_jsonl` | workers>1 could leave merged seat metrics reading incomplete or last-writer-wins game rows |
| `deterministic_mode_ignored_cli_seed` | deterministic root-temperature runs always used the first --seeds entry instead of --seed |

## Root Cause

During the audit, `arena.py` was corrected so each worker writes its own temporary `games.jsonl` and the parent process merges rows by `game_index`. The previous worker-shared path allowed last-writer-wins row loss under multiprocessing, which could corrupt per-game diagnostics and seat-aware metrics derived from `games.jsonl`.
`run_opening_suite_seat_benchmark.py` was also corrected so deterministic root-temperature runs honor `--seed` instead of silently reusing the first `--seeds` value.

## Artifacts

- Report JSON: `/tmp/azlite_opening_suite_determinism_audit/determinism_audit_report.json`
- Audit workdir: `/tmp/azlite_opening_suite_determinism_audit`
- Source artifact: `/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2`
