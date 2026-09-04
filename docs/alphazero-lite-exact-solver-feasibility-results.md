# Exact Solver Feasibility Results

**Classification:** `exact_teacher_not_feasible`

This standalone experiment implemented independent Kalah transitions and a full-depth alpha-beta solver with a bounded transposition table and persistent SQLite exact-value cache. It does not call the neural evaluator, use a heuristic leaf cutoff or PUCT, train, or integrate with production search.

## Validation

| Gate | Result |
| --- | --- |
| Golden rule vectors | 5 of 5 passed |
| Deterministic reachable states | 10,000 of 10,000 passed |
| Legal state transitions compared | 41,245 passed |
| Observed extra turns / captures / terminal sweeps | 5,777 / 3,524 / 485 |
| Independent brute-force tiny positions | 12 of 12 passed across 72 variant and repeat executions |

The transition comparison covered state, terminal status, winner, and settled player-zero final margin. Exactness variants covered ascending and descending move order, transposition-table sizes of 2, 17, and 1,000 entries, disabled cache, and repeat execution.

## Feasibility Corpus

The deterministic fresh corpus contains 96 training-ineligible standard-start reachable states. Every state had a fixed 30-second wall-clock limit. The environment terminated a single 48-minute sequential process before it could serialize its report, so the corpus was rerun as three independently checkpointed 32-state workers. Each worker retained its own persistent cache; no state completed exactly, so this sharding cannot account for the failed coverage threshold.

| Remaining stones | Exact / total | Rate | Required rate |
| --- | ---: | ---: | ---: |
| 17-24 | 0 / 32 | 0% | 100% |
| 25-32 | 0 / 32 | 0% | 75% |
| 33-40 | 0 / 32 | 0% | 25% |

No exact action-margin table or optimal-action set was produced for any corpus state. The projected label rate is therefore zero, below the 50,000 exact labels in 24 CPU-hours target.

## Decision

The rules and exactness gates passed, but the feasibility and dataset-scale gates failed decisively. This solver is not a practical independently stronger teacher at the required coverage, so it must not be used to generate training labels or motivate further ML experiments.

The merged runtime report is retained at `/tmp/opencode/exact_solver_feasibility_271.json` for the local execution record. The versioned summary is `docs/data/alphazero-lite-exact-solver-feasibility-summary.json`.
