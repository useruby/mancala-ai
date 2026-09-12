# Root Q/Visit Capture Audit

## Inherited Inputs

PR #299 machine artifact SHA256: `847ebccb4c6a444ef35d8205c99290f3fb450c5ae2397e5b2d3b26d941cf7d67`. PR #298 attribution SHA256: `fc339be30a8feefa0f9b68d7317e44db56aab4d5b26bf71f3d84c7909d721bf9`.

## Baseline Reproduction

The frozen root-Q-alpha 1.00, zero-FPU production configuration reproduced selected action, regret, visits, and child Q (1e-6) for both decision-critical roots and all six primary checkpoints before telemetry was interpreted.

## Decision-Critical Results

| Seed | Family | State | Stable Q | Stable visits | Q-to-visit lag | Final visit/Q regret | Mechanism |
| ---: | --- | --- | ---: | ---: | ---: | --- | --- |
| 44 | control | capture_available-002 | 46 | 174 | 128 | 0/0 | `search_correct` |
| 44 | control | capture_available-020 | None | 366 | None | 0/4 | `search_correct` |
| 44 | uniform1200 | capture_available-002 | None | None | None | 12/12 | `persistent_q_ranking_error` |
| 44 | uniform1200 | capture_available-020 | None | None | None | 6/4 | `persistent_q_ranking_error` |
| 44 | parent | capture_available-002 | None | None | None | 18/6 | `persistent_q_ranking_error` |
| 44 | parent | capture_available-020 | None | None | None | 6/10 | `persistent_q_ranking_error` |
| 45 | control | capture_available-002 | 52 | 127 | 75 | 0/0 | `search_correct` |
| 45 | control | capture_available-020 | None | None | None | 4/4 | `persistent_q_ranking_error` |
| 45 | uniform1200 | capture_available-002 | 232 | None | None | 12/0 | `visit_hysteresis_after_q_recovery` |
| 45 | uniform1200 | capture_available-020 | None | None | None | 6/4 | `persistent_q_ranking_error` |
| 45 | parent | capture_available-002 | None | None | None | 18/6 | `persistent_q_ranking_error` |
| 45 | parent | capture_available-020 | None | None | None | 6/10 | `persistent_q_ranking_error` |
| 46 | control | capture_available-002 | 38 | 43 | 5 | 0/0 | `search_correct` |
| 46 | control | capture_available-020 | 307 | None | None | 6/0 | `visit_hysteresis_after_q_recovery` |
| 46 | uniform1200 | capture_available-002 | 96 | 96 | 0 | 0/0 | `search_correct` |
| 46 | uniform1200 | capture_available-020 | None | None | None | 4/4 | `persistent_q_ranking_error` |
| 46 | parent | capture_available-002 | None | None | None | 18/6 | `persistent_q_ranking_error` |
| 46 | parent | capture_available-020 | None | None | None | 6/10 | `persistent_q_ranking_error` |

## Capture-002 Timelines

The machine artifact stores all 384 post-backup root-child rows, including pre-selection PUCT components, selected edge, and Q/visit leaders. Q is evaluated only as an ordering; it is never numerically compared with exact margins.

| Seed | Family | Q-correct/visit-wrong interval | Q rank reversals A2/A1 | A2-A1 Q zero crossings | Early debt at 128 | Final debt |
| ---: | --- | --- | ---: | --- | ---: | ---: |
| 44 | control | 2-173 (132) | 2 | [5, 40] | None | None |
| 44 | uniform1200 | None-None (0) | 0 | [] | 55 | 150 |
| 45 | control | 2-126 (80) | 6 | [5, 6, 7, 41, 42, 48] | None | None |
| 45 | uniform1200 | 36-384 (236) | 5 | [36, 64, 123, 209, 211] | 59 | 110 |
| 46 | control | 2-42 (6) | 2 | [3, 38] | None | None |
| 46 | uniform1200 | 4-27 (13) | 5 | [4, 12, 23, 45, 48] | None | None |

## Capture And Global Validation

The artifact retains PR #299 sparse snapshots for every exact capture root and every exact forensic root. These are secondary final-snapshot checks; full-resolution tracing is restricted to the mechanically selected decision-critical set.

| Seed | Family | Capture Q-top | Capture visit-top | Q-correct/visit-wrong | Both wrong |
| ---: | --- | ---: | ---: | ---: | ---: |
| 44 | control | 0.708 | 0.583 | 0.292 | 0.125 |
| 44 | uniform1200 | 0.500 | 0.542 | 0.125 | 0.333 |
| 45 | control | 0.708 | 0.583 | 0.125 | 0.292 |
| 45 | uniform1200 | 0.625 | 0.542 | 0.208 | 0.250 |
| 46 | control | 0.667 | 0.542 | 0.208 | 0.250 |
| 46 | uniform1200 | 0.625 | 0.667 | 0.125 | 0.208 |

Global exact final snapshots are also aggregated in the machine artifact, including Q-argmax minus visit-argmax exact regret. These use decision ordering/regret only, never a numeric Q-to-exact calibration.


## Classification

`capture_failure_mixed_q_and_visit`

Exactly one next experiment: inspect subtree evaluator/backup error for the persistent-Q case FIRST, because changing visit allocation cannot fix a genuinely wrong Q ranking.
