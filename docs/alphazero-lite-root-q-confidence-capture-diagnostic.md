# Root-Q Confidence Capture Diagnostic

## Inherited Result

Root-only parent-Q FPU did not repair uniform seeds 44/45, so this evaluates only the existing root visited-child Q confidence override with zero FPU.

## Baseline

`root_q_alpha_100` reproduced PR #298 before any intervention result was interpreted.

## Decision-Critical Results

| Seed | Family | Alpha | Move/regret on capture_available-002 |
| ---: | --- | ---: | --- |
| 44 | control | 1.00 | 2/0 |
| 44 | control | 0.75 | 2/0 |
| 44 | control | 0.50 | 1/12 |
| 44 | control | 0.25 | 1/12 |
| 44 | control | 0.00 | 1/12 |
| 44 | uniform1200 | 1.00 | 1/12 |
| 44 | uniform1200 | 0.75 | 1/12 |
| 44 | uniform1200 | 0.50 | 1/12 |
| 44 | uniform1200 | 0.25 | 1/12 |
| 44 | uniform1200 | 0.00 | 1/12 |
| 45 | control | 1.00 | 2/0 |
| 45 | control | 0.75 | 2/0 |
| 45 | control | 0.50 | 1/12 |
| 45 | control | 0.25 | 1/12 |
| 45 | control | 0.00 | 1/12 |
| 45 | uniform1200 | 1.00 | 1/12 |
| 45 | uniform1200 | 0.75 | 1/12 |
| 45 | uniform1200 | 0.50 | 1/12 |
| 45 | uniform1200 | 0.25 | 1/12 |
| 45 | uniform1200 | 0.00 | 1/12 |
| 46 | control | 1.00 | 2/0 |
| 46 | control | 0.75 | 2/0 |
| 46 | control | 0.50 | 2/0 |
| 46 | control | 0.25 | 2/0 |
| 46 | control | 0.00 | 1/12 |
| 46 | uniform1200 | 1.00 | 2/0 |
| 46 | uniform1200 | 0.75 | 2/0 |
| 46 | uniform1200 | 0.50 | 1/12 |
| 46 | uniform1200 | 0.25 | 1/12 |
| 46 | uniform1200 | 0.00 | 1/12 |

## Global Exact Safety

| Alpha | Accuracy | Mean regret | Blunder rate | Correction balance |
| ---: | ---: | ---: | ---: | ---: |
| 1.00 | 0.5884 | 3.0986 | 0.4116 | 328 |
| 0.75 | 0.5775 | 3.1633 | 0.4225 | 307 |
| 0.50 | 0.5488 | 3.4293 | 0.4512 | 252 |
| 0.25 | 0.5102 | 3.7913 | 0.4898 | 178 |
| 0.00 | 0.4173 | 4.8524 | 0.5827 | 0 |

## Classification

`root_q_confidence_not_causal`

Exactly one next experiment: inspect root-child Q error versus visit count on the failing capture traces; do not train a model.
