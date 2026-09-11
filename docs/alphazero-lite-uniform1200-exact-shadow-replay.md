# Uniform1200 Exact Shadow Replay

## Matched PR #287 Results

| Seed | Exact-set delta | Exact regret delta | Exact blunder delta | Old / shadow decision |
| ---: | ---: | ---: | ---: | --- |
| 44 | +0.0845 | -1.0235 | -0.0845 | fail / fail |
| 45 | +0.1268 | -1.2489 | -0.1268 | fail / fail |
| 46 | +0.1126 | -1.0141 | -0.1126 | fail / pass |

## Other Historical Candidates

| Candidate | Exact shadow passes |
| --- | ---: |
| PR #288 rowmatched | 2/3 |
| PR #290 B | 2/3 |
| PR #290 C | 2/3 |
| PR #290 D | 1/3 |

## Decision-Changing Positions

| Seed | Old-reference false regressions | Genuine regressions | Hidden regressions |
| ---: | ---: | ---: | ---: |
| 44 | 17 | 16 | 6 |
| 45 | 23 | 14 | 7 |
| 46 | 18 | 12 | 5 |

Largest decision-changing rows, including exact action sets and values, are recorded in `decision_flips` in the JSON artifact.

## Hard Classification

`uniform1200_exact_regression_confirmed`

## One Next Experiment

return to the exact failure families and identify the dominant training mechanism using the now-valid exact oracle.
