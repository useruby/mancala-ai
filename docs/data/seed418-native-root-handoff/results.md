# Seed418 native root-handoff feasibility diagnostic

Retrospective diagnostic only; these results are not strength, promotion, or
production-runtime evidence. The cohort contains 128 fixed, replay-verified
lane-A/challenger-seat-0 states from #416, at most one per opening. Each is the
first nonterminal decision at 17–21 active pit stones, deduplicated by canonical
state and selected by `SHA256("418:" + state_hash)` ordering.

Frozen seed455 was searched once per published case with 384 deterministic PUCT
simulations, `c_puct=1.25`, the established deterministic options, exact leaves
disabled, and the unchanged root-16 policy. The separate native query evaluated
every legal root action through `NativeExactRootTablebase`; all 128 states have
complete exact action coverage. Native action ties use highest legal network
prior, then lowest move index.

## Baseline search results

| Measure | Result |
| --- | ---: |
| Exact-optimal action rate | 80.47% |
| Mean final-margin regret | 0.84375 seeds |
| WDL-inferior choice rate | 3.125% |
| Warm native decision mean | 0.159 ms |
| Warm native decision p95 (nearest rank) | 0.257 ms |
| Native process startup | 0.268 ms |
| Native prewarm | 3,650.2 ms |

WDL-inferior means choosing a loss when a draw or win is available, or choosing
a draw when a win is available. Per-stone-count descriptive values and complete
per-case search/oracle records are in `analysis.json`, `search-records.jsonl`,
and `oracle-records.jsonl`.

| Active stones | N | Mean margin regret | Exact-optimal rate | WDL-inferior rate |
| ---: | ---: | ---: | ---: | ---: |
| 17 | 3 | 0.000 | 100.00% | 0.00% |
| 18 | 5 | 1.200 | 60.00% | 0.00% |
| 19 | 16 | 1.500 | 68.75% | 12.50% |
| 20 | 21 | 1.429 | 66.67% | 4.76% |
| 21 | 83 | 0.578 | 86.75% | 1.20% |

## Preregistered decision

Stop this branch. Although exact coverage is complete and latency is below both
operational limits, WDL-inferior choice rate (3.125%) is below 5% and mean
final-margin regret (0.84375) is below 1 seed. No threshold sweep, sample
extension, runtime experiment, training, arena game, or promotion is authorized
by this result.

The preregistration records an excluded preliminary attempt that used the
decision-state hash in the v2 opening-hash field. It was not used for these
results; the accepted run binds the actual #416 opening-state hash. A later
aggregation-only correction used the already-complete, paired execution
checkpoint and did not repeat any model search or native root query.
After execution, metric calculations were factored into the standalone pure
`seed418_analysis.py` module and the verifier was strengthened to check seed
contexts, tie selection, latency accounting, and native tier implementation.
The frozen per-case evidence was revalidated against that implementation; no
searches or root queries were repeated.
