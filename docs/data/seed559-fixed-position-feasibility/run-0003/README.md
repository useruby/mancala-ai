# Seed559 fixed-position feasibility diagnostic

This is a retrospective diagnostic on first eligible positions reconstructed
from the authenticated #556 game archive. It uses newly selected positions,
not the roots already examined by #557 or #418. No training, arena games,
serving changes, or deployment were performed. Serving's cutoff remains 16.

## Cohort audit

| Quantity | Count |
| --- | ---: |
| Authenticated #556 games replayed | 2,048 |
| Eligible first-per-game roots | 1,903 |
| Eligible candidate occurrences matching #557 exclusions | 295 |
| Eligible candidate occurrences matching #418 exclusions | 0 |
| Eligible candidate occurrences matching both | 0 |
| Eligible unique canonical states before exclusion | 1,663 |
| Unique states excluded by #557 / #418 / both | 256 / 0 / 0 |
| Unique remaining canonical states | 1,407 |
| Distinct openings among remaining states | 941 |
| Selected roots | 512 |
| Selected source side P / T512 | 256 / 256 |

The 295 #557 exclusion matches are candidate occurrences: multiple games can
reach the same canonical root. Among 1,663 distinct eligible states, exactly
256 matched the #557 root set and none matched #418; this reconciles to 1,407
unique remaining states (`count-audit-supplement.json`). Selection uses the
same canonical-state hash as #557; exclusions are state-based. Candidates are
ranked with `SHA256("seed559:" + state_hash)` and
then hash order, with alternating source-side selection and no repeated
opening. The complete root list and provenance are in `cohort-manifest.json`.

## Frozen execution and results

`execution-freeze.json` binds the cohort, authenticated runtime artifacts,
native probe and tier-21 tablebase identities, search settings, routing
boundaries, latency protocol, and gate before any search or oracle query.
Search uses fresh 128-simulation PUCT trees, one worker, fixed P seed504 E4
policy and initializer value, with exact root and leaf solving disabled. The
separate oracle path chooses maximum exact final-score margin, then highest
legal P prior, then lowest move index.

| Baseline measure | Result | Paired opening-cluster 95% interval |
| --- | ---: | ---: |
| Mean exact-margin regret | 0.867 seeds | [0.672, 1.078] |
| WDL-inferior choice rate | 2.15% | [0.98%, 3.52%] |

| Active pit stones | Roots | Mean regret | WDL-inferior rate |
| ---: | ---: | ---: | ---: |
| 17 | 10 | 0.400 | 10.00% |
| 18 | 38 | 1.158 | 5.26% |
| 19 | 55 | 0.836 | 3.64% |
| 20 | 76 | 1.053 | 3.95% |
| 21 | 333 | 0.811 | 0.90% |

All 512 roots have complete exact legal-action oracle coverage. The oracle
decision has zero regret by construction. On the pinned CPU, warm baseline
decision latency was 15.07 ms mean / 16.70 ms p95. Warm exact-decision latency,
including P inference, the native action query, and the prescribed tie-break,
was 0.110 ms mean / 0.127 ms p95. Native process launch and prewarm are
reported separately as 1.60 ms and 3,877.88 ms respectively.

## Gate

**No cutoff-21 arena proposal.** Although root count, oracle coverage, and both
latency limits pass, baseline WDL-inferior rate (2.15%) is below 5%, and mean
regret (0.867 seeds) is below 1 seed. The registered gate therefore fails.
A future arena would require a separate proposal and is not supported by this
result. Routing-boundary checks confirm roots at or below 16 and above 21
remain on baseline behavior; the diagnostic's extended range is 17–21.

`search-records.jsonl`, `oracle-records.jsonl`, `timings.jsonl`, and
`root-metrics.jsonl` contain the per-root evidence. `analysis.json` contains
aggregates. Run the read-only verifier from the repository root:

```bash
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed559_fixed_position_feasibility \
  --data docs/data/seed559-fixed-position-feasibility/run-0003
```
