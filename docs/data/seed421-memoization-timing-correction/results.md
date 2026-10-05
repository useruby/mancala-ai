# Seed421 memoization timing correction

**Decision: stop the memoization branch.** This is the protocol-correction
rerun of #420's exact 32-root retrospective cohort, artifact, budgets, RNG
contexts, capacity, search settings, repetitions, analysis and thresholds. It
does not add roots or capacity conditions. All 64 instrumented parity pairs
and all 192 timed pairs matched exactly; 512 searches completed. The original
#420 measurements remain in their original files, with the timing asymmetry
explained by `../seed420-artifact-evaluator-memoization/erratum-1.md`.

The corrected cache-off timing wrapper only forwards evaluator calls and
increments an integer counter. Detailed traces were collected only for the
untimed parity pairs. Both timed conditions had tracing disabled. External
timing covered complete root searches, including PUCT setup and the cache reset,
key construction, lookups and search work.

| Simulations | Mean off (ms) | Mean on (ms) | Mean paired relative speedup | Root-cluster 95% interval (384 only) | Off / on p95 (ms) |
|---:|---:|---:|---:|---:|---:|
| 384 | 30.645 | 32.656 | -6.25% | [-8.28%, -4.43%] | 33.927 / 37.001 |
| 1,536 | 124.466 | 129.416 | -3.36% | — | 144.454 / 153.583 |

At 384 simulations, mean paired speedup was -8.52% for the >32-stone phase and
-3.97% for the 17–32-stone phase. The 15% advancement threshold, positive
bootstrap lower bound, phase slowdown guardrail and nonnegative 1,536-budget
threshold are not met. The fixed rule therefore returns **stop**.

Timed cache totals at 384 were 1,308 hits, 33,978 misses/neural calls, zero
evictions and 35,286 requests in each condition. At 1,536 they were 9,492 hits,
123,459 misses/neural calls, zero evictions and 132,951 requests in each
condition. Full per-pair evidence, registration binding and analysis are in
`benchmark-ledger.jsonl`, `registration.json`, and `analysis.json`.

The comparison is performance-only on the fixed retrospective cohort and
recorded hardware/software configuration; it makes no playing-strength claim
and authorizes no production integration.
