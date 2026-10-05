# Seed420 artifact-evaluator memoization

**Decision: stop the memoization branch.** All policy/value and PUCT-output
equivalence checks passed. The 384-simulation mean paired relative speedup was
**2.03%**, below the preregistered 15% threshold. Its 10,000-resample,
root-cluster 95% percentile interval was **[0.49%, 3.72%]**. The 1,536 budget
mean paired relative speedup was **4.37%**. No playing-strength claim is made;
this retrospective cohort measures performance only.

## Frozen execution

- 32 unique roots: 16 with more than 32 active pit stones, then 16 with 17–32;
  roots are from #416's published A-lane, challenger-seat-zero trajectories.
- The original seed455 artifact matched all published file hashes: weights
  `f06e3e1e…54551d00c`, metadata `8b9f4d02…f6c649ac`, and search-policy sidecar
  `b13ced03…944b08fb8d`.
- Search budgets were 384 and 1,536, `c_puct=1.25`, deterministic root policy,
  zero FPU, subtree reuse and value normalization disabled, no root noise,
  zero tactical bias, zero temperature, exact leaves disabled, and capacity
  fixed at 4,096.
- Each root/budget had one instrumented cache-off/on pair and three timed pairs.
  Timed order alternated off/on, on/off, off/on. Each pair shared its RNG seed;
  every cached search began with an empty cache.
- Completed work: 64 instrumented pairs (128 searches) and 192 timed pairs
  (384 searches), **512 searches total**.
- Hardware: Intel Core i9-13900; Linux x86_64; Python 3.14.7; NumPy 2.5.3.
  OMP, OpenBLAS, MKL, and NumExpr thread-count environment variables were unset.
  Warmup included artifact load, numerical runtime import, and one network
  evaluation before timings. Timings include complete root-search setup, cache
  reset, key construction, lookups, and PUCT; detailed request tracing was
  confined to untimed parity runs.

## Results

| Simulations | Mean off (ms) | Mean on (ms) | Mean paired speedup | Off p95 (ms) | On p95 (ms) | Cache hits | Misses / neural calls | Evictions | Peak entries |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 384 | 32.588 | 32.031 | 2.03% | 36.012 | 36.131 | 1,308 | 33,978 | 0 | 385 |
| 1,536 | 132.070 | 126.898 | 4.37% | 152.571 | 151.037 | 9,492 | 123,459 | 0 | 1,525 |

The 384-budget paired speedups by fixed phase were **0.32%** (>32 stones) and
**3.74%** (17–32 stones). Neither phase exceeded the 5% slowdown guardrail. The
1,536 mean was nonnegative. Advancement nevertheless fails because the required
384-budget speedup was not at least 15%.

Timed evaluation requests were 35,286 at 384 and 132,951 at 1,536 in each
condition. Cache hits avoided 1,308 and 9,492 underlying neural calls
respectively. The complete per-root medians, paired speeds, phase details,
request counts, and ledger bindings are in `analysis.json` and
`benchmark-ledger.jsonl`.

The cohort is retrospective and is not a strength holdout. This benchmark
supports only the measured performance conclusion on these roots and this
hardware/software configuration. It authorizes no production integration.
