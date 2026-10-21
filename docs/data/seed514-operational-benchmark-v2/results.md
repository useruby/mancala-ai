# Seed514 operational benchmark results

**Interpretation:** local diagnostic measurements on this host only. The cohort is examined seed512 trajectory data, not a holdout. No operational threshold was set or tested.

## Protocol and evidence

- Baseline: complete frozen initializer.
- P: seed504 E4 policy output with its own trunk, composed with frozen initializer value output and its own trunk.
- Existing NumPy/native inference and `evaluate_artifact_position` dispatch; 1,200 simulations, `c_puct=1.25`, one synchronous worker, deterministic root policy, exact-root threshold 16, exact leaf solving disabled.
- 64 unique states (32 with >32 active stones, 16 with 17–32, 16 with ≤16); one warm-up and five timed repetitions per config/state; fixed shared state seeds and deterministic alternating order.
- Complete search-call latency includes exact-root dispatch. Memory snapshot ran separately in a fresh process with no search, tracing, or profiler.
- Cohort SHA-256: `7ce8d73a80cf61db9cfde9c6c213d80a491bbf5c7686a1a175a6ffe44ea2974f`.
- Raw measurements SHA-256: `af7533aad75e150402bf960dfe22c2b488d692b8f3ad4a3dd55d45d55a4f7b15`.
- Protocol freeze SHA-256: `7c2f1bcc682e98505a0a4e7c4423094e932ce0dbcf887606b465fc5bf644f88f`.
- Source/runtime identities, commands, dependencies, hardware, and thread settings: `evidence.json` and `protocol-freeze.json`.

## Results

| Cohort | Configuration | Searches | p50 | p95 | Throughput |
| --- | --- | ---: | ---: | ---: | ---: |
| All | Baseline | 320 | 106.2 ms | 8.071 s | 0.378 searches/s |
| All | P | 320 | 171.8 ms | 7.907 s | 0.373 searches/s |
| >32 stones | Baseline | 160 | 107.7 ms | 114.3 ms | 9.27 searches/s |
| >32 stones | P | 160 | 174.8 ms | 185.3 ms | 5.75 searches/s |
| 17–32 stones | Baseline | 80 | 98.8 ms | 105.2 ms | 11.20 searches/s |
| 17–32 stones | P | 80 | 153.5 ms | 166.5 ms | 7.44 searches/s |
| ≤16 stones, exact-root | Baseline | 80 | 700.3 ms | 90.54 s | 0.0974 searches/s |
| ≤16 stones, exact-root | P | 80 | 696.1 ms | 90.29 s | 0.0975 searches/s |

Across paired observations, P/baseline latency-ratio p50 was 1.582 and p95 was 1.669. Median per-state ratios ranged from 0.980 to 1.717. Per-bucket paired-ratio p50/p95: >32 stones 1.621/1.682; 17–32 stones 1.542/1.679; ≤16 stones 1.016/1.584. Each configuration used 80 exact-root and 240 PUCT paths. PUCT-only strata show material latency cost; exact-root tails dominate the all-state p95 and aggregate throughput.

Evaluator loading was 13.9 ms baseline and 27.1 ms P, including both P source evaluators (13.1 ms incremental for E4). The separate fresh-process model-memory snapshot was 42,070,016 bytes baseline and 43,028,480 bytes P combined resident memory (958,464 bytes incremental). It excludes transient search allocations and native exact-root tablebase working memory.

The host was a 13th Gen Intel Core i9-13900 with Linux 7.2.6, Python 3.14.7, NumPy 2.5.2, and Torch 2.13.0+cu130. Thread environment variables for OpenMP, OpenBLAS, MKL, and NumExpr were 1; Torch reported intra-op 1 and inter-op 24. The measurements are not a production SLO result or pass/fail decision.

## Files

- `cohort.json`: selected full states, identities, source games/plies, and hash-order keys.
- `protocol-freeze.json`: finalized protocol and source/archive/artifact bindings.
- `raw-measurements.jsonl`: all 640 timed searches and per-search exact-root/PUCT dispatch accounting.
- `evidence.json`: measured summary, raw measurements' environment, identities, model-load and process memory details.
- `memory-snapshot.json`: isolated no-search fresh-process resident-memory measurement.
- `correctness.json`: artifact/runtime, component routing, legality, terminal, exact-root, and fixed-seed repeatability results.
- `../seed514-operational-benchmark/failed-correctness-attempt.json`: preserved pre-measurement repeatability-check diagnosis from the superseded freeze attempt.
