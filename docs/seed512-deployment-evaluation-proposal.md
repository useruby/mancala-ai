# Seed512 deployment and evaluation proposal

## Configuration and cost

P is the seed504 E4 policy output and its own trunk paired with the frozen initializer value output and its own trunk; no tensor splicing or training. The opponent is the complete frozen initializer. Search uses 1,200 simulations per side, one worker, deterministic root policy, `c_puct=1.25`, the registered exact-root behavior (threshold 16; final-score-margin objective; network-prior then lowest-move tie break), and no exact leaf solve. Inference computes policy and value through two separate networks/trunks per search evaluation; relative to a single shared-trunk model this duplicates trunk inference (two forward passes and two trunk activations per node), plus each head. Measure end-to-end latency and memory before rollout.

## Evidence and limits

On the examined 1,024-opening/2,048-game suite, seed512 records 947 wins, 259 draws, 842 losses (score 0.52563; one-sided 95% paired-opening bootstrap lower bound 0.51489). The matched 1,200-minus-128 per-opening contrast is 0.00659 with descriptive interval [-0.01318, 0.02637]. These support further evaluation only: seed511's 128-simulation results were already examined, the same openings were reused, and this is not fresh confirmation. The seed461 overlap limitation and separate seed505 failed gate remain in force.

## Smallest integration and rollback

Add a versioned P configuration selecting the two existing component checkpoints and explicit separate policy/value model instances; do not modify checkpoint contents or search defaults. Gate behind a runtime configuration switch, expose candidate-vs-baseline shadow inference first, and monitor latency, memory, errors, legal-move behavior, and score telemetry. Roll back by switching the configuration to the existing initializer model; retain both immutable artifacts and do not overwrite the baseline.

## Before promotion

Run a preregistered, independently held-out opening suite with balanced seats and a prospectively frozen analysis plan; include multiple seeds or an adequately powered sample, report confidence intervals and subgroup/opening sensitivity, and compare at matched compute budgets. Benchmark production hardware for p50/p95 latency, throughput, memory, and reliability. Verify artifact provenance and build/runtime reproducibility; assess operational rollback readiness. Promotion requires the predeclared strength gate and production SLOs to pass.

## Provenance limitation

The seed511 bindings authenticate recorded assets/runtime sources where identities were recorded. Seed512 does not prospectively freeze every transitive imported Python source dependency; no retrospective identity is asserted for those files.

## Seed514 local operational diagnostic

Seed514 measured the complete baseline initializer and fixed P composition on the local host, using the registered NumPy/native `evaluate_artifact_position` path, 1,200 simulations, `c_puct=1.25`, one synchronous worker, deterministic root policy, and the registered exact-root dispatch (threshold 16). P pairs seed504 E4 policy with its own trunk and initializer value with its own trunk. No weights were created or changed. The benchmark is an examined diagnostic, not a holdout, fresh confirmation, operational pass, or deployment qualification.

The deterministic cohort contains 64 unique nonterminal states reconstructed from the archived seed512 trajectories: 32 states with more than 32 active stones, 16 with 17–32, and 16 with at most 16. Selection sorted a SHA-256 ordering of state identity and source game/ply, without model outputs or outcomes. `docs/data/seed514-operational-benchmark-v2/cohort.json` publishes the state identities and source references; `protocol-freeze.json` binds the cohort, seed512 batch identities, artifact/runtime identities, protocol, and finalized benchmark/test/verifier/search-source hashes. These examined states inherit the seed512 archive’s limitations.

On this host (13th Gen Intel Core i9-13900, Linux 7.2.6, Python 3.14.7, NumPy 2.5.2, Torch 2.13.0+cu130), 320 timed searches were recorded per configuration (five repetitions for each of 64 states), with one untimed warm-up per configuration/state. The environment pinned OpenMP, OpenBLAS, MKL, and NumExpr thread environment variables to 1; Torch reported 1 intra-op and 24 inter-op threads. Paired seeds and deterministic alternating order are recorded in the protocol and raw measurements.

| Cohort / configuration | Searches | p50 latency | p95 latency | throughput |
| --- | ---: | ---: | ---: | ---: |
| All states — baseline | 320 | 106.2 ms | 8.071 s | 0.378 searches/s |
| All states — P | 320 | 171.8 ms | 7.907 s | 0.373 searches/s |
| >32 stones — baseline | 160 | 107.7 ms | 114.3 ms | 9.27 searches/s |
| >32 stones — P | 160 | 174.8 ms | 185.3 ms | 5.75 searches/s |
| 17–32 stones — baseline | 80 | 98.8 ms | 105.2 ms | 11.20 searches/s |
| 17–32 stones — P | 80 | 153.5 ms | 166.5 ms | 7.44 searches/s |
| ≤16 stones, exact-root dispatch — baseline | 80 | 700.3 ms | 90.54 s | 0.0974 searches/s |
| ≤16 stones, exact-root dispatch — P | 80 | 696.1 ms | 90.29 s | 0.0975 searches/s |

The paired P/baseline latency ratio was 1.582 at p50 and 1.669 at p95 across the 320 paired observations; per-state median ratios ranged from 0.980 to 1.717. The >32 and 17–32 stone PUCT strata had median paired ratios of 1.621 and 1.542. The ≤16 exact-root stratum’s median ratio was 1.016. Each configuration took 80 exact-root and 240 PUCT paths. Overall p95 and throughput are strongly influenced by the exact-root tail; the ≤16 stratum reached approximately 90 seconds at p95 on this host. The similar overall throughput therefore should not be read as evidence that P’s non-exact search has negligible cost.

Evaluator loading measured 13.9 ms for baseline and 27.1 ms for P (baseline plus the E4 evaluator; incremental E4 loading was 13.1 ms). A separate fresh-process memory snapshot without searches or tracing measured 42.1 MB resident for baseline and 43.0 MB for the two-model P composition, an 0.96 MB incremental snapshot on this host. That snapshot excludes transient search allocations and exact-root native-tablebase working memory; the process high-water mark from the latency process is not treated as a separate memory measurement. Raw rows, environment, identities, cohort, correctness checks, summary, and the independent arithmetic verifier are in `docs/data/seed514-operational-benchmark-v2/`.

These measurements show substantial PUCT latency overhead for the two-trunk composition on this host, while exact-root latency is dominated by the registered solver and has a large tail. They are not production-hardware predictions. Before any rollout, measure on intended production hardware and runtime builds, including configuration/model startup, native exact-root transient memory, tail latency under realistic concurrency/load, reliability, and service-level objectives. Product owners must prospectively define applicable latency, memory, throughput, and reliability SLOs; seed514 does not establish or invent pass thresholds. Preserve the existing defaults until separately authorized evaluation and operational decisions are complete.
