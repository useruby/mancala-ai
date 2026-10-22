# Seed512 deployment and evaluation proposal

## Seed520 opt-in composition integration contract (proposal only)

This document remains proposal-only and does not enable composition or alter defaults. `docs/data/seed520-composition/manifest.json` is retained as the original checkpoint-only v1 proposal. The corrected runtime-input proposal is `docs/data/seed520-composition/manifest-v2.json`; its validator is `ml/alphazero_lite/seed520_composition_manifest.py`. The actual extracted fixture entry point is `ml/alphazero_lite/seed521_golden_fixtures.py`, and the portable bundle is `docs/data/seed521-golden-fixtures/`. No merged weights or replacement exports are part of this proposal.

### Actual integration boundary

In this repository, `ml/alphazero_lite/arena.py` is the artifact loader and NumPy diagnostic consumer; `ArtifactEvaluator` loads checkpoint metadata/weights, while `ml/alphazero_lite/input_encodings.py` / `self_play.py:encode_state` provide `kalah_v3`. `selective_head_evaluator.py:SelectiveHeadArtifactEvaluator` is experiment-only composition evaluation and explicitly owns separate `ArtifactEvaluator` instances, applies each source's own trunk, and rejects non-`residual_v3` architectures and extensions. `runtime_search_policy.py` validates the per-artifact `search_policy.json` sidecar and bound native-probe/tablebase objects. The Rails/Ruby application consumer, its model loader, configuration registry, feature switch, and deployment/rollback machinery are not present in this repository. Therefore the Python path is diagnostic evidence and not an implementation of that consumer.

**Proposed target interface (not existing code):** `CompositionManifest.load(path, root:) -> validated immutable component references`; `EvaluatorFactory.build(policy_artifact:, value_artifact:) -> evaluator`; and configuration `evaluator_mode=initializer|composition`, defaulting to `initializer`, with composition requiring a manifest path. The Ruby implementation must map existing model keys to two independent `residual_v3` model instances; policy head and trunk come exclusively from E4 and value head and trunk exclusively from the frozen initializer. Never splice, retag, or substitute a model silently. Keep the search configuration bound to 1,200 simulations, one worker, `c_puct=1.25`, deterministic root policy, exact-root threshold 16, final-score-margin objective, network-prior then lowest-move tie rule, and no exact leaf solve. The referenced sidecar/native identities must be validated by the target loader against actual bytes, not just labels.

### Manifest and golden evaluator contract

The v1 manifest is explicitly checkpoint-only and is not runtime consumable. V2 binds runtime artifact directories from the authoritative seed512 registration: E4 for policy and initializer for value, including `model.npz`, `weights.json`, `metadata.json`, and each `search_policy.json` with its registered SHA-256. It separately retains checkpoint identity as provenance. Validation reads the authenticated metadata and enforces `kalah_v3`, `residual_v3`, 27 input features, six policy outputs, one value output, the registered residual architecture and `.npz` model input; unsupported extensions and metadata disagreement fail closed. Both independently owned components retain their own trunk and meet the selective-head P architecture. Exact-root probe and tablebase bytes are checked separately from a materialized tablebase object when the local artifact is a pointer. The verifier accepts explicit repository and asset roots; it does not infer either from manifest location. No fallback or digest substitution is allowed.

The seed521 bundle joins seed514 cohort states, seed515 provenance and archived seed518 evaluator output bytes without inference or searches. It preserves all 64 frozen cohort cases in order, state identities and source bindings and includes `policy_f32_hex` and `value_f64_hex`. The archived reference and selective outputs are checked for agreement. Two separately labeled terminal fixtures (one per current player) specify six zero float32 policy values and one zero float64 value. The bundle index binds the three source archives and generated fixture bytes by SHA-256. Field orientation and little-endian encoding are documented in `fixtures.json`. These are archived Python bytes, not claims of cross-language agreement.

Target-runtime test: decode each state using the documented field orientation and `kalah_v3` encoder, evaluate each network's own trunk/head, compare serialized policy/value bytes exactly, check legal-move masking and terminal all-zero behavior, then run deterministic root-search contract cases. The Ruby consumer must publish its own runner/version and report every case/hash/output; only a successful execution in the target runtime establishes compatibility.

### Opt-in, failure, rollback, and observability

Disabled mode continues loading the unchanged initializer by its existing configuration. Enabled mode validates the complete manifest and then loads two independently owned models. Any missing, altered, unsupported, or incompatible component/sidecar/runtime binding is an explicit startup/configuration validation failure; it must not fall back to initializer or any alternate model. Rollback is an explicit configuration change to the unchanged initializer model key, followed by normal reload/restart confirmation. Preserve both immutable artifacts.

Required structured observability: selected mode and manifest digest; policy/value artifact paths and verified hashes; architecture/encoding; sidecar/native/tablebase identities; load/validation failures; evaluator latency by component and total; model memory; search dispatch counts (PUCT/exact-root); legal-action validity; and rollback/config revision. Do not emit a healthy composition status unless both component identities validated.

### Preparation checklist and remaining gates

- [x] Publish an opt-in contract and immutable component identities without modifying artifacts/defaults.
- [x] Add read-only Python validation for registered runtime files, source routing, authenticated metadata, sidecars, probe and pointer/materialized runtime bytes.
- [x] Publish and executable-verify the portable 66-case seed521 golden bundle.
- [ ] Implement Ruby manifest parsing, secure path/materialized-object resolution, byte hashing, sidecar/native identity checks, and two-network ownership; reject unsupported extensions and all silent fallback paths.
- [ ] Add Ruby loader/config tests proving disabled initializer behavior, enabled two-model construction, explicit invalid-config failures, and rollback to unchanged initializer.
- [ ] Run the archived golden evaluator and deterministic search contract in the target runtime; publish exact per-case parity results. Target-runtime parity has not passed here.
- [ ] Measure latency, peak/transient memory, reliability and concurrency on production hardware/runtime, and obtain prospectively defined SLO thresholds. Local seed514/518 diagnostics are not production qualification.
- [ ] Separately authorize strength confirmation and preregister a fresh held-out evaluation before any promotion decision.

Ruby implementation, target-runtime parity, production SLOs and strength confirmation remain explicitly pending. Publishing the Python contract and fixtures does not satisfy these gates.

The seed512 examined suite remains 1,024 openings/2,048 games and its recorded results are not fresh confirmation. Seed461 overlap limitations and the separate seed505 failed gate remain in force. Historical source/provenance limitations stated below and in the evidence supplements remain unchanged.

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

## Seed518 selective-head diagnostic (examined)

Seed518 compared the unchanged composed P evaluator against experiment-only selective-head inference on the exact examined seed514 64-state cohort. The initial frozen parity comparison marked all 64 search outputs as mismatches because `search.root_latency_ms` differed; evaluator policy/value bytes matched in every state. The preserved initial freeze and failed output remain unchanged. A post-gate, pre-measurement comparison-contract amendment is separately bound to the original freeze and archive and excludes only the exact elapsed-wall-clock `root_latency_ms` field (in addition to the originally declared exact-root latency leaf). The source assignment is `self_play.py:1563` (`perf_counter` elapsed milliseconds), copied into the returned result at `arena.py:1347-1349`. No other leaf differed. Post-execution reanalysis of the saved outputs, without rerunning searches, found bitwise evaluator parity and deterministic search parity for all 64 states under that amended contract.

Fresh paired timing used the registered search settings, 1,200 simulations, `c_puct=1.25`, one worker, seeds 518000 plus cohort index, one untimed warm-up/configuration/state, and five alternating repetitions. PUCT median paired selective/reference ratios were 0.901923 (>32 stones; 9.81% faster) and 0.901876 (17–32 stones; 9.81% faster), each exceeding the fixed 5% screen. At ≤16 stones the exact-root path had 80 paired searches/configuration, p50 696.7 ms reference and 695.2 ms selective, p95 90.03 s and 89.88 s, and median ratio 0.996242; these exact-root results are reported separately and are not used to mask PUCT overhead. Overall p50 was 166.2 ms reference and 149.7 ms selective, while p95 was 7.969 s and 7.998 s because exact-root searches dominate the tail. This examined diagnostic supports only drafting an opt-in integration proposal; it is not production qualification, strength evidence, promotion, or deployment approval. Raw measurements, compared outputs, amendment, reanalysis, freezes, and the separate receipt are under `docs/data/seed518-selective-head/`.

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

## Seed515 independent evidence supplement (append-only)

Seed515 independently replayed the hash-bound seed512 game batches and reproduced all 64 seed514 cohort identities in the published order. The replay encountered 69,976 state-occurrence visits, corresponding to 60,333 unique state hashes; these counts are distinct. The separate provenance supplement records the batch-relative path and SHA-256, record/game index, opening index, continuation ply, total ply, state hash, and matching-occurrence count for every selected state. It enumerates every matching locator for the 27 selected identities with duplicate matches. Its deterministic selection-order reference is not a claim about which duplicate occurrence was historically consumed. Selection remains the frozen seed514 selection and was not altered.

The read-only verifier independently checked 640 unique state/configuration/repetition keys (five repetitions for each of 64 states and two configurations), registered seeds, parity-alternating execution order, positive finite durations, exact-root/PUCT routing, and documented `puct_simulations_executed` accounting (0 for exact-root dispatch and 1,200 for PUCT). It independently recomputes aggregate and per-state ratios, every bucket's counts, percentiles, throughput, paired ratios and path counts. Machine-readable full-precision values and the provenance supplement are in `docs/data/seed515-independent-verification/`.

Memory validation confirms recorded initializer/E4 artifact identities and that the P-minus-baseline resident-byte arithmetic equals the recorded incremental RSS. This is a model-resident snapshot only; it is not search/native peak memory. Arithmetic cannot reproduce historical RSS. The seed514 publication has no prospective binding for the independent seed515 verifier; its memory source hash is post-execution evidence rather than a freeze binding. Seed512 registration authenticates the native probe and tablebase pointer; the recovered tablebase object also hashes to its registered SHA-256. These limitations are retained rather than filled with retrospective claims. Seed515 performed no searches, arena games, training, deployment, promotion, artifact replacement, or fresh-confirmation activity.
