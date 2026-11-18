# Seed 604 Frozen Evaluation Results

## Execution identity

- Approved source commit: `441fbda0d67f30d9515b51459178799556cfb335`
- Execution freeze: `execution-freeze.json`
- Execution-freeze SHA-256: `3812d07445d61d91afcf2a47b536b0b93cf915774f51a77f17af4aade37a47aa`
- Owner authorization: `evaluation-authorization.json`
- Authorization SHA-256: `a6c322f04df29c0a32d52a9030e7a4105a91063520fefa104136d4dd1bb886d9`
- Registered cohort: 1,024 openings; opening manifest SHA-256 `445052b818b36f84953f73f280b20f7421c441fac6a74a0e2bb309cf907fb410`
- Assignment ledger: 4,096 games; SHA-256 `bcbccf36657092f4320d86afaf7da7b475db059f3172259157c4f9d0298e65c7`
- B512 checkpoint SHA-256: `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c`
- B2048 checkpoint SHA-256: `e2460325824be6ed5c416022744ffe4f22fe277f7f838fcc47df8732f0a04874`
- Serving P-E4 model SHA-256: `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`
- Runtime: CPython 3.14.7, NumPy 2.5.3, Linux x86-64, CPU; `PYTHONHASHSEED=604` and numerical thread counts fixed at one
- Execution settings: 18 opening-pair workers; one search worker per game; 128 simulations per side; 64 GiB registered memory limit
- Exact-root native probe SHA-256: `d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`
- Complete tablebase SHA-256: `f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7` (709,635,072 bytes)
- Execution: all 4,096 assigned games completed; no capped games; no execution deviations recorded

## Independent verification

`independent-verification.json` records the read-only verifier output. It reports `verified_complete`, exactly 4,096 authenticated games, 1,024 opening clusters, all seat-paired assignments, and successful verification of game replay, reports, commands, openings, and artifact hashes. Every game attempt and create-only result is retained under `results/`.

## Registered playing-strength results

| Measure | Result | Registered gate | Outcome |
| --- | ---: | --- | --- |
| B512 score vs serving P-E4 | 0.499755859375 | Report | — |
| B2048 score vs serving P-E4 | 0.492431640625 | Report | — |
| B2048 one-sided 95% lower confidence bound | 0.482177734375 | Strictly greater than 0.50 | Fail |
| Paired B2048−B512 score | −0.00732421875 | Report | — |
| Paired B2048−B512 two-sided 95% interval | [−0.017822265625, 0.003173828125] | Lower endpoint strictly greater than 0 | Fail |
| Strength gate | Fail | B2048 lower bound > 0.50 | Fail |
| Incremental-improvement gate | Fail | Paired interval lower endpoint > 0 | Fail |
| Preregistered interpretation | `neither` | Frozen four-way interpretation | Neither claim established |

Intervals use the registered paired opening-cluster bootstrap: 20,000 resamples, seed 604, linear quantiles, with both seat outcomes retained within each sampled opening. `analysis.json` is SHA-256 `4c5437bd1d885524da850e4f1a6d605d0c389e06c1ccb9a61359694c3ccb4dfa` and passed independent recomputation.

## Closeout

Seed 604's registered evaluation is complete and closed. Neither candidate is promoted. These results do not authorize another training run; a new training experiment requires its own hypothesis, registration, freeze, and explicit owner authorization.

## One prospective training-data generalization hypothesis

**Hypothesis:** At equal training compute and dataset size, a replay set deliberately diversified across game phase and board occupancy improves playing score on previously unseen opening clusters more than continuing to train on the narrow 2,048-state replay population.

**Test:** Register a matched treatment/control comparison before training. Keep architecture, authenticated parent checkpoint, optimizer initialization and schedule, update count, and total distinct training-state count fixed. The control uses the existing narrow replay-sampling rule; the treatment uses a preregistered phase/occupancy-stratified sampler over eligible training states. Exclude every evaluation opening state and its defined exclusion population from both training arms. Evaluate both checkpoints on a newly registered, disjoint opening cohort against the same frozen P-E4 evaluator, with seat-swapped games and a paired opening-cluster bootstrap. Support the hypothesis only if the treatment's absolute-strength lower bound exceeds 0.50 and the paired treatment-minus-control interval has a strictly positive lower endpoint; otherwise it is not supported. No training or evaluation for this hypothesis is authorized by this closeout.
