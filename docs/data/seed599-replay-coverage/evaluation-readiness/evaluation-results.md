# Seed 599 Frozen Evaluation Results

## Execution identity

- Approved source commit: `eeae3471aa5f1a0864f739305599dbdc5723cc72`
- Execution freeze: `execution-freeze.json`
- Execution-freeze SHA-256: `467cb1ecf4acb47013fce9cd9418b3ac6b1d3929752713e9bd029fceabbcea11`
- Owner authorization: `evaluation-authorization.json`
- Evaluation contract SHA-256: `8d29200a27ae87daa7ea9de9a36ebf5a19d1f0f04b4f0237ff48ba1ab1af7bf4`
- Openings file SHA-256: `3acc395a59d90df57ddf60aa1e9d99aabde220901434f72239eb48a4047bc54a`
- Opening manifest SHA-256: `6ed3c99b35c6f51d03ea59ee2c9ed1c5fb52cf76a51b4cccd50f74ba8f962962`
- Candidate A checkpoint SHA-256: `eec38be82dce5e7791c3b5c0bd414be86d8d83fb66a81390d1283300564e2067`
- Candidate B checkpoint SHA-256: `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c`
- Serving P-E4 model SHA-256: `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`
- Runtime: Python 3.14.7, NumPy 2.5.3, PyTorch 2.14.0+cu130; CPU evaluation
- Execution settings: 18 pair-level workers; one search worker per game; 128 simulations per side
- Workload: 1,024 openings, two seat-swapped games per opening and arm, 4,096 games total
- Execution issues or deviations: none observed. Memory headroom remained adequate; no OOM or assignment failure was observed.

## Independent verification

`independent-verification.json` is the output of `python -m ml.alphazero_lite.verify_seed599_evaluation`.
It reports `verified_complete`, exactly 4,096 authenticated games, 2,048 paired assignments, and 18-way pair parallelism.

| Measure | Result | Registered gate | Outcome |
| --- | ---: | --- | --- |
| Arm A score vs serving P-E4 | 0.483154296875 | Report | — |
| Arm B score vs serving P-E4 | 0.4970703125 | Strength lower bound must be > 0.50 | — |
| B one-sided 95% lower confidence bound | 0.48779296875 | > 0.50 | Fail |
| Paired B−A score | 0.013916015625 | Report | — |
| Paired B−A two-sided 95% interval | [0.000732421875, 0.027099609375] | Lower endpoint > 0 | Pass |
| Strength gate | Fail | B lower bound > 0.50 | Fail |
| Coverage gate | Pass | Paired interval lower endpoint > 0 | Pass |
| Four-way classification | `coverage_only` | Frozen outcome classification | Coverage only |

Confidence intervals use the registered paired opening-cluster bootstrap: 20,000 resamples, seed 599, linear quantiles, retaining both seat outcomes within each sampled opening.

### Trajectory overlap

Canonical and production-float32 identities yielded the same overlap counts.

| Candidate trajectories | Training membership | Overlap | Candidate distinct positions |
| --- | --- | ---: | ---: |
| A | Training A | 3 | 49,140 |
| A | Training B | 54 | 49,140 |
| B | Training A | 5 | 44,920 |
| B | Training B | 58 | 44,920 |

### Secondary diagnostic

Held-out masked policy cross-entropy is diagnostic only and is not a decision gate:

- A: `0.6977924297224092`
- B: `0.662916541666083`
- B−A: `-0.03487588805632613`
- Paired opening-group B−A mean: `-0.03575151962496871`

No promotion decision is made by these results.
