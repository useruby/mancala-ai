# Seed 599 training results

## Authorization and execution identity

- Authorization: explicit project-owner authorization in `training-authorization.json`.
- Independent human review: not claimed.
- Approved worktree commit: `58b9a8293cee563054646ad10a74af7d09ededba`.
- Execution-freeze SHA-256: `bd141778461d7046b067966d7fb87188e5b0a2630d82655a321059b4ce17cad9`.
- Runtime: Python `3.14.7`, NumPy `2.5.2`, PyTorch `2.13.0+cu130`, CPU, one thread, deterministic algorithms enabled; `PYTHONHASHSEED=599`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`.
- Read-only preparation preflight passed before authorization and training. The registered configuration, membership, schedules, source dependencies, and P-E4 parent identity were not modified.
- Deviations: none.

## Results

| Arm | Registered states | Updates | Initial loss | Final loss | Checkpoint SHA-256 | Trace SHA-256 | Independent verification |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| A | 256 | 512 | 0.662121057510376 | 0.4695238471031189 | `eec38be82dce5e7791c3b5c0bd414be86d8d83fb66a81390d1283300564e2067` | `8789eae8d484f6166284063f8bbccaf4f2abae4c0073528ddbd6fcb258f2fdee` | Passed |
| B | 2,048 | 512 | 0.7018150687217712 | 0.4574684500694275 | `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c` | `9f22328080756aa0fa2fa066164d36f935643938363059e0d0c711eb6f8f8955` | Passed |

Losses are descriptive only and were not used to select or reject a candidate.

Each `training-run-{A,B}` directory contains the final checkpoint, complete JSONL trace, executor receipt, and independent verifier report. The verifiers confirmed 512 trace records, exact registered batch indexes and runtime identity, receipt/checkpoint/trace identities, and value-head parameters unchanged from the registered P-E4 parent checkpoint. The executor also recorded its frozen-parameter runtime snapshots as unchanged.

## Evidence file identities

| Artifact | SHA-256 |
| --- | --- |
| `training-authorization.json` | `623148001f2375025e929504d08dab0a6dea23cb9cdd76263d3a6c9ef2a2cc77` |
| `training-run-A/seed599-arm-A-update-512.npz` | `eec38be82dce5e7791c3b5c0bd414be86d8d83fb66a81390d1283300564e2067` |
| `training-run-A/training-trace.jsonl` | `8789eae8d484f6166284063f8bbccaf4f2abae4c0073528ddbd6fcb258f2fdee` |
| `training-run-A/training-receipt.json` | `d927dfcab00549098415af1cc69c047522086f55f7c6b5fba9d9f673f4e1b16b` |
| `training-run-A/independent-verification.json` | `b1e55cff5ac1432a0683f3dd13bacc7c176dda7e5af87aa7823a178785c4f0e2` |
| `training-run-B/seed599-arm-B-update-512.npz` | `bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c` |
| `training-run-B/training-trace.jsonl` | `9f22328080756aa0fa2fa066164d36f935643938363059e0d0c711eb6f8f8955` |
| `training-run-B/training-receipt.json` | `3d81b00c438a678a2e38b9d58dcadc22f4091c0cb8e3ccbb755e1afe19ca41de` |
| `training-run-B/independent-verification.json` | `57fd2623874a0064814b71847763f52ad890cc1833d8daffdaa222786673da46` |

No evaluation games were run. These training results do not authorize evaluation, sealed-suite access, checkpoint selection, additional training, or serving changes.
