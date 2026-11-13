# #582 pair-probe reuse: generation-run validation

## Result

The frozen 16-pair comparison passed the adoption gate. v2 reproduced every
trajectory, terminal result, and retained training row, while reducing total
pair CPU by **34.30%** (174.55 s to 114.68 s). All 32 runner attempts have
completed receipts reconciled to their checkpoints.

| Metric | Original runner (v1) | Probe reuse (v2) |
| --- | ---: | ---: |
| Games | 32 | 32 |
| Retained training rows | 473 | 473 |
| Pair-process CPU | 54.87 s | 54.84 s |
| Reaped native-child CPU | 119.68 s | 59.84 s |
| **Total pair CPU** | **174.55 s** | **114.68 s** |
| Sum of per-pair wall times | 153.94 s | 92.83 s |

The wall values are sums of the per-pair wall measurements, reported separately
from CPU. Pair concurrency was 1 and each game used one search worker; this
comparison does not measure throughput under 18-way concurrency.

## Frozen subset and execution

The subset is bound by
[`freeze.json`](data/seed583-probe-reuse-generation-validation/freeze.json),
SHA-256 `edbc5c72031064dde8cb5491efc4e9e2130300acc35325cfcd6ca92bc8371b77`.
The exact harness source used for execution is archived at
[`execution-source-snapshot/`](data/seed583-probe-reuse-generation-validation/execution-source-snapshot/ml/alphazero_lite/validate_varied_opponent_probe_reuse.py)
and hash-bound by the freeze.
It selects eight original control pairs, four original initializer-treatment
pairs, and four original T512-treatment pairs. Every selected game spec is
copied exactly from #580, including opening moves, seeds, and seat assignments.
For each pair, both runners ran back-to-back, alternating which runner went
first. Checkpoints and attempt receipts are separate by runner and pair.

No-games preflight passed before execution with identical resource settings for
both variants: CPU affinity 14–31, CPU quota 1800%, CPU weight 100, memory max
36 GiB, nice 19, one intra-op/inter-op thread, and one search worker per game.
The 64 games are 32 unique assignments run once by each runner.

## Verification and raw evidence

The read-only verifier checks the freeze/source bindings, checkpoint
assignments, attempt-to-checkpoint receipts, attempt reconciliation, and exact
semantic equality of trajectories, outcomes, search settings, and retained
training rows. Timing and runner metadata are excluded from semantic
comparison. It reported zero mismatches and a 0.343002 CPU reduction ratio.

Raw pair checkpoints, completed attempt receipts, and execution order are in
[`evidence/`](data/seed583-probe-reuse-generation-validation/evidence/).
The evidence directory contains 65 files; the aggregate inventory digest is
recorded in `results.json`. Summary metrics and gate status are in
[`results.json`](data/seed583-probe-reuse-generation-validation/results.json).
Reproduce verification from the repository root:

```bash
.venv/bin/python -m ml.alphazero_lite.verify_seed583_probe_reuse_generation --freeze docs/data/seed583-probe-reuse-generation-validation/freeze.json --expected-freeze-sha256 edbc5c72031064dde8cb5491efc4e9e2130300acc35325cfcd6ca92bc8371b77 --evidence docs/data/seed583-probe-reuse-generation-validation/evidence --results docs/data/seed583-probe-reuse-generation-validation/results.json
```

**Recommendation:** authorize v2 for future generation runs subject to the
normal run-specific authorization and freeze. This result does not alter #580's
decision or expand its pilot. No training, serving changes, or sealed-suite
access occurred; #483 remains closed.
