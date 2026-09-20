# Promotion Gate Generation Rollover

Classification: `promotion_gate_generation_rollover_ready`.

## Before

The canonical shadow gate embedded the seed48 experiment candidate SHA and the
pre-seed48 incumbent SHA/version. Those historical identities correctly
reproduced that experiment, but blocked evaluating generation N+1 against the
promoted seed48 incumbent.

## After

Canonical shadow evaluation verifies each artifact by hashing its actual
`weights.json` bytes and requiring that hash to match
`metadata.json.artifacts.weights_json_sha256`. Gate provenance records each
artifact version, actual weights SHA-256, and metadata SHA-256.

The experiment must pass `--expected-current-weights-sha256`; the gate binds
the current artifact to that explicit pin. The candidate must differ from the
current artifact, and the canonical hard opponent must have the same verified
weights identity as current. Frozen suite SHA and disjointness checks are
unchanged.

For the next iteration, explicitly pin the promoted seed48 incumbent:

```sh
--expected-current-weights-sha256 935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c
```

## Verification

```sh
.venv/bin/python -m unittest ml.alphazero_lite.test_local_promotion_gate
.venv/bin/ruff check script/ai/local_promotion_gate ml/alphazero_lite/test_local_promotion_gate.py
.venv/bin/ruff format --check script/ai/local_promotion_gate ml/alphazero_lite/test_local_promotion_gate.py
pre-commit run --files script/ai/local_promotion_gate ml/alphazero_lite/test_local_promotion_gate.py docs/alphazero-lite-promotion-gate-generation-rollover.md
```

The test suite includes a dry-run with temporary N+1 candidate, seed48 current,
and seed48 hard artifacts. It verifies frozen suites and identity preflight
without launching arena games or producing a promotion result.
