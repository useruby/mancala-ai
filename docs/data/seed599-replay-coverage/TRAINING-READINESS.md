# Seed 599 training readiness

The preparation and evaluation registrations remain unchanged. No training
authorization is included in this readiness package.

## Read-only verification

Run from the repository root in `.venv-azlite`:

```bash
PYTHONHASHSEED=599 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv-azlite/bin/python -m ml.alphazero_lite.seed599_replay_coverage preflight
PYTHONHASHSEED=599 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv-azlite/bin/python -m unittest \
  ml.alphazero_lite.test_seed599_replay_coverage \
  ml.alphazero_lite.test_seed599_training_executor
```

`training-execution-freeze.json` binds the exact hashes of the two frozen
training configurations and the CPU runtime versions/environment. A separate
`training-authorization.json` must be supplied by the project owner after
review. Its `execution_freeze_sha256` must equal the SHA-256 of the exact freeze
file bytes. Missing, malformed, or stale authorization is rejected before
output-directory creation or optimizer updates. No authorization artifact is
created by these instructions.

The executor is opt-in and must not be invoked as part of readiness checks.
Only after a separate authorized execution, the commands are:

```bash
PYTHONHASHSEED=599 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv-azlite/bin/python -m ml.alphazero_lite.seed599_training_executor \
  --arm A --output-dir docs/data/seed599-replay-coverage/training-run-A
PYTHONHASHSEED=599 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  ./.venv-azlite/bin/python -m ml.alphazero_lite.seed599_training_executor \
  --arm B --output-dir docs/data/seed599-replay-coverage/training-run-B
```

Each output directory is create-only. Verify results independently with
`ml.alphazero_lite.verify_seed599_training_result --arm A|B --output-dir PATH`.
The verifier checks frozen membership and schedule identities, exact update
trace indexing, trace/checkpoint hashes, checkpoint tensor compatibility,
parameter grouping, and frozen value-group equality against the authenticated
P-E4 parent.
