# Frozen A5 diagnostic reproduction

The preregistered opening-cluster analysis is reproducible from the published
opening score matrix:

```sh
PYTHONPATH=. .venv/bin/python -m ml.alphazero_lite.reproduce_corrected_order38615_diagnostic \
  docs/data/order38615-a5-frozen-diagnostic-v4/opening-score-matrix.json
```

Verify the post-execution publication, original manifest inputs, snapshots,
AST equivalence, exact runtime/artifact identities, and raw game evidence:

```sh
PYTHONPATH=. .venv/bin/python -m ml.alphazero_lite.publish_order38615_a5_diagnostic verify
```

The regular analyzer and launch validators intentionally require exact live
source hashes from the original registration. They reject the post-execution
formatted delivery files. The publication verifier is separate: it checks
those original source hashes against byte-exact execution snapshots, verifies
AST equivalence for formatted files, recomputes the complete exclusion union,
strictly replays both new suites, and validates completed raw game evidence.
Snapshots are publication evidence only and cannot satisfy a future launch
binding. The per-game outcome ledger is `game-outcome-accounting.jsonl`; its
hash is recorded in `results.json` and in the opening matrix.

The registered excluded-state identities and every source path, byte hash,
count, and identity digest are published in `opening-exclusion-manifest.json`.
Some source bytes are run-local `.tmp` inputs and are not included in this PR:
the canonical reconstruction and five training-replay JSONL files listed in
the manifest. The source inputs were present and hash-verified during
registration, launch, resume validation, and analysis on the execution host.
A fresh clone without those `.tmp` inputs can reproduce the published score
intervals and inspect the full exclusion identity union, but cannot independently
recompute the training-replay identities until those exact source files are
made available.

The result is **FAIL** under the frozen decision rule. It does not alter #392's
canonical gate failure.
