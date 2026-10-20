# Seed508 pre-execution freeze correction 001

The first explicit execution command was:

```bash
/home/alex/Mancala/ai/.venv-azlite/bin/python -m ml.alphazero_lite.seed508_policy_value_crossover --execute
```

It stopped before constructing or invoking any arena command with:

```text
KeyError: 'source_sha256'
```

The runner referenced `source_sha256` while the frozen registration schema names
that map `sources_sha256`. The runner was corrected, the focused tests and lint
were rerun, and a replacement registration/freeze was created before restarting
execution. The superseded first registration and freeze are preserved in
`superseded-pre-execution-freeze-001/`. No games or outcome-bearing batches
existed at the time of correction.
