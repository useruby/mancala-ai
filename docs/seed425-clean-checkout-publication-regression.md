# Seed425 clean-checkout publication regression

Run the portable publication regression with:

```bash
python -m unittest ml.alphazero_lite.test_seed425_publication_portable -v
```

The test builds a relocated fixture from tracked public sources and publication
evidence, explicitly creates its scratch parent, invokes the #424 verifier entry
point from another working directory, checks read-only behavior and published
counts/decision, and rejects mutations to the historical exclusion evidence,
registered source and snapshot, amendment, training results, runtime/candidate
binding, report/outcome binding, and ledger. It requires NumPy for pure analysis
and does not require Torch, checkpoints, runtime artifacts, a native executable,
or a tablebase. The Python Quality workflow runs this command as a dedicated
step.

`test_seed422_publication.py` and `test_seed422_adam_memory.py` are distinct
historical verifier coverage. They exercise the original training/recovery-era
APIs and their path assumptions; they are not excluded or skipped by the
portable regression gate.
