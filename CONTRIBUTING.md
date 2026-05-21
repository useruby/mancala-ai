# Contributing

This repository is public, but it is still early.

The project is an active research and tooling repository for Mancala.com's
AlphaZero-lite Kalah/Mancala bot training, evaluation, and artifact pipeline.
That means maintainers may be selective about scope while the repository is
still changing quickly.

## Before you start

- prefer bug reports, documentation fixes, and small targeted improvements
- discuss larger changes before implementing them
- expect APIs, configs, scripts, and artifact formats to keep evolving

If you are planning a larger feature, workflow change, or experiment-structure
change, open an issue first so maintainers can confirm whether it fits the
current direction.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r ml/alphazero_lite/requirements.txt
```

## Local checks

Run the same baseline checks maintainers use for a quick pass:

```bash
.venv/bin/ruff check ml/alphazero_lite script/ai
.venv/bin/ruff format --check ml/alphazero_lite script/ai
.venv/bin/pre-commit run --all-files
.venv/bin/python -m unittest ml.alphazero_lite.test_run_manifest
```

## Pull requests

- keep changes focused
- explain why the change is needed
- include the verification steps you ran
- call out any follow-up work or known limitations

## Scope notes

While the repository is still early, maintainers may defer or decline changes
that expand project scope, add new framework-like surfaces, or assume the repo
is already a stable general-purpose platform.
