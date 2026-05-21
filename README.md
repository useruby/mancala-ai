# Mancala AI Repository

This repository holds the local AI and supporting tooling for Mancala, centered on the `ml/alphazero_lite` training and artifact pipeline.

## Repo layout

- `ml/alphazero_lite/`: AlphaZero-lite training code, configs, tests, and the detailed operations guide
- `script/ai/`: AI-related entrypoints and helper scripts
- `docs/`: longer-form development and verification notes
- `model-artifact/`: runtime model artifact location
- `test/`: app-side fixtures and tests that support the AI integration contract

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
pip install -r ml/alphazero_lite/requirements.txt
```

Use `requirements-dev.txt` for shared repo tooling such as Ruff and `pre-commit`.

## Lint and pre-commit

```bash
.venv/bin/ruff check ml/alphazero_lite script/ai
.venv/bin/ruff format --check ml/alphazero_lite script/ai
.venv/bin/pre-commit install
.venv/bin/pre-commit run --all-files
```

## Fast smoke test

```bash
.venv/bin/python -m unittest ml.alphazero_lite.test_run_manifest
```

## Detailed docs

- AlphaZero-lite operations guide: `ml/alphazero_lite/README.md`
- AlphaZero-lite development reference: `docs/alphazero-lite-development.md`
