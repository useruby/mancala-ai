# Mancala AI Repository

This repository is public, but still early.

It is the open development repository for Mancala.com's AlphaZero-lite
Kalah/Mancala bot training, evaluation, and artifact pipeline. Expect active
research and tooling work rather than a polished external product.

APIs, configs, scripts, and generated artifacts may change as the pipeline
evolves.

## What this repo is

- an active research and tooling repository for Mancala.com's AlphaZero-lite
  Kalah/Mancala bot work
- the home for model training, evaluation, promotion-gate checks, and artifact
  generation
- a public record of open development while the repository is still early

## What this repo is not

- not a general-purpose AlphaZero framework
- not a stable machine-learning platform
- not yet a polished reusable library or end-user package

## Repo layout

- `ml/alphazero_lite/`: AlphaZero-lite training code, configs, tests, and the
  detailed operations guide
- `script/ai/`: AI-related entrypoints and helper scripts
- `docs/`: longer-form development and verification notes
- `model-artifact/`: runtime model artifact location
- `test/`: app-side fixtures and tests that support the AI integration contract

## Contributor setup

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

## Public repo follow-ups

- review whether the detailed AlphaZero-lite operations guide should trim some provider-specific operational examples in a later docs pass
- review whether any app-integration references should move into narrower internal or integration-focused documentation later
