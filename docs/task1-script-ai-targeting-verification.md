# Task 1 script/ai targeting verification

The final Task 1 targeting is the same for the Ruff hooks, while the generic hooks cover both the Task 1 config files and the Python code scope:

- `ml/alphazero_lite/**/*.py`
- Python-typed entrypoints under `script/ai`
- Excluded from `script/ai`: Bash entrypoints, `.sh` files, and Ruby entrypoints
- Generic pre-commit hooks also process: `requirements-dev.txt`, `pyproject.toml`, `.pre-commit-config.yaml`

Pre-commit uses the shared strategy below for code hooks, which reduces drift with Ruff's `extend-include` plus excludes:

- `files: ^(ml/alphazero_lite/.*|script/ai/.*)$`
- `types_or: [python]`

Pre-commit uses separate generic-hook entries for the config files so they are actually processed:

- `files: ^(requirements-dev\.txt|pyproject\.toml|\.pre-commit-config\.yaml)$`

Ruff discovery proof run from the worktree root:

```bash
FILES=$(./.venv/bin/ruff check script/ai --show-files)
for path in \
  script/ai/diagnose_search_interaction \
  script/ai/promote_runpod_candidate \
  script/ai/runpod_training_experiment \
  script/ai/run_local_mix_ablation \
  script/ai/promote_superhuman_candidate \
  script/ai/runpod_remote_run.sh \
  script/ai/check_superhuman_regressions \
  script/ai/compare_superhuman_regressions
do
  if printf '%s\n' "$FILES" | rg -qx ".*/${path}"; then
    printf '%s INCLUDED\n' "$path"
  else
    printf '%s EXCLUDED\n' "$path"
  fi
done
```

Observed Ruff output:

```text
script/ai/diagnose_search_interaction INCLUDED
script/ai/promote_runpod_candidate INCLUDED
script/ai/runpod_training_experiment INCLUDED
script/ai/run_local_mix_ablation EXCLUDED
script/ai/promote_superhuman_candidate EXCLUDED
script/ai/runpod_remote_run.sh EXCLUDED
script/ai/check_superhuman_regressions EXCLUDED
script/ai/compare_superhuman_regressions EXCLUDED
```

Pre-commit generic hook included-file proof run from the worktree root:

```bash
./.venv/bin/pre-commit run trailing-whitespace --files \
  requirements-dev.txt \
  pyproject.toml \
  .pre-commit-config.yaml \
  script/ai/diagnose_search_interaction \
  script/ai/promote_runpod_candidate
```

Observed included-file generic hook output:

```text
trim trailing whitespace.................................................Passed
```

Pre-commit generic hook excluded-file proof run from the worktree root:

```bash
./.venv/bin/pre-commit run trailing-whitespace --files \
  script/ai/run_local_mix_ablation
```

Observed excluded-file generic hook output:

```text
trim trailing whitespace.............................(no files to check)Skipped
```

Pre-commit Ruff hook included-file proof run from the worktree root:

```bash
./.venv/bin/pre-commit run ruff --files \
  script/ai/diagnose_search_interaction \
  script/ai/promote_runpod_candidate
```

Observed included-file Ruff hook output:

```text
ruff (legacy alias)......................................................Passed
```

Pre-commit Ruff hook excluded-file proof run from the worktree root:

```bash
./.venv/bin/pre-commit run ruff --files \
  script/ai/run_local_mix_ablation
```

Observed excluded-file Ruff hook output:

```text
ruff (legacy alias)..................................(no files to check)Skipped
```

These paired pre-commit runs show the final configured behavior directly: approved Python entrypoints are processed, while excluded Bash entries are skipped as non-targets. Ruff discovery above separately shows the excluded Ruby entrypoints as well.
