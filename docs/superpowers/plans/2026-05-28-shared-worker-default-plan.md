# Shared Worker Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Centralize AlphaZero-lite worker defaults in one reusable module, set the repo default to `24`, append `--workers 24` for supported tools that omit it, and preserve explicit CLI overrides like `run_local_superhuman_recovery --workers`.

**Architecture:** Add one shared helper module that knows which executables support `--workers` and can normalize a command list to either the repo default or an explicit override. Apply that helper at the two command-materialization layers already in use: the general pipeline path in `ml/alphazero_lite/pipeline.py` and the superhuman runtime-config rewrite path in `ml/alphazero_lite/superhuman_runtime_config.py`. Keep the scope tight by using an explicit allowlist instead of dynamic CLI inspection.

**Tech Stack:** Python 3, `unittest`, existing pipeline/runtime-config helpers, existing wrapper scripts

---

### File Map

- Create: `ml/alphazero_lite/worker_config.py`
- Modify: `ml/alphazero_lite/pipeline.py`
- Modify: `ml/alphazero_lite/superhuman_runtime_config.py`
- Modify: `script/ai/run_local_superhuman_recovery`
- Modify: `ml/alphazero_lite/test_pipeline.py`
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`
- Verify: `python -m unittest ml.alphazero_lite.test_pipeline`
- Verify: `python -m unittest ml.alphazero_lite.test_promote_runpod_candidate`
- Verify: `python -m unittest ml.alphazero_lite.test_runpod_wrappers`
- Verify: `.venv/bin/python -m pre_commit run --all-files`

### Task 1: Lock Down Pipeline Worker Behavior With Failing Tests

**Files:**
- Modify: `ml/alphazero_lite/test_pipeline.py`

- [ ] **Step 1: Write the failing test for appending the shared default**

Add a focused unit test near the existing `build_step_command()` coverage:

```python
    def test_build_step_command_appends_default_workers_for_supported_tool(self):
        step = {
            "name": "self_play",
            "command": [
                sys.executable,
                "ml/alphazero_lite/self_play.py",
                "--games",
                "100",
            ],
        }

        self.assertEqual(
            [
                sys.executable,
                "ml/alphazero_lite/self_play.py",
                "--games",
                "100",
                "--workers",
                "24",
            ],
            build_step_command(step),
        )
```

- [ ] **Step 2: Write the failing test for replacing an existing worker literal**

Add a second focused test:

```python
    def test_build_step_command_rewrites_existing_workers_to_shared_default(self):
        step = {
            "name": "arena_confirm_report",
            "command": [
                sys.executable,
                "ml/alphazero_lite/arena.py",
                "--games",
                "120",
                "--workers",
                "6",
            ],
        }

        self.assertEqual(
            [
                sys.executable,
                "ml/alphazero_lite/arena.py",
                "--games",
                "120",
                "--workers",
                "24",
            ],
            build_step_command(step),
        )
```

- [ ] **Step 3: Write the failing test for unsupported commands staying unchanged**

Add the guardrail test:

```python
    def test_build_step_command_leaves_unsupported_command_unchanged(self):
        command = [sys.executable, "ml/alphazero_lite/train.py", "--epochs", "2"]
        step = {"name": "train", "command": command}

        self.assertEqual(command, build_step_command(step))
```

- [ ] **Step 4: Run the focused tests to verify they fail**

Run:

```bash
python -m unittest \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_appends_default_workers_for_supported_tool \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_rewrites_existing_workers_to_shared_default \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_leaves_unsupported_command_unchanged
```

Expected: FAIL because `build_step_command()` currently only handles nested self-play search options and does not normalize worker flags.

- [ ] **Step 5: Commit the red test checkpoint**

Run:

```bash
git add ml/alphazero_lite/test_pipeline.py
git commit -m "test: cover shared worker defaults in pipeline"
```

Expected: commit succeeds with only the new pipeline worker tests.

### Task 2: Implement the Shared Worker Module and Pipeline Integration

**Files:**
- Create: `ml/alphazero_lite/worker_config.py`
- Modify: `ml/alphazero_lite/pipeline.py`
- Test: `ml/alphazero_lite/test_pipeline.py`

- [ ] **Step 1: Create the shared allowlist helper**

Add `ml/alphazero_lite/worker_config.py` with one default, one allowlist, and one command normalizer:

```python
from __future__ import annotations

DEFAULT_WORKERS = 24

WORKER_CAPABLE_EXECUTABLES = {
    "ml/alphazero_lite/self_play.py",
    "ml/alphazero_lite/generate_bootstrap_dataset.py",
    "ml/alphazero_lite/arena.py",
    "ml/alphazero_lite/mcts1200_baseline.py",
}


def normalize_command_workers(
    command: list[str], *, workers: int | None = None
) -> list[str]:
    if len(command) < 2:
        return list(command)

    executable = command[1]
    if executable not in WORKER_CAPABLE_EXECUTABLES:
        return list(command)

    target_workers = str(DEFAULT_WORKERS if workers is None else workers)
    rendered = list(command)
    if "--workers" in rendered:
        worker_index = rendered.index("--workers")
        if worker_index + 1 < len(rendered):
            rendered[worker_index + 1] = target_workers
            return rendered
    rendered.extend(["--workers", target_workers])
    return rendered
```

- [ ] **Step 2: Route `build_step_command()` through the worker helper**

Update `ml/alphazero_lite/pipeline.py` imports and finish `build_step_command()` by normalizing workers after the existing search-option augmentation:

```python
from ml.alphazero_lite.worker_config import normalize_command_workers


def build_step_command(step: dict) -> list[str] | object:
    command = step.get("command", [])
    if not isinstance(command, list) or not command:
        return command
    command = append_step_search_option_flags(step, command)
    return normalize_command_workers(command)
```

- [ ] **Step 3: Run the focused pipeline tests to verify they pass**

Run:

```bash
python -m unittest \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_appends_default_workers_for_supported_tool \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_rewrites_existing_workers_to_shared_default \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_leaves_unsupported_command_unchanged
```

Expected: PASS

- [ ] **Step 4: Update one config-render assertion from `6` to `24`**

In `ml/alphazero_lite/test_pipeline.py`, change the existing aggressive config expectation:

```python
            self.assertIn("--workers", rendered)
            self.assertIn("24", rendered)
            self.assertNotIn("6", rendered)
```

Use the existing test named `test_aggressive_config_uses_python_arena_workers_for_prefilter_and_confirm` rather than adding a duplicate broad config test.

- [ ] **Step 5: Run the targeted pipeline regression block**

Run:

```bash
python -m unittest \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_aggressive_config_uses_python_arena_workers_for_prefilter_and_confirm \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_aggressive_config_uses_dedicated_mcts1200_baseline_driver
```

Expected: PASS and rendered arena / baseline commands now contain `--workers 24`.

- [ ] **Step 6: Commit the shared worker module**

Run:

```bash
git add ml/alphazero_lite/worker_config.py ml/alphazero_lite/pipeline.py ml/alphazero_lite/test_pipeline.py
git commit -m "feat: centralize alphazero worker defaults"
```

Expected: commit succeeds with the new helper and pipeline integration.

### Task 3: Lock Down Superhuman Runtime Config Behavior With Failing Tests

**Files:**
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Add a failing test for default worker injection in dry-run recovery**

Add a new test next to `test_dry_run_workers_override_rewrites_runtime_config`:

```python
    def test_dry_run_without_workers_flag_uses_shared_default_runtime_workers(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-local-recovery-") as tmp:
            output_root = Path(tmp) / "recovery"

            result = subprocess.run(
                [
                    "script/ai/run_local_superhuman_recovery",
                    "--dry-run",
                    "--output-root",
                    str(output_root),
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            runtime_config = json.loads(
                Path(payload["runtime_config_path"]).read_text(encoding="utf-8")
            )
            worker_values = []
            for step in runtime_config["steps"]:
                command = step.get("command")
                if not isinstance(command, list) or "--workers" not in command:
                    continue
                worker_values.append(command[command.index("--workers") + 1])

            self.assertEqual(["24", "24", "24", "24", "24", "24", "24"], worker_values)
```

- [ ] **Step 2: Add a failing test for explicit override precedence**

Keep the existing override test but make it prove the shared helper honors explicit values different from the default:

```python
    def test_dry_run_workers_override_rewrites_runtime_config(self):
        repo_root = Path(__file__).resolve().parents[2]

        with tempfile.TemporaryDirectory(prefix="azlite-local-recovery-") as tmp:
            output_root = Path(tmp) / "recovery"

            result = subprocess.run(
                [
                    "script/ai/run_local_superhuman_recovery",
                    "--dry-run",
                    "--output-root",
                    str(output_root),
                    "--workers",
                    "11",
                ],
                cwd=repo_root,
                capture_output=True,
                text=True,
                check=False,
            )

            self.assertEqual(0, result.returncode, msg=result.stderr)
            payload = json.loads(result.stdout)
            runtime_config = json.loads(
                Path(payload["runtime_config_path"]).read_text(encoding="utf-8")
            )
            worker_values = []
            for step in runtime_config["steps"]:
                command = step.get("command")
                if not isinstance(command, list) or "--workers" not in command:
                    continue
                worker_values.append(command[command.index("--workers") + 1])

            self.assertEqual(["11", "11", "11", "11", "11", "11", "11"], worker_values)
```

- [ ] **Step 3: Run the focused recovery tests to verify they fail**

Run:

```bash
python -m unittest \
  ml.alphazero_lite.test_promote_runpod_candidate.RunLocalSuperhumanRecoveryTest.test_dry_run_without_workers_flag_uses_shared_default_runtime_workers \
  ml.alphazero_lite.test_promote_runpod_candidate.RunLocalSuperhumanRecoveryTest.test_dry_run_workers_override_rewrites_runtime_config
```

Expected: FAIL because runtime-config generation only rewrites workers when the CLI flag is passed and does not yet inject the shared default.

- [ ] **Step 4: Commit the red recovery tests**

Run:

```bash
git add ml/alphazero_lite/test_promote_runpod_candidate.py
git commit -m "test: cover shared recovery worker defaults"
```

Expected: commit succeeds with only the recovery wrapper test changes.

### Task 4: Implement Shared Runtime-Config Rewriting for Recovery and Sweep

**Files:**
- Modify: `ml/alphazero_lite/superhuman_runtime_config.py`
- Modify: `script/ai/run_local_superhuman_recovery`
- Test: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Add a config-level rewrite helper in `superhuman_runtime_config.py`**

Import the shared command normalizer and walk all steps through it:

```python
from ml.alphazero_lite.worker_config import normalize_command_workers


def apply_worker_overrides(config: dict, *, workers: int | None = None) -> dict:
    updated = copy.deepcopy(config)
    for step in updated.get("steps", []):
        command = step.get("command")
        if not isinstance(command, list):
            continue
        step["command"] = normalize_command_workers(command, workers=workers)
    return updated
```

- [ ] **Step 2: Use the shared helper during runtime-config construction**

Update `script/ai/run_local_superhuman_recovery` imports and `build_runtime_config()`:

```python
from ml.alphazero_lite.superhuman_runtime_config import (
    apply_runtime_paths,
    apply_worker_overrides,
    final_iteration,
    load_json,
    python_executable,
    rewrite_python_commands,
    write_json,
)


def build_runtime_config(
    *,
    base_config_path: Path,
    runtime_config_path: Path,
    versions_dir: Path,
    current_path: Path,
    python_bin: str,
    workers: int | None,
) -> dict:
    config = load_json(base_config_path)
    config = rewrite_python_commands(config, python_bin=python_bin)
    config = apply_runtime_paths(
        config,
        versions_dir=versions_dir,
        current_path=current_path,
        parent_artifact_path=current_path,
    )
    config = apply_worker_overrides(config, workers=workers)
    write_json(runtime_config_path, config)
    return config
```

- [ ] **Step 3: Run the focused recovery tests to verify they pass**

Run:

```bash
python -m unittest \
  ml.alphazero_lite.test_promote_runpod_candidate.RunLocalSuperhumanRecoveryTest.test_dry_run_without_workers_flag_uses_shared_default_runtime_workers \
  ml.alphazero_lite.test_promote_runpod_candidate.RunLocalSuperhumanRecoveryTest.test_dry_run_workers_override_rewrites_runtime_config
```

Expected: PASS, with dry-run recovery producing `24` by default and an explicit `11` when requested.

- [ ] **Step 4: Run the full wrapper test module**

Run:

```bash
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate
```

Expected: PASS

- [ ] **Step 5: Commit the runtime-config integration**

Run:

```bash
git add ml/alphazero_lite/superhuman_runtime_config.py script/ai/run_local_superhuman_recovery ml/alphazero_lite/test_promote_runpod_candidate.py
git commit -m "feat: share superhuman worker runtime config"
```

Expected: commit succeeds with the shared runtime-config worker rewrite.

### Task 5: Final Verification and Documentation Sweep

**Files:**
- Modify: `ml/alphazero_lite/test_pipeline.py`
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`
- Modify: `ml/alphazero_lite/pipeline.py`
- Modify: `ml/alphazero_lite/superhuman_runtime_config.py`
- Modify: `script/ai/run_local_superhuman_recovery`
- Create: `ml/alphazero_lite/worker_config.py`

- [ ] **Step 1: Run the main targeted suites**

Run:

```bash
python -m unittest ml.alphazero_lite.test_pipeline
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate
python -m unittest ml.alphazero_lite.test_runpod_wrappers
```

Expected: PASS

- [ ] **Step 2: Run repository formatting and lint hooks**

Run:

```bash
.venv/bin/python -m pre_commit run --all-files
```

Expected: PASS

- [ ] **Step 3: Inspect the final diff before handoff**

Run:

```bash
git diff -- ml/alphazero_lite/worker_config.py ml/alphazero_lite/pipeline.py ml/alphazero_lite/superhuman_runtime_config.py script/ai/run_local_superhuman_recovery ml/alphazero_lite/test_pipeline.py ml/alphazero_lite/test_promote_runpod_candidate.py
```

Expected: diff shows a single shared worker source of truth, pipeline normalization, runtime-config normalization, and matching tests.

- [ ] **Step 4: Commit the verification-ready change set**

Run:

```bash
git add ml/alphazero_lite/worker_config.py ml/alphazero_lite/pipeline.py ml/alphazero_lite/superhuman_runtime_config.py script/ai/run_local_superhuman_recovery ml/alphazero_lite/test_pipeline.py ml/alphazero_lite/test_promote_runpod_candidate.py
git commit -m "feat: default alphazero worker commands to 24"
```

Expected: final commit succeeds after tests and hooks are green.

### Self-Review

- Spec coverage: covered the shared helper module, the pipeline injection point, the superhuman runtime-config injection point, default `24`, append-when-missing behavior, and explicit override precedence.
- Placeholder scan: no `TODO` / `TBD` / deferred implementation notes remain.
- Type consistency: the plan consistently uses `normalize_command_workers()` for command lists and `apply_worker_overrides()` for config-wide rewriting.

Plan complete and saved to `docs/superpowers/plans/2026-05-28-shared-worker-default-plan.md`. Two execution options:

**1. Subagent-Driven (recommended)** - I dispatch a fresh subagent per task, review between tasks, fast iteration

**2. Inline Execution** - Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
