# AlphaZero-lite Memory Speed Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add one reusable memory-speed profile for AlphaZero-lite generation commands so checkpoint self-play gets an auditable evaluator-cache setting and eligible bootstrap hybrid-teacher runs get auditable teacher-search reuse through the existing command normalization layers.

**Architecture:** Extend the existing shared command normalization layer instead of modifying individual experiment scripts. Keep the profile declarative at the config/runtime-config level, then have `ml/alphazero_lite/pipeline.py` and `ml/alphazero_lite/superhuman_runtime_config.py` apply one shared helper that rewrites only supported `self_play.py` and `generate_bootstrap_dataset.py` commands. Preserve explicit command flags unless the chosen profile is documented to set a higher-priority default.

**Tech Stack:** Python 3, `unittest`, existing pipeline/runtime-config helpers, JSON config files, wrapper scripts

---

### File Map

- Modify: `ml/alphazero_lite/worker_config.py`
- Modify: `ml/alphazero_lite/pipeline.py`
- Modify: `ml/alphazero_lite/superhuman_runtime_config.py`
- Modify: `ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json`
- Modify: `ml/alphazero_lite/test_pipeline.py`
- Create: `ml/alphazero_lite/test_superhuman_runtime_config.py`
- Verify: `.venv/bin/python -m unittest ml.alphazero_lite.test_pipeline`
- Verify: `.venv/bin/python -m unittest ml.alphazero_lite.test_superhuman_runtime_config`
- Verify: `.venv/bin/python -m unittest ml.alphazero_lite.test_local_superhuman_workflows`
- Verify: `.venv/bin/python -m pre_commit run --all-files`

### Task 1: Lock Down Shared Memory-Profile Command Rewriting With Failing Tests

**Files:**
- Modify: `ml/alphazero_lite/test_pipeline.py`

- [ ] **Step 1: Write the failing self-play profile test**

Add this near the existing `build_step_command()` tests:

```python
    def test_build_step_command_applies_memory_speed_profile_to_checkpoint_self_play(self):
        step = {
            "name": "self_play",
            "command": [
                sys.executable,
                "ml/alphazero_lite/self_play.py",
                "--games",
                "100",
                "--checkpoint",
                "parent.npz",
            ],
        }

        self.assertEqual(
            [
                sys.executable,
                "ml/alphazero_lite/self_play.py",
                "--games",
                "100",
                "--checkpoint",
                "parent.npz",
                "--workers",
                "24",
                "--evaluator-cache-size",
                "200000",
            ],
            build_step_command(step, memory_speed_profile="high_memory_local"),
        )
```

- [ ] **Step 2: Write the failing non-checkpoint self-play guardrail test**

Add this focused guardrail:

```python
    def test_build_step_command_does_not_add_cache_for_non_checkpoint_self_play(self):
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
            build_step_command(step, memory_speed_profile="high_memory_local"),
        )
```

- [ ] **Step 3: Write the failing bootstrap profile test**

Add the eligible-bootstrap case:

```python
    def test_build_step_command_applies_memory_speed_profile_to_hybrid_teacher_bootstrap(self):
        step = {
            "name": "mcts_bootstrap_dataset",
            "command": [
                sys.executable,
                "ml/alphazero_lite/generate_bootstrap_dataset.py",
                "--games",
                "32",
                "--position-selection-mode",
                "hybrid_teacher",
            ],
        }

        self.assertEqual(
            [
                sys.executable,
                "ml/alphazero_lite/generate_bootstrap_dataset.py",
                "--games",
                "32",
                "--position-selection-mode",
                "hybrid_teacher",
                "--workers",
                "24",
                "--teacher-search-reuse",
            ],
            build_step_command(step, memory_speed_profile="high_memory_local"),
        )
```

- [ ] **Step 4: Write the failing explicit-bootstrap-override guardrail test**

Add the profile precedence check:

```python
    def test_build_step_command_leaves_explicit_bootstrap_teacher_reuse_intact(self):
        command = [
            sys.executable,
            "ml/alphazero_lite/generate_bootstrap_dataset.py",
            "--games",
            "32",
            "--position-selection-mode",
            "hybrid_teacher",
            "--teacher-search-reuse",
        ]
        step = {"name": "mcts_bootstrap_dataset", "command": command}

        self.assertEqual(
            [
                sys.executable,
                "ml/alphazero_lite/generate_bootstrap_dataset.py",
                "--games",
                "32",
                "--position-selection-mode",
                "hybrid_teacher",
                "--teacher-search-reuse",
                "--workers",
                "24",
            ],
            build_step_command(step, memory_speed_profile="high_memory_local"),
        )
```

- [ ] **Step 5: Run the focused failing tests**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_applies_memory_speed_profile_to_checkpoint_self_play \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_does_not_add_cache_for_non_checkpoint_self_play \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_applies_memory_speed_profile_to_hybrid_teacher_bootstrap \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_leaves_explicit_bootstrap_teacher_reuse_intact
```

Expected: FAIL because `build_step_command()` currently does not accept a `memory_speed_profile` argument and cannot apply cache/reuse rewriting.

- [ ] **Step 6: Commit the red test checkpoint**

Run:

```bash
git add ml/alphazero_lite/test_pipeline.py
git commit -m "test: cover memory speed profile command rewriting"
```

Expected: commit succeeds with only the new failing pipeline tests.

### Task 2: Implement the Shared Memory-Speed Profile Helper

**Files:**
- Modify: `ml/alphazero_lite/worker_config.py`
- Modify: `ml/alphazero_lite/pipeline.py`
- Test: `ml/alphazero_lite/test_pipeline.py`

- [ ] **Step 1: Extend the shared command helper module**

Update `ml/alphazero_lite/worker_config.py` to keep worker normalization and add one profile-aware command normalizer:

```python
from __future__ import annotations

DEFAULT_WORKERS = 24
HIGH_MEMORY_LOCAL_PROFILE = "high_memory_local"
HIGH_MEMORY_LOCAL_EVALUATOR_CACHE_SIZE = 200000

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


def normalize_memory_speed_profile(
    command: list[str], *, memory_speed_profile: str | None = None
) -> list[str]:
    rendered = normalize_command_workers(command)
    if memory_speed_profile != HIGH_MEMORY_LOCAL_PROFILE or len(rendered) < 2:
        return rendered

    executable = rendered[1]
    if executable == "ml/alphazero_lite/self_play.py":
        if "--checkpoint" not in rendered:
            return rendered
        if "--evaluator-cache-size" in rendered:
            cache_index = rendered.index("--evaluator-cache-size")
            if cache_index + 1 < len(rendered):
                rendered[cache_index + 1] = str(HIGH_MEMORY_LOCAL_EVALUATOR_CACHE_SIZE)
                return rendered
        rendered.extend(
            ["--evaluator-cache-size", str(HIGH_MEMORY_LOCAL_EVALUATOR_CACHE_SIZE)]
        )
        return rendered

    if executable == "ml/alphazero_lite/generate_bootstrap_dataset.py":
        teacher_mode = "puct"
        if "--teacher-mode" in rendered:
            teacher_mode = rendered[rendered.index("--teacher-mode") + 1]
        position_mode = None
        if "--position-selection-mode" in rendered:
            position_mode = rendered[rendered.index("--position-selection-mode") + 1]
        if (
            teacher_mode == "puct"
            and position_mode == "hybrid_teacher"
            and "--teacher-search-reuse" not in rendered
        ):
            rendered.append("--teacher-search-reuse")
    return rendered
```

- [ ] **Step 2: Thread the profile into pipeline command building**

Update `ml/alphazero_lite/pipeline.py` imports, signature, and call sites:

```python
from ml.alphazero_lite.worker_config import normalize_memory_speed_profile


def build_step_command(
    step: dict, *, memory_speed_profile: str | None = None
) -> list[str] | object:
    command = step.get("command", [])
    if not isinstance(command, list) or not command:
        return command
    command = append_step_search_option_flags(step, command)
    return normalize_memory_speed_profile(
        command, memory_speed_profile=memory_speed_profile
    )
```

Pass the top-level config profile through both `run_step()` and dry-run gate rendering:

```python
    memory_speed_profile = config.get("memory_speed_profile")
```

and:

```python
    command = build_step_command(step, memory_speed_profile=memory_speed_profile)
```

- [ ] **Step 3: Run the focused pipeline tests to verify they pass**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_applies_memory_speed_profile_to_checkpoint_self_play \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_does_not_add_cache_for_non_checkpoint_self_play \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_applies_memory_speed_profile_to_hybrid_teacher_bootstrap \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_build_step_command_leaves_explicit_bootstrap_teacher_reuse_intact
```

Expected: PASS

- [ ] **Step 4: Commit the shared helper implementation**

Run:

```bash
git add ml/alphazero_lite/worker_config.py ml/alphazero_lite/pipeline.py ml/alphazero_lite/test_pipeline.py
git commit -m "feat: add memory speed profile command normalization"
```

Expected: commit succeeds with the new helper and pipeline integration.

### Task 3: Shift Repo-Level Config Assertions To Effective Profile Behavior

**Files:**
- Modify: `ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json`
- Modify: `ml/alphazero_lite/test_pipeline.py`

- [ ] **Step 1: Add the profile key to the superhuman phase2 config**

Update the top of `ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json`:

```json
{
  "run_id": "aggressive-v3-superhuman",
  "seed": 42,
  "iterations": 1,
  "start_iteration": 2,
  "replay_window": 3,
  "memory_speed_profile": "high_memory_local",
  "versions_dir": "/tmp/azlite_v3_superhuman_versions",
```

Also remove these command literals from the config so the profile becomes the source of truth:

```json
        "--evaluator-cache-size",
        "50000"
```

and, if present in a targeted hybrid-teacher config during implementation, remove direct `--teacher-search-reuse` there as well.

- [ ] **Step 2: Replace the repo-wide checkpoint cache literal assertion with rendered-profile coverage**

Update `test_checkpoint_self_play_configs_require_evaluator_cache_size` so it checks effective rendered commands instead of raw config literals for profiled configs:

```python
                    effective_command = build_step_command(
                        step,
                        memory_speed_profile=config.get("memory_speed_profile"),
                    )

                    if "--checkpoint" in effective_command:
                        found_checkpoint_self_play = True
                        self.assertIn("--evaluator-cache-size", effective_command)
                        cache_size_index = effective_command.index("--evaluator-cache-size")
                        self.assertEqual(
                            "200000",
                            effective_command[cache_size_index + 1],
                            msg=(
                                f"{config_path}: checkpoint self_play step must render "
                                "--evaluator-cache-size 200000 under the memory profile"
                            ),
                        )
```

- [ ] **Step 3: Replace the repo-wide bootstrap reuse assertion with rendered-profile coverage**

Update `test_bootstrap_configs_gate_teacher_search_reuse_by_hybrid_teacher_mode` to evaluate `effective_command` instead of the raw config command:

```python
                    effective_command = build_step_command(
                        step,
                        memory_speed_profile=config.get("memory_speed_profile"),
                    )
```

and then keep the existing hybrid-teacher + puct semantics checks against `effective_command`.

- [ ] **Step 4: Add a focused regression for the approved superhuman profile config**

Add this test near the existing aggressive-v3 config coverage:

```python
    def test_superhuman_phase2_profile_renders_memory_speed_flags(self):
        repo_root = Path(__file__).resolve().parents[2]
        config = load_config(
            repo_root / "ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json"
        )
        self_play = build_step_command(
            self.config_steps_by_name(config)["self_play"],
            memory_speed_profile=config.get("memory_speed_profile"),
        )
        bootstrap = build_step_command(
            self.config_steps_by_name(config)["mcts_bootstrap_dataset"],
            memory_speed_profile=config.get("memory_speed_profile"),
        )

        self.assertEqual("high_memory_local", config.get("memory_speed_profile"))
        self.assertEqual(
            "200000",
            self.command_flag_value(self_play, "--evaluator-cache-size"),
        )
        self.assertIn("--tree-reuse-enabled", self_play)
        self.assertIn("--tree-reuse-enabled", bootstrap)
```

- [ ] **Step 5: Run the targeted config regression block**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_checkpoint_self_play_configs_require_evaluator_cache_size \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_bootstrap_configs_gate_teacher_search_reuse_by_hybrid_teacher_mode \
  ml.alphazero_lite.test_pipeline.PipelineScriptTest.test_superhuman_phase2_profile_renders_memory_speed_flags
```

Expected: PASS and the approved superhuman config now expresses the profile declaratively.

- [ ] **Step 6: Commit the config/profile wiring**

Run:

```bash
git add ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json ml/alphazero_lite/test_pipeline.py
git commit -m "test: assert effective memory profile behavior"
```

Expected: commit succeeds with the config and test updates only.

### Task 4: Cover Runtime-Config Rewriting For Recovery Flows

**Files:**
- Create: `ml/alphazero_lite/test_superhuman_runtime_config.py`
- Modify: `ml/alphazero_lite/superhuman_runtime_config.py`

- [ ] **Step 1: Write the failing runtime-config unit test**

Create `ml/alphazero_lite/test_superhuman_runtime_config.py` with this focused test class:

```python
import unittest

from ml.alphazero_lite.superhuman_runtime_config import (
    apply_memory_speed_profile,
)


class SuperhumanRuntimeConfigTest(unittest.TestCase):
    def test_apply_memory_speed_profile_rewrites_supported_generation_steps(self):
        config = {
            "steps": [
                {
                    "name": "self_play",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/self_play.py",
                        "--checkpoint",
                        "parent.npz",
                    ],
                },
                {
                    "name": "mcts_bootstrap_dataset",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/generate_bootstrap_dataset.py",
                        "--position-selection-mode",
                        "hybrid_teacher",
                    ],
                },
                {
                    "name": "train",
                    "command": [
                        ".venv/bin/python",
                        "ml/alphazero_lite/train.py",
                        "--epochs",
                        "2",
                    ],
                },
            ]
        }

        updated = apply_memory_speed_profile(
            config, memory_speed_profile="high_memory_local"
        )

        self.assertEqual(
            "200000",
            updated["steps"][0]["command"][
                updated["steps"][0]["command"].index("--evaluator-cache-size") + 1
            ],
        )
        self.assertIn("--teacher-search-reuse", updated["steps"][1]["command"])
        self.assertEqual(
            [".venv/bin/python", "ml/alphazero_lite/train.py", "--epochs", "2"],
            updated["steps"][2]["command"],
        )
```

- [ ] **Step 2: Run the new runtime-config test to verify it fails**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_superhuman_runtime_config.SuperhumanRuntimeConfigTest.test_apply_memory_speed_profile_rewrites_supported_generation_steps
```

Expected: FAIL because `apply_memory_speed_profile()` does not exist yet.

- [ ] **Step 3: Implement the runtime-config profile application helper**

Update `ml/alphazero_lite/superhuman_runtime_config.py`:

```python
from ml.alphazero_lite.worker_config import (
    normalize_command_workers,
    normalize_memory_speed_profile,
)


def apply_memory_speed_profile(
    config: dict, *, memory_speed_profile: str | None = None
) -> dict:
    updated = copy.deepcopy(config)
    for step in updated.get("steps", []):
        command = step.get("command")
        if not isinstance(command, list) or not command:
            continue
        step["command"] = normalize_memory_speed_profile(
            command, memory_speed_profile=memory_speed_profile
        )
    return updated
```

Keep `apply_shared_worker_normalization()` intact for existing callers that only need worker rewriting.

- [ ] **Step 4: Run the runtime-config test to verify it passes**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_superhuman_runtime_config.SuperhumanRuntimeConfigTest.test_apply_memory_speed_profile_rewrites_supported_generation_steps
```

Expected: PASS

- [ ] **Step 5: Commit the runtime-config helper**

Run:

```bash
git add ml/alphazero_lite/superhuman_runtime_config.py ml/alphazero_lite/test_superhuman_runtime_config.py
git commit -m "feat: apply memory speed profile to runtime configs"
```

Expected: commit succeeds with the new runtime-config helper and unit test.

### Task 5: Wire The Recovery Wrapper To The New Profile Path And Verify End To End

**Files:**
- Modify: `script/ai/run_local_superhuman_recovery`
- Modify: `ml/alphazero_lite/test_local_superhuman_workflows.py`
- Test: `ml/alphazero_lite/test_superhuman_runtime_config.py`

- [ ] **Step 1: Add a failing workflow assertion for cache visibility in dry-run output**

Extend `RunLocalSuperhumanRecoveryTest.test_dry_run_without_workers_flag_uses_shared_default_runtime_workers` with this extra assertion after loading `runtime_config`:

```python
            self.assertEqual(
                "high_memory_local",
                runtime_config.get("memory_speed_profile"),
            )
            self_play_command = next(
                step["command"]
                for step in runtime_config["steps"]
                if step.get("name") == "self_play"
            )
            self.assertEqual(
                "200000",
                self_play_command[
                    self_play_command.index("--evaluator-cache-size") + 1
                ],
            )
```

- [ ] **Step 2: Run the workflow dry-run tests to verify they fail or stay red until wired**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_local_superhuman_workflows.RunLocalSuperhumanRecoveryTest.test_dry_run_without_workers_flag_uses_shared_default_runtime_workers \
  ml.alphazero_lite.test_local_superhuman_workflows.RunLocalSuperhumanRecoveryTest.test_dry_run_workers_override_rewrites_runtime_config
```

Expected: FAIL because the wrapper currently only applies shared worker normalization and does not materialize the memory-speed profile into the runtime config.

- [ ] **Step 3: Update the wrapper to materialize the profile-aware runtime config**

In `script/ai/run_local_superhuman_recovery`, import and use the new helper:

```python
from ml.alphazero_lite.superhuman_runtime_config import (
    apply_memory_speed_profile,
    apply_shared_worker_normalization,
    apply_runtime_paths,
    final_iteration,
    load_json,
    python_executable,
    rewrite_python_commands,
    write_json,
)
```

Then update `build_runtime_config()` so it materializes the profile before the worker rewrite:

```python
    config = apply_memory_speed_profile(
        config, memory_speed_profile=config.get("memory_speed_profile")
    )
    config = apply_shared_worker_normalization(config, workers=workers)
```

This keeps the runtime config auditable because the generated JSON will contain the top-level profile key and the effective command flags.

- [ ] **Step 4: Run the workflow dry-run tests to verify they pass**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_local_superhuman_workflows.RunLocalSuperhumanRecoveryTest.test_dry_run_without_workers_flag_uses_shared_default_runtime_workers \
  ml.alphazero_lite.test_local_superhuman_workflows.RunLocalSuperhumanRecoveryTest.test_dry_run_workers_override_rewrites_runtime_config
```

Expected: PASS and the generated runtime config shows both `memory_speed_profile` and the effective cache/reuse flags.

- [ ] **Step 5: Commit the wrapper integration**

Run:

```bash
git add script/ai/run_local_superhuman_recovery ml/alphazero_lite/test_local_superhuman_workflows.py
git commit -m "feat: materialize memory speed profile in recovery runtime config"
```

Expected: commit succeeds with the wrapper and workflow-test updates.

### Task 6: Run Full Targeted Verification And Clean Up

**Files:**
- Verify only

- [ ] **Step 1: Run the full targeted Python test set**

Run:

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_pipeline \
  ml.alphazero_lite.test_superhuman_runtime_config \
  ml.alphazero_lite.test_local_superhuman_workflows
```

Expected: PASS

- [ ] **Step 2: Run formatting/hooks verification**

Run:

```bash
.venv/bin/python -m pre_commit run --all-files
```

Expected: PASS

- [ ] **Step 3: Inspect worktree before handoff**

Run:

```bash
git status --short
git diff --stat
```

Expected: only the planned implementation files are changed and all verification has already passed.

- [ ] **Step 4: Commit the final verification-safe state**

Run:

```bash
git add ml/alphazero_lite/worker_config.py \
  ml/alphazero_lite/pipeline.py \
  ml/alphazero_lite/superhuman_runtime_config.py \
  ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json \
  ml/alphazero_lite/test_pipeline.py \
  ml/alphazero_lite/test_superhuman_runtime_config.py \
  ml/alphazero_lite/test_local_superhuman_workflows.py \
  script/ai/run_local_superhuman_recovery
git commit -m "feat: add alphazero memory speed profile"
```

Expected: commit succeeds with the final verified implementation.

## Self-Review

- Spec coverage: the plan covers the shared profile boundary, self-play cache handling, bootstrap teacher-search reuse gating, runtime-config auditability, targeted tests, and explicit non-goals around arena/training redesign.
- Placeholder scan: no `TODO`, `TBD`, or deferred implementation markers remain; every code-edit step includes concrete code or exact commands.
- Type consistency: the plan consistently uses `memory_speed_profile`, `high_memory_local`, `normalize_memory_speed_profile()`, and `apply_memory_speed_profile()` across helper, pipeline, runtime-config, and wrapper tasks.
