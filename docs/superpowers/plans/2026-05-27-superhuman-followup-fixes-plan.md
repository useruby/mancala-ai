# Superhuman Follow-up Fixes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the sweep execute distinct challenger budgets, restore explicit baseline-path validation, and align the progress spec with shipped behavior.

**Architecture:** Keep the fixes narrow. The sweep will generate per-lane runtime configs instead of changing `local_promotion_gate`, the RunPod wrapper will add one explicit path check for user-supplied baselines, and the progress probe documentation will be updated to match the existing implementation.

**Tech Stack:** Python scripts, `unittest`, Markdown docs

---

### Task 1: Fix Budget Sweep Lane Rewriting

**Files:**
- Modify: `script/ai/superhuman_budget_sweep`
- Test: `ml/alphazero_lite/test_runpod_wrappers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_superhuman_budget_sweep_execute_budget_rewrites_lane_config(self):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodTrainingExperimentValidationTest.test_superhuman_budget_sweep_execute_budget_rewrites_lane_config`
Expected: FAIL because the lane config is not written or still contains the default `896` values.

- [ ] **Step 3: Write minimal implementation**

```python
lane_config_path = write_lane_config(...)
run_command([... , "--config-path", str(lane_config_path), ...], ...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodTrainingExperimentValidationTest.test_superhuman_budget_sweep_execute_budget_rewrites_lane_config`
Expected: PASS

### Task 2: Restore Explicit Baseline Path Validation

**Files:**
- Modify: `script/ai/runpod_training_experiment`
- Test: `ml/alphazero_lite/test_runpod_wrappers.py`

- [ ] **Step 1: Write the failing test**

```python
def test_runpod_training_experiment_rejects_missing_explicit_promotion_current_path(self):
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodTrainingExperimentValidationTest.test_runpod_training_experiment_rejects_missing_explicit_promotion_current_path`
Expected: FAIL because the script currently exits successfully and silently drops the path.

- [ ] **Step 3: Write minimal implementation**

```python
if args.promotion_current_path != "model-artifact/current":
    validate_existing_include_path(args.promotion_current_path, ...)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodTrainingExperimentValidationTest.test_runpod_training_experiment_rejects_missing_explicit_promotion_current_path`
Expected: PASS

### Task 3: Align Progress Spec Text

**Files:**
- Modify: `docs/superpowers/specs/2026-05-27-local-superhuman-recovery-progress-design.md`

- [ ] **Step 1: Update the spec text**

```md
- `train`
- `arena`
- `completed`
```

- [ ] **Step 2: Re-read the doc for consistency**

Check that the behavior and testing sections match the current script semantics and do not mention a separate emitted `gate` status.

### Task 4: Verify Targeted Coverage

**Files:**
- Test: `ml/alphazero_lite/test_runpod_wrappers.py`
- Test: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Run wrapper tests**

Run: `python -m unittest ml.alphazero_lite.test_runpod_wrappers`
Expected: PASS

- [ ] **Step 2: Run promotion/recovery tests**

Run: `python -m unittest ml.alphazero_lite.test_promote_runpod_candidate`
Expected: PASS
