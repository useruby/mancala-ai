# Local Superhuman Recovery Workers Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--workers` override to `script/ai/run_local_superhuman_recovery` so server-native recovery runs can use higher CPU parallelism without mutating the shared base config.

**Architecture:** Keep the change local to `run_local_superhuman_recovery` by parsing one optional CLI flag and rewriting `--workers` values in the generated runtime config before execution. Cover the behavior with one focused regression test that inspects the produced dry-run payload/runtime config.

**Tech Stack:** Python 3, `unittest`, existing recovery wrapper script

---

### File Map

- Modify: `script/ai/run_local_superhuman_recovery`
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`
- Verify: `python -m unittest ml.alphazero_lite.test_promote_runpod_candidate`

### Task 1: Add failing test for worker override

**Files:**
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Write the failing test**

Add a focused test that runs the wrapper in `--dry-run` mode with `--workers 24`, then reads the emitted runtime config and asserts `--workers 24` is present in the worker-based steps.

- [ ] **Step 2: Run the focused test to verify it fails**

Run:

```bash
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate.RunLocalSuperhumanRecoveryTest.test_dry_run_workers_override_rewrites_runtime_config
```

Expected: FAIL because `run_local_superhuman_recovery` does not yet accept `--workers` or rewrite worker flags.

### Task 2: Implement the minimal override

**Files:**
- Modify: `script/ai/run_local_superhuman_recovery`
- Test: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Add the optional CLI flag**

Parse:

```python
    parser.add_argument("--workers", type=int, default=None)
```

- [ ] **Step 2: Rewrite runtime-config worker flags when provided**

Add a tiny helper that replaces the argument after each `--workers` token in generated step commands.

- [ ] **Step 3: Apply the helper during runtime-config construction**

Keep path/python rewrites intact and only apply the worker rewrite when `args.workers` is not `None`.

- [ ] **Step 4: Run the focused test to verify it passes**

Run:

```bash
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate.RunLocalSuperhumanRecoveryTest.test_dry_run_workers_override_rewrites_runtime_config
```

Expected: PASS

### Task 3: Verify broader wrapper behavior and sync to server

**Files:**
- Modify: `script/ai/run_local_superhuman_recovery`
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Run the wrapper test module**

Run:

```bash
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate
```

Expected: PASS

- [ ] **Step 2: Dry-run the wrapper locally with workers override**

Run:

```bash
script/ai/run_local_superhuman_recovery --dry-run --workers 24
```

Expected: JSON output whose `runtime_config_path` points at a config containing worker-based steps set to `24`.

- [ ] **Step 3: Sync changed files to the server**

Sync:

```bash
rsync -az script/ai/run_local_superhuman_recovery ml/alphazero_lite/test_promote_runpod_candidate.py mancala-ai:/home/alex/Mancala/ai/server-session/
```

Expected: remote server session receives the updated wrapper and test file.

- [ ] **Step 4: Provide the clean restart command**

Use:

```bash
cd /home/alex/Mancala/ai/server-session
pkill -f "script/ai/run_local_superhuman_recovery|ml/alphazero_lite/pipeline.py|ml/alphazero_lite/generate_bootstrap_dataset.py"
script/ai/run_local_superhuman_recovery --run-gate --workers 24
```

Expected: the regenerated runtime config uses `24` workers on the worker-based steps.
