# Local Superhuman Recovery Progress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small progress probe that reports the active stage and milestone outputs for `run_local_superhuman_recovery`, then run it against the server session.

**Architecture:** Implement one focused Python wrapper script that inspects the recovery output directory and infers status from config plus milestone files. Keep the logic local to the script and cover it with a few narrow tests for status transitions.

**Tech Stack:** Python 3, `unittest`, existing recovery output conventions

---

### File Map

- Create: `script/ai/local_superhuman_recovery_progress`
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`
- Verify: `python -m unittest ml.alphazero_lite.test_promote_runpod_candidate`

### Task 1: Add failing tests for progress-status reporting

**Files:**
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Write failing tests**

Add a test class that shells out to `script/ai/local_superhuman_recovery_progress` against a temporary output root and covers:

- config only -> `self_play`
- iter2 self-play plus iter1 bootstrap -> `train`
- gate report present -> `completed`

- [ ] **Step 2: Run the focused tests to verify they fail**

Run:

```bash
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate.LocalSuperhumanRecoveryProgressTest
```

Expected: FAIL because the progress script does not exist yet.

### Task 2: Implement the minimal progress probe

**Files:**
- Create: `script/ai/local_superhuman_recovery_progress`
- Test: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Create the script**

Implement a wrapper that:

- accepts optional `--output-root`
- reads `pipeline_config.json` when present
- derives key milestone paths
- emits JSON with file existence, size, mtime, and status

- [ ] **Step 2: Keep status logic simple**

Use this order:

- `completed` if gate report exists
- `arena` if arena report exists
- `train` if checkpoint or model exists, or if bootstrap is complete
- `bootstrap` if iter2 self-play exists and iter1 bootstrap is still being built
- `self_play` if config exists but later milestones are absent
- `not_started` otherwise

- [ ] **Step 3: Run the focused tests to verify they pass**

Run:

```bash
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate.LocalSuperhumanRecoveryProgressTest
```

Expected: PASS

### Task 3: Verify and run the probe on the server session

**Files:**
- Create: `script/ai/local_superhuman_recovery_progress`
- Modify: `ml/alphazero_lite/test_promote_runpod_candidate.py`

- [ ] **Step 1: Run the full wrapper test module**

Run:

```bash
python -m unittest ml.alphazero_lite.test_promote_runpod_candidate
```

Expected: PASS

- [ ] **Step 2: Dry-run the progress probe locally**

Run:

```bash
script/ai/local_superhuman_recovery_progress
```

Expected: JSON summary or `not_started` style payload, with no crash.

- [ ] **Step 3: Sync the new script and tests to the server**

Run:

```bash
rsync -azR script/ai/local_superhuman_recovery_progress ml/alphazero_lite/test_promote_runpod_candidate.py mancala-ai:/home/alex/Mancala/ai/server-session/
```

Expected: remote server session receives the probe script.

- [ ] **Step 4: Run the probe on the server session**

Run:

```bash
ssh mancala-ai 'cd /home/alex/Mancala/ai/server-session && script/ai/local_superhuman_recovery_progress'
```

Expected: JSON summary showing current stage and milestone outputs for the active recovery job.
