# RunPod Rerun Watch Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent the superhuman recovery watcher from killing a healthy rerun just because no local results have been downloaded yet, and restore the rerun retry flow to the original `cpu3c-16-32` pod profile.

**Architecture:** Introduce one tiny Python helper that encodes the watcher timeout decision so the behavior is testable, then have the shell watcher call that helper rather than embedding the incorrect local-result assumption in shell flow. Keep the retry script structure unchanged except for switching the pod profile back to `cpu3c-16-32`.

**Tech Stack:** Python 3, `unittest`, Bash, existing RunPod wrapper scripts

---

### File Map

- Create: `ml/alphazero_lite/runpod_rerun_watch.py`
- Modify: `ml/alphazero_lite/test_runpod_wrappers.py`
- Modify: `tmp/runpod_superhuman_rerun_watch.sh`
- Modify: `tmp/runpod_superhuman_rerun_retry.sh`
- Verify: `python -m unittest ml.alphazero_lite.test_runpod_wrappers`

### Task 1: Add failing regression test for watcher timeout semantics

**Files:**
- Modify: `ml/alphazero_lite/test_runpod_wrappers.py`
- Create later in Task 2: `ml/alphazero_lite/runpod_rerun_watch.py`

- [ ] **Step 1: Write the failing test**

Add a new test class near the other RunPod wrapper coverage:

```python
class RunpodRerunWatchTest(unittest.TestCase):
    def test_should_keep_waiting_when_marker_missing_but_timeout_not_reached(self):
        from ml.alphazero_lite.runpod_rerun_watch import should_stop_watcher

        self.assertFalse(
            should_stop_watcher(
                result_marker_exists=False,
                elapsed_seconds=900,
                max_seconds=15600,
            )
        )
```

- [ ] **Step 2: Run the test to verify it fails**

Run:

```bash
python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodRerunWatchTest.test_should_keep_waiting_when_marker_missing_but_timeout_not_reached
```

Expected: FAIL with `ModuleNotFoundError` for `ml.alphazero_lite.runpod_rerun_watch`.

- [ ] **Step 3: Commit**

```bash
git add ml/alphazero_lite/test_runpod_wrappers.py
git commit -m "test: cover rerun watcher timeout semantics"
```

### Task 2: Implement the minimal watcher decision helper

**Files:**
- Create: `ml/alphazero_lite/runpod_rerun_watch.py`
- Test: `ml/alphazero_lite/test_runpod_wrappers.py`

- [ ] **Step 1: Write the minimal implementation**

Create `ml/alphazero_lite/runpod_rerun_watch.py` with:

```python
def should_stop_watcher(*, result_marker_exists: bool, elapsed_seconds: int, max_seconds: int) -> bool:
    if result_marker_exists:
        return False
    return elapsed_seconds >= max_seconds
```

- [ ] **Step 2: Run the focused test to verify it passes**

Run:

```bash
python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodRerunWatchTest.test_should_keep_waiting_when_marker_missing_but_timeout_not_reached
```

Expected: PASS

- [ ] **Step 3: Add one timeout-edge regression assertion**

Extend the same test class with:

```python
    def test_should_stop_once_timeout_is_reached_without_result_marker(self):
        from ml.alphazero_lite.runpod_rerun_watch import should_stop_watcher

        self.assertTrue(
            should_stop_watcher(
                result_marker_exists=False,
                elapsed_seconds=15600,
                max_seconds=15600,
            )
        )
```

- [ ] **Step 4: Run both regression tests**

Run:

```bash
python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodRerunWatchTest
```

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add ml/alphazero_lite/runpod_rerun_watch.py ml/alphazero_lite/test_runpod_wrappers.py
git commit -m "fix: define rerun watcher timeout decision"
```

### Task 3: Wire the shell watcher to the tested timeout decision

**Files:**
- Modify: `tmp/runpod_superhuman_rerun_watch.sh`
- Read for context if needed: `script/ai/runpod_remote_run.sh`

- [ ] **Step 1: Update the shell watcher to call the helper**

Replace the inline timeout branch in `tmp/runpod_superhuman_rerun_watch.sh` with a Python call that imports the helper and exits success when the pod should be stopped. The inserted block should look like:

```bash
      if python3 - <<'PY' "$RESULT_MARKER" "$elapsed" "$MAX_SECONDS"
from pathlib import Path
import sys

from ml.alphazero_lite.runpod_rerun_watch import should_stop_watcher

result_marker = Path(sys.argv[1]).is_file()
elapsed_seconds = int(sys.argv[2])
max_seconds = int(sys.argv[3])

raise SystemExit(
    0
    if should_stop_watcher(
        result_marker_exists=result_marker,
        elapsed_seconds=elapsed_seconds,
        max_seconds=max_seconds,
    )
    else 1
)
PY
      then
        log "timeout exceeded after ${elapsed}s without completed download"
        capture_diagnostics
        if [ -n "$ACTIVE_POD_ID" ]; then
          log "stopping pod id=$ACTIVE_POD_ID"
          runpodctl pod stop "$ACTIVE_POD_ID" >> "$LOG_PATH" 2>&1 || true
        fi
        exit 0
      fi
```

- [ ] **Step 2: Keep the early success exit intact**

Verify the script still has this separate early-return block before pod discovery:

```bash
    if [ -f "$RESULT_MARKER" ]; then
      log "result marker detected; watcher exiting"
      exit 0
    fi
```

- [ ] **Step 3: Run the watcher regression tests again**

Run:

```bash
python -m unittest ml.alphazero_lite.test_runpod_wrappers.RunpodRerunWatchTest
```

Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add tmp/runpod_superhuman_rerun_watch.sh ml/alphazero_lite/runpod_rerun_watch.py ml/alphazero_lite/test_runpod_wrappers.py
git commit -m "fix: stop rerun watcher only at timeout"
```

### Task 4: Restore the high-capacity rerun profile and verify the full wrapper suite

**Files:**
- Modify: `tmp/runpod_superhuman_rerun_retry.sh`
- Verify: `ml/alphazero_lite/test_runpod_wrappers.py`

- [ ] **Step 1: Change the retry profile back to the original pod size**

Update this line in `tmp/runpod_superhuman_rerun_retry.sh`:

```bash
POD_PROFILE="cpu3c-16-32"
```

- [ ] **Step 2: Run the focused RunPod wrapper suite**

Run:

```bash
python -m unittest ml.alphazero_lite.test_runpod_wrappers
```

Expected: PASS

- [ ] **Step 3: Dry-run the main superhuman wrapper for a smoke check**

Run:

```bash
script/ai/runpod_superhuman_experiment --config-path ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json --dry-run
```

Expected: JSON output showing the wrapped RunPod plan with no shell or import errors.

- [ ] **Step 4: Commit**

```bash
git add tmp/runpod_superhuman_rerun_retry.sh tmp/runpod_superhuman_rerun_watch.sh ml/alphazero_lite/runpod_rerun_watch.py ml/alphazero_lite/test_runpod_wrappers.py
git commit -m "fix: restore superhuman rerun recovery settings"
```

### Task 5: Launch the corrected recovery flow

**Files:**
- Use: `tmp/runpod_superhuman_rerun_retry.sh`
- Use: `tmp/runpod_superhuman_rerun_watch.sh`
- Observe: `tmp/runpod_superhuman_rerun_retry.log`
- Observe: `tmp/runpod_superhuman_rerun_watch.log`

- [ ] **Step 1: Start the retry loop**

Run:

```bash
bash tmp/runpod_superhuman_rerun_retry.sh
```

Expected: either a capacity-retry log line or a launched RunPod rerun attempt.

- [ ] **Step 2: Start the watcher in parallel**

Run in a second shell:

```bash
bash tmp/runpod_superhuman_rerun_watch.sh
```

Expected: the watcher logs pod creation, waits for either a result marker or true timeout, and no longer treats missing local results as immediate evidence of failure.

- [ ] **Step 3: Confirm the rerun is using the intended pod profile**

Run:

```bash
python - <<'PY'
import json
import subprocess

pods = json.loads(subprocess.check_output(["runpodctl", "pod", "list"], text=True))
for pod in pods:
    if pod.get("name") == "azlite-runpod-training-capacity-retry":
        print(pod.get("id"), pod.get("vcpuCount"), pod.get("memoryInGb"))
        break
PY
```

Expected: the active rerun pod reports `16` vCPU and `32` GB memory once capacity is available.

- [ ] **Step 4: Commit**

```bash
git add docs/superpowers/plans/2026-05-27-runpod-rerun-watch-plan.md
git commit -m "docs: add rerun watch recovery plan"
```
