# RunPod Rerun Watch Design

## Goal

Recover the `aggressive-v3-superhuman-iter2` candidate by rerunning the existing superhuman launcher on RunPod without prematurely killing a healthy rerun.

## Problem

The current recovery flow uses two local shell scripts:

- `tmp/runpod_superhuman_rerun_retry.sh`
- `tmp/runpod_superhuman_rerun_watch.sh`

The retry script currently falls back to `cpu3c-2-4`. The watch script stops the pod after `4h20m` if a local result marker has not appeared. That is not a valid health signal for this launcher, because `runpod_training_experiment` downloads results only after the remote command exits. A long-running but healthy remote job can therefore have no local result marker for most or all of its execution.

## Recommended Approach

Use the minimal recovery fix:

1. Change the retry script back to the original higher-capacity profile, `cpu3c-16-32`.
2. Keep the watch script's hard timeout, but treat it only as a maximum pod lifetime after pod creation.
3. Keep the local result marker as an early-success exit only.
4. Do not add SSH-based remote progress polling in this change.

This keeps the fix focused on the confirmed root cause while avoiding extra moving parts.

## Alternatives Considered

### Remote progress-aware watcher

Poll remote `remote_run.log` over SSH and extend the run while the log is still advancing.

Trade-off: more accurate, but introduces SSH-health, credential, and remote-path handling into the watcher.

### Generalized recovery framework

Turn the retry and watch scripts into reusable parameterized helpers.

Trade-off: useful later, but adds scope unrelated to recovering this one candidate.

## Design

### Retry script

`tmp/runpod_superhuman_rerun_retry.sh` will continue to:

- poll every `300s`
- retry only on RunPod capacity failures
- exit immediately on non-capacity failures

The only behavior change is the pod profile:

- from `cpu3c-2-4`
- to `cpu3c-16-32`

### Watch script

`tmp/runpod_superhuman_rerun_watch.sh` will continue to:

- wait for pod creation
- record the first detected pod id
- stop the pod when the allowed lifetime is exceeded
- capture diagnostics before stopping

The important semantic clarification is:

- missing local result marker means only "remote run has not been downloaded yet"
- it does not mean "pod is stalled"

The script will still exit early if the local result marker appears. Otherwise it will wait until the pod either disappears or reaches the maximum allowed lifetime.

## Testing

Add one focused regression test that covers the watch decision semantics:

- if the result marker is absent
- and elapsed time is below the maximum
- the watcher should continue waiting rather than stop the pod

Keep test coverage minimal and local to this recovery flow.

## Error Handling

- Capacity errors continue to trigger retry sleep-and-retry behavior.
- Non-capacity launcher failures still stop the retry loop immediately.
- Watcher diagnostics remain in `tmp/runpod_superhuman_rerun_watch_diagnostics/` when a timeout stop happens.

## Scope Boundaries

This change does not:

- add remote SSH progress inspection
- generalize the scripts for other jobs
- change the main RunPod orchestration code
- change the `4h20m` timeout value itself

## Success Criteria

- The retry script requests `cpu3c-16-32`.
- The watch script no longer encodes the false assumption that missing local results implies no progress.
- Regression coverage exists for the corrected watcher behavior.
