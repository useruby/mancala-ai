# Seed 629 — execution supervisor readiness

**Preparation and synthetic unit tests only. No Seed 624 state/game generation,
inference, search, solving, training, production keys, signatures, or authorized
freeze were created. The production owner allowlists remain empty.**

## Supervisor contract

`ml.alphazero_lite.seed629_execution_supervisor` retains the planned API and
validates a pre-created, dedicated, empty cgroup-v2 hierarchy, controller set,
ownership, canonical path, writable `cgroup.procs`, `cgroup.kill`, and memory,
CPU and PID controls. Its empty-group `cgroup.kill` probe does not establish
successful enforcement. Final-accounting classification covers CPU, memory
events/peak, monotonic wall duration, residual descendants, and external signal
termination. Cgroup memory is not equivalent to RSS; the planned cap is
conservative. CPU accounting uses cumulative `usage_usec`, not `CPUQuota`.

**The launch path is unconditionally disabled.** `run_authorized()` raises
before authorization parsing, path creation, or worker creation. The
preflight also rejects unconditionally because there is no verified hard quota
backend. Caller-supplied quota fields, environment variables, and synthetic
flags cannot enable it. The Seed 624 proposal template records
`disabled_pending_verified_hard_quota_backend`.

Output and scratch paths require a hard combined quota. The current supervisor
does not implement or trust caller-provided quota assertions; absence of a
verified quota backend permanently blocks this launch path. Unsupported quota,
cgroup, or privilege configurations never fall back to unconfined execution.
Future resource termination records require the existing Seed 624 frozen,
create-only `external_resource_limit_termination` receipt.

## Evidence verification

Run `python -m ml.alphazero_lite.verify_seed629_supervisor RESULT FREEZE
--receipt RECEIPT` on a physically separate copy. It checks record-format and
internal consistency only, and labels its result
`record_consistent_not_os_enforcement_verified`; it never asserts OS
enforcement based on publisher JSON. A claimed resource termination requires
the exact frozen Seed 624 failure receipt. Future independent OS verification
must additionally consume raw cgroupfs controls/statistics/events and
filesystem-native quota records collected by a separate privileged,
root-owned collector and authenticated outside the publisher. Kernel identity
and collector provenance must also be independently pinned.

## Synthetic report and current status

The synthetic worker supports bounded success, allocation, CPU, sleep,
file-write and child-process modes. Privileged integration was not run: this
host has no delegated writable worker cgroup or hard quota setup. No limit is
OS-enforcement verified.

| Limit | Status | Evidence |
| --- | --- | --- |
| Process memory | Implemented; unit-tested classification; OS enforcement blocked | cgroup preflight/final OOM evidence logic; privileged OOM run unavailable |
| Cumulative CPU seconds | Implemented; unit-tested boundaries; OS enforcement blocked | final `cpu.stat` classification; delegated cgroup unavailable |
| Monotonic wall deadline | Implemented; unit-tested boundary; OS enforcement blocked | monotonic final-duration classification; privileged worker termination unavailable |
| Cross-process concurrency | Implemented; unit-tested lock lifecycle; OS enforcement unverified | `flock` slot helper; cross-process contention integration not run |
| Combined output/scratch quota | Blocked; launch disabled | No verified hard quota backend or privileged exhaustion test |

Production readiness remains blocked on implementing and independently
authenticating a hard quota backend, host path/ownership policy, completing the
privileged synthetic matrix (including restart, receipt collision, partial
preservation and copied evidence), and independent review. The freeze template remains
`proposal_not_ready_or_approved`; this change neither signs nor authorizes it.
