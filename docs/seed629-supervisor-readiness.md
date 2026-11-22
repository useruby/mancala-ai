# Seed 629 — execution supervisor readiness

**Preparation and synthetic unit tests only. No Seed 624 state/game generation,
inference, search, solving, training, production keys, signatures, or authorized
freeze were created. The production owner allowlists remain empty.**

## Supervisor contract

`ml.alphazero_lite.seed629_execution_supervisor` accepts the existing Seed 624
authenticated authorization binding; it does not accept keys or substitute an
authorization scheme. It requires a pre-created delegated cgroup v2 with CPU,
memory and pids controllers, writable `cgroup.procs`, and `cgroup.kill`. It
applies `memory.max`, `memory.oom.group=1`, CPU accounting and `pids.max`, moves
the gated worker before releasing it, polls `cpu.stat`, `memory.current`'s
companion peak, and `memory.events`, and kills the whole cgroup on detected
limits. The cgroup memory ceiling is 80% of the frozen RSS value by default;
this is a conservative cap, not an assertion that cgroup memory accounting
equals process RSS. CPU is cumulative `usage_usec`, not `CPUQuota`. Wall time is
monotonic and enforced by the supervisor. Separate OS `flock` slots coordinate
independent supervisor processes.

Output and scratch paths must be create-only and placed on a pre-provisioned
quota-limited filesystem. The caller must provide the exact quota identity and
limit matching the freeze. A production operator must establish that quota
outside the process and independently audit its kernel/filesystem record; no
generic quota-control interface exists across supported Linux filesystems.
Unsupported quota, cgroup, or privilege configurations are blockers and must
not fall back to unconfined execution. The result record is exclusive-create,
durable JSON; partial paths and failure receipt hash are retained. Externally
terminated worker receipts use Seed 624's existing
`external_resource_limit_termination` category.

## Evidence verification

Run `python -m ml.alphazero_lite.verify_seed629_supervisor RESULT FREEZE
--receipt RECEIPT` on a physically separate copy. The verifier reconstructs
the exact freeze digest and run identity, compares frozen limits, checks cgroup
measurements and resource classification, checks receipt identity/hash, and
checks artifact paths for symlinks. It does not use a publisher success flag.

## Synthetic report and current status

The dedicated synthetic worker supports bounded success, allocation, CPU,
sleep, file-write and child-process modes. Available local checks are the
unit/fail-closed tests. In this environment, privileged integration was not
run: actual delegated cgroup v2 and hard project-quota setup have not been
established. Therefore none of the five limits is reported as OS-enforcement
verified. CPU, memory, wall time, concurrency and storage are **unit-tested
only** for configuration/rejection logic; actual OS enforcement is **blocked**.
The current tests intentionally do not claim enforcement based on mocked
measurements.

| Limit | Status | Evidence |
| --- | --- | --- |
| Process memory | Blocked | Requires delegated memory controller and privileged OOM integration run |
| Cumulative CPU seconds | Blocked | Requires delegated cgroup and privileged controlled CPU integration run |
| Monotonic wall deadline | Blocked | Requires privileged cgroup termination integration run |
| Cross-process concurrency | Unit-tested only | OS `flock` slots implemented; separate-process contention test still required |
| Combined output/scratch quota | Blocked | Hard quota backend and privileged quota exhaustion test are host/filesystem-specific |

Production readiness remains blocked on a verified quota backend, fixing and
auditing the host path/ownership policy, completing the privileged synthetic
matrix (including restart, receipt collision, partial preservation and copied
evidence), and independent review. The freeze template remains
`proposal_not_ready_or_approved`; this change neither signs nor authorizes it.
