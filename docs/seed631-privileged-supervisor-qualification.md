# Seed 631 — privileged synthetic enforcement qualification

## Outcome

**Overall production readiness: `blocked`.** No privileged experiment was run.
The current environment is not a disposable Linux VM provisioned for Seed 631,
and no explicit operator authorization to provision or modify an environment was
provided. Root availability alone is not authorization. No synthetic evidence is
represented as kernel-verified, and no production benchmark, inference, search,
solving, training, or production authorization was used.

The Seed 629 `run_authorized()` launch guard remains unconditional. This task
does not alter it.

## Environment and execution record

Read-only inspection of the available environment found:

| Requirement | Observation | Result |
| --- | --- | --- |
| Privileged worker setup | Process UID is 1000 (`alex`), not root | Blocked |
| Cgroup v2 | Unified cgroup v2 is mounted; current cgroup is the user service cgroup | No dedicated writable worker subtree; blocked |
| Required controllers and `cgroup.kill` | No dedicated Seed 631 subtree was provisioned for examination | Blocked; no enforcement claim |
| Disposable XFS project quota | Repository root is `/dev/md3` on ext4; no disposable XFS volume is provisioned | Blocked |
| Test-only Ed25519 key and independent pins | Not provisioned | Blocked |
| Physical-copy independent verification | No privileged evidence directory exists | Blocked |

These are missing-environment blockers, not results of failed enforcement
experiments. No automatic retry was attempted. The existing Seed 630 report
(`docs/seed630-hard-quota-qualification.md`) provides the same pre-run capability
assessment. An operator must provision and explicitly authorize a disposable VM
before privileged setup or experiment execution.

## Five-limit matrix

| Limit | Frozen limit | Observed kernel measurements | Mechanism and outcome | Evidence / independent verifier | Overshoot or uncertainty | Classification |
| --- | --- | --- | --- | --- | --- | --- |
| Process memory | No Seed 631 freeze issued | No cgroup `memory.max`, baseline/final `memory.events`, or worker exit observation | No OOM experiment; unavailable VM/cgroup | None | Unknown; no run | `blocked` |
| Cumulative CPU | No Seed 631 freeze issued | No `cpu.stat usage_usec` delta | No cumulative-budget experiment; `CPUQuota` is not treated as proof | None | Unknown; no run | `blocked` |
| Wall time | No Seed 631 freeze issued | No monotonic deadline/termination observation | No worker deadline experiment | None | Unknown; no run | `blocked` |
| Execution concurrency | No Seed 631 freeze issued | No overlapping supervisor intervals | Existing flock lifecycle has unit coverage; cross-process contention and abnormal recovery were not run | No signed OS evidence | Unit tests do not establish maximum OS-observed concurrency | `unit_tested_only` |
| Combined output/scratch storage | No Seed 631 freeze issued | No XFS project report or directory identity observations | No `EDQUOT` experiment; no quota mount exists | None | Unknown; no run; `ENOSPC` cannot be excluded from absent evidence | `blocked` |

No resource is classified `os_enforcement_verified`. In particular, a signed
manifest is not available and would not, by itself, establish enforcement.

## Runner boundary and remaining qualification work

`ml/alphazero_lite/seed631_privileged_runner.py` adds a one-shot entry point
restricted to the registered Seed 630 modes. It requires an externally supplied
exact freeze digest, validates the synthetic namespace and unique UUID run ID,
rejects production authorization, checks the frozen paths/configuration, and
reserves an exclusive root-private attempt record before invoking the existing
Seed 630 harness. The harness remains responsible for XFS/cgroup preflight and
worker source-digest validation. There is no retry path, arbitrary command
interface, or call to Seed 629 `run_authorized()`.

The runner's fail-closed boundary has unit tests only. The privileged matrix and
failure/recovery cases remain unqualified: normal success, memory/CPU/wall
termination, child termination, worker failure, supervisor interruption and
recovery, missing cgroup controls, disabled quota enforcement, wrong project
identity/limit, receipt collision, collector failure, signed-evidence tampering,
and physical-copy verification. Each actual attempt must use distinct frozen
identities and preserve its raw collector output and partial artifacts. The
existing collector/verifier still need integration qualification across these
attempts and must be independently pinned (public-key fingerprint and exact
source-bundle digest) before the resulting copied evidence can support any
enforcement claim.

The runner's reserved `seed631-synthetic-attempt-v1` record is metadata, not raw
OS evidence. The runner does not yet automatically invoke the privileged
collector for each attempt, and the current collector expects a completed Seed
630 run record. Collector-failure and pre-completion failure capture remain
unimplemented integration work; these records must never be represented as
collector-authenticated evidence.

The required next step is an explicitly authorized disposable VM matching the
Seed 630 readiness prerequisites. Until the full five-limit matrix and all
acceptance-path evidence are captured and independently verified from a
physical copy, supervisor production readiness remains blocked.
