# Seed 630 — hard quota and privileged enforcement qualification

## Result and host capability

No prospective benchmark, inference, search, solving, or training was run. The
Seed 629 production `run_authorized()` guard remains unconditional. No host mount,
cgroup delegation, or production key was changed.

Observed on the development host during qualification:

| Capability | Observation | Conclusion |
| --- | --- | --- |
| Kernel | Linux; unified cgroup v2 mounted at `/sys/fs/cgroup` | Mount alone does not imply delegation |
| Worker privilege | UID 1000 (`alex`), not root | Cannot create and independently attest privileged test controls |
| Current cgroup | `/user.slice/user-1000.slice/user@1000.service/app.slice/herdr.service` | No dedicated writable Seed 630 child cgroup was provisioned |
| Repository filesystem | `/dev/md3`, ext4, mount `/` | Not XFS; no project-quota filesystem for the requested test |
| Isolated disposable XFS | Not provisioned | Storage exhaustion experiment blocked |

These are environment privilege/capability blockers, not evidence of a Seed 629
implementation defect. No experiment is reported as OS-enforcement verified.

## Implementation and execution boundary

`seed630_xfs_project_quota.verify_shared_project_quota()` is a read-only,
root-only verifier for two disjoint directories sharing one XFS project ID and
project-inheritance flag. It checks the real mount source/type/options, rejects
symlink and nested-mount escape paths, reads project IDs with `FS_IOC_FSGETXATTR`,
requires accounting and enforcement both `ON` in `xfs_quota state -p`, and
requires an exact finite hard limit from `xfs_quota report -p -n -b`. The
`xfs_quota(8)` report documents block quantities as 1K blocks; the parser
converts them using 1024 bytes per block. It does not configure quotas and does
not accept a caller-provided enforcement flag. Preflight also rejects active
user/group quota enforcement and project inode limits so a storage denial can be
attributed to the project byte quota. The worker distinguishes `EDQUOT` from
`ENOSPC` with a fixed JSON failure record and exit code; the verifier requires
`EDQUOT`, active project enforcement, project usage near the hard limit, and
matching raw filesystem records.
The synthetic harness uses a dedicated namespace and only the registered
`seed629_synthetic_worker`; production authorization is never accepted by that
entry point. The freeze binds the worker source digest, and the harness starts
its absolute source path in isolated Python mode with a sanitized environment.
A forked child blocks before exec until the supervisor attaches it to cgroup;
stdout/stderr go to anonymous memfds and the worker is waited/reaped without
communicating with a closed stdin pipe. Run records are create-only, fsynced,
and stored outside worker-writable paths.

The production Seed 629 path remains blocked pending verified quota integration.
The new harness is an integration-test implementation, not a claim that this
host successfully ran it. Resource experiments require an operator-created
test cgroup and a disposable project-quota filesystem before launch.

## Five-limit report

| Limit | Classification | Evidence / reason |
| --- | --- | --- |
| Memory | `blocked` | No delegated dedicated cgroup; no `memory.max`/`memory.events` OOM run |
| Cumulative CPU | `blocked` | No delegated dedicated cgroup; no frozen-budget `cpu.stat` termination run |
| Wall time | `blocked` | No privileged worker process-tree run or independent cgroup termination evidence |
| Concurrency | `unit_tested_only` | Seed 629 tests cover flock descriptor lifecycle; no independent supervisor contention/recovery experiment was run |
| Combined storage | `blocked` | Host filesystem is ext4; no isolated XFS project quota or denial test |

The harness currently emits synthetic supervisor observations only. They are
not independently collected OS evidence. The collector and verifier
implementations have not been exercised on an isolated privileged host. No
`os_enforcement_verified` status is warranted. The complete failure/recovery
matrix and a physical-copy verification run remain explicit blockers.

## Safe isolated-host operator setup

Run only on a disposable Linux VM with XFS tools installed, root access, and no
production data on the test disk. These commands create a loopback test image;
they do not reconfigure an existing filesystem or the host's production cgroup.
Choose a dedicated location with sufficient free space and set a unique project
ID. Stop if the image, mountpoint, cgroup parent, or project ID is already in use.

1. Provision a sparse test image on the disposable VM's scratch disk, format it
   as XFS with project quotas, and mount it at a newly created empty mountpoint:

   ```sh
   truncate -s 2G /var/tmp/seed630-test.img
   mkfs.xfs -f /var/tmp/seed630-test.img
   mkdir -m 0700 /mnt/seed630-test
   mount -o loop,prjquota /var/tmp/seed630-test.img /mnt/seed630-test
   ```

2. Create a fresh project-map entry (for example ID `63001`, name
   `seed630-test`), apply that project ID and inheritance flag to an empty test
   root, and create disjoint `output` and `scratch` children. Configure one
   project hard limit equal to the frozen combined byte budget. For example,
   after manually confirming these paths and the unique project ID:

   ```sh
   BUDGET_BYTES=1073741824  # Replace with a freeze value divisible by 1024.
   printf '63001:/mnt/seed630-test/tree\n' >> /etc/projects
   printf 'seed630-test:63001\n' >> /etc/projid
   mkdir -m 0770 /mnt/seed630-test/tree
   mkdir -m 0770 /mnt/seed630-test/tree/output /mnt/seed630-test/tree/scratch
   xfs_quota -x -c 'project -s seed630-test' /mnt/seed630-test
   xfs_quota -x -c "limit -p bhard=${BUDGET_BYTES}B seed630-test" /mnt/seed630-test
   xfs_quota -x -c 'report -p -n -b' /mnt/seed630-test
   xfs_quota -x -c 'state -p' /mnt/seed630-test
   ```

   Set root ownership and verify project ID plus inheritance on both child
   directories (for example with `xfs_io -c 'stat'`). Do not proceed unless
   report readback converts to the exact frozen byte value and `state -p` shows
   project accounting and enforcement `ON`. Accounting-only and disabled states
   fail preflight. These `/etc` changes are for the disposable VM only.

3. Provision a delegated cgroup-v2 subtree specifically for the test worker,
   with `cpu`, `memory`, and `pids` controllers enabled by its parent. Verify
   the dedicated child is empty and writable and supports `cgroup.kill`,
   `memory.max`, `memory.oom.group`, `cpu.max`, `cpu.stat`, and `memory.events`.
   Use the disposable VM's service manager/delegation mechanism. Do not write to
   `/sys/fs/cgroup` outside that delegated subtree.

4. Create separate, empty output and scratch directories under the one XFS
   project tree, set the same project ID and inheritance on each, and ensure
   there are no nested mounts. Run quota preflight as root, then run each
   synthetic experiment with a new create-only run ID/freeze. Preserve the full
   raw cgroup controls/events, filesystem quota report, worker status, freeze,
   and partial artifacts before cleanup.

5. Provision an Ed25519 signing key restricted to root and independently retain
   its public key and the collector source digest. The verifier requires the
   pinned key and digest. Verify evidence from a physical copy on another
   machine. A worker-produced JSON result alone is never OS evidence. After
   evidence is copied and checked,
   terminate remaining descendants, confirm the cgroup is empty, and remove only
   the dedicated test cgroup, project tree, mount, and loopback image created
   for this run.

This procedure is preparation guidance, not a record that the privileged setup
or experiments have been performed. The collector and verifier still require
privileged integration qualification before any limit can be classified
`os_enforcement_verified`.
