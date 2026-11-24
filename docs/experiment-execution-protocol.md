# Mancala AI experiment execution protocol

**Status:** proposed common operating procedure. An experiment still requires its
own explicit owner authorization and scientific preregistration. This protocol
does not authorize any workload or candidate promotion.

## Purpose and conventions audited

The repository already uses isolated Git worktrees for long-running research
branches (for example `ai-seed604-*`), frozen JSON registrations and execution
freezes, immutable attempt ledgers, hash-bound archives, and separate replay or
publication verifiers. `ml/alphazero_lite/run_manifest.py` provides a minimal
legacy run record; `runpod_experiment.py` captures logs and bundles source for
its separate cloud workflow, while historical archive helpers use ordinary
tar.gz. Those remain available for their existing uses; the new utility follows
the same manifest/hash patterns without adding a scheduler or cloud dependency.
Experiment-specific scripts also own the resource controls
that matter to their workload: pair-level concurrency, search workers,
checkpoint cadence, exact-solver budgets, and stop conditions. Existing verifiers
often reconstruct results from a physical copy and are the precedent for
scientific verification.

The common utility is `ml.alphazero_lite.experiment_lifecycle`. Its default
worktree root is the repository's already ignored `./.worktrees/`; set
`MANCALA_EXPERIMENT_WORKTREE_ROOT` or pass `--worktree-root` to change it. Set
`MANCALA_EXPERIMENT_ARCHIVE_ROOT` or pass `--archive-root` to choose external
archive storage. The utility uses ordinary Git, Python, and `zstd` when
available; without `zstd` it uses standard-library gzip tar archives. Runtime
attempt files are under `.experiment/runtime/` and ignored by the experiment
worktree's local `.experiment/.gitignore`. Registration and publication files
remain ordinary reviewable repository files.

These pieces remain useful: scientific registrations/freezes, independent
input authentication, per-attempt identities, append-only failure history,
domain-specific resource controls, exact-oracle validation, and independent
verification. The Seed 629–631 requirement for privileged per-experiment
cgroups, XFS quotas, a disposable VM, root collection, and test signing keys is
proposed for supersession by the separate amendment
[`seed623-631-execution-contract-amendment.md`](seed623-631-execution-contract-amendment.md).
Historical implementations and evidence remain in place. The Seed 629
production `run_authorized()` guard is not changed by this protocol.

## Identities (do not conflate)

| Identity | Meaning | Lifetime |
|---|---|---|
| `experiment_id` | Stable research question/registration; unique in the configured worktree registry | One registered study, including its explicit retries |
| `attempt_id` | One immutable execution of that study | New ID and directory for every retry, resume, or corrected run |
| Random seed(s) | Inputs to the experiment's PRNG(s), declared by registration and frozen per attempt | May be shared across arms only when protocol specifies it |
| `source_commit` | Exact Git commit checked out when the attempt was frozen | May change between implementation and execution; never overwritten in an existing attempt |
| Checkpoint identity | Independently authenticated checkpoint identity/hash, if applicable | Separate from source, experiment, attempt, and seed identities |

A checkpoint is an input artifact; a seed is a randomization input; neither
identifies a run. A new source or changed configuration requires a new attempt
freeze. A scientific protocol change requires a new/revised registration under
its declared amendment procedure.

## Lifecycle, ownership, artifacts, and failure behavior

| Stage | Required inputs and responsible person | Durable outputs | Failure behavior |
|---|---|---|---|
| 1. Register and approve | Owner/researcher states hypothesis, objective, type, estimand, cohort, exclusions, acceptance/stop rules, seeds, resource budget, required inputs, and artifact-retention class. Owner records explicit execution authorization. | `registration.json`, approvals/authorization reference, protocol version. | No worktree workload execution before approval. Ambiguity or missing provenance blocks registration. |
| 2. Create worktree | Operator supplies repository, unique experiment ID/branch, exact base commit, registration, and input hashes. | Dedicated branch/worktree, `.experiment/registration.json`, pinned `.experiment/base-commit.json`, standard ignored `.experiment/runtime/attempts/`. | Existing experiment ID, branch, or path is a hard collision. Never reuse a different worktree or silently retarget the base. |
| 3. Implement and regress | Implementer uses only that worktree, runs applicable tests/lint, commits implementation and registration. | Focused commits, tests, implementation and verification source. | Failed checks block freeze. Do not implement from or execute in the main checkout. |
| 4. Authenticate inputs | Experiment owner/verifier checks every input, checkpoint, cohort, rules/tablebase, dataset, and prior receipt with the relevant experiment-specific verifier and SHA-256. | Authenticated input inventory and receipts referenced by registration/freeze. | Missing, changed, ambiguous, or unavailable identity stops before workload. |
| 5. Freeze attempt | Researcher binds clean worktree commit/tree, exact argv, complete configuration, inputs, checkpoint, seeds, runtime/dependency versions, CPU/GPU/device, worker count, wall/disk/resource budget, acceptance criteria, and outputs. | Create-only `manifest.json` validated by [`mancala-experiment-attempt-v1.schema.json`](schemas/mancala-experiment-attempt-v1.schema.json). | Invalid input hash, dirty source, duplicate attempt ID, or missing budget blocks launch. Edit means supersede with a new attempt; never rewrite a frozen manifest. |
| 6. Execute | Immediately before spawn, the operator utility rechecks the experiment/attempt IDs, exact source commit and tree, a clean worktree, and every registered input hash. It then runs the frozen argv in that linked worktree, capturing logs, wall timeout, process-group cleanup, disk preflight and basic child resource usage. Domain scripts retain their own limits/checkpoints. | `started.json`, outputs, exactly one `completed.json`, `failed.json`, or `interrupted.json`. | Changed source, input, identity, or dirty worktree blocks launch and leaves an explicit failed attempt. Timeout, interruption, disk preflight, nonzero exit, missing required output, and process cleanup failure are explicit failures. Retry only under a new attempt ID. |
| 7. Verify results | Independent reviewer runs lifecycle hash checks and the domain-specific verifier; verifies scientific estimand, cohort, inputs, coverage, acceptance criteria, and failure ledger. Historical verification authenticates the recorded commit/tree from Git objects; it does not require current `HEAD` to equal an earlier attempt's source. | Verification report and final scientific classification. | Infrastructure success is not scientific success. Missing or inconsistent evidence is failed/blocked, never inferred successful. |
| 8. Archive artifacts | Operator sets external archive root, separates Git outputs from reproducibility data and temporary files, creates an archive, and verifies hashes plus safe extraction. | Create-only `.tar.zst` when `zstd` exists, otherwise `.tar.gz`; sidecar binds source, attempt-manifest digest, and complete runtime file inventory; the PR tracks copies of the reference and sidecar. | Existing destination, symlink, unsafe member, hash mismatch, extraction mismatch, or inventory drift stops. Never delete originals before verification. Keep failed attempts when needed for interpretation. |
| 9. Prepare and publish PR | Author includes registration/specification, implementation, tests, frozen manifest, logs/outputs intended for Git, scientific report, aggregate results, verification code/procedure, and external archive manifest/reference. | Focused PR against `main`; PR binds external archive URI/path, size, digest, and file manifest. | Do not report unverified numbers or claim scientific success from passing infrastructure checks. No automatic merge. |
| 10. Independent review | Reviewer examines protocol and approval, source/input/checkpoint identities, command/config freeze, attempt history, verifier, archive and scientific conclusions. | Review comments and accepted/rejected classification. | Address substantive issues in a new commit/attempt as appropriate; do not rewrite prior evidence. |
| 11. Merge | Authorized maintainer merges reviewed PR under project policy. | Merged publication and retained archive. | A passing run does not grant merge or candidate promotion. |
| 12. Safe cleanup | Operator checks merged commit ancestry, clean worktree, terminal record, committed PR reference/sidecar, and exact runtime-to-archive inventory equality for every started attempt, then calls `safe_remove_worktree`. | Removed worktree; identity registry and external archive remain. | Dirty/unpublished changes, missing terminal record/artifact, substituted reference, changed/new runtime file, failed archive verification, or unmerged commit refuses removal. |

### Manifest and attempt records

The JSON schema defines the attempt freeze fields. In addition to those fields,
the runtime writes:

- `started.json`: UTC start and monotonic timestamp, create-only.
- One terminal record: attempt/experiment/source identities, freeze SHA-256,
  exit status, completion time, elapsed seconds, free space, resource-usage
  snapshot, stdout/stderr digests, and output digests.
- `archive-reference.json`: external archive and sidecar paths/digests after
  archive verification.

Execution reauthenticates inputs immediately before launch. Historical
verification checks the recorded commit/tree object and registration without
requiring the worktree's current `HEAD` to match; input files/directories are
still rehashed. A failed prelaunch attempt whose input changed can be archived
with an explicit `input_artifacts_verified: false` disposition so its logs and
failure receipt are preserved. It cannot be treated as a successful or
input-verified run.

Configuration is JSON so the exact configuration is reviewable and hash-bound.
`command` is an argv array invoked without a shell. Environment-dependent
settings are captured as explicit environment overrides. The process inherits
only `PATH`, `HOME`, `LANG`, and `LC_ALL` before applying those frozen overrides;
other host environment variables are not passed through. Do not put secrets in
the manifest. Absolute input paths plus hashes are recorded. Directory inputs
use SHA-256 over sorted relative member paths and file SHA-256s. Do not rely on a
mutable symlink or a filename alone. `checkpoint_identity` remains a separate
object and should identify the checkpoint's digest and provenance.

The common runner records child CPU-time deltas and disk/wall measurements. Its
`RUSAGE_CHILDREN` maximum RSS is a process-wide high-water mark that may include
prior children in a long-lived caller; the record labels that scope rather than
claiming an attempt-specific peak. This is operational bookkeeping, not
kernel-level enforcement evidence. Training, self-play,
evaluation, solver, and GPU workloads must continue to use and record their
experiment-specific safe resource controls. Set worker count and execution
parallelism in the frozen configuration; repository policy requires arena
pair-level concurrency and per-game worker count to be frozen before launching.

## Operator workflow

### Configure roots and create an experiment worktree

The default worktree root `./.worktrees/` is already Git-ignored. Keep archive
storage outside the repository, on backed-up storage selected by the operator:

```sh
export MANCALA_EXPERIMENT_ARCHIVE_ROOT="$HOME/mancala-ai-experiment-archives"
python -m ml.alphazero_lite.experiment_lifecycle create \
  --repo "$PWD" \
  --registration registration.json \
  --branch experiment/my-study \
  --base-commit 956f2d927ec40fca19f0680a383f6297c0212e36
```

The registration input is JSON with these required fields:

```json
{
  "experiment_id": "my-study",
  "experiment_type": "retrospective_diagnostic",
  "hypothesis": "State the falsifiable hypothesis.",
  "objective": "Name the primary question and estimand.",
  "authorization_reference": "owner-approved issue or decision record",
  "random_seeds": [63501],
  "input_artifacts": {"/trusted/input.json": "<64-character-sha256>"},
  "checkpoint_identity": null,
  "acceptance_criteria": ["Predeclared criterion"]
}
```

Then work only in the printed worktree, commit the registration and
implementation, run regression checks there, and authenticate inputs using the
existing verifier for each artifact domain. Inspect active registered worktrees
with:

```sh
python -m ml.alphazero_lite.experiment_lifecycle list --repo "$PWD"
```

The worktree registry reserves an experiment ID even after its worktree is
removed. Do not recycle IDs; use a new identity for a genuinely new study.

### Freeze, execute, and verify an attempt

Create a JSON freeze spec containing `attempt_id`, `command`, `configuration`,
`runtime_versions`, `device`, `worker_count`, `expected_runtime_seconds`,
`wall_time_limit_seconds`, `minimum_free_disk_bytes`, `outputs`,
`input_artifacts`, `random_seeds`, and optional `additional_resource_limits`
(for example registered CPU seconds, memory bytes, or GPU-hours), plus
`publication_outputs`, a subset of declared outputs chosen for Git publication.
Input hashes and seeds must exactly match the registration. Example:

```sh
FROZEN=$(python -m ml.alphazero_lite.experiment_lifecycle freeze \
  --worktree /path/to/.worktrees/my-study --spec attempt-0001.json)
ATTEMPT_DIR=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["attempt_dir"])' "$FROZEN")
MANIFEST_SHA256=$(python -c 'import json,sys; print(json.loads(sys.argv[1])["manifest_sha256"])' "$FROZEN")
python -m ml.alphazero_lite.experiment_lifecycle run \
  "$ATTEMPT_DIR" --manifest-sha256 "$MANIFEST_SHA256"
python -m ml.alphazero_lite.experiment_lifecycle verify \
  "$ATTEMPT_DIR"
```

`freeze` returns the attempt path and manifest SHA-256; retain that digest
outside the runtime directory before launch. `freeze` refuses uncommitted source
changes, changed inputs/seeds, and reused attempt IDs. `run` requires the
independently retained manifest digest, refuses a repeated attempt, runs no shell string, applies the
wall limit and disk preflight, and terminates the command process group on
timeout/interruption. It does not install privileged controls or override a
domain runner's own authorization. A retry gets a new `attempt_id` and preserves
the old attempt. Verify the result and evaluate scientific criteria separately.

### Archive, publish, and clean up

```sh
python -m ml.alphazero_lite.experiment_lifecycle archive ATTEMPT_DIR \
  --archive-root "$MANCALA_EXPERIMENT_ARCHIVE_ROOT"
python -m ml.alphazero_lite.experiment_lifecycle verify-archive ARCHIVE.tar.zst ARCHIVE.archive-manifest.json
python -m ml.alphazero_lite.experiment_lifecycle prepare-publication \
  ATTEMPT_DIR docs/data/my-study/attempt-0001
```

When `zstd` is present, the helper creates `.tar.zst`; otherwise it creates
`.tar.gz`. The archive sidecar records every member's relative path, byte size,
and SHA-256, archive byte size/SHA-256, source commit, experiment ID, attempt ID,
and attempt-manifest digest. The helper verifies both the compressed file and
the extracted member hashes before writing its create-only reference. The
publication helper copies the attempt manifest, terminal record, declared
`publication_outputs`, stdout/stderr logs no larger than 1 MiB each, archive
reference, and external archive manifest into a create-only PR directory. Large
logs and non-Git outputs remain in the archive. Add the scientific report,
domain verifier and results aggregation to the PR directory as applicable.

Do not change backup configuration as part of an experiment. The operator should
ensure that the selected external archive root (default shown above) is covered
by the existing backup policy. Cleanup after merge only:

```sh
python -m ml.alphazero_lite.experiment_lifecycle remove \
  --repo "$PWD" --worktree /path/to/.worktrees/my-study --merged-into origin/main
```

Cleanup refuses dirty worktrees, unmerged commits, started attempts lacking
exactly one terminal receipt, and attempts without independently verified archive
references. External archives and registry history are retained.

## Artifact classes and retention

1. **Git publication:** specifications/registrations, attempt manifest and
   terminal receipts, concise reports and aggregates, verifier/tests, relevant
   logs and small outputs, external archive references/manifests. Commit these
   in the experiment PR.
2. **Reproducibility archive:** large raw predictions, trajectories, checkpoints,
   intermediate tensors, model files, large logs, and any attempt output needed
   for replay. Store outside the worktree in the configured archive root and
   publish its URI/path, byte size, SHA-256, and member manifest in Git.
3. **Discardable temporary data:** caches and intermediates not listed as
   required by registration or verifier. Delete only after the result and
   archive manifest establish they are not required. Failed attempts are
   retained whenever they affect interpretation, stop rules, or retries.

## Scientific reporting and authorization

Every experiment PR reports (a) registered hypotheses and estimands, (b)
observations with evidence links, (c) acceptance criteria and whether each
passed, (d) final scientific classification, (e) limitations, and (f) the next
recommended experiment. Distinguish registration, execution, infrastructure
verification, and scientific outcome. A successful command, clean archive, or
passing regression suite is not a positive scientific result.

Training, evaluation, self-play, retrospective diagnostics, exact solving, and
benchmarks all use this common lifecycle but retain their own domain-specific
preregistration and verification. Explicit owner authorization is required at
the scientific gates relevant to each study. Nothing here authorizes Seed 623,
generation, inference, search, exact solving, training, or evaluation games.
