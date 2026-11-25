# Mancala AI experiment execution protocol

**Status:** mandatory and authoritative for every research experiment in Mancala
AI. A merged experiment registration PR authorizes only the scope stated in that
registration; this protocol does not authorize a particular workload or
candidate promotion by itself.

All experiments—including training, self-play, evaluation, benchmark
generation, exact solving, retrospective diagnostics, ablations, and feasibility
studies—MUST follow this document. Ordinary unit tests and synthetic
implementation regression tests are not research experiments and do not require
registration.

## Normative language

`MUST` and `MUST NOT` state mandatory requirements. `SHOULD` and `MAY` state
recommended or optional practices. Every lifecycle stage below is mandatory;
recommendations do not replace a registration, owner merge, freeze, verification,
or archive gate. Experiment-specific instructions may add scientific
requirements but MUST NOT weaken this protocol.

## Recommended practices

The mandatory lifecycle is the execution contract. In addition, operators
SHOULD use an externally backed-up archive root, preserve source copies until
archive verification passes, use physical-copy verification where practical,
and keep reports concise while retaining complete replay evidence. A
recommendation does not substitute for or weaken a mandatory gate.

## Purpose and scope

This protocol applies to all research experiments, including training,
self-play, evaluation, benchmark generation, exact solving, retrospective
diagnostics, ablations, and feasibility studies. It defines mandatory ownership,
registration, execution, verification, publication, and cleanup requirements.
An experiment-specific protocol MUST add scientific detail without weakening
these requirements.

The common utility is `ml.alphazero_lite.experiment_lifecycle`. Its default
worktree root is the repository's already ignored `./.worktrees/`; set
`MANCALA_EXPERIMENT_WORKTREE_ROOT` or pass `--worktree-root` to change it. Set
`MANCALA_EXPERIMENT_ARCHIVE_ROOT` or pass `--archive-root` to choose external
archive storage. The utility uses ordinary Git, Python, and `zstd` when
available; without `zstd` it uses standard-library gzip tar archives. Runtime
attempt files are under `.experiment/runtime/` and ignored by the experiment
worktree's local `.experiment/.gitignore`. Registration and publication files
remain ordinary reviewable repository files.

The permanent, reviewable registration is stored at
`docs/experiments/<experiment_id>/registration.json`. The worktree-local
`.experiment/registration.json` and `.experiment/base-commit.json` are
operational metadata only and MUST remain ignored; they are never copied into
the permanent registration directory or committed to `main` at fixed shared
paths.

Experiment-specific scripts MUST retain the resource controls that matter to their
workload, including pair-level concurrency, per-game/search workers, checkpoint
cadence, solver budgets, and stop conditions. The privileged Seed 629
production `run_authorized()` guard and its historical evidence remain
unchanged; this ordinary research protocol does not call or bypass that guard.

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

## Mandatory lifecycle

Every stage MUST occur in this order. A failed gate stops the experiment; there
is no implicit authorization, parameter change, or retry.

| Stage | Mandatory requirements | Durable outputs and failure behavior |
|---|---|---|---|
| 1. Create a dedicated experiment worktree | The operator MUST create a unique linked worktree from the exact current `main` base commit using `experiment_lifecycle create`. A draft registration may seed worktree creation and MUST be finalized in the worktree. Implementation and experiment workloads MUST NOT run in the main checkout. | Dedicated branch/worktree, local `.experiment/registration.json` pointer and `.experiment/base-commit.json` operational metadata, permanent registration path `docs/experiments/<experiment_id>/registration.json`, and ignored runtime directory. An identity, branch, or permanent-path collision is a hard stop; never retarget or reuse another worktree. |
| 2. Prepare the registration, implementation, and tests | In the worktree, the researcher MUST specify the hypothesis, estimand, cohort, exclusions, inputs, seeds, resource budgets, retention class, acceptance criteria, and stop conditions in the permanent experiment-specific registration. The researcher MUST implement and run relevant unit, synthetic-regression, and static checks, then commit the registration and implementation. | Reviewable permanent registration, implementation, tests, and verification source. Failed checks block PR publication. Synthetic implementation tests may run before authorization; research workloads may not. |
| 3. Publish the registration PR | The author MUST publish a PR against `main` containing the registration, execution scope, protocol/input identities, resource budgets, acceptance criteria, stop conditions, implementation, and tests. | Registration PR and GitHub metadata. Do not launch or resume the registered experiment while this PR is open or unmerged. |
| 4. Obtain independent review | A reviewer other than the PR author MUST review the registration, scientific protocol, scope, source, tests, inputs, and resource/stop criteria. The author MUST address review feedback before requesting owner merge. | Review record and focused follow-up commits. Unresolved scientific or implementation blockers stop the process. |
| 5. Project owner merges the registration PR | For `useruby/mancala-ai`, only repository owner `useruby` merging the registration PR into `main` authorizes its explicitly registered scope. The execution entry point MUST verify GitHub PR metadata and registration/protocol blobs in the merge commit. When that entry point requires the exact merge commit as source, stage 6 MUST freeze that commit. The PR author MUST NOT merge their own PR. No separate owner comment, manually prepared approval JSON, copied hash, or authorization reference is required. | Automatically recorded PR number, merge commit, merged-by identity, merge timestamp, and registration digest. A PR not merged by the owner, into `main`, or with mismatched committed objects is unauthorized. Never auto-merge. |
| 6. Freeze and authenticate the execution configuration | After owner merge, the operator MUST reauthenticate all registered inputs with the relevant independent verifier and freeze the clean source commit/tree, exact command/configuration, runtime, dependencies, device, workers, seeds, resource limits, output paths, and acceptance/stop rules. Configuration MUST conform to the merged registration. | Create-only `manifest.json` validated by [`mancala-experiment-attempt-v1.schema.json`](schemas/mancala-experiment-attempt-v1.schema.json). Changed/missing inputs, dirty source, duplicate attempt identity, or missing budgets block execution. Frozen parameters MUST NOT be changed after observing results. |
| 7. Execute the registered experiment | The operator MUST execute only through `experiment_lifecycle run`. Immediately before spawn, the runner rechecks identities, exact source commit/tree, clean worktree, and registered input hashes. Domain scripts MUST retain registered resource controls and checkpoints. | `started.json`, outputs, logs, and exactly one terminal record: `completed.json`, `failed.json`, or `interrupted.json`. Timeout, interruption, resource/disk failure, nonzero exit, missing output, or cleanup error is an explicit failed attempt. Each authorized retry/resume requires a new attempt ID and must be allowed by the registration; never retry automatically or overwrite prior attempts. |
| 8. Independently verify results | A reviewer independent of execution MUST run lifecycle hash checks and the domain-specific verifier, then check the estimand, cohort, inputs, coverage, acceptance criteria, stop/failure ledger, and final classification. | Verification report and evidence. Infrastructure success is not scientific success; missing or inconsistent evidence is blocked/failed, never inferred successful. |
| 9. Archive non-Git artifacts and verify integrity | The operator MUST archive every required non-Git artifact, including failed/partial evidence needed to interpret the attempt, using the standard lifecycle. The operator MUST verify archive hashes, member inventory, and safe extraction before relying on the archive. | Archive, sidecar manifest, and reference binding source, attempt-manifest digest, and complete runtime inventory. Preserve originals until verification passes. |
| 10. Publish the experiment results PR | The author MUST publish a results PR against `main` containing the scientific report, results, verification procedure/output, registration and freeze references, small Git-publishable outputs, and verified archive reference. | Focused results PR with archive URI/path, size, digest, and member manifest. Do not claim results that lack verification. |
| 11. Obtain independent review and merge results | A reviewer other than the PR author MUST review execution evidence, source/input identities, freeze, attempt history, verifier, archive, and scientific conclusions. An authorized project maintainer merges only after review. Authors MUST NOT merge their own experiment PRs. | Reviewed and merged results publication. A successful run or archive does not imply scientific qualification or candidate promotion. |
| 12. Safely clean up the worktree | After results PR merge, the operator MUST verify the worktree is clean, all attempts are terminal and accounted for, and every required artifact is represented in the committed publication and verified archive before using `safe_remove_worktree`. | Removed linked worktree; registration identity registry and external archive remain. Dirty/unpublished work, unaccounted runtime paths, invalid terminal records, missing artifact/archive references, inventory mismatch, or unmerged publication blocks cleanup. |

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
  "random_seeds": [63501],
  "input_artifacts": {"/trusted/input.json": "<64-character-sha256>"},
  "checkpoint_identity": null,
  "acceptance_criteria": ["Predeclared criterion"],
  "registration_metadata": {
    "scientific_protocol": "Versioned estimand, cohort, and method",
    "resource_budgets": {"wall_seconds": 3600, "cpu_seconds": 3600},
    "stop_conditions": ["Stop on any missing required input or verifier mismatch"],
    "artifact_retention_class": "Git report plus verified external archive"
  }
}
```

`experiment_lifecycle create` writes the permanent registration under its
experiment-specific path and a separate local pointer under `.experiment/`.
Later experiments therefore add a new permanent path and cannot overwrite or
collide with earlier registrations.

Then work only in the printed worktree: finish and commit the registration,
implementation, and tests there. Authenticate the inputs after owner merge and
before freezing, using the existing verifier for each artifact domain. Inspect
active registered worktrees with:

```sh
python -m ml.alphazero_lite.experiment_lifecycle list --repo "$PWD"
```

The `list` command resolves each active worktree's local metadata pointer and
reports its permanent registration path. The worktree registry reserves an
experiment ID even after its worktree is removed. Do not recycle IDs; use a new
identity for a genuinely new study.

### Publish and authorize the registration (stages 3–5)

After the registration, implementation, and tests are committed in the worktree,
the author MUST publish the registration PR against `main`. The PR MUST state
its exact execution scope and include scientific criteria, resources, and stop
conditions. An independent reviewer MUST review it before owner merge. For
`useruby/mancala-ai`, only `useruby` merging it into `main` authorizes the
registered scope. No separate approval comment or JSON record is needed. Do not
freeze or run research workload before this merge.

The execution entry point verifies the PR metadata and committed registration
and protocol blobs. It automatically records the PR number, merge commit,
merged-by identity, merge timestamp, and registration digest. If the experiment
entry point requires the exact merge commit as source, update the experiment
worktree to that merge commit before freezing. Any subsequent source change
requires an applicable new reviewed, owner-merged registration PR and a new
attempt; never edit a frozen source or configuration.

### Freeze, execute, and independently verify an attempt (stages 6–8)

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

The lifecycle `verify` command checks execution and artifact identities; it does
not replace the experiment-specific independent scientific verifier. The
operator MUST run both and publish the domain verifier's results with the
results PR.

`freeze` returns the attempt path and manifest SHA-256; retain that digest
outside the runtime directory before launch. `freeze` refuses uncommitted source
changes, changed inputs/seeds, and reused attempt IDs. `run` requires the
independently retained manifest digest, refuses a repeated attempt, runs no shell string, applies the
wall limit and disk preflight, and terminates the command process group on
timeout/interruption. It does not install privileged controls or override a
domain runner's own authorization. A retry MUST be explicitly allowed by the
merged registration; otherwise stop and obtain a revised registration PR before
retrying. Every authorized retry/resume gets a new `attempt_id` and preserves the
old attempt. Never retry automatically.

### Archive non-Git artifacts and verify integrity (stage 9)

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

Do not change backup configuration as part of an experiment. Operators SHOULD
ensure that the selected external archive root is covered by the existing backup
policy.

### Publish results, obtain review, merge, and clean up (stages 10–12)

After archive verification, prepare the results PR. Include the scientific report,
registered-versus-observed acceptance table, lifecycle manifest and terminal
record, domain verifier and results, relevant small outputs, and archive
reference and sidecar. An independent reviewer other than the PR author MUST
review the source, inputs, freeze, attempt history, verification, archive
integrity, and conclusions. An authorized maintainer merges after review;
authors MUST NOT merge their own experiment PRs.

Cleanup is allowed only after the results PR is merged and the checks below pass.

```sh
python -m ml.alphazero_lite.experiment_lifecycle remove \
  --repo "$PWD" --worktree /path/to/.worktrees/my-study --merged-into origin/main
```

Cleanup enumerates the runtime root rather than only discovered manifests. It
refuses unaccounted runtime files, attempt directories without a valid identity
and manifest, frozen attempts with no execution disposition, inconsistent start/
log/terminal records, dirty worktrees, unmerged commits, and attempts without a
matching committed archive reference/sidecar whose inventory exactly covers the
current attempt files. External archives and registry history are retained.

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

In `useruby/mancala-ai`, the owner-merged registration PR in mandatory stage 5
is the sole ordinary research execution authorization. The PR MUST be merged
into `main` by repository owner `useruby` and contain the exact registration,
execution scope, scientific protocol, resource budgets, acceptance criteria,
and stop conditions. The execution entry point MUST authenticate PR metadata
and the committed registration/protocol blobs with Git objects and MUST record
the PR number, merge commit, merged-by identity, merge timestamp, and
registration digest automatically. Where an experiment authorization entry
point binds the source to the registration merge commit, the attempt MUST use
that exact commit as its source. No separate owner comment, manually authored
approval JSON, copied hash, or authorization reference is required.

Authorization applies only to the scope declared in the merged registration.
Each later scientifically gated stage requires its own registration PR merged
by the owner. This ordinary research mechanism does not authorize or modify the
Seed 629 privileged production launch guard.

Every experiment PR reports (a) registered hypotheses and estimands, (b)
observations with evidence links, (c) acceptance criteria and whether each
passed, (d) final scientific classification, (e) limitations, and (f) the next
recommended experiment. Distinguish registration, execution, infrastructure
verification, and scientific outcome. A successful command, clean archive, or
passing regression suite is not a positive scientific result.

Training, evaluation, self-play, retrospective diagnostics, exact solving, and
benchmarks all use this common lifecycle but retain their own domain-specific
preregistration and verification. Owner-merged registration is required at the
scientific gates relevant to each study. A Seed 636 registration authorizes only
its named population-generation and membership-screening scope, not inference,
search, exact solving, training, or evaluation games.
