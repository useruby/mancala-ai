# Seed 626 — production execution readiness

**Status: preparation only. No production states, games, inference, search,
solving, or training were run. The freeze remains non-authorized.**

## Protocol and owner approval gate

Merged PRs #623–#625 leave the active protocol at a finite selected population
of exactly 512 states, with fixed weight `1/5` per stone bin. Opening-cluster
resampling intervals are descriptive stability intervals only. PR #625's body
explicitly required project-owner approval before merge. The PR review record
currently contains no reviews, requested reviewers, or approval comments that
independently establish that approval. **Explicit independently recorded owner
approval is therefore a blocking gate.** Neither this document nor the merge
itself supplies it. The owner decision must record identity, timestamp, exact
protocol commit/hash, and explicit acceptance of the estimand amendment.

The implementation's absent production trust-anchor entry remains intentional.
The proposed selected-population contract and PR #625 do not waive any separate
authorization requirement.

## Failure receipts and attempts

The production caller must use `execute_with_failure_receipts`. It emits one
JSON receipt via exclusive create (`O_EXCL`, mode 0600), flush, and `fsync`,
then re-raises the original execution failure. The receipt identifies freeze
digest, authorized run identity, category, exception and message, partial
output path/state, and the complete structured shortfall report when available.
Categories distinguish `opening_reserve_exhaustion`,
`insufficient_stone_bin_supply`, `mathematical_infeasibility`,
`selector_resource_exhaustion`, `verification_or_identity_failure`, and
`unexpected_execution_exception`. Resource limits are never reported as
mathematical infeasibility. Opening rejection attempts are retained in the
structured reserve report. A receipt collision fails without replacement; an
attempt is never retried automatically. Any already-written output files are
left in place as forensic evidence. External `SIGKILL`/OOM termination cannot
be caught in-process; the supervisor must write the corresponding receipt.

Freeze `authorized_run_identity` and a dedicated create-only receipt path before
authorization. Receipt storage must be outside the population directory and on
the same durable evidence volume; the supervisor additionally records its
termination signal/reason. Preserve partial output trees without cleanup.

## External resource enforcement (required harness)

Python allocator limits are diagnostics only. The authorized operator must
freeze positive integer values for process RSS bytes, CPU seconds, wall seconds,
maximum concurrent executions, combined output-plus-temporary storage bytes,
and per-run output/temporary paths. Enforce them outside Python in an isolated
Linux cgroup v2 scope or equivalent OS job object. For Linux, use a dedicated
systemd transient service/scope with `MemoryMax`, `CPUQuota`/`CPUAccounting`,
`RuntimeMaxSec`, and `TasksMax`; bind execution concurrency with a supervisor
semaphore whose capacity is exactly the freeze value. Place output and scratch
on a dedicated quota-limited filesystem/project quota; monitor combined
allocated bytes and stop the cgroup before the frozen cap. `ulimit` and
`tracemalloc` alone do not satisfy these limits. The supervisor polls cgroup
`memory.events`, CPU accounting, wall deadline, concurrency slot, and storage
usage; on breach it terminates the complete cgroup, preserves partial artifacts,
and writes a create-only failure receipt. Capture the exact supervisor version,
host/kernel, cgroup configuration, filesystem quota, and observed termination
evidence in the run record. Synthetic fixtures must prove each termination and
receipt path before a production freeze can be authorized.

## Owner signing-key provisioning

No owner key is provisioned by this change. When explicitly requested by the
project owner: (1) owner generates Ed25519 keypair outside the repository and
delivers only the raw 32-byte public key over an independently authenticated
channel; (2) two reviewers independently verify identity-to-key binding with a
live challenge signed by the private key and confirm the owner identity through
a separate known channel; (3) record the verification date, identities, and
challenge result in the review; (4) add only public key hex and stable key ID
`sha256(raw_public_key)[:16]` plus exact owner identity to both producer and
independent-verifier source trust maps in a focused reviewed commit; (5) tests
must reject invalid signature, wrong scope, wrong freeze digest, mismatched
owner identity, and unknown key. Never put private key material in repository,
arguments, logs, CI output, or receipts. Key provisioning grants no execution
authorization: an independently reviewed, signed exact freeze and separately
recorded owner approval remain mandatory. Current production allowlists remain
empty.

## Authenticated inputs and P-E4 limitation

The read-only preparation input sources are the authenticated Seed 599/615
membership records, protected-holdout inventory, and source records bound by the
Seed 621/622 verifiers. Preparation must rerun those independent verifiers,
extract canonical source states (not publisher-supplied derived IDs), rebuild
canonical JSON SHA-256 and `kalah_v3` little-endian float32 identities, sort by
canonical bytes, and publish create-only B512, S615, and protected-holdout
membership snapshots plus input-file digests and row counts. A second verifier
must authenticate source artifact digests and reconstruct every row and both
identities without trusting a `authenticated` flag or publisher assertion.
Missing sources, failed source verifiers, duplicate identities, or mismatched
hashes fail closed. Preserve the Seed 622 conclusion: 764 membership nonmatches
remain unresolved; the identity-incomplete P-E4 ancestry means no result may
claim training disjointness or absence of inherited exposure.

No authenticated snapshot publication is included here: source extraction is
gated on rerunning the pinned source verifiers, including the unavailable pinned
Seed 615 environment. The proposal must not be treated as evidence snapshots.

## Proposed freeze contract

`proposed_freeze_schema()` is a template only and must remain
`proposal_not_ready_or_approved`. The finalized schema must bind: source commit
and every producer/verifier/harness file hash; runtime, OS, Python, NumPy, PRNG,
crypto and dependency versions; rules and `kalah_v3` hashes; B512/S615/holdout
snapshot hashes and provenance source hashes; exact master seed, opening reserve,
prefix rule, game count, continuation action limit and output schema; five
stone-bin quotas/weights, candidate rank, cluster caps, diversity, optimizer
version/limits/one worker; all five external limits and supervisor identity;
exclusive-create output and failure-receipt paths; independent verifier command
and physical-copy procedure; required owner-approval identity, key ID, exact
freeze digest, scope and signature. Reject absent fields and any status other
than explicit authorized state only after all gates pass.

Preflight must check (before opening reserve consumption) complete fields,
current source/runtime hashes, trusted key and owner binding, valid signature on
the exact freeze digest/scope, create-only non-existing output/receipt paths,
authenticated membership/provenance snapshots, and external harness limits
including storage and concurrency. Any failure terminates with a durable
failure receipt; no generation begins. Independent verification must be run on
a byte-for-byte physical copy of the complete evidence directory.

## Remaining gates before production generation

1. Independent review and explicit, recorded project-owner approval of PR #625.
2. Rerun and independently verify authenticated source inputs in pinned
   dependencies; publish identity-reconstructed snapshots and provenance.
3. Resolve the P-E4 ancestry limitation or retain its unresolved language and
   apply the approved protocol's eligibility rule without claiming clearance.
4. Review/freeze source and runtime identities, production dependency versions,
   all external resource caps, supervisor and quota configuration.
5. Implement and verify full external resource supervisor termination and
   durable receipts on synthetic failures, including supervisor-killed runs.
6. Owner-authorized, independently witnessed key provisioning; add only its
   public trust anchor in a separately reviewed source change.
7. Final independent freeze review, exact-digest signature, and separate
   explicit execution authorization. This PR authorizes neither merge nor run.
