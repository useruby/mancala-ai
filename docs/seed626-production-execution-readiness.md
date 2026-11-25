# Seed 626 — production execution readiness

**Status: preparation only. No production states, games, inference, search,
solving, or training were run. The freeze remains non-authorized.**

**Protocol note:** this document preserves the production-specific readiness and
approval record as assessed at the time. Ordinary research experiment
registration and authorization now follow the mandatory
[`experiment-execution-protocol.md`](experiment-execution-protocol.md). This
does not change or waive the privileged Seed 629 production launch guard or its
separate production requirements.

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

The production caller must use `execute_with_failure_receipts`. Before running,
it authenticates the exact freeze bytes and owner signature, then requires the
caller run identity and receipt path to match the signed freeze fields exactly.
An authenticated receipt's run identity, freeze digest, and output path are read
only from that authenticated binding, never caller data. It uses exclusive
creation (`O_EXCL`, mode 0600), flushes and fsyncs file contents, then fsyncs the
pre-existing parent directory. A collision leaves the prior receipt untouched;
the receipt writer does not create directories or alter other frozen paths.

Preflight errors before successful authentication use the distinct
`seed624-unauthenticated-preflight-failure-v1` record schema, with null freeze
digest and null authorized run identity. They must never imply that an
authorized run began. Authenticated execution failures use
`seed624-authenticated-execution-failure-v1`. The complete category enum shared
by implementation, freeze template, and tests is: `opening_reserve_exhaustion`,
`insufficient_stone_bin_supply`, `mathematical_infeasibility`,
`selector_resource_exhaustion`, `verification_or_identity_failure`,
`unexpected_execution_exception`, `external_resource_limit_termination`, and
`preflight_failure`. Resource limits are never classified as mathematical
infeasibility. Opening rejection attempts are retained in structured reserve
diagnostics. No automatic retry occurs. If receipt writing fails, the original
execution exception is re-raised with the receipt-write exception attached as
an exception note; neither error is discarded and no evidence is overwritten.
External `SIGKILL`/OOM termination is recorded by the supervisor as
`external_resource_limit_termination`, using the same signed binding.

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
