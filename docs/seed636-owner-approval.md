# Seed 636 — owner approval record and instructions

**Purpose:** record explicit project-owner approval before the separately
authorized Seed 623 prospective population-generation stage. This migration PR
and its merge are not execution authorization.

## Approved scope

The approval decision must explicitly cover all four items:

1. Seed 625's finite-cohort estimand amendment: deterministic hash-selected
   512-state population, equal `1/5` weight per stone-count bin, and no
   design-unbiased inference claim.
2. Seed 635's worktree-based experiment execution contract.
3. Acceptance of `known-exposure-screened; inherited P-E4 exposure unresolved`;
   no training-disjoint, unseen, or contamination-free claim.
4. Model-independent prospective population generation and membership screening
   only. No inference, MCTS, exact solving, training, or model-based games.

## Protocol artifacts

The owner comment and registration must bind these exact artifacts:

| Artifact | Revision | SHA-256 |
|---|---|---|
| `docs/seed624-estimand-amendment-proposal.md` | Seed 625 merge `72a266ed73db02da2cebb7ffe99575d4761ccaaa`, with authorization notes proposed in this PR | `9fe2b129b748fc050526a99ea2ac896ecf7ead23536835381cea0337120ba170` |
| `docs/seed623-631-execution-contract-amendment.md` | Seed 635 merge `f400d9757bee792370306a4448914f2049ff77d8`, with GitHub authorization binding proposed in this PR | `b0ceb1254a990b830ddc2f256fccca27ea898645f569bee162a70364dd35df05` |
| `docs/experiment-execution-protocol.md` | Seed 635 merge `f400d9757bee792370306a4448914f2049ff77d8`, with GitHub owner mechanism proposed in this PR | `93ecf7a16b6a5f1925deaff0c58402d45ea1213b85ac272a0f1c149de5053255` |

The SHA-256 values above are for the exact proposal documents in this PR. The
eventual Seed 636 registration must pin those files again from the exact
approved main commit. The approval's `approved_protocol_revision` is the full
main commit SHA used as the experiment's base, not a mutable branch name.

## Required project-owner GitHub comment

Post a GitHub issue/PR comment as repository owner `useruby` in
`useruby/mancala-ai`. Replace the bracketed values with the exact registration
and document identities, then post the same JSON object as the comment body
following the marker line. The generator independently fetches the comment via
`gh api` and requires the recorded identity, URL, timestamp, text and hashes to
match.

```text
SEED636-APPROVAL-V1
{
  "marker": "SEED636-APPROVAL-V1",
  "experiment_id": "seed636-prospective-population",
  "registration_sha256": "<sha256 of committed .experiment/registration.json>",
  "approved_protocol_revision": "<full pinned main commit SHA>",
  "protocol_documents": {
    "docs/seed624-estimand-amendment-proposal.md": "9fe2b129b748fc050526a99ea2ac896ecf7ead23536835381cea0337120ba170",
    "docs/seed623-631-execution-contract-amendment.md": "b0ceb1254a990b830ddc2f256fccca27ea898645f569bee162a70364dd35df05",
    "docs/experiment-execution-protocol.md": "93ecf7a16b6a5f1925deaff0c58402d45ea1213b85ac272a0f1c149de5053255"
  },
  "approved_decisions": [
    "seed625-finite-cohort-estimand-amendment",
    "seed635-worktree-execution-contract-amendment",
    "known-exposure-screened-inherited-PE4-unresolved",
    "model-independent-population-generation-and-screening-only"
  ],
  "execution_scope": "model-independent-generation-and-screening-only"
}
```

The `.experiment/seed636-owner-approval.json` record in the experiment worktree
copies the exact comment URL, author identity (`useruby`),
`created_at` timestamp, protocol revision/document map, registration digest, and
scope. The execution entry point rejects a missing record, wrong owner, URL or
timestamp mismatch, changed registration/document, malformed approval body, or
different scope. Add and commit this record before freezing the attempt; the
entry point requires it to be in the frozen source commit. Do not create or
backdate this record before the owner comment exists.

When creating the worktree with `experiment_lifecycle create`, put these fields
in the registration input's `registration_metadata` object:

```json
{
  "approved_protocol_revision": "<full pinned main commit SHA>",
  "protocol_documents": {
    "docs/seed624-estimand-amendment-proposal.md": "9fe2b129b748fc050526a99ea2ac896ecf7ead23536835381cea0337120ba170",
    "docs/seed623-631-execution-contract-amendment.md": "b0ceb1254a990b830ddc2f256fccca27ea898645f569bee162a70364dd35df05",
    "docs/experiment-execution-protocol.md": "93ecf7a16b6a5f1925deaff0c58402d45ea1213b85ac272a0f1c149de5053255"
  },
  "approved_decisions": [
    "seed625-finite-cohort-estimand-amendment",
    "seed635-worktree-execution-contract-amendment",
    "known-exposure-screened-inherited-PE4-unresolved",
    "model-independent-population-generation-and-screening-only"
  ]
}
```

The per-attempt Seed 635 freeze separately binds the clean source commit/tree,
configuration, runtime, command, registered inputs, random seed, resource
budget, and output paths. GitHub approval does not replace attempt freezing, and
attempt freezing does not grant owner authorization.

For the population attempt, put the complete output of
`seed624_prospective_benchmark.proposed_freeze_schema()` (with every required
identity and resource limit finalized) under
`configuration.seed624_freeze`. Set `configuration.execution_scope` to the scope
above and `configuration.registration_sha256` to the raw-byte digest of the
committed registration. Freeze argv invoking
`python -m ml.alphazero_lite.seed624_prospective_benchmark --attempt-dir
<absolute-attempt-directory> --output
.experiment/runtime/attempts/<attempt-id>/population`; declare `population` as
the output directory, and bind that path in `seed624_freeze.create_only_output_paths`.
Keep the lifecycle worker count at one. Execute only through
`experiment_lifecycle run`; the generator requires the runner-created
`started.json` and committed attempt source. Lifecycle terminal records preserve
completed, failed, and interrupted attempts without automatic retry.
