# Seed 604 historical exclusion inventory audit

**Result: incomplete; evaluation cohort generation is blocked.** This audit is
read-only. It does not generate openings, train, or run evaluation games. The
machine-readable report is `exclusion-inventory-audit.json`.

## Authenticated baseline and scope

The published Seed 561 historical inventory (PR #565, commit
`b2f2bb092`) independently verifies the exact source-file set pinned by the
Seed 511 registration. Its published union contains 534,033 distinct identities
under each of the canonical and production `kalah_v3` float32 schemes. It
includes authenticated historical identity snapshots and replayed Seed 488,
499, 505, 508, and 500 evidence. This is a bounded inventory, not a complete
project-wide census.

Seed 561 explicitly records two scope limitations: Seed 461 continuation states
were not reconstructed or counted, and other historical populations outside
the exact Seed 511-pinned scope remain uncovered with unknown counts. Seed 511's
own registration and results make the same incomplete-scope disclosure. Those
limits prevent asserting that the union is complete for Seed 604.

The Seed 604 read-only preflight separately authenticated the P-E4 parent,
Seed 599 B membership and targets, B512 checkpoint, Seed 599 evaluation
contract/openings, and the 2,048-update schedule. These are valid Seed 599
identities but do not close the older historical gaps.

## Exact unresolved Seed 461 evidence

Seed 485 (PR #485, commit `0c1767676`) examined 58 receipt-matched Seed 461
streams. Seed 486 (PR #486, commit `1701142bb`) published a bounded recovery
ledger and rechecked the same 58 streams. Their published verifiers report:

- **58/58** lack a run-bound binding for the execution-time arena writer and
  Kalah rules source.
- **58/58** lack explicit state-trace rows; the uncovered population is
  1,129,643 recorded trajectory action pre-states, with no unique identity count
  asserted.
- **58/58** lack resume/chunk mappings sufficient to bind run, suite index,
  seat initialization, and per-game ranges.
- Ten fixed-E4 streams additionally lack authenticated registration and suite
  receipts/bytes: `order_38411_A/B`, `order_38412_A/B`, `order_38413_A/B`,
  `order_38414_A/B`, and `order_38415_A/B`.

The exact per-stream game-log receipt hashes, binding references, suite
references, local-artifact references, and unresolved fields are recorded in
`docs/data/seed486-seed461-recovery-ledger.json` (SHA-256
`cc803b7313851e35adf3191437d7c3b91c490f57bbe491e302bbc1024782d238`). The
recovery ledger's artifact request specifies the missing run-bound source
snapshots/hashes, explicit state traces, command logs, resume/chunk manifests,
and fixed-E4 registration/suite evidence. Nearby source revisions and local
`.tmp` run files are not substitutes for those run-bound receipts.

## Verification and decision

The Seed 561 published inventory verifier passed and reported the limitations
above. The Seed 485 and Seed 486 published verifiers also passed against
Git-tracked JSON inputs after excluding one tracked JSONL file named
`identity-inventory.json` from their recursive JSON-document scan. The
unmodified verifier invocation also encounters untracked local evidence in this
worktree. The independent Seed 604 read-only preflight passed and
reported no Seed 604 freezes or authorizations.

Because historical coverage remains incomplete, this publication does **not**
register a Seed 604 opening seed or cohort and does not claim exclusion
disjointness. It makes no use of Seed 599 evaluation outcomes. Resolve the exact
Seed 461 evidence request and audit the remaining populations outside the
Seed 511-pinned inventory before attempting cohort generation.

No Seed 604 execution freezes or authorization records were created. No training
or evaluation games were run, and no Seed 599 evidence was modified.
