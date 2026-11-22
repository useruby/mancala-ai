# Seed 628 — authenticated membership handoff to Seed 624

**Status: integration and independent verification only. Execution remains
blocked.** No production benchmark states were generated. No inference, MCTS,
solving, training, key provisioning, or execution authorization occurred.

## Adapter and verification

`seed628_authenticated_screening.authenticate_snapshot_bundle()` consumes the
three exact Seed 627 snapshot files and
`source-identity-lineage-manifest.json` read-only. It checks pinned file-byte
SHA-256 values, calls Seed 627's independent verifier to reconstruct source
rows, traces, receipts, lineage and source-verifier records, and recomputes all
5,449 row identities with the proposed Seed 624 identity implementation. The
adapter then supplies the rows and evidence hashes to Seed 624 screening; it
does not produce or modify snapshots.

Seed 627's authentication runtime records NumPy 2.5.2; the proposed Seed 624
runtime pins NumPy 2.5.3. The historical verifier is launched as a separate
`.venv/bin/python` process, with its actual interpreter and installed package
versions compared against the source manifest and the SHA-256-pinned historical
requirements lock. The complete verifier output and actual runtime probe are
recorded in the evidence object. No runtime metadata is monkeypatched. A
separate process running the proposed Seed 624 NumPy 2.5.3 implementation
recomputes every canonical and float32 identity; equality is required for every
row in B512 (2,048), S615 (2,048), and protected holdout (1,353). This
establishes identity equivalence for these exact inputs; it does not certify
other platforms, NumPy versions, or an execution freeze.

Seed 624 production loading reads evidence only through this pinned adapter,
not a caller-supplied evidence JSON. The freeze binds canonical membership
hashes, raw snapshot-file hashes, and the source-manifest hash. The independent
Seed 624 verifier separately reconstructs Seed 627 evidence from the pinned
source files and checks the complete embedded evidence object before screening.

The independent Seed 628 verifier reconstructs the adapter evidence and
classification outcomes without calling Seed 624's screening producer. It
fails closed on altered snapshot/manifest bytes, bad row or aggregate hashes,
missing lineage, unauthenticated evidence, a forged crosswalk status, and
classification mismatches. Exact canonical **or** float32 matches in B512,
S615, or protected holdout are `confirmed_known_exposure` and are ineligible
for selection. A nonmatch against those exact inventories is labeled
`known_exposure_nonmatch_inherited_PE4_unresolved`; it is never labeled clear.

## Provenance interpretation and protocol gate

Seed 627 authenticates historical source provenance but establishes no
crosswalk between its source game/opening records and Seed 624's prospective
`game_id`/`opening_identity`. This is different from missing or unauthenticated
provenance. The adapter records `authenticated_non_crosswalkable`; it supplies
no fabricated source-game matches (`source_provenance_match` is null when no
crosswalk permits the comparison). A verified crosswalk could support
provenance exclusions, but this authenticated evidence does not.

The existing Seed 623 approved known-exposure contract explicitly says to
compare provenance where a complete crosswalk exists and to keep unavailable
crosswalks `unresolved`, never `cleared`. Therefore this history can be
accommodated under the existing protocol only with the unresolved inherited
P-E4 exposure label. Exact identity and protected-holdout exclusions still
apply. It does not require an amendment to claim source-provenance clearance,
because no such clearance is claimed. If the scientific target requires
training-disjoint states or cleared source provenance, the design cannot meet
that target without new crosswalk evidence and an explicit protocol revision.

Seed 624's completeness check has been corrected: an authenticated provenance
hash proves the provenance record's integrity, not that its source-game
crosswalk is complete. Only an authenticated `complete` crosswalk can be used
for game/opening linkage. Authenticated non-crosswalkable evidence permits the
known-exposure-screened-but-inherited-unresolved classification; missing or
unauthenticated provenance remains invalid/insufficient. The source status is
carried into each screening result.

## Outstanding execution blockers

The independent project-owner approval gate for PR #625 remains absent, as
recorded in `seed624-implementation-readiness.md`. Merging PR #625 did not
constitute that approval. The membership handoff does not supply approval,
execution authorization, a finalized runtime freeze, or a trusted signing key.
Do not generate a production cohort or proceed to inference until those
separate gates are independently satisfied.
