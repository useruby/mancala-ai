# Seed 604 bounded evaluation exclusion policy

**Proposal for project-owner review.** This amendment preserves the historical
audit in PR [#605](https://github.com/useruby/mancala-ai/pull/605) and defines an
explicitly bounded exclusion population for the future Seed 604 cohort. The
machine-readable source and identity bindings are in
`bounded-exclusion-policy.json`.

No evaluation opening seed has been generated or selected, and no cohort,
execution freeze, authorization, training, or evaluation game has been created.

## Boundary

Before considering any Seed 604 root, the future generator must authenticate and
exclude the union of:

1. Seed 599 control A training membership (256 states).
2. Seed 599 treatment B training membership (2,048 states).
3. Seed 599 provenance-held-out membership (1,353 states).
4. All 1,024 Seed 599 evaluation opening roots.
5. The complete authenticated Seed 561 inventory: 534,033 canonical and
   534,033 production `kalah_v3` float32 identities, validated from its two
   externally pinned indexes.

For every future candidate opening, recompute both identities from its
registered resulting state. Reject canonical collisions and production-float32
collisions independently, and require uniqueness in both schemes within the
new cohort. Identity-set digests use sorted lowercase SHA-256 strings joined by
LF without a final LF. Source bytes, registration bindings, extraction rules,
record counts and per-scheme identity-set hashes are pinned in the JSON policy.

The authenticated bounded union contains 537,698 distinct identities per
scheme. The 256 A states are a subset of B, so they are separately authenticated
and reported without double-counting the union. The known Seed 599 evaluation
roots intersect the Seed 561 inventory once in each scheme; those roots remain
included as exclusions. Detailed source-to-source intersections are recorded
in the policy artifact.

This boundary includes Seed 599's 1,024 roots, not all post-root states from its
evaluation games. It therefore makes no claim of Seed 599 trajectory-state
novelty.

## Relationship to prior protocols

- **Seed 511:** its registered exclusion contract screened roots against the
  reconstructable historical union and its explicitly named Seed 488/499/505/508
  and Seed 500 sources, in canonical and production-float32 identity schemes.
  Its registration says directly that the Seed 461 replay gap remained open and
  does not claim complete historical exclusion.
- **Seed 561:** it authenticates exactly the source set pinned by the Seed 511
  registration, adds the published Seed 562–564 subtotal, and independently
  validates both identity indexes. Its recorded union includes the authenticated
  Seed 482–484 snapshots, Seed 488/499/505/508 replay populations, Seed 500
  pilot/self-play records, and Seed 562–564 source families. It explicitly
  leaves Seed 461 continuation identities unresolved and other out-of-scope
  historical populations uncounted.
- **Seed 599:** training/evaluation preparation rejected generated roots that
  collided with selected B training states in canonical or production-float32
  form; A is nested in B. Its registered evaluation cohort did not declare the
  held-out membership or Seed 561 inventory as opening-root exclusion inputs.
  Seed 604 adopts a stricter bounded rule by adding A explicitly, held-out
  states, all Seed 599 evaluation roots, and the entire authenticated Seed 561
  inventory. The observed Seed 599 overlap with Seed 561 is reported, not
  hidden or used to select outcomes.

## Seed 461 gap and scientific claims

The unresolved Seed 461 continuation states do not logically prevent a paired
Seed 604 comparison **within this bounded cohort**: the registered B2048 and
B512 policies use the same openings, seat-swapped design and P-E4 evaluator,
and the preregistered statistical gates remain unchanged. However, those
unknown states could overlap future roots. Therefore the comparison cannot be
described as fully historically independent, globally untouched, or evidence
of generalization to never-seen states. Report any result as conditional on the
authenticated bounded exclusion set and registered cohort. The missing Seed 461
population remains unresolved; no state reconstruction may assume a writer or
rules revision that is not bound to the historical run.

## Preserved Seed 604 experiment contract

This amendment does not change the P-E4 parent, B512 checkpoint, B2048 training
intervention, fresh-Adam restart and exact B512 reproduction gate, 2,048-update
budget, B512/B2048 comparisons against serving P-E4, common P-E4 value evaluator,
or either registered statistical gate. The authoritative contract remains
`registration.json`.

The Seed 561 inventory verification and Seed 599 source/identity checks are
read-only. Cohort generation and all later execution steps remain deferred.
