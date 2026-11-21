# Seed 624 implementation readiness

## Status

Implementation proposal and synthetic checks only. No production population was
generated, frozen, inferred on, searched, solved, or trained on.

## Sampling and estimand amendment required

The proposed selector is deterministic and constrained by stone-bin quotas,
game caps, opening caps, and opening diversity. Its unequal inclusion
probabilities are not calculated. The defensible implementation contract is
therefore option B: the target is the deterministic, hash-standardized selected
population, with fixed weight 1/5 for each stone bin. There is no claim of
design-unbiased inference over the broader random-generation population, and no
ad hoc state-level design weights are permitted. This narrows/changes the
protocol's PR #623 language about reporting candidate inclusion probabilities
and applying candidate inclusion weights. A protocol amendment and owner
acceptance are required before real generation.

## Proposed freeze status

`proposed_freeze_schema()` emits a schema template only, marked
`proposal_not_ready_or_approved`. It is not an execution freeze. Before an
authorized freeze, the repository still needs:

- Independent review and acceptance of the estimand amendment above.
- Final source/runtime identities, including a supported Python/NumPy/PRNG
  contract and finalized selector runtime/resource bounds.
- Authenticated, identity-complete snapshots for B512, S615, protected holdout,
  and available compatible source-game/opening provenance. Missing evidence
  must fail closed; a nonmatch remains unresolved for inherited P-E4 exposure.
- An independent review of the exhaustive selector's worst-case runtime and
  independent reproduction of its exact rank-lexicographic selection.
- Project-owner authorization bound to the final source hashes, snapshot hashes,
  create-only output paths, verification commands, and resource limits.

The production entry point requires explicit authorization, a freeze digest,
owner identity, and membership-evidence path. The implementation has not called
that entry point.

## Exposure language

Only `confirmed_known_exposure`,
`known_exposure_nonmatch_inherited_PE4_unresolved`, and
`invalid_or_insufficiently_authenticated_evidence` are used. No
`training_disjoint`, unseen, contamination-free, or equivalent status is
produced. Identity comparison uses canonical JSON and exact production
little-endian `kalah_v3` float32 bytes.

## Remaining review blockers

1. Validate the selected-population estimand change.
2. Review and benchmark the exact selector on an independently generated
   synthetic constraint fixture before approving its frozen resource limit.
3. Exercise the independent verifier from a separately copied, complete
   synthetic artifact bundle, including its optimizer rank reconstruction.
4. Bind runtime and source identities in the final freeze; this proposal is not
   ready or approved for production.
