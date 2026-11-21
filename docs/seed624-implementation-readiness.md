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

The explicit proposed amendment is recorded in
[`seed624-estimand-amendment-proposal.md`](seed624-estimand-amendment-proposal.md).
It is not approved and does not alter merged PR #623. Its incompatible
inclusion-probability and design-unbiased-inference requirements remain in force
until the project owner approves and merges that amendment.

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
- An independently reviewed project-owner Ed25519 public key must be added to
  the source-pinned trust allowlist. The allowlist is currently empty, so the
  production entry point rejects every approval record. Caller-provided keys
  and identity flags are not trusted.
- Project-owner signature over the exact finalized freeze SHA-256 and the
  `model-independent-generation-and-screening` scope, bound to the finalized
  source hashes, snapshot hashes, create-only output paths, verifier commands,
  and resource limits.

The production entry point verifies an Ed25519 approval against a repository-
pinned key, exact freeze digest, and exact execution scope. The implementation
has not called that entry point. The reviewed key is not yet provisioned.

## Opening-prefix reserve semantics

An opening seed whose random prefix does not produce exactly eight legal,
nonterminal actions consumes that reserve index and is skipped. The scan
proceeds to the next fixed index; no seed is reused or replaced. The bundle
records each consumed index, accepted opening, invalid-prefix rejection (seed,
partial actions, terminal flag, resulting state, and both identities), and
duplicate-opening rejection. The independent verifier replays the entire
consumed reserve ledger and requires the published openings to be the first
distinct valid entries under this rule.

## Synthetic selector resource benchmark

`seed624-dfs-bounded-v2` uses a deterministic rank-first exact search with
candidate-count ceiling 10,240, recursively measured Python input-object ceiling
128 MiB, 250,000 search nodes, 30 seconds, and 64 MiB traced optimizer working
memory. On this host, a synthetic feasible 4x-supply pool of 2,048 candidates
selected 512 states in 0.308 s wall time (0.115 s in the optimizer), 518 nodes,
4,528,318 measured input bytes, and 3,257,184 traced working bytes; 227 opening
clusters were represented. Rows included synthetic state objects, 64-action
trajectory lists, and repeated-bin visit metadata. A synthetic infeasible
4x-supply pool of 2,048 candidates concentrated in 100 openings failed its
opening-capacity bound in 0.0630 s (game capacity 2,048; opening capacity 400;
target 512). These are synthetic measurements, not production estimates. Other
hard instances terminate with a structured node/time/memory shortfall rather
than claiming infeasibility.

Additional adversarial fixtures exercise backtracking rather than the aggregate
prechecks: a feasible five-bin fixture selects a later-ranked alternative after
the early-rank path dead-ends; a six-bin fixture passes hard supply, total game
capacity, opening capacity, and cluster-count gates but exhausts the exact
search as mathematically infeasible. Repeated node-limit runs return the same
node count and `optimizer_resource_limit`; an injected deterministic clock
returns a time-limit shortfall at the same search node. Neither resource-limit
result is classified as mathematical infeasibility.

The 128 MiB deep-object estimate and 64 MiB tracemalloc ceiling are Python
allocation guards, **not hard process RSS limits**; tracemalloc excludes some
interpreter, native-library, and operating-system memory. A production freeze
must separately register host/process RSS, CPU, and wall-time caps in its
`production_resource_limits` record, and the authorized execution harness must
enforce those external caps. Those values are intentionally unset in this
proposal. Selector-specific synthetic limits do not replace execution resource
caps.

## Exposure language

Only `confirmed_known_exposure`,
`known_exposure_nonmatch_inherited_PE4_unresolved`, and
`invalid_or_insufficiently_authenticated_evidence` are used. No
`training_disjoint`, unseen, contamination-free, or equivalent status is
produced. Identity comparison uses canonical JSON and exact production
little-endian `kalah_v3` float32 bytes.

## Remaining review blockers

1. Validate the selected-population estimand change.
2. Review the bounded selector semantics, resource limits, and synthetic
   feasible/infeasible benchmark results.
3. Approve and merge the estimand amendment; do not drop the existing
   inclusion-probability wording before that approval.
4. Provision an independently reviewed project-owner public key in the
   allowlist, then approve a signed, finalized freeze before production.
5. Bind final runtime/source/dependency identities in the signed freeze. This
   implementation is not ready for production authorization.
