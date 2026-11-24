# Seed 624 implementation readiness

## Status

Implementation proposal and synthetic checks only. No production population was
generated, frozen, inferred on, searched, solved, or trained on.

## Active statistical contract and approval gate

The active statistical contract in merged PR #625 is the finite 512-state
selected-population estimand with fixed weight 1/5 for each stone bin. Opening-
cluster resampling intervals are descriptive stability intervals, not inferential
confidence intervals. No claim of design-unbiased inference over the broader
random-generation population is made.

The explicit proposed amendment is recorded in
[`seed624-estimand-amendment-proposal.md`](seed624-estimand-amendment-proposal.md).
PR #625 merged, but the available PR review record has no independent
project-owner approval. The amendment's merge is not evidence of owner approval.
The Seed 636 migration accepts only a GitHub owner comment that binds the exact
experiment-registration digest, protocol revision/document hashes, and
generation/screening-only scope. Until such approval is recorded, generation
remains blocked.

## Proposed freeze status

`proposed_freeze_schema()` emits a schema template only, marked
`proposal_not_ready_or_approved`. It is not an execution freeze. Before an
authorized freeze, the repository still needs:

- Explicit, independently recorded project-owner approval of PR #625's
  amendment (blocking; absent in the inspected PR record).
- Final source/runtime identities, including a supported Python/NumPy/PRNG
  contract and finalized selector runtime/resource bounds.
- Authenticated, identity-complete snapshots for B512, S615, protected holdout,
  and available compatible source-game/opening provenance. Missing evidence
  must fail closed; a nonmatch remains unresolved for inherited P-E4 exposure.
- An auditable GitHub project-owner comment must approve the exact registration
  digest, approved protocol revision and document hashes, and
  `model-independent-generation-and-screening-only` scope.
- A standard Seed 635 lifecycle freeze must bind the clean worktree source tree,
  exact command/configuration, inputs, runtime, resource budget, and output paths.

The production entry point verifies the GitHub comment through `gh api`, then
checks its owner identity, exact comment URL/timestamp/body, registration digest,
protocol revision/document hashes, and execution scope. It also requires the
Seed 635 lifecycle runner's frozen and started attempt, current source tree, and
input identities. Failed attempts are preserved by the lifecycle terminal
record and attempt directory. The legacy Seed 629 production launch guard is
unchanged and is not called by this generation path.

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

1. Obtain and record explicit GitHub project-owner approval of the Seed 625
   estimand amendment, Seed 635 execution-contract amendment, unresolved P-E4
   exposure limitation, and generation/screening-only scope (blocking).
2. Review the bounded selector semantics, resource limits, and synthetic
   feasible/infeasible benchmark results.
3. Freeze the final runtime/source/dependency/input identities and resource
   budget in the standard lifecycle attempt before any population generation.
4. Preserve the failed attempt and stop if any preregistered gate fails. This
   migration does not itself authorize execution.
