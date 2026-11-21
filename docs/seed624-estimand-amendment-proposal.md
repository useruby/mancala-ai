# Proposed amendment to Seed 623 sampling and estimand language

**Status: proposed only; not approved.** This proposal does not change the
merged Seed 623 protocol. Until the project owner approves and merges a protocol
amendment, the Seed 623 inclusion-probability and candidate inclusion-weight
requirements remain authoritative and this implementation must remain draft.

## Requested decision

Approve option B for the Seed 623 primary population: define it as the
deterministic, hash-standardized cohort selected from the generated,
known-exposure-screened candidate inventory under the frozen exact quotas and
cluster constraints. Assign each of the five stone-count bins a fixed weight of
`1/5`. Do not claim design-unbiased inference over the broader random-generation
population. Do not calculate, report, or apply candidate-level inclusion
probabilities or design weights. Report descriptive unweighted results as
sensitivity summaries only.

## Proposed replacement text

Replace the sentences in Seed 623 that require publishing candidate inclusion
probabilities, applying candidate inclusion weights, and describing the
primary endpoint as inclusion-weighted with:

> The constrained, hash-ranked selection defines a deterministic standardized
> selected-state population. The primary target is the equal-weighted mixture
> of the five eligible stone-count bins, using fixed weight `1/5` per bin.
> Selection is not a probability sample from the broader random-generation
> population; no candidate-level inclusion probabilities or design weights are
> claimed or applied. Estimates and uncertainty statements are conditional on
> the frozen selected population and its opening-position clusters. Report the
> complete generated candidate inventory, all exclusions, all ranks, the exact
> selected set, per-bin counts, and cluster membership so the selected target
> can be reconstructed. These results do not support design-unbiased inference
> over all states reachable under the generation process.

Replace the primary-endpoint wording `cluster- and inclusion-weighted mean`
with `opening-cluster-aware, fixed-stone-bin-standardized mean`, retaining the
registered paired contrast and the fixed `1/5` stone-bin weights.

## Required amendment record

Before this proposal is considered approved, record the approver's project
identity, approval timestamp, exact amended protocol commit/hash, and the
explicit decision to withdraw the inclusion-probability and design-unbiased
inference requirements. Approval must be independent of this implementation
PR. No generation authorization is requested here.
