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

The 512 selected roots are the complete finite population for the amended
primary estimand. Compute the paired B−A and C−A contrasts exactly on that
finite selected population after all required outcomes are available. Do not
interpret a bootstrap interval as uncertainty about a larger stochastic
population.

Opening-position cluster bootstrap resampling (20,000 resamples under the frozen
analysis seed) may be retained only as a **descriptive stability analysis**:
resample the frozen opening clusters and report how the standardized contrast
varies under that perturbation. Label the output `cluster-resampling stability
interval`, not a confidence interval. It describes sensitivity to the observed
cluster composition and does not have a confidence-coverage or design-based
interpretation for the random-generation population.

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

Replace the primary improvement, no-improvement, inconclusive, and WDL
non-inferiority rules that use interval endpoints as inferential gates with this
finite-cohort descriptive rule:

1. Report `observed selected-population regret threshold met` if and only if the
   exact B−A finite-cohort mean regret difference is at most −0.25 seeds and the
   exact finite-cohort S615-minus-B512 WDL-inferior rate difference is at most
   +1 percentage point.
2. Otherwise report `observed selected-population regret threshold not met`,
   with a separate flag if the finite-cohort WDL-inferior difference exceeds
   +1 percentage point.
3. If any required exact action coverage or outcome is missing, report
   `unavailable`; do not substitute a bootstrap endpoint or another population.

Report the cluster-resampling stability interval beside the exact finite-cohort
contrast, but never use it to decide the labels above. These labels describe
only the frozen selected population. They do not establish improvement, no
improvement, non-inferiority, or generalization for the broader generation
distribution. Any such stochastic-population claim requires a separately
registered sampling design and estimand.

## Required amendment record

Before this proposal is considered approved, record the approver's project
identity, approval timestamp, exact amended protocol commit/hash, and the
explicit decision to withdraw the inclusion-probability and design-unbiased
inference requirements. Approval must be independent of this implementation
PR. No generation authorization is requested here.

For execution under Seed 636, record the decision as a GitHub comment on an
issue or pull request in `useruby/mancala-ai`. The approval artifact must bind
the GitHub owner identity, comment URL and timestamp, exact protocol revision,
SHA-256 for this proposal and the Seed 623–631 execution-contract amendment,
the experiment-registration SHA-256, and the explicit
`model-independent-generation-and-screening-only` scope. A merge alone is not
approval. The Seed 636 approval template provides the required comment body and
record structure; until the owner comment is present and verified, the Seed 624
execution entry point remains blocked.
