# Seed428 frozen analysis plan

This plan is frozen before computing the replay-source decomposition. The analysis
uses only the already published seed427 predictions and membership evidence. It is
a retrospective decomposition of the same seed416 validation population, not a
new independent holdout; weighted copies and appearances across sources are not
independent observations.

## Primary cohort and estimands

The primary cohort is `unseen/>32` (2,607 weighted positions, 1,242 canonical
identities). For each of the five sources registered by seed416, count published
weighted rows and distinct canonical identities, then calculate E4-minus-initializer
policy-loss contribution row by row.

For exposure-weighted attribution, sum source row-loss differences times policy
weight and divide by the **global cohort policy-weight denominator**, including
sources with no rows as zero contributions. Contributions must sum to the global
exposure-weighted delta.

For equal-canonical-identity attribution, within every global identity sum each
source's weighted row-loss differences and divide by that identity's total policy
weight across all sources. Average these identity-specific source values over all
global identities. Identities appearing in multiple sources retain their global
denominator and contribute to each such source. Contributions must sum to the
global equal-identity delta.

Reconcile to seed427's published deltas: exposure-weighted `+0.05520672117819547`
and equal-identity `+0.03754168053554596`, with absolute tolerance `1e-12`.

## Secondary cohorts and reporting

Repeat counts, source contributions, overlap, and concentration summaries for
`seen/>32`, `unseen/17-32`, and `unseen/<=16`. Show signed contributions, including
positive and negative sources, number of source identities, and identity-level
concentration (largest absolute identity contribution and share of positive
identity-level contribution). Report source-membership overlap as the pairwise
shared-identity counts and the number of identities represented by multiple
sources.

The fixed descriptive concentration rule applies separately by aggregation: a
source is “concentrated” only when its positive contribution is at least 75% of
the sum of positive source contributions; otherwise report “no single-source
concentration.” This classification is descriptive only: it establishes no
causal explanation and authorizes no reweighting or training.

## Constraints and limitations

Do not train, run new forward passes, generate targets, search, play arena games,
change replay, select checkpoints, or promote a model. Do not revise seed427
estimates absent a demonstrated calculation defect; document any such defect
separately. Source memberships overlap at canonical-identity level and compact
rows can have weighted copies, so neither source counts nor identity-level
contributions are independent-sample estimates. The source decomposition is
accounting attribution of published loss differences, not a causal analysis.
