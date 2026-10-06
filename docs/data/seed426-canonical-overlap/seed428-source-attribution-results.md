# Seed428 replay-source attribution

This is a retrospective decomposition of the already-published seed427
predictions. It is not a new independent holdout. The frozen method is
[`seed428-analysis-plan.md`](seed428-analysis-plan.md); the complete machine
readable source-by-cohort matrix is
[`seed428-source-contributions.json`](seed428-source-contributions.json).

## Primary cohort: unseen >32

There are 2,607 weighted positions and 1,242 canonical identities. Global policy
weight denominator is 2,607. Contributions use that global denominator for
exposure attribution. Equal-identity contributions retain each identity's
denominator across all of its source memberships.

| Replay source | Weighted positions | Identities | Initializer loss | E4 loss | Difference | Exposure contribution | Equal-identity contribution |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh | 859 | 851 | 1.2211021003 | 1.2434937878 | +0.0223916875 | +0.0073780052 | +0.0164289089 |
| generic_bootstrap | 1,044 | 259 | 1.1647652075 | 1.2300115789 | +0.0652463714 | +0.0261285814 | +0.0143458266 |
| opening_disagreement | 368 | 46 | 1.2628487526 | 1.3739155311 | +0.1110667784 | +0.0156780109 | +0.0040695060 |
| random_teacher | 8 | 8 | 1.5406170755 | 1.4448337729 | -0.0957833025 | -0.0002939265 | -0.0006169617 |
| stability | 328 | 82 | 0.9451763619 | 0.9953774068 | +0.0502010449 | +0.0063160501 | +0.0033144007 |
| **Reconciled total** | **2,607** | **1,242 global** | — | — | — | **+0.0552067212** | **+0.0375416805** |

Both totals reconcile to the seed427 published policy deltas within the frozen
absolute tolerance `1e-12`. Four global identities appear in multiple sources;
source identity counts therefore overlap. The pairwise overlap counts appear in
the machine-readable matrix. Source contributions include positive and negative
terms.

The fixed 75% positive-contribution rule yields **no single-source concentration**
for either exposure-weighted or equal-identity aggregation. Positive source
contributions total 0.0555006477 (exposure) and 0.0381586422 (equal identity);
the largest shares are 47.08% and 43.05%, respectively.
Across individual canonical identities, the largest positive identity contributes
2.996% of total positive identity deltas; the largest absolute identity-level
delta is +1.7188885106. These identity deltas are descriptive weighted averages,
not independent observations.

## Secondary cohorts

| Cohort | Weighted positions | Identities | Exposure delta | Equal-identity delta | Exposure concentration | Equal-identity concentration | Largest positive identity share |
|---|---:|---:|---:|---:|---|---|---:|
| seen >32 | 3,166 | 774 | -0.0210745499 | -0.0426795149 | concentrated | no single-source concentration | 0.6005% |
| unseen 17–32 | 4,013 | 2,322 | +0.0646683688 | +0.0305547362 | no single-source concentration | no single-source concentration | 1.1293% |
| unseen ≤16 | 4,086 | 2,116 | -0.0394254422 | -0.1096590307 | concentrated | no single-source concentration | 1.4235% |

Each secondary cohort's five source rows, signed contributions, membership
overlap, identity counts, and concentration shares are in the JSON matrix.

## Interpretation and limitations

These are accounting contributions to observed loss differences, not causal
effects. Replay-source memberships overlap at identity level, and repeated
weighted rows are not independent observations. In the primary cohort, the four
cross-source identities were retained with their global identity denominators;
the source counts should not be summed to infer unique identities. This
descriptive decomposition establishes no causal explanation and authorizes no
reweighting, training, checkpoint selection, or promotion. The population is a
retrospective subset of the historical seed416 validation set, not an independent
holdout. No new forward passes, targets, searches, or games were used.
