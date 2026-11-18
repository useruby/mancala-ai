# Seed 615 — Source-opening diversity feasibility

**Decision: `train` is recommended for owner review.** A deterministic
source-opening-balanced sample increases the effective number of opening groups
by 17.46 (13.61%) over the immutable Seed 599 B / Seed 604 B512 membership,
while preserving all three phase quotas and all original teacher targets. No
training or evaluation was run for this preparation.

## Authentication and sample definition

The preparation reran the read-only Seed 599 preflight, including the published
#580/#585 source verifiers and target reconstruction. It reproduced the
registered held-out provenance split and verified its canonical membership
digest. The B512 membership has 2,048 unique canonical and production-float32
states and matches the membership digest bound by the Seed 604 registration;
the immutable B512 checkpoint hash also matches that registration. The
authenticated eligible source pool after held-out exclusions contains 5,287
unique states across 150 `(source, opening_id)` groups. The complete available
counts for each source-opening group and phase are in `feasibility.json`.

For each phase, the treatment orders groups by
`SHA256("seed615-group:615:<phase>:<source>:<opening_id>")`, and orders each
group's eligible states by
`SHA256("seed615-state:615:<phase>:<canonical_identity>")`. It takes one
successive state per group per round until the quota is met. Exhausted groups
are skipped. Hash ties break lexicographically by group tuple or canonical
identity. Selection fails closed if a phase cannot supply its exact quota.
The final membership is sorted by `(phase, canonical_identity)`.

The selected set has exactly 512 / 1,024 / 512 states in the registered
`above_32` / `17_to_32` / `at_most_16` bins, with 2,048 unique identities under
both canonical and production-float32 encodings. It is disjoint from the
provenance-held-out membership under both identities. Each row retains the
authenticated source record and its unchanged reconstructed target.

## Diversity result

| Measure | B512 | Candidate |
|---|---:|---:|
| States | 2,048 | 2,048 |
| Phase quotas (`>32 / 17–32 / <=16`) | 512 / 1,024 / 512 | 512 / 1,024 / 512 |
| Distinct source-opening groups | 150 | 150 |
| Effective groups, `1 / sum(p_group^2)` | 128.3133 | 145.7773 |
| Largest-group share | 1.855% | 0.830% |
| Top-10-group share | 13.086% | 7.861% |
| Shared canonical states with the other membership | 928 | 928 |
| Membership-specific canonical states | 1,120 | 1,120 |

The number of represented groups is already saturated at 150 for B512; the
treatment's substantive change is a more even allocation across those groups.
Effective group count rises by 17.4640, a ratio of 1.1361. This is a
pre-training coverage comparison, not evidence of improved playing strength.
The memberships share 928 examples and each contributes 1,120 states not in the
other set, so the experiment contrasts both group allocation and a substantial
membership change.

## Secondary differences and possible confounders

Both memberships have 1,536 `artifact_puct` and 512 `exact_root_tablebase`
targets. No target is regenerated. The selected targets' mean entropy differs
(0.26052 to 0.27480 nats), as does mean maximum target probability (0.89844 to
0.89141). These reflect different original teacher records and are a potential
secondary difference, even though teacher-source totals and targets themselves
are preserved.

Stone-count, legal-move-count, and empty-pit histograms are reported by phase in
`feasibility.json`. They remain within the exact phase quotas but are not
identical. For example, the candidate selects one state with one active pit
stone in the `at_most_16` phase (B512 selects none); the `above_32` candidate
contains no one-legal-move state (B512 contains one). These occupancy and
teacher-target composition changes should be treated as possible confounders
when interpreting any future training contrast.

## Prospective registration and precision

`registration.json` fixes the existing B512 checkpoint as control and the
candidate membership digest as treatment. The prepared independent evaluation
cohort contains 1,024 legal random 12-ply opening roots and is disjoint in both
identity schemes from eligible/held-out source states and authenticated Seed
599/604 evaluation roots. It binds 4,096 seat-swapped games; the cohort is
recorded in `evaluation-openings.jsonl`, and game seats/seeds are frozen in
`evaluation-assignments.jsonl`. Both canonical digests are bound in the
registration. These are assignments only; no games were executed.

The initial design's bounded-outcome precision assessment uses no historical
game outcomes. At 1,024 opening clusters, using the standard bounded
independent-cluster Hoeffding planning assumption, the radius for the paired
treatment-minus-B512 cluster contrast in
`[-1, 1]` is 0.0849 at two-sided 95% confidence. For a candidate score in
`[0, 1]` against P-E4, the one-sided 95% Hoeffding radius is 0.0382. These are
conservative bounds, not predicted intervals or a guarantee to detect a small
effect. The registered analysis uses 20,000 paired opening-cluster bootstrap
resamples, preserving both seat outcomes; the design remains the requested
initial 1,024-opening / 4,096-game design.

The primary confirmatory gate is the two-sided paired 95% interval for
treatment-minus-B512 seat-averaged playing score, with its lower endpoint
strictly above zero. Separately, the treatment's one-sided 95% lower bound
against P-E4 must be strictly above 0.50 before it can be considered stronger
than serving P-E4.

The independent verifier and synthetic negative tests pass. Training and
evaluation remain unauthorized; this report and registration are preparation
artifacts only.
