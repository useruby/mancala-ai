# Seed483 corrected prospective confirmation protocol

**Supersedes only the conflicting analysis choices in
`seed482-prospective-confirmation.md`. Seed482's prospective preparation and
distribution publication remain in force. This protocol is frozen before any
new generation or candidate inference. No such execution is authorized here.**

## Population and sampling unit

The confirmation population is the selected seed482 `>32` active-stone state
population. Do not pool it with any historical cohort. Exactly one eligible
position is selected per complete independent game, before candidate inference;
therefore game ID and selected identity are the same sampling unit. No game may
contribute more than one selected state. Exclusions apply to both canonical
state identity and exact little-endian float32 `kalah_v3` input identity.

## Estimands and bootstrap

For each selected game/identity, compute per-record policy cross-entropy for F,
initializer, and M against the frozen policy target, and per-record value MSE
for F and initializer against the frozen value target. Contrasts are candidate
minus comparator, so negative values favor F. Exposure weighting is the
arithmetic mean over selected identities. Equal-input weighting first groups
records by exact float32 input identity, averages within each identity, then
takes the arithmetic mean over identities. Since selection enforces unique
float32 identities, the two estimands coincide for this one-position-per-game
cohort; both are still reported and gated separately as registered.

Use 20,000 nonparametric identity/game-cluster bootstrap replicates with RNG
seed 481. For each replicate, draw N game IDs with replacement from the N
selected IDs using NumPy `Generator(PCG64(481))`; each draw contributes its
single selected record, including repeated records when a game is drawn more
than once. Recompute each estimand and contrast on that resampled multiset.
Do not deduplicate bootstrap draws. Use the empirical inverse-CDF quantile
convention: sort the 20,000 replicate statistics and select order statistic
`ceil(p * 20000) - 1` (zero-based), with endpoint clamping. The one-sided upper
bound is the 98.75th or 97.5th percentile as specified below. Report point
estimates and these bounds; do not replace them with descriptive/two-sided
intervals or point-only gates.

## Frozen gates

All four CE contrasts must pass, separately:

| Contrast | Point improvement | One-sided upper bound |
|---|---:|---:|
| F − initializer, exposure weighted | improvement >= 0.001 | 98.75% upper bound < 0 |
| F − M, exposure weighted | improvement >= 0.001 | 98.75% upper bound < 0 |
| F − initializer, equal-input weighted | improvement >= 0.001 | 98.75% upper bound < 0 |
| F − M, equal-input weighted | improvement >= 0.001 | 98.75% upper bound < 0 |

“Improvement” is the negative of the F-minus-comparator CE contrast. Both value
MSE contrasts, F − initializer under exposure and equal-input weighting, must
have one-sided 97.5% upper bound <= 0.01. Equality passes the value bound;
equality at zero does not pass the strict CE upper-bound requirement. No other
legacy pooled or training-trajectory gate is part of this confirmation decision.

Advance to `advance_to_registered_strength_evaluation` only if all six gates
pass. Otherwise decide `stop_fresh_confirmation`. No strength evaluation is
authorized by this preparation publication.

## Exclusion authorization is independent of labels

An experiment-ID list, a `complete` coverage string, or a nonempty source list
is not proof of historical exclusion coverage. Generation must remain blocked
until a verifier independently reconstructs each source contribution from its
authenticated bytes, validates any reused-cohort subset proof against its
parent population, records explicit empty contributions only where the
source's no-probe status is evidenced, independently rebuilds the canonical +
float32 identity union, and binds the exact final manifest digest. The coverage
table must distinguish an inspected source with no new identities from an
uninspected source. The exact correction package and source-by-source evidence
inventory are published separately as seed483 exclusion-closure evidence.
