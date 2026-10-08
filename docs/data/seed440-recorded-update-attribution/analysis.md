# Seed440: recorded-update attribution

This is a retrospective observational decomposition of the two archive-defined
seed435 paths. The endpoint changes had already been observed before this
diagnostic was run: A16 minus initializer was approximately +0.01705
exposure-weighted and +0.02310 equal-input policy CE. No training, optimizer
steps, audit replay, searches, games, target changes, or promotion occurred.

The authoritative seed427 unseen >32 population contains 2,607 exposures over
1,242 exact-input identities. Equal-input weighting first averages within each
identity while retaining the exposure multiplicity, then gives each identity
equal mass. The step ledger uses the production legal mask and CE, float64
gradient accumulation from float32 autograd chunk gradients, and float32
parameter additions in the registered seed435 parameter order.

| Arm | Weighting | D (endpoint change) | S (sum of first-order dots) | R (signed residual sum) |
|---|---|---:|---:|---:|
| A | Exposure-weighted | +0.01704685 | -0.57653065 | +0.59357750 |
| A | Equal-input | +0.02310250 | -0.52486084 | +0.54796334 |
| B | Exposure-weighted | +0.15330187 | -2.86909801 | +3.02239988 |
| B | Equal-input | +0.13132338 | -2.80483675 | +2.93616013 |

The fixed primary A classification is **finite_step_residual_dominant** under
both weightings: D is positive and R is at least 75% of D. S is negative under
both weightings. This says the full-step residual more than offsets the
favorable first-order direction. Signed residuals include nonlinear behavior
and numerical effects; they are not identified as pure curvature. Midpoint
CEs and linear-prediction residuals are in `step-ledger.json`.

All 16 step changes telescope exactly in the published arithmetic to the
endpoint changes; S+R reconciles to D, and the trunk/policy/value parameter-group
dot contributions reconcile to each step's S (the value-head policy-gradient
contribution is zero). The final arrays match the original A16/B16 checkpoint
arrays bit-for-bit. The listed per-state hashes are hashes of
**archive-defined reconstructed paths**, not independently logged intermediate
checkpoints.

Limitations: this attribution does not establish causal mechanism, generalize
beyond the fixed validation population, establish playing strength, or
authorize any intervention. B is secondary and does not alter A's fixed
classification.
