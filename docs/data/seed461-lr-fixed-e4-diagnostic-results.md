# PR #384 retrospective fixed-E4 diagnostic

Retrospective only; the shared holdout is consumed. PR #384's rejection of LR 0.0005 is preserved.

Mean fixed-E4 effect (B−A): **+0.0719** (95% opening-cluster bootstrap interval +0.0631 to +0.0805).
Mean selection interaction: **-0.0084** (95% interval -0.0209 to +0.0035); **no_clear_average_interaction**.

| Order | Selected effect | Fixed E4 effect | Interaction | A selected−E4 | B selected−E4 |
|---:|---:|---:|---:|---:|---:|
| 38411 | +0.1201 | -0.0674 | +0.1875 | +0.0000 | +0.1875 |
| 38412 | +0.1602 | +0.1543 | +0.0059 | -0.0059 | +0.0000 |
| 38413 | +0.1035 | +0.1035 | +0.0000 | +0.0000 | +0.0000 |
| 38414 | -0.0859 | +0.0352 | -0.1211 | +0.1211 | +0.0000 |
| 38415 | +0.0195 | +0.1338 | -0.1143 | +0.1172 | +0.0029 |

Between-order ranges: selected [-0.0859, +0.1602]; fixed E4 [-0.0674, +0.1543]; interaction [-0.1211, +0.1875].

Epoch-fixed heterogeneity persists descriptively: the five fixed-E4 effects span both signs and a wide range. A subsequent experiment should vary training order/permutation seed at fixed LR and epoch, using a fresh preregistered holdout.

No checkpoint was selected from these games, and this diagnostic does not change the original rejection or imply promotion.

Hash-bound inputs and per-run checkpoint/artifact/report/game identities are in `seed461-lr-fixed-e4-evidence.json`; compact numerical output is in `seed461-lr-fixed-e4-diagnostic-results.json`. Detailed games and exports are preserved under `.tmp/seed461-lr-sensitivity/`.
