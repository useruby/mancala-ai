# Seed462 recovery: retrospective reconstruction, no experiment decision

The registered 16-step C and F archives were independently reconstructed from the
registered initializer and minibatches. All per-step objective gradients, clipped
gradients, Adam moments, proposals, and post-states match the archived tensors.
Control C matches seed442 A exactly: 16/16 updates and 352/352 post-state parameter
tensors. The original attempt remains preserved with its invalid/unverified status.

The endpoint forward passes and validation-target metrics below are retrospective.
They reuse the registered membership file and are not new holdout evidence.

| Cohort / weighting | Initializer CE / MSE | C CE / MSE | F CE / MSE | F−C CE / MSE |
|---|---:|---:|---:|---:|
| Pooled / exposure | 1.170699 / 0.476411 | 1.187746 / 0.484873 | 1.234367 / 0.488482 | +0.046621 / +0.003609 |
| Pooled / equal input | 1.198160 / 0.616998 | 1.221263 / 0.621018 | 1.239726 / 0.626683 | +0.018463 / +0.005665 |
| Fresh / exposure | 1.221102 / 0.719705 | 1.250081 / 0.716730 | 1.243930 / 0.725162 | −0.006151 / +0.008432 |
| Fresh / equal input | 1.223056 / 0.724216 | 1.251720 / 0.722187 | 1.244640 / 0.730438 | −0.007080 / +0.008251 |

The reconstructed full-training fresh-policy objective decreases from 1.053216 at
the initializer to 1.004433 for F. Numeric gates versus the initializer fail for
fresh CE, pooled CE, pooled MSE, and fresh MSE; F also fails the fresh-MSE guard
relative to the initializer. These values do not support advancement.

**No registered experiment decision is assigned.** The registration binds the full
validation-membership file but does not explicitly freeze the pooled/fresh `>32`
selection rules or the exposure-weighted and equal-input aggregation definitions.
Adding those estimands after observing results would retrofit the protocol. The
numeric gate calculations are therefore descriptive recovery diagnostics only.

The endpoint predictions retain occurrence order and each source record's own
policy/value target, including repeated identities with differing targets. Recovery
details, source restoration history, exact control reconciliation, machine-readable
metrics, and the post-execution receipt are in this directory.
