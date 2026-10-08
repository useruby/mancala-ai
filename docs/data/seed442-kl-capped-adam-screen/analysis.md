# Seed442 KL-capped Adam screen

This is a short retrospective-population screen and is not evidence of playing
strength. Arm A used ordinary Adam; arm B capped each Adam proposal by its
immediate pre-step legal-policy KL on the current minibatch and the fixed
training-only guard cohort.

## Fixed decision

**Classification: `close_kl_capped_step_branch`.** The KL cap activated on the
screen, with at least one nonzero proposal accepted below scale 1 and no
nonzero proposals rejected. The screen does not satisfy every advancement
clause, so it does not authorize a strength experiment.

| Metric | Initializer | A: ordinary Adam | B: KL-capped Adam |
| --- | ---: | ---: | ---: |
| Full frozen-training objective | 1.0214983345 | 1.0195943288 | 1.0019462621 |
| Unseen policy CE, exposure weighted | 1.1706991593 | 1.1877460084 | 1.1558827694 |
| Unseen policy CE, equal input | 1.1981604519 | 1.2212629525 | 1.1965274705 |
| Unseen value MSE, exposure weighted | 0.4764106647 | 0.4848726256 | 0.4809968564 |
| Unseen value MSE, equal input | 0.6169980428 | 0.6210176897 | 0.6208880654 |

B improved policy CE over A under both weightings and decreased the full
training objective. It did not reach the registered policy improvement versus
initializer under equal-input weighting, and its value MSE exceeded the
initializer guard under both weightings.

The complete per-proposal, per-trial, optimizer, prediction, and decision
records are in `evidence.json`, `step-tensors.npz`,
`optimizer-state-reconstruction.npz`, the three `*-predictions.npz` archives,
and `receipt.json`. The optimizer-state reconstruction stores independently
replayed post-step Adam moments from the archived clipped gradients, alongside
the once-advanced optimizer step records.
