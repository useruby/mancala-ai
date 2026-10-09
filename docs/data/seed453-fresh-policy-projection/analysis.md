# Seed453 — fresh-policy direction-projection screen

Prospective sixteen-update screen using the seed442 initializer and zero Adam
moments. C reproduced seed447 T's accepted scales and all post-step parameter
tensors exactly. Both arms used all sixteen proposals and evaluated all eight
scales at every proposal. The frozen fresh-training guard contains 1,399 unique
inputs. Projection first activated on an accepted proposal at step 4; steps
4–16 were active. No nonzero proposal was rejected.

## Fixed decision

**`close_fresh_policy_projection_branch`**

Projection activated, but B's fresh strict-unseen policy CE exceeded the
initializer under both weightings, failing the fixed `initializer + 2e-6`
fresh-cohort gate. The full training objective decreased and the pooled policy
and value gates passed. No follow-up, strength claim, export, or promotion is
authorized by this screen.

## Endpoint metrics

| Population | Weighting | Initializer CE | Ordinary Adam A CE | C CE | B CE | B−C CE | Initializer MSE | B MSE | B−C MSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict unseen >32 pooled | Exposure | 1.1706991593 | 1.1877460084 | 1.1472948612 | 1.1465885047 | -0.0007063565 | 0.4764106647 | 0.4759013209 | +0.0003345275 |
| Strict unseen >32 pooled | Equal input | 1.1981604519 | 1.2212629525 | 1.1968499526 | 1.1928895282 | -0.0039604244 | 0.6169980428 | 0.6123529477 | +0.0003364458 |
| Seed451 fresh unseen | Exposure | 1.2211021015 | — | 1.2368540808 | 1.2303543190 | -0.0064997618 | 0.7197047284 | 0.7100550229 | +0.0002382524 |
| Seed451 fresh unseen | Equal input | 1.2230564668 | — | 1.2397133994 | 1.2332080165 | -0.0065053829 | 0.7242157055 | 0.7149855712 | +0.0002804136 |

The seed451 fresh cohort has 859 exposures and 851 exact input identities.
Pooled strict unseen has 2,607 exposures and 1,242 exact input identities.
For B−C, the full-training-objective difference is `-0.0008774642`.

## Frozen decision gates

- Pooled B CE beats seed442 A by at least 0.01 and initializer by at least
  0.005 under both weightings: **pass**.
- Pooled B value MSE is within initializer +0.002 under both weightings:
  **pass**.
- Fresh B CE is at most initializer +2e-6 under both weightings: **fail**.
- Full training objective decreases: **pass**.
- Projection activates on an accepted nonzero proposal: **pass**.
- No nonzero proposal is rejected: **pass**.

The corrected independent verifier reports `valid` and independently rebuilds
the frozen inputs, guard, gradients, Adam moments and proposals, projection,
every trial, trajectories, metrics, predictions, and fixed classification. The
separate post-execution verifier correction is recorded in
`verifier-correction.json`; it did not modify the frozen registration or
training evidence. Historical seed447/448 decisions remain unchanged.
