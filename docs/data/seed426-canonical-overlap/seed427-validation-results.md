# Seed427 retrospective validation correction

This publication supplements, and does not alter, #426's original figures or
chronology. The metrics compare the exact seed455 initializer used by #416's
training lineage with the fixed #416 lane A E4 checkpoint. The subsets are
retrospective subsets of historical validation, not a fresh independent
holdout; no game-family independence is inferred.

## Frozen split and identity reconciliation

The original validation partition remains unchanged at 14,946 weighted
positions. Its retrospective subsets contain 4,240 seen positions and 10,706
unseen positions. Canonical-state and exact float32 network-input exclusions
agree on every validation position. The exact position/row/identity membership
is in `seed427-validation-membership.jsonl.gz`; the counts are in
`seed427-validation-subsets.json`.

## E4 parity and primary comparison

E4's full-validation replay matched the #416 training record within the
predeclared absolute tolerance `1e-6` and relative tolerance `1e-5`:

| Metric | Recomputed E4 | #416 record | Difference |
|---|---:|---:|---:|
| Legal-masked policy cross-entropy | 0.9556284594510852 | 0.9556283950805664 | 0.0000000643705188 |
| Huber value loss, delta 1 | 0.20779636462911685 | 0.20779633522033691 | 0.0000000294087799 |
| Total (`policy + 0.3 × value`) | 1.0179673688398203 | 1.0179673433303833 | 0.0000000255094370 |

The primary comparison is E4 minus initializer on unseen `>32` positions:

| Aggregation | Policy difference | Value difference | Total difference |
|---|---:|---:|---:|
| Exposure-weighted | +0.05520672117819547 | +0.004363925747575298 | +0.05651589890246811 |
| Equal canonical identity | +0.03754168053554596 | −0.002249060767965716 | +0.036866962305156514 |

There are 2,607 weighted positions and 1,242 canonical identities in this
primary subset. Both models have policy-weight denominator 2,607. This is a
validation-loss comparison only: it makes no playing-strength claim and does
not select a checkpoint.

## Complete results and reproducibility

`seed427-evaluation-results.json` reports both models for all, seen, and unseen
subsets, overall and in `>32`, `17–32`, and `≤16` buckets. Every cell includes
exposure-weighted and equal-canonical-identity policy, value, and total losses,
weighted-position, policy-weight, and identity denominators, plus E4-minus-
initializer differences. For equal-identity means, position losses (including
historical multiplicities) are averaged within each canonical identity first;
identity means are then averaged equally.

`seed427-prediction-evidence.jsonl.gz` contains the two models' per-position
logits and values, targets, legal masks, weights, and identity/membership links,
so losses and aggregates can be reconstructed without either checkpoint.
`seed427-evaluation-manifest.json` was frozen before the accepted forward pass
and binds model/checkpoint/loaded-parameter identities, source target hashes,
subset membership, inference settings, loss definitions, tolerances, and
execution-source snapshots. `seed427-evaluation-results.json` binds that
manifest and the prediction evidence, records the full-validation E4 parity,
and confirms both checkpoint byte hashes were unchanged before and after.

Run the read-only, checkpoint-free verification from the repository root:

```bash
python ml/alphazero_lite/verify_seed427_evaluation.py
```

No training or arena follow-up was performed.
