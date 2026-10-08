# Seed437 training-versus-unseen gradient alignment

**Classification:** `no_persistent_training_objective_opposition`

This is a retrospective local-gradient diagnostic, not strength evidence. It
does not establish causality and does not authorize an intervention, training,
search, play, target change, or model promotion.

The frozen population reconstructs 87,625 compact source rows into 149,448
replay exposures, partitioned into 134,502 training and 14,946 validation
exposures by the committed #416 v3 split. The #427 strict unseen `>32` subset
contains 2,607 validation exposures over 1,242 exact-input identities.

For each checkpoint, `P` is coefficient-weighted legal-masked training policy
cross entropy, `0.3V` is 0.3 times the training Huber value-loss gradient with
delta 1, and `T=P+0.3V`. `U_exposure` weights each unseen exposure equally;
`U_equal` gives each exact input identity equal total mass while averaging its
exposures internally. The reported unit-distance first-order CE change is
`-dot(U,G)/||G||`; negative values indicate a local decrease in unseen CE
along unit descent in the indicated training objective.

| Checkpoint | U weighting | Objective | Dot(U,G) | Cosine | Unit-distance unseen CE change |
|---|---|---:|---:|---:|---:|
| Initializer | Exposure | P | 2.232048 | 0.851289 | -1.950702 |
| Initializer | Exposure | 0.3V | 0.006192 | 0.049916 | -0.114381 |
| Initializer | Exposure | T | 2.238240 | 0.850110 | -1.948000 |
| Initializer | Equal input | P | 0.987634 | 0.597241 | -0.863145 |
| Initializer | Equal input | 0.3V | 0.005141 | 0.065710 | -0.094966 |
| Initializer | Equal input | T | 0.992776 | 0.597860 | -0.864039 |
| Adam A16 | Exposure | P | 2.554817 | 0.853922 | -2.008680 |
| Adam A16 | Exposure | 0.3V | 0.010798 | 0.107616 | -0.253146 |
| Adam A16 | Exposure | T | 2.565615 | 0.855728 | -2.012929 |
| Adam A16 | Equal input | P | 1.595831 | 0.651599 | -1.254694 |
| Adam A16 | Equal input | 0.3V | 0.006612 | 0.080495 | -0.154999 |
| Adam A16 | Equal input | T | 1.602442 | 0.652922 | -1.257243 |

The fixed opposition criterion requires `cosine(T,U) <= -0.05` for both
validation weightings at both checkpoints. All four observed cosines are
positive, so the rule yields `no_persistent_training_objective_opposition`.
Policy/value attribution is secondary: the policy component supplies most of
the positive dot in these measurements; the weighted value component's dot is
also positive and its cosine is small. These local derivatives say nothing
about finite updates, later optimization behavior, playing strength, or
causal effects.

`manifest.json` freezes checkpoint, registration, source snapshot, split,
membership, reconstruction and execution-source identities, parameter layout,
objectives, denominators, tolerances, and the decision rule. `results.json`
contains metrics and additive source/bucket/parameter-group dot contributions.
The complete full-parameter vectors and component vectors are in
`gradient-vectors.npz`; per-vector hashes are recorded in `results.json`.
Recompute the report with:

```bash
python -m ml.alphazero_lite.verify_seed437_training_unseen_gradient_alignment
```

The independent verifier checks frozen input bindings, checkpoint immutability,
vector hashes, norms/dots/cosines/unit-distance changes, additive
decompositions, parameter accounting, population membership, and the fixed
classification.
