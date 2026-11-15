# Seed 599 — Replay Coverage Scaling (preparation)

**Status:** data and contracts prepared; training and evaluation have not been run.

## Authenticated source inventory

The read-only published verifiers replayed #580 and #585 successfully before
extraction. #580 uses freeze digest
`8da846678e89568e17e7bd81270287ef373e1f8acdfcbc892317d08385ad4c36` and
evidence-manifest digest
`7c8d32ced1fd56ae71f261b9dcbc482b5248a62ce825409513bb9ad78aed2c0c`. #585 uses
freeze digest
`3f2ccca4fae605f826b4fb00e0adb6f88c8c87ad8ca1373152aea051133f75b5` and
evidence digest
`e1cf047b1e8587d673ae711c11e29eb9125240391d945a581a27ce47ce37cd75`.

Only completed games' accepted designated-P rows are extracted. Every policy
target is reconstructed with `varied_opponent_pilot.reconstruct_policy_target`
against the state-derived legal moves. Conflicting targets are counted by
canonical state identity; all identities in the authenticated union agree.

| Inventory measure | Count |
|---|---:|
| Eligible rows | 7,794 |
| Unique canonical states | 6,640 |
| Unique production float32 states | 6,640 |
| Rows: >32 active stones | 1,606 |
| Rows: 17–32 active stones | 2,803 |
| Rows: ≤16 active stones | 3,385 |
| Unique states: >32 active stones | 1,189 |
| Unique states: 17–32 active stones | 2,430 |
| Unique states: ≤16 active stones | 3,021 |
| Distinct source games | 504 |
| Distinct opening groups | 188 |
| Conflicting targets for identical canonical states | 0 |

The phase-balanced 2,048-state dataset is feasible. The deterministic subset
selects 512/1,024/512 states, and Control A is its nested 256-state
64/128/64 subset. Full row provenance, target evidence, identities, per-file
artifact hashes, and verifier reports are recorded in `preparation.json` and
the membership manifests.

## Frozen preparation contracts

`training-A.json` and `training-B.json` bind the same P-E4 parent checkpoint,
residual-v3 architecture, trainable/frozen groups, masked policy-only target,
Adam settings, 512 updates, batch size 32, gradient clipping, and seed 599.
Only the explicit training memberships differ. Both configurations remain
`frozen-pretraining-not-authorized`. The exact 512 minibatch index lists are
frozen in `batch-schedule-A.json` and `batch-schedule-B.json` and hash-bound by
their respective configurations.

`heldout-membership.json` reserves a deterministic fifth of source opening
groups before training selection. Training and this generalization benchmark are
provenance-disjoint, including states repeated across group boundaries. The
registered generalization statistic is masked policy cross-entropy on this
held-out cohort, paired by opening group.

`evaluation-contract.json` and `evaluation-openings.json` register 1,024 unique
legal random 12-ply openings from seed 599, paired seat swaps for A-vs-P-E4 and
B-vs-P-E4, and unchanged common serving P-E4 values. The frozen search uses 128
simulations per side, `c_puct=1.25`, deterministic root policy, a 200-ply cap,
and native exact-root threshold 16 with the serving margin objective and
prior/index tie rule. Model, sidecar, native-probe, and tablebase hashes are
recorded. The contract fixes 20,000 paired opening-cluster bootstrap resamples
with seed 599, a B-minus-A strength non-inferiority margin of 0.02, and lower
held-out policy cross-entropy as the generalization gate.
Opening-start identity overlap with training is rejected during generation;
trajectory-level overlap is necessarily pending evaluation and must be
reported from observed game traces.

## Read-only preflight and tests

From the repository root:

```bash
python -m ml.alphazero_lite.seed599_replay_coverage preflight
python -m unittest ml.alphazero_lite.test_seed599_replay_coverage
```

The preflight reauthenticates both source publications, reconstructs all
targets and memberships in memory, compares the registered artifacts, and
does not train, evaluate, or write files.
