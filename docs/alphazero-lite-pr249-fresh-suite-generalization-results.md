# PR #249 Fresh-Suite Generalization Evaluation

**Classification:** `canonical_strength_signal_generalizes`

This evaluation consumed three sealed 128-opening suites and did not train, generate self-play, alter targets, or promote a model.

## Frozen Inputs

| Suite | Preregistered seed | SHA-256 | Status |
| --- | ---: | --- | --- |
| A | 1042 | `c8277e659c7a4e137140d83c187781f40e6b25c4b1dff5ec4da3f2e09fdcc6ab` | CONSUMED |
| B | 2042 | `1f4c17eb7df21af75bc29c3274b3951899ae5fb2522f762d5270d58ddf93b37e` | CONSUMED |
| C | 3042 | `b56783a2a2bbf63168cfb642f2b878badc80ace0279a9bbb7778757a4e4ba90d` | CONSUMED |

Suites used the canonical generator's legal-prefix enumeration, state deduplication, stratification, and selection procedure. The selection pool excluded previously sealed suites and exact encoded training-replay states, as required for disjointness. There were zero duplicates within suites, across suites, against canonical SHA `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`, or against either training replay.

The full candidate checkpoint, adapter, optimizer, replay, and batch-plan manifest is in `docs/data/alphazero-lite-pr249-fresh-suite-generalization-summary.json`.

## Canonical Invariant

The old suite was used only as an invariant. All required effects reproduced exactly: seed-45 fixed-768 `+0.041016`, seed-45 fixed-1024 `-0.019531`, seed-46 fresh-1024 `+0.041016`, and seed-46 fresh-768 `-0.019531`.

## Fresh Results

All games used ordinary deterministic PUCT at `1200:1200`, `c_puct=1.25`, seat swapping, arena seed 42, and one matched P1-vs-P1 control per suite.

| Candidate | A effect / W-D-L | B effect / W-D-L | C effect / W-D-L |
| --- | --- | --- | --- |
| seed45 fixed768 | +.003906 / 236-44-232 | +.005859 / 240-38-234 | +.011719 / 240-44-228 |
| seed45 fixed1024 | -.023438 / 208-72-232 | -.031250 / 202-76-234 | -.023438 / 204-80-228 |
| seed46 fresh1024 | +.003906 / 236-44-232 | +.005859 / 240-38-234 | +.011719 / 240-44-228 |
| seed46 fresh768 | -.023438 / 208-72-232 | -.031250 / 202-76-234 | -.023438 / 204-80-228 |

| Primary contrast | A | B | C | Pooled 384 CI95 | Hierarchical CI95 | Positive suites |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| seed45 fixed768 - fixed1024 | +.027344 | +.037109 | +.035156 | +.033203 [+.024740, +.041667] | [+.017578, +.050781] | 3/3 |
| seed46 fresh1024 - fresh768 | +.027344 | +.037109 | +.035156 | +.033203 [+.024740, +.041667] | [+.017578, +.050781] | 3/3 |

Both preregistered generalization rules pass: the pooled lower 95% bound is positive and each contrast is positive in at least two suites.

## Diagnostics

The historical positive pair and the historical negative pair were outcome-identical but trajectory-different on the old canonical suite; they were not completely game-record identical.

Fresh first-divergence telemetry was concentrated in challenger seat 1. For seed45, move divergence was 7.81%, 10.55%, and 9.38% for A/B/C; first divergence was predominantly ply 7, with additional ply-20 changes. For seed46, it was 5.47%, 7.42%, and 7.03%, all first diverging at ply 7. Final outcomes changed in 5.47%, 7.42%, and 7.03% of games respectively.

## Next Experiment

Compare the two successful adapter parameter deltas and full-batch training gradients to identify a common update direction independent of target budget.
