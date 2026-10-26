# Seed504 policy-only training ablation — strength screen in progress

## Outcome classification

**Development/training: valid. Strength screen: incomplete. Advancement: no.**
The policy-only arm completed under a prospectively frozen protocol, and the
historical control reproduced all four saved parameter states exactly. The
1,024-opening arena is running on the host, outside CI and the API container, in
a CPUQuota=25%, CPUWeight=1, nice=19 scope pinned to CPU 31. The API container's
separate CPU quota and cpuset were read from its host cgroup before launch. The
read-only verifier currently validates completed pairs incrementally; no final
strength result is claimed until all 2,048 games pass replay and binding checks.
Incomplete or capped games are not ordinary draws.

Seed511's authenticated 1,024-opening suite is the intended reused suite. Its
known seed461 historical-overlap limitation remains in force; this is not fresh
confirmation. Passing that screen would authorize only proposing a separate
1,200-simulation evaluation, not deployment or promotion.

## Frozen protocol and control

`registration.json` and `execution-freeze.json` were created before arm T
training. They bind the initializer, eight seed500 corpus shards, seed504
manifest, filtered split, epoch permutations, all historical checkpoints,
execution sources, and actual Python/dependency environment. Training used the
production loader and trainer, residual_v3 (96, 3), 27 inputs, four epochs,
batch size 512, Adam at 0.0001 with (0.9, 0.999), epsilon 1e-8, no weight decay
or scheduler, norm clipping 1, seed504 ordering and exact seed504 permutations.
T used the unchanged production policy CE/targets/weights/legality and
`value_loss_weight=0`; value-head parameters were marked non-trainable and
asserted unchanged after training. The optimizer and policy minibatch
normalization were otherwise unchanged.

The repository's Ruff formatter changed the newly added training runner during
the commit hook, after its prospective freeze and training. Its exact frozen
bytes are retained at the publication root in
`execution-source-snapshots/run_seed504_policy_only.py.txt`. The arena
registration similarly binds the screen sources before launch; their exact
registered bytes are retained under `screen/execution-source-snapshots/` when
later read-only/cap validation hardening changes the working copies. Preflight
authenticates those snapshots against the registered digests.

| Arm | Objective | Policy trunk/head | Value head |
|---|---|---|---|
| C | Policy CE + 0.5 × Huber(value, delta 1) | trained | trained |
| T | Policy CE only | trained | frozen at initializer |

C was independently rerun before T. Parameter arrays in all four epoch
checkpoints were bitwise equal to the authenticated seed504 archives. Archive
hashes also matched: E1 `6187b0e702a2118579c82fc0950398babe255e95f1f64a4f695d52b3e56144c3`,
E2 `d1ef62ad445ad33a532a9a5efa8aadefe5657235af1706fdba4b1c1c8579d796`,
E3 `e049d2a7e6702b68cf4bf2d230f5966f86e4dfc8e4ed928a534b61f7d7804842`,
E4 `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
The reproduced C checkpoint archives are included under
`control-reproduction-checkpoints/`; the verifier reloads all 22 parameter
arrays per epoch and compares them bitwise with the authenticated historical
archives.
The available deterministic runtime was Python 3.14.7, NumPy 2.5.3, and Torch
2.14.0+cu130 (CPU execution), rather than the seed504 manifest's NumPy 2.5.2 and
Torch 2.13.0+cu130. Exact parameter-array reproduction was required and passed;
the environment difference is reported, not concealed.

## Training objective

Per epoch, 13,724 examples and 27 updates were processed. The logged value loss
is the diagnostic unweighted Huber value; its coefficient in T's training
objective is zero.

| Epoch | Policy CE | Diagnostic value Huber | T objective | Updates |
|---|---:|---:|---:|---:|
| E1 | 0.72310726 | 0.21232570 | 0.72310726 | 27 |
| E2 | 0.68067016 | 0.21714321 | 0.68067016 | 27 |
| E3 | 0.66208611 | 0.22014768 | 0.66208611 | 27 |
| E4 | 0.64943353 | 0.22210265 | 0.64943353 | 27 |

## Development diagnostics

All three candidates were evaluated on the unchanged filtered seed504
development rows. The complete per-row identities, targets, masked predictions,
counts, metrics, and T−C differences are in `development-results.json`.
Equal-input weighting averages repeated observations within each exact
production float32 input (retaining their different targets), then weights
identities equally. Lower CE is better.

| Cohort | Rows / exact inputs | Initializer exposure / equal-input | C exposure / equal-input | T exposure / equal-input | T−C exposure / equal-input |
|---|---:|---:|---:|---:|---:|
| Pooled | 2,671 / 2,615 | 0.753460 / 0.755281 | 0.655992 / 0.656697 | 0.654496 / 0.655159 | −0.001496 / −0.001538 |
| >32 stones | 548 / 529 | 0.948154 / 0.954791 | 0.901285 / 0.906067 | 0.897226 / 0.901786 | −0.004059 / −0.004281 |
| 17–32 stones | 992 / 968 | 0.822509 / 0.825354 | 0.712685 / 0.717427 | 0.711157 / 0.716108 | −0.001528 / −0.001319 |
| ≤16 stones | 1,131 / 1,118 | 0.598562 / 0.600207 | 0.487415 / 0.486120 | 0.487189 / 0.485691 | −0.000226 / −0.000430 |

Development CE is descriptive and does not select E4 or gate an arena. The
observed improvements do not establish stronger playing strength.

## Composition and arena status

The planned opponent is the existing P composition (seed504 E4 policy plus
initializer value). The challenger is T E4 policy plus the identical initializer
value. Each component retains its own trunk; no tensors are spliced. The
prospective registration binds the copied seed511 suite, P/T/initializer runtime
artifacts and sidecars, arena/native/tablebase source identities, command
settings, and bootstrap protocol. Complete opening pairs are atomically
committed to the ledger; complete raw pairs can be recovered after interruption,
while partial/capped pairs remain uncommitted. The fixed seed511 suite and its
historical reuse limitation are bound at
`docs/data/seed511-fixed-composition-screen-v4/`.

The registered decision would use exactly two seat-swapped games per opening,
128 simulations per side, score `(wins + 0.5 × draws) / 2,048`, and a paired
opening bootstrap with 20,000 resamples, seed 539, linear quantiles. Advancement
requires all games valid and the one-sided 95% lower bound strictly above 0.50.
Until then the strength outcome remains **incomplete**, not negative.

## Verification

Run from the repository root with the Python environment used for the
experiment:

```bash
.tmp/venv/bin/python -m ml.alphazero_lite.verify_seed504_policy_only
.tmp/venv/bin/python -m ml.alphazero_lite.verify_seed504_policy_only_screen
.tmp/venv/bin/python -m pytest ml/alphazero_lite/test_seed504_policy_only.py \
  ml/alphazero_lite/test_run_seed504.py \
  ml/alphazero_lite/test_seed504_policy_only_screen.py -q
```

The verifier is read-only and recomputes development predictions and metrics,
validates all available source/input/checkpoint bindings, and verifies that the
T value-head arrays equal the initializer. The screen verifier is read-only,
replays raw trajectories, checks pair receipts and runtime settings, recomputes
bootstrap statistics when complete, and rejects capped games even if their
ordinary receipt hashes were rebound.
