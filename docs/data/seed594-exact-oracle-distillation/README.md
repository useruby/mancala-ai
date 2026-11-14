# Seed 594 — exact-oracle policy-target preparation

**Status: frozen preparation; training not authorized or run.** Seed 594 tests
whether fine-tuning serving P's policy on uniform exact-optimal-action targets
improves raw-policy decisions over serving P and the descriptive archived #571
PUCT-target treatment. The treatment differs from #571 only in its target
distribution. This is exploratory public qualification; #570 has prior
experimental use and cannot support an independent generalization claim.

## Data and target contract

The 256 ordered roots are the authenticated #557 cohort. Each root joins exactly
once by canonical state hash to a complete #557 `oracle-records.jsonl` row.
`legal_moves` and all action-margin keys must equal the current player's legal
move indices `0..5`; these are player-relative policy slots, not absolute pit
indices. The target assigns equal mass to every legal action whose exact
final-score margin attains the legal maximum, and zero elsewhere. The checked-in
`exact-targets.jsonl` is the deterministic materialization; its row order follows
the frozen #557 cohort order. No search is performed.

Qualification binding is identity-only: the 423-root #570 membership is used
solely to verify disjoint opening, canonical-state, and production-float32
identities. Qualification oracle values, predictions, and results are not read.

## Matched training contract

The initialization is the exact authenticated P-E4 checkpoint. The trainable
parameters are `input_layer.*`, `residual_layers.*`, `policy_hidden_layer.*`,
and `policy_head.*`; the value hidden/head remain frozen. Architecture,
`kalah_v3` encoding, legal mask, masked soft-target cross-entropy, and all other
training semantics match #571: Adam, seed 571, learning rate `1e-5`, betas
`(0.9, 0.999)`, epsilon `1e-8`, zero weight decay, batch size 32, one seeded
permutation of 256 roots cycled in contiguous batches, 512 updates, gradient
clip 1.0. Only update 512 is eligible.

The exact ordered batch permutation is recorded in `training-configuration.json`.
Runtime is CPython 3.12.8, NumPy 2.2.6, CPU PyTorch 2.7.1, one thread,
deterministic algorithms, `PYTHONHASHSEED=571`, and OMP/MKL thread counts 1.
Preflight rejects any source/model/configuration/runtime mismatch, target or
cohort mismatch, nonfinite training value, changed frozen parameter, incomplete
update trace, or pre-existing output directory. Any failure stops the run;
there is no retry, alternate checkpoint selection, or parameter tuning.

## Qualification gate (future, separately frozen)

Stage 1 compares the frozen candidate with serving P on all 423 #570 roots using
raw policy only. Report mean exact final-margin regret, exact-optimal top-action
selection rate, WDL downgrade rate (selected-action W/D/L category below the
best exact-action category), exact-optimal prior mass, and paired opening-cluster
bootstrap 95% confidence intervals. Include a descriptive comparison with the
archived #571 checkpoint; it is not eligible for post-hoc selection. Resample the 423 opening
clusters with replacement 20,000 times using RNG seed 594 and NumPy linear
quantiles. Advance only if candidate-minus-P raw regret is at most `-0.10`
seeds, its CI upper endpoint is strictly below zero, and WDL downgrade rate
does not increase. Failure stops the experiment.

Only if Stage 1 passes may a separately frozen Stage 2 use the existing
#570/#577 paired 128-simulation PUCT contract (shared initializer value,
registered search seeds, native root-16 routing, paired roots). The original
post-search gate is mean regret improvement at least `0.10` seeds, paired 95%
CI upper endpoint below zero, and no increase in WDL downgrade rate. No sealed
suite access, arena games, model promotion, serving change, or reopening #483
is permitted. A positive result requires separately registered fresh
confirmation before any strength claim. Seed 588's failed decision is unchanged.

## Commands

Read-only preparation verification (pinned runtime and required environment):

```bash
PYTHONHASHSEED=571 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python -m ml.alphazero_lite.seed594_exact_oracle_distillation --verify-only
python -m pytest ml/alphazero_lite/test_seed594_exact_oracle_distillation.py
```

The future training entry point requires an additional separately reviewed
execution freeze with status `authorized-for-pretraining`. This preparation
does not authorize invoking `--run`.
