# Seed542 C reproduction diagnosis

This diagnostic record does not change the seed504 or seed542 registrations,
historical checkpoints, or either preserved C attempt. D was not run.

## Production reference

The unmodified `run_seed504.run(..., do_train=True)` was run into
`production-reference/` with the authenticated initializer, original corpus,
and frozen permutations. Its E1 arrays do **not** reproduce historical seed504
E1 bitwise. Parameter names, shapes, and dtypes match; 4,486 of 73,159 elements
differ, with a maximum absolute difference of `5.960464477539063e-8`.

The historical seed504 manifest records Python 3.14.7 and PyTorch
`2.13.0+cu130`. The required restored seed542 runtime is Python 3.14.7 and
PyTorch `2.14.0+cu130`. The reference run used CPU affinity `[31]`, one Torch
intra-op thread and one inter-op thread; OMP, MKL, OpenBLAS, BLAS, and NumExpr
thread environment variables were unset. The historical seed504 manifest does
not record those CPU/backend settings. These are observed differences and
possible hypotheses, not an established cause of the checkpoint mismatch.

## Production versus corrected C path

The registered corrected-C run and production training path were replayed from
identical initializer arrays and fresh Adam state. Their E1 parameter arrays
are bitwise identical. Instrumentation compared all 108 updates. Every update
matched in ordered row identities, batch size, tensor bytes and metadata, legal
masks, forward outputs, policy/value/objective values, raw and clipped
gradients, Adam options/state, and parameters before and after the update.

The first diagnostic report labeled update 1 as divergent only because its
diagnostic implementation computed the production-style scalar norm with a
single aggregate Torch reduction. Production computes that telemetry as a
Python-float sum of per-parameter squared-gradient sums. At update 1 the
production telemetry is `3.031172305684415`, while the diagnostic's aggregate
reduction gave `3.031172275543213`. The actual `clip_grad_norm_` return is
`3.031172275543213` in both paths. The policy loss (`0.8450915217399597`), value
loss (`0.21561069786548615`), total objective (`0.9528968930244446`), clipped
gradients, Adam state, and resulting parameters match. The report's per-update
statuses show that this was the only unequal field across the 108 updates; it
was a diagnostic norm-calculation discrepancy, not a training-path divergence.

The raw first-update tensor evidence is in
`first-divergence/first-divergence.json`; the prospective diagnostic source and
settings binding is in `first-divergence/diagnostic-registration.json`.
Production reference source and runtime binding is in
`production-reference/diagnostic-registration.json`; its checkpoint comparison
is in `production-reference/production-reference-comparison.json`.

## Outcome

The confirmed finding is that both production and corrected C execution in the
restored runtime differ from historical seed504 E1, while matching each other
at E1 and at every update in the stepwise replay. The earliest available
historical comparison point is E1; seed504 did not publish intermediate
checkpoints or update traces, so an earlier divergence from history cannot be
located from the authenticated evidence. The reason for the historical/runtime
reproduction mismatch remains unresolved. No tolerance was changed and no
PyTorch version was switched. The fixed-update comparison is incomplete; no D
training or development evaluation was performed.
