# Seed 604 B2048 training readiness

Status: **implementation-only; not execution-ready**.

The guarded executor and read-only result verifier are prepared, but the
execution freeze is intentionally not published. Production authorization,
optimizer updates, checkpoint creation, and evaluation games remain prohibited
until execution-readiness review and separate project-owner authorization.

Freeze authentication is fail-closed and checks exact values for the registered
Seed 604 and Seed 599 inputs, runtime, source set and digests, v5 cohort byte
identities, fixed output contract, update count, and B512 gate contract. The
runtime is compared with the executor's pinned contract, not accepted from the
freeze alone. At the B512 boundary, all named model tensors must match exactly;
all trainable parameters must have finite, complete Adam state at step 512.
The report fingerprints that state and records the same optimizer's state at
step 513.

## Exact runtime requirement

The exact-reproduction runtime is CPython 3.14.7, NumPy 2.5.2, and PyTorch
2.13.0+cu130 executing on CPU, with one PyTorch thread, deterministic
algorithms enabled, and `PYTHONHASHSEED=599`, `OMP_NUM_THREADS=1`, and
`MKL_NUM_THREADS=1`. Version substitution is not accepted for the bitwise B512
gate. This checkout has CPython 3.14.7 and NumPy 2.5.3, but PyTorch is absent;
the required runtime cannot currently be certified. See
`ml/alphazero_lite/requirements-seed604-training.txt` for pinned Python package
versions. The CUDA wheel index/platform availability must be provisioned and
authenticated separately.

## Authenticated read-only inputs

The Seed 604 preflight authenticates the P-E4 parent, Seed 599 B512 checkpoint,
Seed 599 B membership, B512 schedule, and deterministic 2,048-update schedule.
The merged PR #607 opening-manifest SHA-256 is
`445052b818b36f84953f73f280b20f7421c441fac6a74a0e2bb309cf907fb410`; its
independent verifier passes and reports the registered bounded exclusion
unions. No opening cohort or frozen Seed 599 target is regenerated.

The exact update-512 parameter-tensor reproduction has **not** been run or
claimed. A future authorized run must stop at update 512, compare all tensors
exactly, then continue with the live Adam state only if every comparison passes.
