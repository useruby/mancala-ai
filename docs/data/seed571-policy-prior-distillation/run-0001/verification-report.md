# Seed571 policy-prior distillation — run 0001 verification

## Authorization and runtime

- Executed from the isolated results worktree at merged #573 revision `34e58dce7b0c57831850962f7bc1e6b5ab16badb` (worktree merge-result commit `b6bf1b390b785bcb9e0843d753a936492b4dcd19`; identical tracked tree).
- `--verify-only` passed before execution. The separate execution freeze was also checked and reported `authorized-for-pretraining` with all bindings verified.
- Pinned interpreter: CPython 3.12.8, NumPy 2.2.6, PyTorch 2.7.1+cpu; CPU, deterministic algorithms, and one thread as frozen.
- Run command used the frozen `PYTHONHASHSEED=571`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` settings.

## Run evidence

- 256 ordered serving-P teacher targets, searched at 128 PUCT simulations per root.
- All 512 Adam updates completed in the frozen contiguous batch order; update 512 is the sole checkpoint candidate.
- `teacher-visits.jsonl`, `targets.sha256`, `training-trace.jsonl`, `seed571-policy-distillation-update-512.npz`, and `checkpoint-identity.json` are retained in this directory.
- Target SHA-256: `56d2d1e5f56f786a92304d6a6ed741c1f885a3c61d474ef03b2950797561b15a`.
- Checkpoint SHA-256: `4e16fed2550a64a9413b588f84e8df1eadde7e0b5889b4932cb3783e3cb648e4`.
- The identity record binds 22 named tensors with shapes and dtypes and records the authenticated parent policy and initializer-value checkpoint hashes. The checkpoint hash and every tensor's name, shape, dtype, and finiteness were independently checked.

## Independent audit and reproduction

Independently verified root order against the frozen 256-root cohort; recalculated each canonical state hash and search seed; checked each legal-action mask, six-action visit vector, 128-visit sum, zero illegal-action visits, and normalized target. Checked each of the 512 trace update numbers and batch assignments against the fixed seeded permutation, with finite losses and gradient norms.

Re-ran the deterministic pipeline to the separate scratch directory `.tmp/seed571-reproduction-0001`. The teacher records, target digest, full update trace, checkpoint, and checkpoint identity record were byte-identical to run 0001.

## Scope

This is the verified pretraining checkpoint only. The 423-root qualification cohort was not consumed; its qualification requires a separate evaluation freeze. No arena games, sealed-suite access, or serving changes were performed. Issue #483 remains closed.
