# Seed 604 evaluation preparation

This directory prepares the registered Seed 604 comparison. It does not
authorize evaluation games. The only game entry point is
`python -m ml.alphazero_lite.seed604_evaluation --run`; it refuses to start
without a separate owner authorization bound to the published execution-freeze
bytes.

## Inputs and composition

- B512: immutable Seed 599 Arm B update 512.
- B2048: verified Seed 604 update 2048, bound to its training receipt.
- Both candidate policies use their own exported policy trunk. The arena
  receives serving P-E4 as each candidate's value artifact and as the complete
  opponent. The P-E4 value trunk is therefore shared and unchanged.
- Runtime candidate bundles preserve the original checkpoint bytes exactly;
  their `model.npz` digests equal the authenticated source checkpoint digests.
- Every candidate bundle is checked against P-E4 value evaluation on synthetic
  legal positions, including states after extra turns and states where either
  player is to move. These checks do not use registered openings or produce
  experimental results.

## Registered design

The frozen v5 cohort is consumed in manifest order. The assignment ledger has
exactly one B512 and one B2048 game for each candidate seat per opening, for
4,096 unique game IDs. Seeds are derived from the opening ID, candidate, and
seat. Pair scheduling runs the two seat-swapped games together with one arena
worker per game and 18 pair-level workers.

Search settings and both statistical gates are copied from the registered
Seed 604 and Seed 599 contracts and independently checked before freezing.
Inference is by opening cluster; each resample applies the same sampled
opening indices to both arms and retains both seat outcomes. The analysis uses
20,000 resamples, seed 604, linear quantiles, and the preregistered strict
thresholds.

## Exact-root materialization

The repository-tracked tablebase pointer remains an LFS pointer in Git history.
For preparation, Git LFS 3.7.1 materialized its referenced object at
`model-artifact/runtime/kalah_v1_21.kvtb`. The complete object was verified at
709,635,072 bytes with SHA-256
`f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7`.
The original tracked LFS pointer is preserved (pointer-file SHA-256
`a82ca3d03265950a23aae8a1088fa86a93db9acea7bca68ea33303292eb0202c`); readiness
evidence binds both the pointer OID and the materialized binary digest.
The native probe SHA-256 is
`d898ed68e5d8a5aade35c2f148efd758478ef1c27bed934ba601b9aa353b1e48`.
Readiness evidence binds these identities and the materialization command.

## Verification and recovery

`verify_seed604_evaluation --preflight` is read-only. It independently
reconstructs the assignment ledger, checks checkpoint and runtime artifact
identities, verifies result trajectories and report configuration, and rejects
missing, duplicate, conflicting, or inconsistent records. Results are
create-only. An interrupted attempt is retained under its game ID; a later
attempt gets a new attempt directory. If an earlier attempt already emitted a
game or report but lacks its final result record, execution stops for
reconciliation rather than silently rerunning or choosing an outcome.

Completed result analysis is a separate create-only action and requires all
4,096 independently verified games. No Seed 604 evaluation authorization is
included here.

The existing training-freeze verifier is scoped to the pre-authorization
training stage and refuses the completed run's authorization artifact. The
published training freeze contract is independently validated in that scope;
the completed B2048 training result is authenticated with its independent
training-result verifier.
