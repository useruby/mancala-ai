# Seed466 replay-input bundle

This is a portable, read-only input publication for future experiments. It is not a training run, model export, promotion, or model-strength claim. The historical seed464 attempt remains classified `invalid_attempt_dependency_closure_incomplete`; seed466 records a separate bundle commitment and does not upgrade that execution evidence.

## Contents

- `training.npz`: compact encoded states, legal masks, policy/value targets, policy coefficients, compact/source/raw-line provenance, source multiplicities, eligibility and effective value-mode codes, expanded source-major replay indices, frozen train/validation positions, and the exact first 8,192 positions of the frozen epoch-1 permutation (16 batches of 512).
- `evaluation.npz`: all 2,607 strict-unseen (`subset == unseen`, `active_stones > 32`) membership occurrences in source membership order. Each occurrence retains its own policy/value targets from the matching canonical source/raw line, input identity, compact row, and source/raw-line provenance. Identity groups do not collapse repeated occurrences or their possibly differing targets.
- `manifest.json`: input and decompressed input identities, array order/dtypes/shapes/hashes, source/helper closure, environment, recipe, counts, and policy masses.

All arrays are in NumPy NPZ files and load with `allow_pickle=False`. `ml.alphazero_lite.seed466_replay_bundle.read_bundle` needs only Python and NumPy plus the published bundle. It checks the fixed manifest identity, file hashes, exact array inventory, dtype, shape, and ordered C-byte hashes before returning arrays. It does not need the original checkout or source packet.

## Reproduction contract

Sources are ordered `fresh`, `generic_bootstrap`, `random_teacher`, `opening_disagreement`, `stability`, with registration weights `[1, 4, 1, 8, 4]`. The lane-A derivatives are authoritative. Their compressed packet bytes and decompressed JSONL bytes are checked against the seed464 packet manifest before decoding. Policy targets are the stored distributions; no additional sharpening or coefficient normalization is applied. Effective policy-mode metadata prefers `policy_target_actual_mode`, then `policy_target_mode`, and is validated against the frozen sharpened request (including the loader's exact-root mode rules). The per-source registered value mode is passed to the loader; any row-declared value mode must agree, while a missing declaration uses the registered mode. The exact coefficient from production semantics is 1.0 for every row at the default exact-root policy weight of 1.0. Every derivative row remains eligible.

The expanded vector is source-major, made by tiling each source's compact row indices by its registered multiplicity. It contains 149,448 occurrences. The already-frozen seed416 source-row split selects 134,502 training positions and 14,946 validation positions; validation is excluded from training counts and masses. Training contains 60,015 fresh occurrences. With coefficients retained at 1.0, the training policy mass is 134,502 and its fresh-policy mass is 60,015, matching the existing correction's denominators.

No split or permutation is regenerated. The bundle carries the frozen source-row positions and the exact first 8,192 positions of the registered epoch-1 permutation.

## Evaluation membership

The pooled evaluation membership has 2,607 occurrences and 1,242 distinct full input identities. Fresh has 859 occurrences and 851 identities. Evaluation targets come from each occurrence's canonical source JSONL raw line, not from compact-row substitution. Group IDs are metadata for equal-identity summaries only; occurrence arrays retain their original multiplicity and targets.

## Commands

From a checkout containing the verified seed464 packet, build once (publication refuses an existing output):

```bash
python -m ml.alphazero_lite.seed466_replay_bundle build
```

Check runtime consumption using only the bundle:

```bash
python -m ml.alphazero_lite.seed466_replay_bundle read --bundle docs/data/seed466-replay-input-bundle
```

The independent verifier reconstructs expected arrays from the pinned packet inputs and recipe, and therefore additionally requires that packet and declared project-source closure:

```bash
python -m ml.alphazero_lite.verify_seed466_replay_bundle --bundle docs/data/seed466-replay-input-bundle
```
