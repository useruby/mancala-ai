# Seed438 erratum: correction to the seed437 gradient diagnostic

This correction follows observation of the original seed437 results. The seed437
manifest, source, results, and gradient archive are historical evidence and are
retained byte-for-byte. The seed437 result remains labeled as originally
published; this erratum supersedes its value-gradient interpretation only.

## Defects and corrected calculation

Seed437 supplied row weights already containing `0.3` for the value objective,
then multiplied the value loss by `0.3` again in `flat_grad`. Its published
`0.3V` vectors therefore represented `0.09V`. In addition, its verifier only
reconciled contribution totals instead of independently recomputing each
source, bucket, and parameter-group dot. Finally, its relocation test used
symlinks into the originating checkout and invoked verification from that
checkout's working directory.

Seed438 uses seed437's bound initializer, Adam A16 checkpoint, sources, split,
membership, masks, denominators, and populations. It computes unweighted `V`
explicitly, applies `0.3` once in the weighted component, and forms `T=P+0.3V`.
Unknown objective names are rejected. The frozen protocol and corrected source
identities precede corrected gradient execution. `protocol.json` records the
original evidence hashes and states explicitly that the correction follows
observation of those results.

## Results

The complete corrected vectors, source and bucket component vectors,
parameter-group indices, contribution dots, metrics, and classifications are
bound in `corrected-results.json` and `corrected-gradient-vectors.npz`.
Classification is `no_persistent_training_objective_opposition` at both
checkpoints, the same classification as the historical seed437 results.
The fixed decision is persistent opposition only when cosine(T,U) is at most
`-0.05` under both validation weightings at both checkpoints.

| Checkpoint | Metric | P cosine (exposure / equal-input) | 0.3V cosine (exposure / equal-input) | T cosine (exposure / equal-input) |
|---|---|---:|---:|---:|
| Initializer | Corrected | 0.851289 / 0.597241 | 0.049916 / 0.065710 | 0.840364 / 0.594310 |
| Adam A16 | Corrected | 0.853922 / 0.651599 | 0.107617 / 0.080495 | 0.856242 / 0.653189 |
| Initializer | Historical seed437 | 0.851289 / 0.597241 | 0.049916 / 0.065710 | 0.850110 / 0.597860 |
| Adam A16 | Historical seed437 | 0.853922 / 0.651599 | 0.107617 / 0.080495 | 0.855728 / 0.652922 |

For `P`, `U_exposure`, and `U_equal_input`, the corrected vectors are exactly
equal to their original counterparts (maximum absolute discrepancy `0`). For
corrected `0.3V` versus `(10/3)` times the historical mislabeled `0.3V`, the
maximum absolute / L2 discrepancies are `1.2062e-9 / 1.0059e-8` at initializer
and `2.4484e-9 / 9.5143e-9` at Adam A16. Declared comparison tolerances are
absolute `2e-5`, relative `2e-5`; all comparisons pass.

The corrected verifier independently recomputes every source, bucket, and
parameter-group contribution from its bound component vector or parameter
slice, in addition to vector sums, parameter coverage, geometry, and the fixed
classification rule. No optimizer steps, training, searches, games, target
changes, or promotion were performed. This is retrospective observational
diagnostic evidence only, not causal or playing-strength evidence.

## Verification and relocation

The independent verifier is `ml.alphazero_lite.verify_seed438_seed437_gradient_correction`.
The receipt binds the original evidence, frozen inputs, corrected code, vectors,
and results. Seed437 evidence hashes and both checkpoint hashes are checked
before and after correction execution.
