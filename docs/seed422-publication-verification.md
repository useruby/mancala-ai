# Seed422 publication verification

The corrected #422–423 publication has a separate portable verifier. From the
repository root, run:

```bash
PYTHONPATH=. python ml/alphazero_lite/verify_seed422_publication.py
```

To verify a relocated checkout or evidence bundle, pass its repository root:

```bash
PYTHONPATH=/path/to/checkout python \
  /path/to/checkout/ml/alphazero_lite/verify_seed422_publication.py \
  --root /path/to/checkout
```

The verifier is read-only and requires NumPy for analysis, but does not require
Torch, checkpoints, exported runtime artifacts, the native executable, or the
tablebase. It rebuilds the published historical exclusion union from the
public #415–421 evidence, validates the registered execution snapshots and
the registration-to-outcome binding chain, and replays the published games.
Historical absolute paths remain provenance strings; known public evidence is
resolved by explicit repository-relative identity mappings.

The separate supplemental receipt binds this verifier and its exclusion
helper to the existing #422–423 correction receipt and final published
evidence identities.
