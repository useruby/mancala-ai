# Seed451 independent verification

The original seed451 registration and measured evidence were frozen before this
completion. The independent gradient reconstruction is a post-execution
verification; it did not rerun the publisher or alter the original evidence.

From the repository root, run:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed451_gradient_reconstruction --root .
```

The read-only verifier reconstructs the lane-A derivatives from hash-bound
compressed sources, invokes the production replay loader, verifies source and
split accounting, recomputes the full-parameter objectives and unseen gradients
at both registered checkpoints, then compares every vector and the derived
geometry with the archived evidence. Its JSON output is archived as
`independent-verification-report.json`; its hashes and verification-source
bindings are recorded separately in `supplemental-verification-receipt.json`.

**Verified classification:** `mixed_or_checkpoint_dependent_alignment`.
At the initializer, fresh equal-input unseen policy aligns with the fresh
training policy gradient and opposes the historical policy gradient, while its
joint-objective cosine is below the positive threshold. At T-final the
historical-policy cosine is positive and the joint-objective cosine is positive.
These are fixed-checkpoint first-order descriptions, not causal evidence about
replay weighting and not a strength or training claim.
