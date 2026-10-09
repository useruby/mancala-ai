# Seed448: seed447 publication-verification correction

Seed448 supplies a separate, read-only verification entry point:

```bash
python -m ml.alphazero_lite.verify_seed448_seed447_publication --root .
```

It validates the frozen seed447 registration and original receipt identities, all registered source snapshots, the amendment source and frozen-verifier identities named by the existing amendment receipt, the post-execution publication amendment and supplemental hashes, and source/input identities embedded in the seed442 registration. It then calls `verify(root)` on the existing verifier amendment directly. It does not call either receipt publisher and prints its JSON report only.

The verifier retains seed447's full replay of gradients, Adam moments and proposals, all eight trial scales, feasibility/selection, accepted trajectories, control reproduction, targets, predictions, and endpoint metrics. Seed448 independently recomputes the fixed gates and classification precedence.

**Result:** `close_joint_output_cap_branch` (unchanged). The keys `T_minus_C_*_policy` in the frozen evidence actually compare T with seed442 ordinary-Adam A; this historical naming mismatch is documented here and preserved without renaming or changing those fields.

The seed447 registration predates seed448's correction, and seed442 dependency source identities are only those recorded in the embedded seed442 registration. Seed448 checks those historical hashes against current files but does not claim an independent trusted timestamp or cryptographic signature for either prior registration. The supplemental seed448 receipt is generated only after the source, test, report, and verification run are finalized.
