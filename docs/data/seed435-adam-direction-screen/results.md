# Seed435 Adam-direction screen

This is a retrospective validation screen. It does not establish game-strength
improvement and does not justify promotion.

The protocol was registered before either training arm. Both arms consumed the
same 16 consecutive 512-row batches from the frozen #416 first-epoch
permutation. See `registration.json`, `A-updates.json`, `B-updates.json`,
`initializer.npz`, and the final checkpoints for machine-readable evidence.

## Result

The fixed screen classifies **close optimizer-direction branch**. No longer-run
experiment is authorized.

| Measurement | Initializer | A: Adam | B: matched first moment |
|---|---:|---:|---:|
| Full training objective on frozen training partition | 1.021498 | 1.019594 | 1.139742 |
| Unseen >32 policy CE, exposure-weighted | 1.170699 | 1.187746 | 1.324001 |
| Unseen >32 policy CE, equal-input | 1.198160 | 1.221263 | 1.329484 |
| Unseen >32 value MSE, exposure-weighted | 0.476411 | 0.484873 | 0.488574 |
| Unseen >32 value MSE, equal-input | 0.616998 | 0.621018 | 0.632229 |

All fixed advance conditions fail. B regresses against both A and the
initializer on unseen policy CE; its full training objective also increases.
This result closes the optimizer-direction branch under the registered screen.

## Post-execution verification addendum

Read-only prediction archives and the `supplemental-receipt.json` were produced
after the experiment. The receipt's independent verifier reconstructs metrics
from the stored legal-masked logits/value predictions and authoritative source
rows, validates #427 membership independently, and binds all post-run sources
and evidence. The missing per-step vectors were recovered by deterministic
replay of the unchanged fixed protocol; both recovered final parameter arrays
match the original A16/B16 checkpoints bit-for-bit. Those telemetry files are
explicitly supplemental diagnostics, not independent experimental evidence.

The original `results.json` is preserved. Its recorded full-objective values
differ from the independent training-partition recomputation by +0.0002061248
(initializer), -0.0004664489 (A), and -0.0004695535 (B). The original metric
aggregated the full replay array, including frozen validation positions; the
supplemental recomputation is restricted to the training partition as
registered. All unseen-validation metrics reproduce within tolerance, and the
fixed close-branch decision is unchanged.

## Verification

Run `python -m ml.alphazero_lite.verify_seed435_publication` from the
repository root. The command is read-only. The #434 source verifier was also
run successfully against the frozen five-source reconstruction.
