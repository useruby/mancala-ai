# Seed441: seed440 publication completion

This is a **post-execution publication completion** of seed440. It supplements
and does not replace any seed440 file. The reproduced objects are only the
already-defined A/B reconstructed paths, their 16 midpoints, and per-step
weighting-specific gradients; there was no optimizer update, training, audit
replay, new trajectory, search, game, target change, or promotion.

The source protocol and hashes are in `protocol.json`. It binds seed440's
original protocol, evidence bindings, ledger, and path-hash archive, plus the
initializer actually loaded by seed440, both delta archives and final
checkpoints, exact membership and replay source/split inputs, the parameter
layout registration, and reconstruction/model/loss/mask dependencies. The
historical seed435 initializer file is independently retained and hash-bound;
seed440's actual computation loads the seed429 initializer checkpoint.

The evidence archives contain policy logits and value predictions at all 17
states and 16 midpoints per arm, both exposure-weighted and equal-input
gradients at every pre-step state, selected compact-row and input-identity
records, and the exact parameter layout. The portable verifier reconstructs
the paths, compares predictions and gradients using seed440's registered
absolute/relative tolerance (3e-5), checks 16-step and two-weighting coverage,
reconciles the complete historical numeric ledger under those tolerances, and
checks bit-identical endpoint parameters and the unchanged historical hashes.

Reproduction matched all 779 numeric fields in seed440's published ledger
exactly (maximum absolute difference 0.0). Both final reconstructed parameter
sets are bit-identical to A16/B16. No historical evidence hash changed.
Seed440's signed interpretation is retained: `R = D - S`, including
nonlinear behavior and numerical effects; it is not isolated curvature.

The fixed primary arm A classification is
**finite_step_residual_dominant**. Under both weightings A has positive total
CE change and `R >= 0.75*D`; this does not identify causal mechanism, generalize
beyond the fixed validation population, establish playing strength, or
authorize intervention. B remains secondary.

Portable execution was tested from an unrelated working directory with source
code in a source-only checkout and evidence copied physically to a different
`--root`, with no symlinks. The verifier did not modify copied evidence.
