# Shadow-Target Policy Distillation

**Classification:** `invariant_failure`

**Recommended follow-up:** Repair the preregistered auxiliary-gradient scale contract before considering another distillation run.

## Stop

The deterministic frozen target cache and selector were built before any training. The first `shadow_sensitive` batch produced a primary gradient norm of `0.00797855947` and auxiliary CE gradient norm of `0.151517481`, or `18.990581x`. The fixed protocol declares any ratio above `10x` invalid, so training stopped without retuning the weight.

No checkpoints, frozen-root diagnostics, canonical arenas, or P0 gates were run after this stop. This is not evidence for or against policy distillation.

## Frozen Inputs

| Input | SHA-256 |
| --- | --- |
| P1 checkpoint | `e4a9c95302c17b63405107d26b4cd4f90755d2b21980846ac6e9f0baabf7efe9` |
| A16 step 16 state | `0b322e0996a4902cb8737ff32a429bfd45803ee4a55d713e2960ea9e9faf5068` |
| Fresh-P1 replay | `892827d8ee67a66e6324a2aaec7011df1a21625fc3f6bcd87cab39ce655d2a88` |
| Target cache | `526a1e06fda3f96575c886b3d52a1e2a06a75a572c51203b0b7e61806090cecb` |
| Selector manifest | `2bf821c1debc78538961467657ff3c5757b50ccbcd37094dcf932edda5c4b4b3` |

The cache contains ordinary A16, exact PR #229 A16-with-P1-shadow, and ordinary P1 1,200-simulation root visits, moves, root Qs, and normalized targets. Searches used `c_puct=1.25`, zero FPU, no normalization, no root noise, and deterministic root selection.

## Population

The 4,096-root PR #221/#230 manifest was checked against the fresh-P1 replay. The eligible population contained 4,015 roots after 40 exact frozen amplified roots, 40 washed controls, and one canonical opening root were excluded. The two PR #220 held-out hashes were excluded by rule but were not members of this manifest. The frozen sensitive and fixed-seed random selectors each contain 1,003 roots.

The complete machine-readable stop record is `docs/data/alphazero-lite-fresh-p1-shadow-target-distillation-summary.json`.
