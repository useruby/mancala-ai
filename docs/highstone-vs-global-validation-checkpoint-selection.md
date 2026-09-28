# High-Stone Versus Global Validation Checkpoint Selection

**Classification:** `highstone_checkpoint_selector_not_aligned`

The opt-in `best_high_stone_validation_v1` selector evaluates the normal supervised objective on the existing validation replay appearances with active pit stones above 32. It does not use arena results or parent-model outputs.

## Result

On the exact seed461 reproduction, global validation and high-stone validation both selected E2. The selected final checkpoint exactly equals E2 (`73914b801e61be7eada0366619fbc2defb23999fb163ef0da49134594ec44488`), so its frozen arena score is also the global-BV score: `0.5625`. The persisted opening-paired comparison is identically zero (10,000 resamples, seed 373) because both sides are that same checkpoint: `docs/data/highstone-vs-global-validation-seed461-paired.json`.

The independent seed455 reproduction exactly recovered global-BV E3 and selected E4 under the fixed high-stone selector. On the preregistered non-promotion suite, E3 scored `0.6211` versus the seed48 parent and E4 scored `0.4277`. The paired opening effect, high-stone minus global-BV, was `-0.1934` (10,000 resamples, seed 374, 95% CI `[-0.2344, -0.1523]`). This is a clear control regression.

Complete phase and shadow metrics are in `.tmp/highstone-checkpoint-selection/seed461/phase_metrics.json` and `.tmp/highstone-checkpoint-selection/seed455/phase_metrics.json`. The machine-readable comparison provenance is `docs/data/highstone-vs-global-validation-checkpoint-selection.json`.

No new self-play, canonical games, or promotions occurred.
