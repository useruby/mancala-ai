# Seed556 exploratory arena: T512 versus frozen serving P

This screen compares #554 T512 update 108 policy/trunk against #552 frozen serving P (seed504 E4) policy/trunk. Both sides use the same frozen initializer as value network and value trunk through `ComposedArtifactEvaluator`.

The screen played all 1,024 indexed opening pairs (2,048 games) from the reused seed511 v4 suite. Arena settings remain 128 simulations per side, `c_puct=1.25`, one search worker per game, deterministic root policy, native exact-root threshold 16 with final-score-margin objective and the registered prior/index tie rule, exact-leaf solving disabled, and a 200-ply cap. Per-game search settings were unchanged while 18 opening pairs ran concurrently.

The suite is reused; canonical and production-float32 overlap was recomputed against T512 training, P policy training, and development membership. The unresolved seed461 continuation-overlap limitation remains. This is exploratory evidence and does not authorize deployment. The preregistered lower-bound gate did not pass; no separate 1,200-simulation screen is proposed.

See `results.md`, `verification.json`, `verification-receipt.json`, and `physical-copy-verification.json` for reconstructed analysis and verification records.
