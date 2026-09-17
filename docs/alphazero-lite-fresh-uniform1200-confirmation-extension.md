# Fresh Uniform1200 Six-Seed Extension

Inherited PR #324 classification: `uniform1200_fresh_seed_heterogeneous`. This extension preserved its parent (`8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`) and all four fixed replay byte hashes at weights `4:1:8:4`; no replay source was regenerated.

## Results

New checkpoint SHA256 pairs (control/uniform1200) were seed50 `7a9a733d.../5da18b26...`, seed51 `e462e6bf.../e6f5ce2c...`, and seed52 `d2d7faac.../78084a42...`.

The six direct scores are `0.5000, 0.5000, 1.0000, 0.5000, 0.7500, 1.0000`. Effects are `0, 0, +0.5, 0, +0.25, +0.5`: mean `+0.2083`, median `+0.1250`, three positive, three neutral, and zero negative.

Exact W/D/L top1, policy mass, regret, and blunder rate are recorded per seed in `docs/data/alphazero-lite-fresh-uniform1200-confirmation-6seed.json`. Generic forensic top1, regret, and blunder rate improved in all six pairs. No true win-to-loss regression repeated in four of six seeds; seven canonical states had repeated improvements at that threshold.

For the new exact-solved rows, capture-available top1/regret/blunder changed control to uniform as follows: seed50 `.7917/.2917/.2083` to `.8333/.2083/.1667`; seed51 `.7917/.2500/.2083` to `.8333/.2083/.1667`; seed52 `.7500/.3750/.2500` to `.7917/.2500/.2083`. Sparse-endgame changed seed50 `.9583/.0417/.0417` to `1.0000/0/.0000`; seed51 and seed52 each `1.0000/0/.0000` to `.9583/.0417/.0417`.

Production classifications for seeds50–52 were `seat_artifact_only`, `high_search_breakthrough`, and `high_search_breakthrough`. The outcome-aligned shadow gate remains separate and passed six of six. The R61 panel was secondary only and never used for training or checkpoint selection.

New replay rows were control/uniform: seed50 `64314/70779`, seed51 `63956/70368`, seed52 `63941/71032`. New self-play search work was `14.81M/84.93M`, `14.74M/84.44M`, and `14.73M/85.24M` simulations respectively.

Hard classification: `uniform1200_six_seed_exact_gain_arena_uncertain`.

Exactly one next experiment: run a high-power paired arena on the six existing control/uniform checkpoints at unchanged evaluation search. No promotion occurred.
