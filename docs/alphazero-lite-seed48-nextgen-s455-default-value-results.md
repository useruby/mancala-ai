# Seed48 Generation N+1, Seed455 Default Values

`seed48-nextgen-s455-default-value` trained one fresh challenger from the pinned
seed48 incumbent (`935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`).
Fresh self-play used 1,600 uniform-1200 games, six workers, seeds 454/455/456,
and canonical `default` value targets. Historical replay was unchanged at weights
`1,4,1,8,4` and retained its sharpened target metadata.

The candidate weights SHA256 is
`f06e3e1e46815e674bd64a9437a8d8eb3a76f62632f3cbaab19771454551d00c`.

**Exact diagnostics**

| Metric | Raw | MCTS-384 |
| --- | ---: | ---: |
| Exact-optimal top-1 | 0.6900 | 0.7400 |
| Optimal mass | 0.6173 | 0.7047 |
| Top-move regret | 1.7000 | 1.2300 |
| Expected regret | 2.3391 | 1.5296 |

Value WDL MAE was 0.2403 and sign accuracy was 0.8900. The superhuman
regression suite passed.

**Canonical Gate**

The frozen prefilter passed at 0.5869 (minimum 0.55), and the independent hard
arena passed at 0.5576. The downstream MCTS-relative score tied the incumbent
(0.6625 each). The forensic check failed: overall and sparse-endgame blunder
rates regressed. The final standard decision is rejected with semantic
classification `seed455_default_value_downstream_blocked`; `model-artifact/current`
remains seed48.

For descriptive context only, sequential forward prefilter scores were seed443
sharpened 0.5458984375, seed449 sharpened 0.5283203125, and seed455 default
0.5869140625. These are not matched replicates and are not statistically pooled;
the canonical gate is authoritative.
