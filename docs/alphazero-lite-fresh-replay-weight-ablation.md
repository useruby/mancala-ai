# Fresh Replay Weight Ablation

Classification: `fresh_replay_upweighting_no_clear_benefit`.

All six independent uniform1200 sources (401, 407, 413, 419, 443, and 449) passed SHA256 validation. Both arms use the seed48 parent, training seed 443, final checkpoint semantics, and exactly 1,084 optimizer updates. The runner validates all declared uniform1200 bytes before every arm and stops with `fresh_replay_weight_source_provenance_unavailable` if fewer than five sources are available.

| Metric | fresh_w1 | fresh_w4 | Paired delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Effective fresh replay share | 0.4618 | 0.7744 | 0.3126 | row-derived |
| Arena score | 0.4814 | 0.5550 | 0.0736 | [-0.0036, 0.1439] |
| Raw optimal mass | 0.6232 | 0.6270 | 0.0039 | [-0.0048, 0.0100] |
| Raw expected regret | 2.3005 | 2.2570 | -0.0435 | [-0.0909, 0.0274] |
| MCTS-384 optimal mass | 0.6965 | 0.6853 | -0.0113 | [-0.0223, -0.0012] |
| MCTS-384 expected regret | 1.6385 | 1.6982 | 0.0597 | [-0.0065, 0.1276] |

The raw metrics do not meet the preregistered one-sided CI requirements, and MCTS-384 optimal mass materially regresses. All superhuman regression-suite runs passed. The complete required per-source table, compact row counts, weighted-index fractions, model hashes, losses, exact metrics, arena records, selected moves, and paired bootstrap output are in `docs/data/alphazero-lite-fresh-replay-weight-ablation/results.json`.

No self-play, canonical promotion suite, selection, or promotion was performed.
