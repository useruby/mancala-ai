# Seed48 Uniform384 vs Uniform1200 Training Ablation

Classification: `uniform1200_training_advantage_confirmed`.

This non-promoting matched training-seed experiment compares uniform 384 and uniform 1200 self-play simulations. Both lanes use 1,600 games, the promoted seed48 parent (`935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`), and byte-identical frozen replay inputs at weights `4,1,8,4`.

The six matched training seeds are 401, 407, 413, 419, 425, and 431. Their three-seed self-play sweeps are non-overlapping. The generated lane configs remove all opening-minimum options, so `--simulations` is the only lane difference.

The PR #340 exact corpus remains evaluation-only. Preflight requires `training_eligible: false`, rejects it in the fixed replay, and records natural fresh-self-play collisions separately.

All 12 children completed from the pinned parent. No artifact was promoted.

This report was regenerated from the completed workers6 manifests, self-play logs, training logs, checkpoints, and evaluation JSON. The machine-readable result stores the extracted commands, step durations, cache counters, checkpoint and exported-weight SHA256 values, training losses, and perspective-audit reports for every lane.

| Seed | 384 search work | 1200 search work | 384->parent | 1200->parent | 384-vs-1200 | Raw optimal-mass delta | Raw regret delta | Regression status |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 401 | 50,976,000 | 170,083,200 | 0.387 | 0.328 | 0.352 | -0.041 | +0.342 | fail/pass |
| 407 | 50,949,888 | 169,658,400 | 0.443 | 0.621 | 0.357 | -0.013 | +0.106 | pass/pass |
| 413 | 50,780,160 | 170,450,400 | 0.291 | 0.453 | 0.297 | -0.022 | +0.137 | pass/pass |
| 419 | 50,807,808 | 171,081,600 | 0.648 | 0.594 | 0.465 | -0.019 | +0.183 | pass/pass |
| 425 | 51,343,104 | 170,076,000 | 0.289 | 0.566 | 0.363 | -0.026 | +0.141 | fail/pass |
| 431 | 50,539,776 | 170,620,800 | 0.355 | 0.527 | 0.465 | -0.005 | +0.010 | pass/pass |

`384-vs-1200` is the uniform384 score. Raw deltas are uniform384 minus uniform1200, so positive optimal mass and negative regret favor uniform384.

| Metric | uniform384 | uniform1200 | paired delta | 95% CI |
| --- | ---: | ---: | ---: | --- |
| Self-play work | 50,899,456 | 170,328,400 | 0.299x work ratio | - |
| Parent arena score | 0.402 | 0.515 | -0.113 | [-0.207, -0.017] |
| Sibling score | 0.383 | 0.617 | -0.117 effect | [-0.164, -0.070] |
| Raw exact optimal mass | 0.611 | 0.632 | -0.021 | [-0.031, -0.013] |
| Raw exact expected regret | 2.405 | 2.252 | +0.153 | [+0.080, +0.241] |
| MCTS-384 exact optimal mass | 0.694 | 0.708 | -0.014 | [-0.027, -0.001] |
| MCTS-384 exact expected regret | 1.641 | 1.552 | +0.089 | [-0.043, +0.222] |

Uniform1200 has a clear sibling advantage beyond the non-inferiority margin, stronger parent-relative performance, better raw exact metrics, and no regression failures. Uniform384 failed the regression suite for seeds 401 and 425. Do not promote either child; retain 1200 for the next full AlphaZero iteration.

Artifacts, exact commands, hashes, per-seed telemetry, and deterministic bootstrap output are in `docs/data/alphazero-lite-seed48-uniform384-vs-1200/results.json`.
