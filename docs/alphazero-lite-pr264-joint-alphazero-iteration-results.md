# PR #264 Joint AlphaZero Iteration Results

**Classification:** `joint_full_network_does_not_beat_incumbent`.

## Frozen Contract

- Exact A16 was verified from snapshot SHA-256
  `f38a3a9db15a095870c0294150fe81e4719cf77102c4b3de316fcae5e1b9c3ff`, checkpoint
  SHA-256 `8eaa5765786468df9fade11727449af2263a5d0d6da0fa2138040ef2b8629d34`, and
  artifact weights SHA-256
  `74e160734554dba9ecf4ecf3ac13b3e44526c4f6e7e5e52d9fe8bde0054eb789`.
- Five independent ordinary reused-tree 384-simulation replays were generated
  from A16 for seeds 63 through 67, with 700 games per seed.
- Both lanes trained all `73,741` parameters for exactly one epoch using fresh
  Adam, LR `1e-6`, zero weight decay, Huber value loss, unit value weight, and
  gradient clipping `1.0`.
- The 10 frozen candidates have aggregate model-state SHA-256
  `0d066f78bf92c8ba19c86319ec066dc061c42464b0e06ee7db2c824de115597e`.
- AK/AL/AM were sealed at seeds 37042/38042/39042 and added to the consumed
  registry. Preflight found zero final-state, prefix, replay-state, and mutual
  overlap.

## Primary Arena: 1200:1200 vs A16

| Lane | Mean | Median | SD | Range | Positive seeds | Hierarchical CI95 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| joint_anchor95 | -.014974 | -.005859 | .048142 | [-.094401, +.037109] | 1/5 | [-.058594, +.018359] |
| joint_pure_az | -.007422 | -.016276 | .020227 | [-.022135, +.027995] | 1/5 | [-.024219, +.013281] |

Neither lane passes the incumbent replacement screen. No candidate is promoted.

## Secondary Arena: 384:384 vs A16

| Lane | Mean | Median | SD | Range | Positive seeds | Hierarchical CI95 |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| joint_anchor95 | +.007161 | +.008464 | .015386 | [-.013021, +.022135] | 3/5 | [-.007031, +.021484] |
| joint_pure_az | +.038411 | +.050130 | .018235 | [+.017578, +.054688] | 5/5 | [+.017969, +.057812] |

Pure AlphaZero targets improved shallow equal-budget play but did not produce a
robust positive high-budget replacement effect.

## Recommended Next Experiment

Run a joint-training scale experiment varying only the number of unique
self-play games and/or epochs, while retaining pure AlphaZero targets. Do not
return to isolated-head fitting.
