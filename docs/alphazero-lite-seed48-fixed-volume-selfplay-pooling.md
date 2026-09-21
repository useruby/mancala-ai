# Seed48 Fixed-Volume Self-Play Pooling

Classification: `fixed_volume_pooling_no_clear_benefit`.

This non-promoting experiment held fresh self-play volume at 71,115 rows. It
compared the five recorded `single443` training-seed realizations with five
models trained on an equal 14,223-row contribution from each independently
generated uniform1200 trajectory: seed401, seed407, seed413, seed419, and
seed443. No self-play was generated, no canonical suite was run, no model was
selected, and no model was promoted.

All source hashes matched the PR #345 pins. The deterministic pool uses seed
346, SHA-256 ranking within each source, then a second SHA-256 final shuffle.
It has SHA-256 `a7016fe0e360f81ba98613c52f04510e1449352bcc6d9f20cbfbfa647706e400`.
The committed manifest records every source and selected-subset hash, source
row count, and byte-for-byte regeneration algorithm.

Pool diagnostics: 51,579 unique canonical states, 27.47% duplicate-state rate,
and 3,834 states shared by more than one source. Each source contributes exactly
14,223 rows. The ply-bucket contribution is recorded in `pool_manifest.json`.
Naturally repeated training positions were retained.

| Seed | single443 arena | pooled arena | Delta | single raw optimal mass | pooled raw optimal mass | regression status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 443 | 0.5586 | 0.4805 | -0.0781 | 0.6303 | 0.6156 | passed |
| 1001 | 0.5664 | 0.5039 | -0.0625 | 0.6259 | 0.6183 | passed |
| 1003 | 0.4160 | 0.7070 | +0.2910 | 0.6132 | 0.6124 | passed |
| 1009 | 0.5391 | 0.4844 | -0.0547 | 0.6311 | 0.6217 | passed |
| 1013 | 0.4277 | 0.2930 | -0.1348 | 0.6305 | 0.6210 | passed |

| Metric | single443 | pooled5 fixed-volume | paired delta / ratio |
| --- | ---: | ---: | ---: |
| Arena mean | 0.5016 | 0.4938 | -0.0078 |
| Arena SD | 0.0735 | 0.1468 | 2.00x |
| Arena range | 0.1504 | 0.4141 | 2.75x |
| Raw optimal mass | 0.6262 | 0.6178 | -0.0084 |
| Raw expected regret | 2.2749 | 2.3678 | +0.0929 |
| MCTS-384 optimal mass | 0.6975 | 0.6963 | -0.0012 |
| MCTS-384 expected regret | 1.6071 | 1.6323 | +0.0252 |

The paired arena mean delta is -0.0078125 and median is -0.0625. A bootstrap
over the five training-seed pairs (10,000 resamples, seed 346) gives 95% CI
`[-0.105859375, +0.146484375]`. The interval overlaps zero and pooled dispersion
increased rather than reaching the pre-registered 0.75x stability threshold.

Exact-oracle changes are recorded per training seed in `results.json`. Raw
optimal mass fell by 0.0084 on average and raw expected regret rose by 0.0929;
the MCTS-384 changes were smaller. All five current superhuman regression suites
passed. These diagnostics do not support a material policy-quality improvement.

The fixed-volume diversity intervention therefore has no clear benefit. The next
experiment should isolate data volume directly before generating another
170M-simulation challenger.
