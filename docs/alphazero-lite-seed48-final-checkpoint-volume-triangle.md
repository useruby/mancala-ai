# Seed48 Final-Checkpoint Volume Triangle

**Primary classification:** `final_checkpoint_triangle_inconclusive`

This non-promoting audit compares only final-state checkpoints. A small capped
fixture (seed 443, cap 8, `--final-checkpoint final`) produced the identical
checkpoint SHA256 twice:
`342957886de85f03f3686a8366b705817b41b0b1bc24afa1ae1fb13fb92bc643`.

All A/C runs re-verified the seed48 parent weights
`935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`,
the fixed replay sources, and the diagnostic-suite SHA256 before training.
B_final is reused from PR #348 after artifact, pool, replay, parent, update,
and final-checkpoint provenance validation. No self-play, pool construction,
canonical gate, selection, or promotion was performed.

| Seed | A_final arena | B_final arena | C_final arena | B-A | C-B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 443 | 0.6133 | 0.3320 | 0.7227 | -0.2812 | 0.3906 |
| 1001 | 0.5918 | 0.6406 | 0.6934 | 0.0488 | 0.0527 |
| 1003 | 0.3789 | 0.3730 | 0.5957 | -0.0059 | 0.2227 |
| 1009 | 0.2637 | 0.5449 | 0.4941 | 0.2812 | -0.0508 |
| 1013 | 0.2930 | 0.6484 | 0.7305 | 0.3555 | 0.0820 |

Paired effects are mean differences over the five training seeds; intervals
are 10,000 paired bootstrap resamples with seed 349.

| Metric | A_final 71k/1084 | B_final 71k/3068 | C_final 353k/3068 | Update effect B-A | Volume effect C-B |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fresh rows | 71,115 | 71,115 | 353,455 |  |  |
| Optimizer updates | 1,084 | 3,068 | 3,068 |  |  |
| Examples processed (mean) | 0.554m | 1.567m | 1.570m |  |  |
| Exposure ratio (mean) | 3.60 | 11.32 | 3.60 |  |  |
| Checkpoint policy | final | final | final |  |  |
| Arena mean | 0.4281 | 0.5078 | 0.6473 | 0.0797 (-0.1137, 0.2684) | 0.1395 (0.0172, 0.2895) |
| Arena SD | 0.1649 | 0.1482 | 0.1010 |  |  |
| Arena range | 0.3496 | 0.3164 | 0.2363 |  |  |
| Raw optimal mass | 0.6165 | 0.6302 | 0.6341 | 0.0138 (0.0021, 0.0237) | 0.0039 (-0.0047, 0.0128) |
| Raw expected regret | 2.3445 | 2.2002 | 2.1636 | -0.1444 (-0.2314, -0.0503) | -0.0366 (-0.1044, 0.0303) |
| MCTS-384 optimal mass | 0.6929 | 0.6841 | 0.6899 | -0.0088 (-0.0141, -0.0045) | 0.0059 (-0.0037, 0.0140) |
| MCTS-384 expected regret | 1.6564 | 1.7030 | 1.6759 | 0.0465 (0.0128, 0.1045) | -0.0270 (-0.0777, 0.0293) |
| Regression status | 5/5 passed | 4/5 passed (seed 1001) | 5/5 passed |  |  |

Natural-boundary accounting was exact: A used 271 batches per full epoch,
four completed epochs, and zero partial-final-epoch batches. C used 767,
four, and zero respectively.

The matched-volume arena effect is positive but remains small-sample uncertain
at the registered material threshold, and the exact-policy evidence is mixed:
raw metrics weakly improve with added optimizer work while MCTS-384 metrics
worsen. The C-B arena interval excludes zero but raw and MCTS exact intervals
do not provide consistent additional support. The prescribed classification is
therefore inconclusive rather than an attribution to either mechanism.

`docs/data/alphazero-lite-seed48-final-checkpoint-volume-triangle/results.json`
contains per-seed checkpoint and artifact hashes, all raw and MCTS-384
top-1/optimal-mass/regret metrics, regression reports, source provenance, and
the complete paired bootstrap output. The plan is recorded alongside it. The
reused B seed 1001 report contains the one recorded regression failure; it was
not selected around or retrained.
