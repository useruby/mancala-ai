# Seed48 Update-Matched Fixed-Volume Control

**Primary classification:** `update_matched_control_inconclusive`

The frozen PR #346 pool was verified as 71,115 rows with SHA256
`a7016fe0e360f81ba98613c52f04510e1449352bcc6d9f20cbfbfa647706e400`.
All parent, fixed-replay, and diagnostic-suite hashes were verified before
each new cell. The update cap reproduction of PR #346 seed 443 produced the
same checkpoint SHA256: `8fdfd3c16a280509cf581ffd12c18de8df56c758a85ea37fbeb8dbd80efa00a3`.

| Seed | A arena | B arena | C arena | B-A | C-B | A raw mass | B raw mass | C raw mass |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 443 | 0.4805 | 0.3320 | 0.7227 | -0.1484 | 0.3906 | 0.6156 | 0.6368 | 0.6309 |
| 1001 | 0.5039 | 0.6406 | 0.6934 | 0.1367 | 0.0527 | 0.6183 | 0.6315 | 0.6248 |
| 1003 | 0.7070 | 0.3730 | 0.5957 | -0.3340 | 0.2227 | 0.6124 | 0.6199 | 0.6398 |
| 1009 | 0.4844 | 0.5449 | 0.4941 | 0.0605 | -0.0508 | 0.6217 | 0.6339 | 0.6354 |
| 1013 | 0.2930 | 0.6484 | 0.7305 | 0.3555 | 0.0820 | 0.6210 | 0.6291 | 0.6396 |

| Metric | A: 71k/1084 | B: 71k/3068 | C: 353k/3068 | Update effect B-A (95% CI) | Matched-volume effect C-B (95% CI) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Optimizer updates | 1,084 | 3,068 | 3,068 |  |  |
| Examples processed | about 0.553m | 1.567m mean | about 1.570m |  |  |
| Exposure ratio | about 3.60 | 11.32 | about 3.60 |  |  |
| Arena mean | 0.4938 | 0.5078 | 0.6473 | 0.0141 (-0.2027, 0.2109) | 0.1395 (0.0172, 0.2895) |
| Arena SD | 0.1468 | 0.1482 | 0.1010 |  |  |
| Arena range | 0.4141 | 0.3164 | 0.2363 |  |  |
| Raw optimal mass | 0.6178 | 0.6302 | 0.6341 | 0.0124 (0.0087, 0.0169) | 0.0039 (-0.0047, 0.0129) |
| Raw expected regret | 2.3678 | 2.2002 | 2.1636 | -0.1677 (-0.2098, -0.1106) | -0.0366 (-0.1092, 0.0296) |
| MCTS-384 optimal mass | 0.6963 | 0.6841 | 0.6899 | -0.0122 (-0.0190, -0.0064) | 0.0059 (-0.0031, 0.0140) |
| MCTS-384 expected regret | 1.6323 | 1.7030 | 1.6759 | 0.0707 (0.0414, 0.0983) | -0.0270 (-0.0777, 0.0329) |

B used `--final-checkpoint final`, so it did not use validation loss for
checkpoint selection. This exposed a historical-contract caveat: the reused
A and C artifacts were produced by the trainer's legacy default
`best_validation` restore. The requested no-selection B is therefore not
checkpoint-selection-identical to those committed cells. Arena intervals for
B-A remain wide and the exact metrics split direction (raw improves while
MCTS-384 degrades), so the registered classification is inconclusive rather
than attributing PR #347 to either update count or replay volume.

No self-play, replay construction, canonical promotion suite, candidate
selection, or promotion was run. All five B superhuman regression reports
passed. Full machine-readable provenance, per-cell accounting, seed table,
and paired 10,000-sample bootstrap results are in
`docs/data/alphazero-lite-seed48-update-matched-fixed-volume/results.json`.
