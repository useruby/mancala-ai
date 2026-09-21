# Seed48 Full-Volume Self-Play Pooling

Classification: `full_volume_pooling_improves_stability_only`.

This non-promoting experiment compared the PR #346 equal-contribution
71,115-row pool against an equal-contribution 353,455-row pool from the same
five validated uniform1200 sources. No self-play was generated, no canonical
promotion suite was run, no candidate was selected, and no model was promoted.

The source and seed48 parent SHA-256 pins all matched. The pool uses seed 347,
SHA-256 ranking within source, and a SHA-256 final shuffle. Its SHA-256 is
`a815b29353e98a3f625e056cec2013b227201a34292b482e44e273db36662d21`.
The manifest records byte-for-byte regeneration details and subset hashes.

Pool diagnostics: 215,635 unique canonical states, 38.99% duplicate-state rate,
and 24,973 states shared across sources. This is 164,056 more unique states
than the 71,115-row pool. Repeated states were retained.

| Seed | fixed71115 arena | full353455 arena | Delta | fixed raw optimal mass | full raw optimal mass | regression status |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 443 | 0.4805 | 0.7227 | +0.2422 | 0.6156 | 0.6309 | passed |
| 1001 | 0.5039 | 0.6934 | +0.1895 | 0.6183 | 0.6248 | passed |
| 1003 | 0.7070 | 0.5957 | -0.1113 | 0.6124 | 0.6398 | passed |
| 1009 | 0.4844 | 0.4941 | +0.0098 | 0.6217 | 0.6310 | passed |
| 1013 | 0.2930 | 0.7305 | +0.4375 | 0.6210 | 0.6439 | passed |

| Metric | fixed71115 | full353455 | delta / ratio |
| --- | ---: | ---: | ---: |
| Fresh rows | 71,115 | 353,455 | 4.97x |
| Unique canonical states | 51,579 | 215,635 | +164,056 |
| Duplicate rate | 27.47% | 38.99% | +11.52 pp |
| Weighted training examples | 153,830 | 436,170 | 2.84x |
| Optimizer updates | 1,084 | 3,068 | 2.83x |
| Training duration | not retained in PR #346 | 76.6-78.3 s | n/a |
| Arena mean | 0.4938 | 0.6473 | +0.1535 |
| Arena SD | 0.1468 | 0.1010 | 0.69x |
| Arena range | 0.4141 | 0.2363 | 0.57x |
| Raw optimal mass | 0.6178 | 0.6341 | +0.0163 |
| Raw expected regret | 2.3678 | 2.1636 | -0.2042 |
| MCTS-384 optimal mass | 0.6963 | 0.6899 | -0.0064 |
| MCTS-384 expected regret | 1.6323 | 1.6759 | +0.0437 |

The five paired arena deltas have mean +0.153515625 and median +0.189453125.
The 10,000-sample paired bootstrap (seed 347) 95% CI is
`[-0.002734375, +0.312890625]`. The interval overlaps zero, so strength is not
established under the preregistered rule. The full-volume SD is 0.69x the
fixed-volume SD, meeting the stability criterion. Raw exact quality improved;
MCTS-384 metrics changed slightly in the adverse direction. All five current
superhuman regression suites passed their sole selected move (move 1).

The training recipe was unchanged, but the full-volume runs consumed more
weighted examples and optimizer updates. This is not a pure unique-data result.
Per-run compact row counts, exact split counts, optimizer work, and measured
wall-clock durations are recorded in `results.json`.
