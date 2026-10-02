# Order 38615 A5 canonical prefilter reproduction

The #392 canonical prefilter remains a **failure**: 222 wins, 50 draws, and 240
losses in the original 512 seat-paired games, for score `0.482421875` against
the required `0.55`. Its original 95% opening-pair bootstrap interval is
`[0.4521484375, 0.5126953125]`.

The public matrix was reconstructed from the original run files, without new
games. It contains one score for each of the 256 canonical openings and binds
the original report, game rows, suite, and search ledgers by SHA-256.

Reproduce it from the checkout containing the original `.tmp` evidence:

```sh
PYTHONPATH=. .venv/bin/python -m ml.alphazero_lite.reproduce_order38615_a5_canonical_gate
```

The command verifies the original report and game-row hashes recorded in
`docs/data/order38615-a5-promotion-review-result.json`, recomputes every pair
score from the two seat-swapped games, and reruns the gate's Python
`random.Random(331)` bootstrap with 20,000 resamples and the original order
statistics (indices 500 and 19,499). It then checks the score and interval
against the original gate evidence.

Public matrix: [`opening-score-matrix.json`](opening-score-matrix.json).
Matrix SHA-256: `66e50f5706f38ed0d09b671a4789c05bd1e6beff48b46b4552596d9b94e60b35`.

The original arena report, game rows, and seed/search ledgers remain run-local
under `.tmp/order38615-a5-promotion-review/`; their exact hashes are published
in the matrix and #392 review result. The later hard-arena, MCTS, regression,
and forensic components were not run after this prefilter failure.
