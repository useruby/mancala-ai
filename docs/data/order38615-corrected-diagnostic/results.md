# Corrected frozen order 38615 E4 diagnostic

Observed score-threshold outcome: **FAIL**. Post-run audit found 14 declared-state intersections for suite 393 and 12 for suite 394 with prior consumed suites, despite zero reconstructed-actual-state overlap. Protocol validity: **invalid_for_preregistered_exclusion_contract**; do not interpret these estimates as corrected-distribution performance. Details are in [`post-run-validity.json`](post-run-validity.json).

| Suite | Score | Wins | Draws | Losses | 95% opening-cluster interval | Rule |
|---:|---:|---:|---:|---:|---:|:---|
| 393 | 0.4912 | 450 | 106 | 468 | 0.4692–0.5122 | FAIL |
| 394 | 0.5010 | 449 | 128 | 447 | 0.4805–0.5220 | FAIL |

Pooled equal-weight score: **0.4961**; stratified 95% interval 0.4807–0.5107 (FAIL).

The preserved observed result does not override the canonical promotion failure and is not a valid corrected-distribution estimate because the full declared-state exclusion contract was missed.

The score matrix contains 512 paired-opening cluster scores per suite. Reproduce all intervals from the committed matrix with `.venv/bin/python -m ml.alphazero_lite.reproduce_corrected_order38615_diagnostic docs/data/order38615-corrected-diagnostic/opening-score-matrix.json`. `analyze_corrected_order38615_diagnostic.py` additionally verifies local raw evidence hashes.

Execution: 2048 games, 1,572,864 requested search simulations, 384 per side, 24 workers; no training or model export.

| Suite | Move mean (ms) | Move p95 (ms) | Summed PUCT (ms) | Summed exact-root (ms) |
|---:|---:|---:|---:|---:|
| 393 | 49.08 | 129.99 | 1640108.2 | 3024.6 |
| 394 | 50.42 | 130.54 | 1670795.7 | 2985.8 |
