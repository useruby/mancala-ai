# Corrected frozen order 38615 E4 diagnostic

Observed score-threshold outcome: **FAIL**. Protocol validity: **valid**.

| Suite | Score | Wins | Draws | Losses | 95% opening-cluster interval | Rule |
|---:|---:|---:|---:|---:|---:|:---|
| 395 | 0.4868 | 450 | 97 | 477 | 0.4658–0.5068 | FAIL |
| 396 | 0.4839 | 444 | 103 | 477 | 0.4629–0.5044 | FAIL |

| Suite | Challenger seat | Games | Wins | Draws | Losses | Score |
|---:|---:|---:|---:|---:|---:|---:|
| 395 | 0 | 512 | 261 | 54 | 197 | 0.5625 |
| 395 | 1 | 512 | 189 | 43 | 280 | 0.4111 |
| 396 | 0 | 512 | 264 | 42 | 206 | 0.5566 |
| 396 | 1 | 512 | 180 | 61 | 271 | 0.4111 |

Pooled equal-weight score: **0.4854**; stratified 95% interval 0.4709–0.5002 (FAIL).

This diagnostic estimates only the corrected opening distribution and does not override the canonical promotion failure.

The score matrix contains 512 paired-opening cluster scores per suite; `analyze_corrected_order38615_diagnostic.py` verifies raw evidence hashes and reproduces 10,000-resample intervals.

Published per-game outcome accounting: `game-outcome-accounting.jsonl` (SHA-256 `2beb361f469eb3ced4a1ebf49f981a3a0429fb45a0e1b7b64f401a1df5ab39e2`; 2048 rows). Registration, immutable exclusion manifest, suite bytes, raw run evidence, and runtime identities are hash-bound in the accompanying JSON records.

Execution: 2048 games, 1,572,864 requested search simulations, 384 per side, 24 workers; no training or model export.

| Suite | Move mean (ms) | Move p95 (ms) | Summed PUCT (ms) | Summed exact-root (ms) |
|---:|---:|---:|---:|---:|
| 395 | 50.65 | 130.72 | 1690295.0 | 3017.3 |
| 396 | 51.16 | 130.90 | 1683258.2 | 2988.7 |
