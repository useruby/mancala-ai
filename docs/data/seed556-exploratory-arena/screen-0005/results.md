# Seed556 exploratory results

## Matchup and execution

The challenger was #554 T512 update 108 (policy and its own trunk, SHA-256 `8940cbff36976801e74542f5ee2d6c5caffe0cafd57c7e787cdb375f3bbe485e`). The opponent was #552 frozen serving P, seed504 E4 (policy and its own trunk, SHA-256 `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`). Both used the frozen initializer value network and trunk (SHA-256 `7b0491f93570cf87bade59a4797795734d061f9dc9db4f38c25378fb3fc2d94c`).

All 1,024 indexed opening pairs (2,048 games) were newly played for this final screen from the reused seed511 v4 suite. The successful frozen runtime used 18 concurrent opening pairs while retaining one search worker per game and the registered 128-simulation game settings. Earlier interrupted runs are preserved separately; their game outcomes were not reused.

## Reconstructed analysis

| Measure | Result |
|---|---:|
| T512 wins | 923 |
| Terminal draws | 193 |
| T512 losses | 932 |
| T512 score (win=1, draw=0.5, loss=0) | 0.497802734375 |
| One-sided 95% opening-cluster bootstrap lower bound | 0.486572265625 |
| Descriptive 95% interval | [0.484130859375, 0.510986328125] |
| Bootstrap | 20,000 resamples, seed 545, NumPy linear quantiles |

All 2,048 games were independently replay-validated and terminal. The preregistered lower-bound gate requires a value strictly above 0.50. Since the observed lower bound is below 0.50, the recorded decision is **`do_not_advance_T512`**. No 1,200-simulation screen is proposed, and deployment is not authorized.

## Overlap and limitations

Canonical and production-float32 opening overlap counts were both 0/1,024 for T512 training and P policy training. One indexed opening overlapped development membership in each identity. The suite is reused, and the seed461 continuation-overlap limitation remains unresolved; this is exploratory evidence, not a fresh-suite generalization claim.

## Verification note

Concurrent workers append validated pairs in completion order. The first finalization verifier compared this append order directly with opening-index order and stopped with `seed556_outcome_ledger_reconstruction_mismatch`; it did not replay games or change the frozen run. The additive correction verifier normalized ledger rows by `(opening_index, game_position)`, replayed all trajectories, reconciled all attempts, commands, pair receipts, artifacts, and analyses, and left the original evidence bytes and mtimes unchanged. It also passed from a physical copy while the original worktree and original materialized-tablebase path were inaccessible.
