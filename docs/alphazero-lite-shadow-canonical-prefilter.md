# Shadow Canonical Prefilter Gate

Classification: `shadow_canonical_hard_arena_blocker`.

PR #331 established `production_prefilter_budget_asymmetry_confound_confirmed_strength_retained`: equal-budget canonical scores for P47, P48, P49, P50, P51, P52, and P_INC were all above the production threshold, with seed48 P_INC at `0.8388671875`. Production defaults are unchanged: without `--shadow-canonical-prefilter`, the gate retains its original 120-game repeated-start, 384/256 prefilter and early exit behavior.

Candidate seed48 uniform1200 weights SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`.
Incumbent SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`, version `azlite-balanced-w8s4-policy-head-e1`.
Frozen PR #330 suite SHA256: `811feb7d208467cdfa9a42244eb75adcc8877db5bc03f87dacbd3379e08bd4bf`.

## Shadow Prefilter

The opt-in initial arena used the frozen opening JSONL, `--games 512`, `--games-per-opening 2`, `--challenger-simulations 384`, and `--current-simulations 384`; the complete recorded command is in `docs/data/alphazero-lite-shadow-canonical-prefilter/local_promotion_gate.json`.

It reproduced PR #331 P_INC exactly: W/D/L `417/25/70`, raw score and opening-pair mean `0.8388671875`, pair median `1.0`, candidate P0/P1 scores `0.853515625` / `0.82421875`, and pair bootstrap CI95 `[0.8115234375, 0.8662109375]` using 20,000 resamples and seed 331. The 256 opening pairs were challenger-favored/neutral/current-favored `186/70/0`. Provenance ledger hashes are persisted in the aggregate report and the game, seed, search-configuration, and search-outcome artifacts are in the adjacent data directory.

This is a reproduction of known PR #331 strength evidence, not a new holdout measurement.

## Unchanged Downstream Gate

The production hard arena remained the 120-game repeated-start 384/256 arena and failed: W/D/L `60/0/60`, score `0.5000`, threshold `0.55`. Its state design remained 60 opening and 60 late games; this reproduces the low-diversity seat pathology.

The unchanged MCTS1200 checks both scored `0.6625` over 40 games (26 wins, 13 losses, 1 draw), so the exact relative condition `candidate_score >= current_score` passed.

The unchanged regression check failed `missed_capture_f67bd4k0_move_28`: expected move 1, selected move 5. No fixture was modified.

The unchanged production forensic suite passed all thresholds. Overall challenger-current deltas were top1 `0.0000` (threshold `-0.02`), average regret `+0.0001` (threshold `+0.02`), and blunder rate `-0.0179` (threshold `+0.01`). Sparse-endgame deltas were `0.0000/-0.0080/-0.0416`; capture-available deltas were `0.0000/-0.0215/-0.1250`; both critical buckets passed.

Frozen PR #328 corrected W/D/L context remains favorable and was not rerun: candidate top1 about `0.9296` versus incumbent `0.8732`, regret `0.1174` versus `0.2207`, and blunder rate `0.0704` versus `0.1268`. It does not override the production gate.

All production failures were retained: `regression_check_failed` and `candidate_not_stronger_than_hard`. Registered precedence makes hard arena the first blocker.

## Next Experiment

Calibrate the hard-arena design using the same frozen known-positive/identity methodology, comparing its current state population and search asymmetry against canonical equal-budget evaluation. Do not modify it yet.
