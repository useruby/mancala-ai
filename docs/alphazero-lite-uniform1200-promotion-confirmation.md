# Uniform1200 Promotion Confirmation

## Inherited Evidence

PR #327 classification: `uniform1200_high_power_strength_confirmed`. Its six seed scores were 0.6396484375, 0.681640625, 0.6572265625, 0.6328125, 0.6240234375, and 0.6357421875; aggregate mean effect was +0.1451822917 with all six positive. PR #325 exact W/D/L improved in 6/6 pairs with no repeated true win-to-loss regression and its shadow gate passed 6/6.

## Frozen Selection

Selection was persisted before the holdout gate. The ranking rule was mean high-power score, CI95 lower bound, PR #325 outcome-optimal top1, lower mean W/D/L regret, then lower seed. It selected seed48 only.
- Candidate weights JSON SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`
- Incumbent weights JSON SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a` (`azlite-balanced-w8s4-policy-head-e1`)
- `selection_frozen_before_holdout`: `True`

## Production Gate

The unmodified `script/ai/local_promotion_gate` ran once, with no config override, no skip-MCTS flag, no stub reports, and no PR #327 opening suite. Its direct arena was 60/0/60 over 120 games, score `0.5000` vs threshold `0.55`; move-time mean/p95 were `24.95`/`41.49` ms.
The gate prefilter failed with `arena_score_below_threshold`. Consequently hard arena, MCTS1200 comparisons, regression positions, and production forensic suite were intentionally not produced by the gate.

## Outcome-Aligned Shadow

This separate W/D/L exact-reference evaluation did not affect production pass/fail. Overall candidate/incumbent: top1 `0.9296`/`0.8732`, policy mass `0.4078`/`0.3577`, mean regret `0.1174`/`0.2207`, blunder rate `0.0704`/`0.1268`, win-to-draw `5`/`5`, win-to-loss `10`/`20`. Capture-available candidate/incumbent top1 was `0.8750`/`0.7917`; sparse-endgame was `1.0000`/`1.0000`.

## Classification

`uniform1200_promotion_arena_failed`

Next experiment: run ONE independent canonical-unique opening-suite arena against the incumbent, not matched control, to determine whether the production start-state arena is the disagreement source.
