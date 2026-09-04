# Generation-3 Turn-Completion PUCT Plan

Preregistered before corpus execution. This diagnostic-only preflight compares ordinary deterministic exact-budget PUCT with `TurnCompletionPUCT` at 384 frozen neural evaluations per root. The read-only artifact is `model-artifact/current`, version `azlite-balanced-w8s4-policy-head-e1`, and `weights.json` SHA-256 `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`.

The candidate subclasses the established exact-budget PUCT adapter. It preserves A16 priors and values, averaged Q, `c_puct=1.25`, zero FPU, deterministic root selection, player-identity backup signs, visit selection, no noise, and no reuse. With turn completion disabled it is byte-equivalent to ordinary exact-budget PUCT. When an unexpanded nonterminal child retains its parent player, it is expanded and its value discarded; unchanged PUCT selection then continues through consecutive extra turns. It backs up only an exact terminal value or a value from the first player-change boundary. Calls inside extensions consume the same budget. If budget ends before the boundary, the final simulation has no backup. A repeated state or 48 actions is an invariant failure.

The fresh 96-state legal-prefix corpus uses seed `275001`: 64 root-extra-turn and 32 control states, each with at least two legal moves, canonical deduplication and canonical-hash sorting. It covers both players, midgame and late game, capture availability, and multiple legal-action counts; every row is diagnostic-only and not training eligible. Before search, state hashes are audited against PR #270, PR #274, every suite resolved through the immutable consumed-suite registry, and every registered Generation-3 diagnostic corpus. Any overlap aborts execution. The registry source hash is required unchanged.

Seeds are `275101`, `275102`, and `275103`; bootstrap seed is `275001`, with 10,000 paired hierarchical samples that resample states then one paired seed. Each forced root action is referenced with ClassicMCTS for 2,400 simulations using seed derived from `275001:{state_hash}:{action}`, with tablebase and exact solve disabled. Values and regret are always transformed to the original root player perspective.

Qualification uses the user-specified 75% primary, 90% full-corpus, confidence-interval, agreement, catastrophic-rate, runtime, and preregistered-slice gates. Classification priority is invariant failure, budget contract failure, subset regression, qualified, then no search gain. The runner validates A16 closeout and artifact hashes before and after, requires a clean worktree at execution start, records the committed implementation SHA, and prohibits artifact, registry, closeout, training, replay, arena, tablebase, solver, implicit minimax, Gumbel, and follow-up variant changes.

## Reproduction

```bash
PYTHONPATH=. .venv/bin/pytest -q ml/alphazero_lite/test_turn_completion_puct.py
PYTHONPATH=. .venv/bin/python ml/alphazero_lite/run_generation3_turn_completion_puct.py
```
