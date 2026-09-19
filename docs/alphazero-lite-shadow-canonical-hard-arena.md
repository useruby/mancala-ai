# Shadow Canonical Hard Arena

Inherited #333 classification: `hard_arena_duplicate_of_legacy_prefilter_calibration_transfers`.

## Inputs And Defaults

Candidate seed48 uniform1200 weights SHA256: `935e3cf6cc0fa2d1d848e74ec3a23f71309f4fc7b129e873b2c187aef56abf0c`.

Incumbent weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`; version: `azlite-balanced-w8s4-policy-head-e1`.

`--shadow-canonical-hard-arena` is disabled by default. With both shadow flags absent, the legacy prefilter and legacy hard command builders, thresholds, ordering, and early exits are unchanged. The shadow hard builder uses the production arena search-flag plumbing, preserving PUCT `c_puct=1.25`, zero FPU, deterministic root policy, disabled subtree reuse/value normalization, zero tactical bias/root temperature, transforms, and `azlite_eval_seed_v2` seed contract.

## Independent Shadow Arenas

The prefilter used PR #330 suite `docs/data/alphazero-lite-production-prefilter-calibration-openings-v1.jsonl`, SHA256 `811feb7d208467cdfa9a42244eb75adcc8877db5bc03f87dacbd3379e08bd4bf`.

The hard arena used PR #329 suite `docs/data/alphazero-lite-uniform1200-incumbent-holdout-openings-v1.jsonl`, SHA256 `5d1f5982c990d00bce0a57161c1ae710bed6b8ced831ac2c1551af5217d2716c`.

Both suites contain 256 canonical-unique openings. Their canonical-state intersection is `0`, so `shadow_prefilter_hard_evidence_independent = true`.

The unchanged PR #330 shadow prefilter reproduced `0.8388671875` exactly, with CI95 `[0.8115234375, 0.8662109375]`.

The new PR #329 shadow hard arena ran 512 games, two opposite challenger seats per opening, at 384/384 simulations. W/D/L was `424/22/66`; raw score and opening-pair mean both equal `0.849609375`, pair median is `1.0`, P0/P1 scores are `0.873046875`/`0.826171875`, and the deterministic 20,000-resample seed-329 CI95 is `[0.822265625, 0.876953125]`. Challenger/neutral/current-favored opening counts are `188/66/2`.

The frozen PR #329 asymmetric 384/256 comparator was `0.8837890625`; equalizing the current budget changes the score by `-0.0341796875`. This is descriptive only. The unchanged hard threshold is `0.55`, which the shadow hard result passes.

## Downstream Gate

Candidate and incumbent MCTS1200 scores were both `0.6625`, so the relative check passed.

Production forensic passed. Overall deltas were top-1 agreement `0.0`, average regret `0.0001`, and blunder rate `-0.0179`; sparse-endgame and capture-available buckets also passed.

The unchanged regression fixture failed exactly one row: `missed_capture_f67bd4k0_move_28`, expected move `1`, selected move `5`. Its full production-search policy, visits, Q values, and selection telemetry are retained in `docs/data/alphazero-lite-shadow-canonical-hard-arena/candidate_regression_suite.json`.

All remaining failure codes: `regression_check_failed`.

## Classification

`shadow_canonical_hard_passed_regression_blocker`

Exactly one next experiment: audit only the failing regression position, recovering its game state and corrected exact W/D/L labels where feasible; compare fixture and candidate actions, raw policy, production-search visits/Q, and exact outcomes for every legal action. Do not modify the fixture in that audit.

No training, self-play, promotion, incumbent change, threshold change, or production-default change was performed.
