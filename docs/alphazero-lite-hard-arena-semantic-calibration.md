# Hard Arena Semantic Calibration

Inherited #332 classification: `shadow_canonical_hard_arena_blocker`.

## Contracts

Both current default invocations use `model-artifact/current`, 120 games, threshold 0.55, 384/256 simulations, seed 42 under `azlite_eval_seed_v2`, deterministic PUCT (`c_puct=1.25`, zero FPU, no subtree reuse/value normalization, zero tactical bias/root temperature), no opening prefixes, zero random opening plies, and one worker.

Normalized evaluator contracts identical: `True`. Path equality/default and #332: `True`/`True`; incumbent artifact SHA: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`.

## Artifact Reproduction

#328 versus #332 semantic fields and recomputed committed #328 ledger files match: `True`. Wall-clock latency is excluded because no latency threshold is enabled.

## Hard Buckets

`hard_suite_buckets` is telemetry about game trajectories, not evidence of a distinct hard-position starting-state suite. The #332 opening/midgame/late counts (60/0/60) are completion classification after challenger phase visitation from the same repeated initial state, not 60 opening and 60 late starting states.

## Independence

Repeated-start diversity: {"distinct_challenger_seats": 2, "distinct_search_configurations": 1, "distinct_starting_states": 1, "repeated_outcome_pattern": "60 challenger wins as player 0 and 60 current wins as player 1", "unique_complete_trajectories": 2}. The 120 nominal games use one start state under deterministic search. `prefilter_hard_evidence_independent = False`.

## Inherited Calibration

`pr330_pr331_calibration_transfer_valid = True`. Data source for every row: `inherited_from_semantically_identical_evaluator`.

| Pair | 384/256 repeated-start | >=0.55 | 384/384 canonical | CI95 | Positive evidence | Identity status |
| --- | --- | --- | --- | --- | --- | --- |
| P47 | 0.5000 | False | 0.6133 | [0.5811, 0.6455] | True | None |
| P48 | 1.0000 | True | 0.6807 | [0.6465, 0.7148] | True | None |
| P49 | 0.5000 | False | 0.6953 | [0.6621, 0.7275] | True | None |
| P50 | 1.0000 | True | 0.6553 | [0.6250, 0.6855] | True | None |
| P51 | 1.0000 | True | 0.6445 | [0.6123, 0.6777] | True | None |
| P52 | 1.0000 | True | 0.6582 | [0.6230, 0.6924] | True | None |
| P_INC | 0.5000 | False | 0.8389 | [0.8115, 0.8662] | True | None |
| I_INC | 0.5000 | False | 0.5000 | [0.5000, 0.5000] | None | True |
| I_48 | 0.5000 | False | 0.5000 | [0.5000, 0.5000] | None | True |

Known-positive repeated-start passes: `4/7`; equal-budget positive evidence: `7/7`; asymmetric canonical identity bias: `True`; equal-budget identity symmetry: `True`. The repeated-start controls remain 0.5 because they repeat seat-determined initial-state trajectories; the asymmetric canonical controls are the registered #330 bias evidence.

No new games were required; no fallback calibration ran. The fallback nine-pair plan is retained but is ineligible unless equivalence fails. `local_promotion_gate` was not modified.

## Classification

`hard_arena_duplicate_of_legacy_prefilter_calibration_transfers`

Exactly one next experiment: Add a DISABLED-BY-DEFAULT shadow canonical equal-budget HARD-arena mode to local_promotion_gate, reusing a frozen canonical suite with 384/384 opening-pair scoring; preserve production defaults, change only hard-arena evaluation, and rerun seed48 through the complete gate once.
