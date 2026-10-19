# Seed499 exploratory search-budget comparison

## Protocol and interpretation

Seed499 used the exact seed488 F and initializer runtime artifacts, 512 frozen
opening roots, both seats per opening, `c_puct=1.25`, the registered
`azlite_eval_seed_v2` derivation (`488 + opening index`), one worker, 200-ply
cap, and unchanged runtime/search sidecar. The only arena setting changed was
simulations per side: 1,200 to 128. Exactly 1,024 new games were run. Seed488's
completed 1,200-simulation results are used only as the historical matched
comparison. No training or promotion occurred.

## Primary matched-budget contrast

For each of the 512 openings, the two seat results were averaged for F at each
budget, then the historical seed488 paired score was subtracted from the new
128-simulation paired score. The mean opening-cluster difference was
**−0.0048828125**. The 20,000-resample opening-cluster percentile bootstrap
(seed 499) gives a one-sided 95% lower bound of **−0.025390625** and two-sided
95% interval **[−0.029296875, 0.02001953125]**. The lower bound is not above
zero, so the budget-sensitivity hypothesis is **unsupported** by this result.

## Standalone 128-simulation result

F scored **0.484375** across the 1,024 games: **457 wins, 78 draws, 489 losses**.
The paired-opening 20,000-resample bootstrap (seed 499) has two-sided 95%
interval **[0.4638671875, 0.5048828125]** and one-sided 95% lower bound
`0.46728515625`. No games reached the 200-ply cap.

Recorded search time totals were **181,687.21926631406 ms** for normal PUCT,
**1,144.7704387828708 ms** for exact-root solving, and **182,831.98970509693 ms**
combined. These are sums of the per-game recorded fields.

A positive budget contrast alone would not demonstrate strength above the
initializer. This experiment does not establish superiority, authorize
promotion, or make a fresh-confirmation claim.

## Verification and preservation

- Seed488's canonical completed-publication verifier passed before freezing and
  again immediately before execution: 1,024 games, historical score
  `0.4892578125`, no advance.
- Synthetic runner/recovery, replay, artifact/runtime, duplicate/missing pair,
  resume/committed-pair skip, and matched-bootstrap tests passed before freeze.
- The seed499 publication verifier independently replayed all 1,024 trajectories
  and reproduced `analysis.json` from the raw evidence.
- The frozen verifier's initial final-publication invocation found a verifier
  label-normalization defect (`outcome_replay:0:0`); its source remains exactly
  as frozen. `verify_seed499_publication.py` is the separately identified
  read-only correction: it compares the frozen game log's arena-native
  `challenger/current` labels with independent replay. No outcome, setting, or
  historical file was changed. `verification-correction.json` binds the
  original and corrected verifier identities and final evidence hashes.
- Full opening-level evidence is in `batches/`; `outcomes.jsonl` is the 1,024-row
  ledger; `registration.json`, `freeze.json`, and receipts bind inputs and
  sources. The reproducible analysis is `analysis.json` and
  `ml/alphazero_lite/seed499_budget_analysis.py`.

## Limitation

The known seed461 overlap limitation remains: non-overlap with the unresolved
seed461 continuation population cannot be certified. This is exploratory
evidence only, not a fresh confirmation or an independently untouched holdout.
