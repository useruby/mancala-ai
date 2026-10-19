# Seed499 read-only verification supplement

This supplement adds semantic checks without editing the seed499 registration,
pre-execution freeze, archived outputs, publication receipt, or either frozen
verifier. No games were run.

## Verification performed

The supplemental verifier first runs
`ml.alphazero_lite.verify_seed499_publication`, then checks all 512 archived
paired batches against seed499's registration and the authenticated seed488
registration/runtime records. It validates canonical one-opening batch bytes,
parent-suite association and state identity, both player seats, copied runtime
artifact hashes, runtime source identities, both sides' search options, budgets,
seed derivation, random-opening setting, and exact-root policy. The reconstructed
current builder command specifies a 200-move cap, and every archived game is
shorter than 200 moves. However, the exact imported builder identity used during
seed499 execution is not authenticated, and archived reports do not themselves
record the requested move cap; the executed cap therefore remains unverified.
The exact-root values are read from the authenticated registered search-policy
artifact; exact-leaf solving must be disabled.

All 1,024 ledger rows are matched to raw games by opening, seat, and batch
position. Winner, game length, normal-search time, and exact-root timing used by
the published analysis must agree. Reported per-batch wins/losses/draws are
recomputed from the raw games. The published analysis is recomputed from the
ledger and historical seed488 outcomes.

The reproduced results are 457 wins, 78 draws, and 489 losses (1,024 games),
standalone score 0.484375, zero games at the 200-move cap, and recorded search
time totals 181,687.21926631406 ms normal PUCT plus 1,144.7704387828708 ms exact
root. The matched contrast is -0.0048828125, with one-sided 95% lower percentile
bound -0.025390625. Budget sensitivity remains unsupported; promotion and fresh
confirmation remain false. The unresolved seed461 non-overlap limitation is
retained.

## Dependency-binding audit

* The seed488 execution freeze prospectively bound an earlier digest of
  `ml/alphazero_lite/seed488_strength_evaluation.py`, containing
  `build_arena_command`. That historical digest differs from current checkout
  bytes. Seed499's registration/freeze binds the seed499 runner that imports
  this module, but does not separately bind the imported module. Therefore the
  exact seed488 command-builder identity used by seed499 execution was **not**
  prospectively authenticated. Current bytes are reported only as a comparison,
  not as evidence of what was executed.
* The seed488 registration/runtime identity records and seed499 runtime
  registration prospectively bind `ml/alphazero_lite/arena.py` and
  `ml/alphazero_lite/kalah_rules.py`; their present bytes match those records.
  Runtime model/policy/tablebase identities are likewise checked against
  registrations and copied-artifact manifests.
* The archives are not a trusted process transcript. They cannot establish
  which executable process loaded the builder/runtime files or independently
  attest that the recorded command was actually invoked. Python/NumPy version
  strings are recorded, but no process or executable/environment binary digest
  authenticates the runtime process. This supplement does not claim execution
  provenance, alter the freeze, or manufacture a retrospective one.

## Preservation and interpretation

This is an append-only verification record. Original seed488/seed499
registrations, freezes, source identities, game outcomes, and receipts remain
the controlling evidence. The older frozen seed499 verifier remains unchanged.
