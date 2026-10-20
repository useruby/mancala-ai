# Seed505 publication correction 1

This append-only supplement independently replays the archived 1,024 raw game
trajectories across all 512 opening/seat pairs and recomputes the registered
20,000 paired-opening bootstrap (seed 505). The recomputed score is
0.50830078125 (474 wins, 93 draws, 457 losses); the one-sided lower bound is
0.48974609375 and the descriptive two-sided interval is
[0.48583984375, 0.5302734375]. The strict lower-bound > 0.50 criterion fails.
The original failed strength-screen decision is retained.

The supplemental verifier checks command inputs, full report settings,
resolved artifacts and search/exact-root settings, raw trajectory replay,
report counts, ledger bindings, and every published result field. Run it with:

```bash
python -m ml.alphazero_lite.verify_seed505_publication_supplement --root .
```

Archived commands record the 200 move cap and zero random-opening plies. They
record the opening suite identity, both artifact roots, simulation counts,
workers, `c_puct`, search options, exact-root tablebase identities/settings,
and the per-opening seed/contract. The game trajectories themselves do not
independently encode the runtime seed or cap; these settings are therefore
established by recorded commands and reports, not reconstructed from moves.

The frozen runner's resume validation has a defect: it compares the number of
distinct opening indexes with the number of distinct `(opening_index, seat)`
pairs. A valid complete ledger has 512 indexes and 1,024 pairs, so resume
rejects it before reaching its existing skip-completed-opening branch. A
synthetic regression invokes that actual guard with a complete pair and
observes the erroneous `partial_or_duplicate_committed_pair` rejection; it
also asserts that no arena process is invoked. Intended behavior is to
validate each complete pair, then skip it without another arena invocation.
The historical runner is preserved and was not resumed or used to run games.
This correction is verification-only; no training, games, 1,200-simulation
evaluation, promotion, or fresh-confirmation claim was made.
