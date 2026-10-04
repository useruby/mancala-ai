# FF 1,536-search first-action exploratory comparison

This retrospective experiment uses the same ordered 64 states from #405 and the frozen FF 1,536-simulation action snapshot. It reuses 120 matching #406/#408 outcomes bound to source-ledger hashes and case identities, and adds exactly eight continuations. The design uses previously observed outcomes; it provides no promotion or overall-strength evidence.

- Primary mean gain over FF384: **0.070312**.
- Paired bootstrap 95% interval (10,000 resamples, seed 411): **[0.015625, 0.128906]**.
- Reference means: seed455 **0.085938**; original O0 E4 **0.054688**.
- Decision: **nominate_fresh_confirmation_experiment**.

Store-margin changes versus FF384 and descriptive differences versus SS384 are included by reference in `analysis.json` and case-by-case in `provenance-matrix.json`. `new-outcomes.jsonl` contains all eight complete trajectories. The 128-case matrix includes each trajectory and source provenance.

Reproduce the read-only publication check with `PYTHONPATH=. python -m ml.alphazero_lite.verify_seed398_ff1536_confirmation`.
