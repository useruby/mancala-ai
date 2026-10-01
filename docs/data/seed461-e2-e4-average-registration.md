# Seed461 fixed E2–E4 checkpoint average

Prospective comparison of the original constant-LR E4 checkpoint (A) against
the uniform arithmetic mean of E2, E3, and E4 (B), using the five constant-LR
PR #386 trajectories for orders 38611–38615. No new training was performed.
PR #386's cosine-LR rejection remains unchanged.

The registration freezes source checkpoint hashes, the float64-accumulate then
cast-to-source-dtype averaging rule, analysis metrics, bootstrap settings, and
all decision thresholds before candidate construction. The seed-387 suite
contains 256 fresh nonterminal openings with more than 32 active-pit stones.
Its SHA-256 and a zero state-identity intersection proof against the
conservative historical/replay/consumed-suite exclusion union are in
`seed461-e2-e4-average-registration.json`.

All ten candidates were constructed and exported, and their checkpoint,
artifact-file, opponent, and resolved runtime identities are frozen in
`seed461-e2-e4-average-candidate-binding.json`. Parameter L2 distances are
published separately as descriptive evidence in
`seed461-e2-e4-average-parameter-distances.json`; they do not establish a
mechanism for any possible gain.

From the repository root, with the frozen PR #386 local training artifacts
available under `.tmp/seed461-cosine-lr-ablation/`:

```sh
.venv/bin/python -m ml.alphazero_lite.run_seed461_e2_e4_average register
.venv/bin/python -m ml.alphazero_lite.run_seed461_e2_e4_average construct
```

Registration and candidate construction are idempotent and refuse changed
source, suite, artifact, opponent, or runtime identities. Arena games have not
been run. The registered evaluation budget is exactly 5,120 games; no
outcome-dependent extensions are allowed. Inference is conditional on these
five orders and this frozen dataset, and no model promotion or production
checkpoint-selection change is permitted.
