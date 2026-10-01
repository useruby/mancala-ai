# Seed461 constant-LR E3 vs E4 preregistration

This comparison uses the five constant-LR A trajectories from PR #386 (orders
38611–38615), with no new training. A is each trajectory's original E4
checkpoint; B is its original E3 checkpoint. Epoch count is the treatment;
learning rate, scheduler, data, initialization, split, architecture, and losses
are inherited unchanged. Exact per-checkpoint SHA-256s and update/exposure
counts at E3 and E4 are recorded in `seed461-e3-e4-registration.json`.

The fixed evaluation uses 256 fresh nonterminal openings, suite seed 389,
active-pit stones greater than 32, two seat-paired games per opening, 384
simulations per side, c_puct 1.25, arena seed 389, and
`azlite_eval_seed_v2`. The suite identity and zero-overlap proof against the
conservative historical population, frozen replay states, and all consumed
suites through #388 are in the registration.

Bootstrap analysis uses 10,000 shared-opening cluster resamples, seed 389,
with a 95% percentile interval. The frozen thresholds and conditional
inference scope are recorded before candidate binding and games. Advancement
requires every threshold to pass; there are no outcome-dependent extensions.
The cosine and checkpoint-averaging rejections remain in force, and no model
is promoted.

Reproduce from the repository root:

```sh
.venv/bin/python -m ml.alphazero_lite.register_seed461_e3_e4
.venv/bin/python -m ml.alphazero_lite.bind_seed461_e3_e4
.venv/bin/python -m ml.alphazero_lite.run_seed461_e3_e4_arena
.venv/bin/python -m ml.alphazero_lite.analyze_seed461_e3_e4
.venv/bin/python -m ml.alphazero_lite.reproduce_seed461_score_matrix docs/data/seed461-e3-e4-opening-score-matrix.json --seed 389
```

The registration, candidate binding, evaluation binding, opening suite, and
results are committed under `docs/data/seed461-e3-e4-*`. Detailed per-game
records remain in repo-local `.tmp/seed461-e3-e4/arena/`; the committed compact
opening score matrix is sufficient to reproduce the estimates and bootstrap
interval.
