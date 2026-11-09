# Seed560 paired first-action diagnostic

Retrospective paired comparison using the authenticated 2,048-game #556
archive. Eligible states are the first post-prefix nonterminal decision per
source game with 33–44 active pit stones and at least three legal moves.
Canonical-state deduplication and seed560 SHA-256 ranking select one root per
opening, balanced 128/128 by source player (`P` / `T512`). The cohort,
provenance, artifact identities, seed schedules, runtime, and analysis plan
were frozen before searching in `execution-freeze.json`.

The unchanged #556 P-E4 policy/initializer composition searched fresh trees at
128 and 512 simulations with a matched seed. Each chosen action was forced
from the same root; the same P composition then played both seats at 128
simulations, exact-root cutoff 16, and exact-leaf solving disabled. The
registered seed schedule is matched across branches and Kalah extra turns are
honored. Equal first actions share one physical continuation, and their
zero-difference roots remain in the primary analysis.

## Result

- 256 complete openings; 512 arm records representing 313 distinct physical
  continuations; 57 roots changed first action (22.27%).
- Mean continuation score difference (`512 - 128`): **+0.0137**.
- Paired 95% opening-bootstrap interval (20,000 resamples, seed 560):
  **[-0.0176, +0.0449]**.
- Changed-action-root score difference: **+0.0614** (descriptive).
- Mean incremental root-search latency: **0.0544 seconds**.
- Decision: **no selective budget-rule test proposed**; the mean is below
  +0.03 and the interval includes zero.

This result is retrospective and specific to the fixed P continuation
policy. It does not establish full-game strength or authorize a serving
change. The serving model and cutoff 16 were unchanged; no training,
full-game arena matchup, or deployment was performed.

## Verification

```sh
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed560_paired_first_action \
  --data docs/data/seed560-paired-first-action/run-0004
PYTHONPATH=. python -m unittest ml.alphazero_lite.test_seed560_paired_first_action -v
PYTHONPATH=. python -m ml.alphazero_lite.verify_seed560_physical_copy
```

The read-only verifier reconstructs the cohort from the authenticated archive,
checks freeze and artifact bindings, replays every continuation through
terminal scoring, validates seed/action/runtime semantics and identical-action
reuse, and recomputes the analysis without modifying evidence.
The physical-copy command clones the committed checkout, hides the execution
worktree while the copied verifier runs, and records the passing receipt in
`physical-copy-verification.json`.
