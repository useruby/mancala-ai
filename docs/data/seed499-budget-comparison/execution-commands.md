# Seed499 execution and verification commands

Environment: registered `.venv-azlite` (Python 3.14.7, NumPy 2.5.2).

Preparation and explicit execution:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.seed499_budget_comparison --prepare
.venv-azlite/bin/python -m ml.alphazero_lite.seed499_budget_comparison --execute
```

The run wrote one two-seat batch per opening under `batches/`. Resume validates
and skips committed pairs; any uncommitted outcome-bearing batch is rejected
and retained for explicit recovery.

Canonical historical comparison verification:

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed488_strength_evaluation --root .
```

Final seed499 verification (including complete independent replay):

```bash
.venv-azlite/bin/python -m ml.alphazero_lite.verify_seed499_publication --root .
```

The frozen verifier's first final-publication check rejected the arena-native
winner label at opening 0. That read-only verifier correction and both source
identities are recorded in `verification-correction.json`; no game was rerun.
