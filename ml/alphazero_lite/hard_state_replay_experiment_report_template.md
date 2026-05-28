# Hard-State Replay Experiment Report

## Shared Dataset

- Number of hard states:
- Source distribution:
- Label entropy:
- Top teacher move distribution:
- Train replay path:
- Holdout validation path:

## Variants

### Weight 1

- Hard-state pass rate:
- Arena score:
- MCTS1200 score:
- Promotion benchmark:

### Weight 2

- Hard-state pass rate:
- Arena score:
- MCTS1200 score:
- Promotion benchmark:

### Weight 4

- Hard-state pass rate:
- Arena score:
- MCTS1200 score:
- Promotion benchmark:

## Guardrails

- Mined train states and holdout validation inputs must stay separate by filename and seed.
- Do not mine from the hard-state validation holdout path.
- Do not promote a model that only improves the mined hard-state set.
