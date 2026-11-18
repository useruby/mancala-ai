# Seed 615 — Source-opening diversity

Preparation-only artifacts for the feasibility test. The recommended next
step is `train`, subject to review and separate authorization. No model training
or evaluation games have been run.

- `FEASIBILITY.md` — decision, results, confounders, and precision assessment.
- `feasibility.json` — authenticated inventories and phase/group occupancy and
  teacher-target distributions for B512 and candidate.
- `candidate-membership.jsonl` — all 2,048 authenticated selected rows and
  unchanged teacher targets.
- `evaluation-openings.jsonl` — independent 1,024-root evaluation cohort.
- `evaluation-assignments.jsonl` — all 4,096 deterministic seat/game-seed
  assignments; assignments are not executed games or authorization.
- `registration.json` — proposed fixed training and confirmatory evaluation
  contract; both authorization fields remain false.
- `independent-verification.json` — read-only verifier result and bound hashes.

From the repository root, using an environment with the repository's pinned
NumPy/PyTorch requirements:

```bash
python -m ml.alphazero_lite.seed615_opening_diversity verify
python -m ml.alphazero_lite.verify_seed615_opening_diversity
python -m unittest ml.alphazero_lite.test_seed615_opening_diversity
```

`prepare` deterministically rebuilds the preparation manifests. It runs source
verification and membership/target reconstruction only; it has no training or
arena execution path.
