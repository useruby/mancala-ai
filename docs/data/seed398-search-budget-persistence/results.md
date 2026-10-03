# Seed398 search-budget persistence diagnostic

This retrospective diagnostic uses 64 selected states from the published seed398 SS/FF trajectories. They are not a fresh holdout. It addresses decision stability only and does not measure move quality or playing strength.

**Primary descriptive classification: `budget_sensitive_disagreement`.** SS and FF selected the same action at 1,536 simulations in 32 of 64 states (50%). Seventeen states met the stricter persistence description: each model retained its own 384-simulation action at both later snapshots, with both normalized top-two visit gaps at least 0.10 at 1,536.

| Secondary mechanism comparison | Same action as SS at 1,536 |
|---|---:|
| FS | 42 / 64 |
| SF | 40 / 64 |

The probes extend one search per state and treatment to 1,536 simulations and record snapshots at 384, 768, and 1,536. Each search uses the original 384-search `azlite_eval_seed_v2` derived seed without reseeding across snapshots. Snapshot-prefix equivalence was checked against standalone 384 searches; the SS/FF standalone actions were checked against the recorded arena trajectories.

The threshold classification describes persistence of search choices only. Neither classification establishes which model is better or authorizes promotion.

## Reproduction and integrity

From the repository root, use the torch-enabled environment:

```bash
./.venv/bin/python -m ml.alphazero_lite.verify_seed398_publication
./.venv/bin/python -m ml.alphazero_lite.verify_seed398_search_budget_persistence
```

The portable verifier checks selection/probe counts, legal-action telemetry, visit totals, normalized gaps, summary reconciliation, and publication ledger/suite hashes. The raw per-state/per-treatment probe data are in `raw-probes.json`; the exact retrospective state and ledger provenance are frozen in `registration.json`.

| Artifact | SHA-256 |
|---|---|
| Registration | `9761d5908d7059c3b3794f1ac87e41127a6be982ecd7244aadee15f908541179` |
| Raw probes | `fc8210da9c70fb828d20cfb5ab19304f6560cbf56f7fe1a0f23ec4342d233e0f` |
| Summary | `a54e46477adcab7e8dc002cbc91e7f0f45c652c4d6a4535ebd6a24c3c7c83409` |
