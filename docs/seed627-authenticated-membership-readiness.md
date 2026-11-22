# Seed 627 — authenticated membership snapshot readiness

**Status: historical membership inputs authenticated; snapshot publication is
create-only. No prospective cohort was generated.**

## Source authentication

The required historical verifier environment was available and used:
CPython 3.14.7, NumPy 2.5.2, PyTorch 2.13.0+cu130, CPU execution. The Seed 615
freeze pins that runtime and the Seed 604 dependency-lock/wheel manifest.
Rerun verifiers passed for Seed 599 B512 training results, Seed 615 training
results and freeze, Seed 615 membership/opening evidence, and the independent
Seed 622 historical provenance audit. The B512 and S615 traces each bind 512
updates of 32 membership indexes (16,384 index references), with all 2,048
membership indexes consumed. Both receipts bind the registered schedule and
shared P-E4 parent. Checkpoint identities are B512
`bd181ef6c9ea99902d0a02bec1fa30b7b73c0827f5a38b06e839d1ab3375824c`, S615
`cdf1e36c4c9374f28b182757b12471b3f00567eb59825bb938435cb56cf8d780`, and
parent `2ecc98ac7e7cd3968d08b599be96da883387b62f9c47b134b9ff11b2ac3edbad`.
The protected holdout contains 1,353 records.

Published derived identities were not accepted as truth: producer and
independent verifier reserialize every canonical source state, recompute
canonical JSON SHA-256, and recompute little-endian float32 `kalah_v3` input
SHA-256. Training rows are reconstructed from the exact traces and registered
schedule, not from archive presence.

## Provenance scope

Membership and source artifacts are authenticated. Historical game/opening
labels in source membership are not equivalent to Seed 624 `game_id` or
`opening_identity`; no independently supported crosswalk is available. This is
recorded as authenticated but non-crosswalkable provenance, with no crosswalk
rows. Missing provenance is not classified as non-exposure. P-E4 inherited
training ancestry remains unresolved because no identity-complete ledger is
available.

## Published evidence

The three snapshots are B512 consumed training membership (2,048 rows), S615
consumed training membership (2,048 rows), and protected holdout (1,353 rows).
The adjacent manifest records raw input digests, verifier results, lineage,
counts, snapshot hashes, runtime dependencies, and provenance status. The
independent verifier reconstructs rows and aggregates from historical sources;
it does not accept a manifest authentication flag as evidence. The bundle must
also be verified after physical copying using the command in its README.

This preparation does not authorize generation, resolve P-E4, or establish
source-game/opening non-exposure. No inference, MCTS, solving, training,
promotion, key provisioning, or execution authorization was performed.
