# Seed555: T512 versus C128 policy-strength screen

## Decision

**`propose_T512_vs_serving_P_screen`**. The exploratory screen completed with all 2,048 games valid and terminal, with no ply-cap games. T512 scored **0.510498046875** (958 wins, 175 draws, 915 losses). The opening-cluster bootstrap's one-sided 95% lower bound is **0.5009765625**, strictly above 0.50; the descriptive interval is **[0.4990234375, 0.52197265625]**.

This permits proposing a separate comparison against serving P. It does not establish superiority to P and does not authorize promotion or deployment. No serving-P or higher-budget follow-up was run in this task.

## Protocol and interpretation

T512 is the #554 treatment update 108 checkpoint; C128 is the regenerated temperature-1.0 control update 108 checkpoint. Each side used its own policy checkpoint and trunk. Both used the same frozen initializer value network and initializer trunk through `ComposedArtifactEvaluator`. The teacher budgets distinguish the trained policies; both policies played at the same 128-simulation arena budget.

The screen used exactly 1,024 seat-swapped pairs / 2,048 new games, with every row from `docs/data/seed511-fixed-composition-screen-v4/openings.jsonl` replayed and retained. The suite SHA256 is `27a003322e5469507a508010a9743b40062928140e6a4b9d5628b0290e32c21d`. No historical game or outcome was imported. This is exploratory evidence on a reused suite, not a fresh holdout or confirmation.

The registered arena settings were 128 simulations per side, `c_puct=1.25`, one worker, deterministic root policy, zero temperature/noise, FPU zero, no subtree reuse or value normalization, and zero tactical root bias. Native exact-root solving used threshold 16, objective `final_score_margin`, and highest legal network prior then lowest move index for ties. Exact-leaf solving and random opening plies were disabled; games were capped at 200 plies and used seed `545 + opening_index` under `azlite_eval_seed_v2`.

## Population overlap and limitations

All 1,024 suite rows were retained. Canonical-state and production-float32 overlap counts were:

| Population | Rows | Canonical overlap openings | Float32 overlap openings |
| --- | ---: | ---: | ---: |
| T512 training derivative | 13,818 | 0 | 0 |
| C128 training derivative | 13,818 | 0 | 0 |
| Unchanged development membership | 4,738 | 1 | 1 |

The unresolved seed461 continuation-overlap limitation remains. The #554 full-search receipt and its search ledger were authenticated through `training-input-binding.json`; no relabeling or training was done. C128 remains distinct from historical C108 evaluated in #552.

## Verification and resources

The read-only seed555 verifier independently replayed all 2,048 trajectories, reconciled all 1,024 seat pairs, raw reports, completion receipts, and ledger rows, reconstructed the bootstrap and decision, and confirmed evidence bytes and mtimes were unchanged. A physical copy was independently verified from an unrelated working directory while the original execution worktree and original materialized-tablebase path were inaccessible. See `verification.json`, `verification-receipt.json`, and `physical-copy-verification.json`.

The numerical environment and resource contract are frozen in `registration.json` and `execution-freeze.json`; the live runtime receipt records the readable `cpu.max` path and value. The repository's 134-byte Git LFS pointer identity is separate from the materialized tablebase object's SHA256 and 709,635,072-byte size. Tracked runtime and serving assets were unchanged.
