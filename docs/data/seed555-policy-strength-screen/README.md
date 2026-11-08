# Seed555: T512 versus C128 policy-strength screen

Exploratory comparison of the #554 update-108 T512 challenger and regenerated C128 control. Each policy uses its own exported checkpoint and trunk; both sides use the same frozen initializer value network and initializer trunk through `ComposedArtifactEvaluator`. The teacher budgets distinguish the trained policies; arena play uses the same 128-simulation budget on both sides.

Arena settings: one worker, c_puct 1.25, deterministic root policy, zero temperature and noise, FPU zero, no subtree reuse or value normalization, and zero tactical root bias. Native exact-root threshold 16 uses final-score-margin objective and highest legal network prior then lowest move index for ties. Exact-leaf solving is disabled; openings use zero random plies, a 200-ply cap, and seed `545 + opening_index` under `azlite_eval_seed_v2`.

The preregistered screen contains 1,024 indexed seat-swapped opening pairs (2,048 new games) from the reused seed511 v4 suite. Every indexed opening is retained and replayed. No historical game or outcome is imported. Canonical-state and production-float32 overlap is reported against both #554 training derivatives and the unchanged development membership.

The unresolved seed461 continuation-overlap limitation remains. This result is exploratory and does not establish superiority to serving P or authorize deployment. A positive result permits only proposing a separate comparison against serving P.

Analysis scores T512 wins as 1, terminal draws as 0.5, and losses as 0, averaging both seats within each opening. It uses 20,000 opening-cluster bootstrap resamples with replacement, seed 545, NumPy linear quantiles, a one-sided 95% lower bound at 0.05, and a descriptive interval at 0.025/0.975. Only 2,048 valid terminal games with lower bound strictly above 0.50 permit `propose_T512_vs_serving_P_screen`; a complete screen otherwise yields `do_not_advance_teacher_budget_branch`. A capped/nonterminal game blocks a valid advancement decision.

The #554 published full-replay receipt, search ledger, and derivative shards are hash-bound through `training-input-binding.json`. The repository tablebase LFS pointer identity is distinct from the separately materialized tablebase object SHA256 and byte size.
