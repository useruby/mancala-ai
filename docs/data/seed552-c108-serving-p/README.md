# Seed552: C108 versus frozen serving P

Exploratory higher-budget evaluation of the #550 initializer-self-play control C, update 108, against serving P (historical seed504 E4). C is a newly registered exploratory candidate; #550's candidate designation and #551's `do_not_advance_source_evaluator_branch` decision are preserved. The C-versus-T result does not imply strength against P.

The screen uses exactly 1,024 seat-swapped pairs (2,048 new games) from the indexed seed511 v4 suite. Every opening is replayed and retained. Canonical-state and production-float32 overlap is reported against C training/development and P policy-training populations. The suite is reused, the limitation of unresolved seed461 continuation overlap remains, and no historical game/outcome is imported.

C policy and trunk are C108; P policy and trunk are historical E4; both values and value trunks are the frozen initializer through `ComposedArtifactEvaluator`. Each exported component retains its own trunk. Serving release component and sidecar identities are authenticated. The tablebase's seed551 historical LFS pointer identity and current repository/materialized object bytes are bound separately.

The preregistered analysis gives C wins 1, terminal draws 0.5 and losses 0, averaging both seats within each opening. It uses 20,000 opening-cluster bootstrap resamples with replacement, seed 545, NumPy linear quantiles, one-sided 95% lower bound at 0.05, and descriptive 95% interval at 0.025/0.975. Any capped/nonterminal game blocks a valid advancement decision. No result authorizes promotion or deployment.
Two durable-runner invocations failed the pre-game runtime gate because systemd's `AllowedCPUs=30` did not restrict process scheduler affinity. A separate no-game runtime diagnostic observed affinity 0-31; `taskset -c 30` is now included in the frozen launcher. Both failed attempts and the diagnostic are preserved in `execution-attempts.json`; no arena games started.
