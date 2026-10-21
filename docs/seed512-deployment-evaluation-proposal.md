# Seed512 deployment and evaluation proposal

## Configuration and cost

P is the seed504 E4 policy output and its own trunk paired with the frozen initializer value output and its own trunk; no tensor splicing or training. The opponent is the complete frozen initializer. Search uses 1,200 simulations per side, one worker, deterministic root policy, `c_puct=1.25`, the registered exact-root behavior (threshold 16; final-score-margin objective; network-prior then lowest-move tie break), and no exact leaf solve. Inference computes policy and value through two separate networks/trunks per search evaluation; relative to a single shared-trunk model this duplicates trunk inference (two forward passes and two trunk activations per node), plus each head. Measure end-to-end latency and memory before rollout.

## Evidence and limits

On the examined 1,024-opening/2,048-game suite, seed512 records 947 wins, 259 draws, 842 losses (score 0.52563; one-sided 95% paired-opening bootstrap lower bound 0.51489). The matched 1,200-minus-128 per-opening contrast is 0.00659 with descriptive interval [-0.01318, 0.02637]. These support further evaluation only: seed511's 128-simulation results were already examined, the same openings were reused, and this is not fresh confirmation. The seed461 overlap limitation and separate seed505 failed gate remain in force.

## Smallest integration and rollback

Add a versioned P configuration selecting the two existing component checkpoints and explicit separate policy/value model instances; do not modify checkpoint contents or search defaults. Gate behind a runtime configuration switch, expose candidate-vs-baseline shadow inference first, and monitor latency, memory, errors, legal-move behavior, and score telemetry. Roll back by switching the configuration to the existing initializer model; retain both immutable artifacts and do not overwrite the baseline.

## Before promotion

Run a preregistered, independently held-out opening suite with balanced seats and a prospectively frozen analysis plan; include multiple seeds or an adequately powered sample, report confidence intervals and subgroup/opening sensitivity, and compare at matched compute budgets. Benchmark production hardware for p50/p95 latency, throughput, memory, and reliability. Verify artifact provenance and build/runtime reproducibility; assess operational rollback readiness. Promotion requires the predeclared strength gate and production SLOs to pass.

## Provenance limitation

The seed511 bindings authenticate recorded assets/runtime sources where identities were recorded. Seed512 does not prospectively freeze every transitive imported Python source dependency; no retrospective identity is asserted for those files.
