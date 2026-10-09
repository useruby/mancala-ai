# Seed454 — retrospective full-guard compliance audit

Seed453 was requested to use seed447's original 2,048-input training guard for both policy-KL and value-movement constraints. The executed seed453 runner instead used the 1,399-input fresh subset for both caps and for B's projection, and its corrected verifier repeated that choice. Seed454 audits the archived trajectory; it does not retrain or infer later states for a hypothetical corrected run.

## Verified outcome

**`archived_trajectory_matches_full_guard_rule`**

The full guard contains 2,048 unique training inputs: 1,399 fresh and 649 historical. Across both arms, 16 archived pre-states per arm, and all eight scales (256 trials), none of the 32 recorded accepted states violates the requested minibatch/full-guard KL, value-movement, or B fresh-gradient-dot constraints. Every recorded selected scale equals the largest feasible scale under the requested rule when all eight trials are evaluated without a monotonicity assumption. There is no first divergence.

This establishes retrospective compatibility of the archived trajectory with the full-guard constraints and selection rule. It does not change which guard seed453 used prospectively. The existing `close_fresh_policy_projection_branch` classification is preserved, and no subsequent corrected-run trajectory is inferred.

The superseding audit freezes its sources and inputs after seed453 execution and before seed454 diagnostic forward passes. It reconstructs replay inputs from committed compressed sources using the verified seed450 loader semantics, retaining each row's declared target modes and frozen order/split dependencies. The read-only verifier has an independent prediction and metric implementation.

Detailed per-trial measurements, executed/requested feasibility, maxima, and scale comparisons are in `superseding-v7/audit.json`. Source snapshots and bindings are in `superseding-v7/audit-registration.json`. The initial draft and superseded attempts are preserved with their hashes and failure notes in the adjacent attempt receipts.
