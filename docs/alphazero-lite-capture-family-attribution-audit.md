# Capture Family Attribution Audit

## Classification

`capture_regression_search_induced`

The decision-critical separation is produced by production PUCT in seeds 44 and 45: both raw students choose the same non-optimal action, but only the control search repairs it. Seed 46 is the negative control: production PUCT repairs both. This is not evidence that uniform1200 supplied worse policy targets or that its raw student alone introduced the regression.

Exactly one next experiment: a capture-focused PUCT diagnostic varying **root FPU only**, with all checkpoints, replay, oracle, and other search settings frozen.

The programmatic mechanism precedence is recorded in `mechanisms`. It applies a category only when uniform searched regret is strictly worse than matched control; seed-46 improvements are retained as the negative control rather than counted as failures. The cross-seed dominant mechanism is `inference_search_degradation`: it appears in seed 44 (`capture_available-002`) and seed 45 (`capture_available-002`, `capture_available-020`).

## Verification

The audit is evaluation-only: no training, self-play, replay mutation, promotion, or tablebase generation occurred. The machine-readable result is `docs/data/alphazero-lite-capture-family-attribution-audit.json`.

All six PR #287/#297 primary checkpoints were SHA-verified against the recorded hashes before evaluation. The artifact records checkpoint, replay, exact-reference, and exact-shadow paths and SHA256 values. The primary exact capture population is 24/24 solved positions.

## Critical Set

The set is mechanically extracted from the PR #297 exact-shadow `genuine_regressions`: a capture row must have positive uniform-minus-control exact regret in at least two seeds. The resulting set is:

`capture_available-002`, `capture_available-020`

Its ordered-ID SHA256 is `58c7ca4677136570b7e5154b1bf3d531772839bd6a66117e6beab84e3f883c19`.

## Raw Versus Search

Production forensic search uses the existing deterministic evaluation profile: 384 simulations, `c_puct=1.25`, zero FPU, no value transform, deterministic root selection, no tactical bias, and seed 42. The result retains legal-masked raw policies, raw values, root priors, visits, and child Q values for every primary checkpoint/state.

The full 24-state by-checkpoint table is the `evaluations` collection in the machine result. Each row includes its raw/search regrets and can be aggregated without replacing the frozen positions.

| State | Seed | Control raw/search regret | Uniform raw/search regret | Result |
| --- | ---: | ---: | ---: | --- |
| capture_available-002 | 44 | 12 / 0 | 12 / 12 | control-only PUCT repair |
| capture_available-002 | 45 | 12 / 0 | 12 / 12 | control-only PUCT repair |
| capture_available-002 | 46 | 12 / 0 | 12 / 0 | both repaired |

## `capture_available-002` Trace

Exact action values are `{0: 2, 1: -4, 2: -16, 3: -10, 4: -6}` for player 1, so the exact optimal set is `{2}`.

| Seed | System | Raw top/regret | Search selected/regret | Root child Q and visits `(move: Q/visits)` |
| ---: | --- | --- | --- | --- |
| 44 | control | 1 / 12 | 2 / 0 | 0: -.012/41, 1: -.013/121, 2: .043/167, 3: .000/24, 4: .005/31 |
| 44 | uniform | 1 / 12 | 1 / 12 | 0: .101/19, 1: .105/239, 2: .075/89, 3: .051/8, 4: .099/29 |
| 45 | control | 1 / 12 | 2 / 0 | 0: .020/50, 1: .014/110, 2: .080/165, 3: .022/28, 4: .029/31 |
| 45 | uniform | 1 / 12 | 1 / 12 | 0: .067/23, 1: .167/218, 2: .201/108, 3: .151/15, 4: .159/20 |
| 46 | control | 1 / 12 | 2 / 0 | 0: -.027/28, 1: .003/109, 2: .041/183, 3: .005/28, 4: -.008/36 |
| 46 | uniform | 1 / 12 | 2 / 0 | 0: -.028/17, 1: .006/122, 2: .097/199, 3: .050/15, 4: .073/31 |

The retained dynamic replay gives `capture_available-002` control/uniform occurrence counts of 1/1 (seed 44), 3/1 (seed 45), and 1/0 (seed 46). For seeds 44 and 45, the uniform stored target places 0.9679 mass on exact action 2 versus the control target's 0.5461. The target is therefore better, while both raw students still choose action 1; PUCT Q/visit allocation determines whether the error is repaired.

Across exact-labeled dynamic replay rows, uniform expected target regret is lower in every seed: 1.4318 vs 2.6292 (44), 2.1320 vs 3.5172 (45), and 1.3442 vs 3.4898 (46). The exact-labeled row counts are 35/47, 35/41, and 27/44 (uniform/control). The machine result also records fixed-source quality and paired shared-state teacher/student inversion results.

Unique-state paired target deltas (uniform minus control) likewise favor uniform in expected exact regret: -2.6871 over 11 shared states (44), -2.9558 over 8 (45), and -2.0427 over 9 (46). This rules out a systematic target-quality deficit for the repeated `capture_available-002` failure.

## Mechanism Trace

| Seed | Critical state | Control raw/search regret | Uniform raw/search regret | Primary mechanism |
| ---: | --- | ---: | ---: | --- |
| 44 | capture_available-002 | 12 / 0 | 12 / 12 | inference_search_degradation |
| 44 | capture_available-020 | 0 / 0 | 6 / 6 | target_quality_deficit |
| 45 | capture_available-002 | 12 / 0 | 12 / 12 | inference_search_degradation |
| 45 | capture_available-020 | 10 / 4 | 6 / 6 | inference_search_degradation |
| 46 | capture_available-002 | 12 / 0 | 12 / 0 | negative control |
| 46 | capture_available-020 | 6 / 6 | 0 / 4 | negative control |

## Neighborhood Coverage

The runner consumes the already generated PR #291--#296 exact neighborhood cohort without creating any synthetic states. It records unique state coverage, row-weighted occurrences, and fractions for radii 0, 1, and 2 for both all 24 capture anchors and the decision-critical subset. These fields are under `neighborhood_coverage` in the machine result and preserve zero coverage where no retained cohort state maps to an anchor.

## Parent And Secondary Scope

The shared parent snapshot is retained as `parent_init_checkpoint.npz` alongside each primary run and is SHA-verifiable through the run manifests. The runner evaluates parent raw and frozen-production PUCT traces for all 24 capture states, enabling parent/control/uniform comparisons in `parent_evaluations`.

PR #288 rowmatched checkpoints are retained and are SHA-verified through their artifact metadata and recorded weights SHA. The runner evaluates their raw and production-search behavior on the decision-critical set under `secondary.rowmatched`. The recovered PR #290 B/C/D checkpoints are SHA-verified against their historical exact-shadow weights SHA and evaluated under `secondary.pr290`; B maps to `control_like_exposure__unsharpened`, C to `uniform_exposure__unsharpened`, and D to `control_like_exposure__sharpened`.

## Seed 46 Negative Control

Seed 46 differs at the search transition, not raw action: uniform raw still chooses action 1 with regret 12, but its root Q for action 2 is .097 and receives 199 visits, selecting the exact action. In seeds 44/45, uniform selects action 1 despite the raw tie with control. This isolates the decision-critical difference to inference-time search telemetry.
