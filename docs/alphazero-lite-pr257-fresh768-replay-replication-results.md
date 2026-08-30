# PR #257 Fresh768 Replay Replication Results

**Classification:** `fresh768_not_robust`

## Frozen Contract

- Five preregistered independent self-play replay seeds: 48, 49, 50, 51, and 52.
- Each used 700 games, 24 workers, ordinary reused-tree A16 gameplay at 384
  simulations, `c_puct=1.25`, visit-count root policy, zero FPU, unnormalized
  values, and Kalah v3.
- The only two policy-target views were the authoritative reused gameplay tree
  and a rootless, non-reused 768-simulation fresh search. No other target
  budget, semantic rule, shadow Q, or checkpoint selection was used.
- A16, initial Adam, P1, and evaluator hashes all matched the frozen contract.
- All ten step-16 candidate hashes were frozen before S/T/U generation.

## Replay And Suite Invariants

- Every reused/fresh768 pair had matching policy-excluded state/outcome SHA,
  row order, metadata, exclusions, and deterministic batch plan.
- Eligible rows were 27,658 (seed48), 27,877 (seed49), 28,198 (seed50),
  27,299 (seed51), and 27,896 (seed52).
- Repeated fresh768 lanes and reversed lane order were byte-identical for every
  replay seed. The pristine Adam SHA remained unchanged; every step-16 lane had
  fit fraction 1.0.
- S/T/U were sealed at seeds 19042/20042/21042 with SHAs
  `59224ca54c9102363cc00e50d424497485e13372909e59e0454a2ce2ae31f619`,
  `e73cb13277168cf0d449f942525d16dd032cdd8e396393f598ac35af612526dd`, and
  `efcab1d3aecff06c223245a1b03f5679c2038168de6b4c46ffa557385dcf349a`.
  Preflight found zero final-state, prefix, or seed48-52 replay-state overlap.

## Target Diagnostics

Across replay seeds, reused versus fresh768 mean legal L1 was .118236
(range .116823-.119536), mean JS .020737 (range .020457-.021019), and top-1
disagreement .062992 (range .061577-.063851). These descriptive target-space
differences did not translate into a measurable model-strength difference.

## Primary 1200:1200 Arena

All arenas used ordinary deterministic 1200:1200 PUCT, zero FPU, unnormalized
values, seat swapping, arena seed 42, and one matched P1-vs-P1 control per
suite. S/T/U were pooled within replay seed before replay-seed analysis.

| Replay seed | S | T | U | delta fresh768 - reused | 95% CI |
| --- | ---: | ---: | ---: | --- |
| 48 | 0 | 0 | 0 | 0 | [0, 0] |
| 49 | 0 | 0 | 0 | 0 | [0, 0] |
| 50 | 0 | 0 | 0 | 0 | [0, 0] |
| 51 | 0 | 0 | 0 | 0 | [0, 0] |
| 52 | 0 | 0 | 0 | 0 | [0, 0] |

The replay-seed mean, median, SD, min, and max were all 0. The hierarchical
replay-seed -> suite -> opening 95% CI was [0, 0], with 0/5 positive replay
seeds. The primary success rule therefore fails conditions A, B, and C.

## Secondary 384:256 Arena

After the primary results were frozen, the same ten candidates were evaluated
on S/T/U at 384:256. Every replay-seed contrast was also 0, giving mean 0 and
hierarchical CI [0, 0]. There is no material shallow-search harm, but the
primary high-budget criterion did not pass.

## Compute And Decision

- Reused generation: 384 gameplay simulations/move.
- Fresh768: 384 gameplay + 768 target simulations/move = 1,152 = 3.0x baseline.
- Five replays generated 140,618 authoritative states and consumed 161,991,936
  search simulations under the fresh768 accounting.

Fresh768 did not improve mean strength or reduce absolute-strength variance:
both lane variances were .0002471924. Historical seeds45-47 remain context
only and were excluded from the primary CI and decision. Close the separate
target-search branch; do not promote fresh768. The next experiment should
return to ordinary targets and focus on replay aggregation, iteration count,
or model capacity.

## Historical Context Only

These selected historical replays are not primary replicates and do not enter
any estimate or confidence interval above.

| Replay seed | Comparable fresh768 vs reused context |
| --- | --- |
| 45 | Approximately equal measured 1200 strength |
| 46 | Approximately equal measured 1200 strength |
| 47 | Fresh768 materially stronger on P/Q/R |

## Frozen Artifacts

- Frozen manifest SHA-256:
  `6c44a90123e3854a65bf7a3a509b50ce719a08cdf9078cce4942f0060d6d4b4b`
- Primary 1200 result SHA-256:
  `ee37bb762f71554fceb5f9c499bc35d0e1ae0b7e87473a4ddcdad58087069cbf`
- Preflight SHA-256:
  `25d78a61f45ed2cf1f10e6a6b960f3871f67b1a872efebea93c1359d7d97a023`
