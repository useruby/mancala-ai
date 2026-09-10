# Uniform1200 Tier-20 Unresolved Feasibility

## Scope

This evaluation-only experiment changed one variable from PR #293: canonical
tablebase tier 19 to tier 20. It did not train a model, generate self-play,
modify replay, promote a checkpoint, alter the frozen 1,190-state cohort, or
overwrite PR #292/#293 artifacts.

## Inputs And Generation

| Input | Value |
| --- | --- |
| Frozen cohort SHA-256 | `2ae5376b06e26ec382e31b4ea9bad49a21dadd055891a863c407fde9fbe164ce` |
| Tier-20 request cohort SHA-256 | `171c16b86e722d0d7208b3ad6110de49c0877163b7df75330507080c6bbbf5d7` |
| Request cohort | 53 PR #293 tier-19 timeout rows only |
| Native probe SHA-256 | `c93e3fc532b58ca83da21e69941ff2a2d1a9cf58c161fd32b5b845edcc5c4973` |
| Request limit / workers | 120 seconds / 4 deterministic primary-anchor shards |
| Generation command | `.tmp/kalah_v1_tablebase_build/kalah_v1_tablebase generate 20 .tmp/kalah_v1_20.kvtb` |
| Tier-20 SHA-256 | `6af75fd010bd3a554befe8b5e060a019cb09ea9980af3b943d0033629a19945d` |
| States / bytes | 451,585,680 / 451,586,096 |
| Generation wall / CPU | 429.8 s / 429.2 s |
| Generation peak RSS | 1,564 MiB |

Resource preflight found 1.6 TiB free disk, 42 GiB available memory, and a
524,288 file-descriptor limit. The generated artifact passed the pinned SHA,
KVTB1 magic, and tier-20 header checks. The binary is not committed.

## Tier-20 Result

| Radius | Solved | Timeout | Error |
| --- | ---: | ---: | ---: |
| 0 | 3 | 3 | 0 |
| 1 | 9 | 6 | 0 |
| 2 | 15 | 17 | 0 |
| Total | 27 | 26 | 0 |

The 53-state solve rate was 50.9%. Solved latency was p50 97.92 s, p90
116.19 s, p95 118.54 s, and max 119.40 s. Successful requests accumulated
19,687,005,664 forward nodes, 61,847,264,799 TT probes, 32,945,840 TT hits,
10,878,567,392 tablebase hits, and a maximum tablebase hit tier of 20. Active
stones were 44:2, 45:3, 46:16, 47:28, and 48:4.

For the same 53 states, tier-19 timeout to tier-20 solved was 27 and timeout
to timeout was 26. Search-node reduction is not measurable because tier-19
timeout rows have no completed native metrics. Tier-19 timeout latency p50 was
120.10 s; tier-20 solved latency is reported above.

## Combined Coverage And Decision

| Radius | Exact solved | States | Coverage |
| --- | ---: | ---: | ---: |
| 0 | 35 | 38 | 92.1% |
| 1 | 193 | 199 | 97.0% |
| 2 | 936 | 953 | 98.2% |

The original PR #291 gate remains unmet: it requires 99% at radius 0 and
radius 1, plus 95% at radius 2. Radius 2 passes, but radii 0 and 1 do not.
The gated scientific forensic analysis was therefore not run.

## Hard Classification

`tier20_oracle_effective_but_insufficient`

## Next Experiment

Evaluate tier 21 on only the still-unresolved frozen states.
