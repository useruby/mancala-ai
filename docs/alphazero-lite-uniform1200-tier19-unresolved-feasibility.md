# Uniform1200 Tier-19 Unresolved Feasibility

## Scope

This oracle-engineering feasibility run evaluated the immutable PR #292 unresolved
states with a newly generated canonical KVTB1 tier-19 tablebase. It did not modify
the 1,190-state cohort, forensic anchors, the PR #292 artifact, exact labels, MCTS,
training, self-play, or promotion state.

## Inputs

| Input | Value |
| --- | --- |
| Frozen cohort SHA-256 | `2ae5376b06e26ec382e31b4ea9bad49a21dadd055891a863c407fde9fbe164ce` |
| PR #292 baseline | 248 solved, 24 timeout, 918 error |
| Native probe SHA-256 | `c93e3fc532b58ca83da21e69941ff2a2d1a9cf58c161fd32b5b845edcc5c4973` |
| Tier-18 SHA-256 | `6e65399387f4b9c13bd83568c60a35fbe474a6bdb8ec868002b783eb0abb02ac` |
| Tier-19 SHA-256 | `53cf399e8a00e0eb4bbd9d7ae20d528f6e2d91f9350269047de49d2c4a4099fd` |
| Tier-19 generation | 282,241,050 states; 282,241,450 bytes; 254.5s wall; 976 MiB peak RSS |
| Request limit | 120 seconds |
| Workers | 4 deterministic primary-anchor shards |

## Result

The tier-19 runner completed all 942 PR #292 unresolved nonterminal rows. It
solved 889 and timed out on 53. The source artifact and its attempt histories are
unchanged; all new labels and native metrics are in the separate feasibility JSON.

| Radius | Combined exact solved | States | Coverage |
| --- | ---: | ---: | ---: |
| 0 | 32 | 38 | 84.2% |
| 1 | 184 | 199 | 92.5% |
| 2 | 921 | 953 | 96.6% |
| Total | 1,137 | 1,190 | 95.5% |

Tier 19 materially improved radius-2 coverage but did not meet the original
radius-0/1 99% acceptance gate. No scientific forensic classification was run,
because the original coverage gates remain unmet.

## Artifact

`docs/data/alphazero-lite-uniform1200-tier19-unresolved-feasibility.json`

## Next Experiment

Evaluate canonical tier 20 on the remaining 53 frozen unresolved states under the
same 120-second request limit.
