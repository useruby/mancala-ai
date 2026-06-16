# AlphaZero-Lite Control EP2 PUCT Scope Ablation Results

**Date**: 2026-06-16
**Classification**: `broader_scope_destructive`
**Schema**: `azlite_control_ep2_puct_scope_ablation_v1`

## Summary

Tested whether the fixed PR #117 capped PUCT replay can safely update more than the policy head without breaking the large-suite disadvantaged-seat breakthrough.

Result: no broader-scope lane was safe.

- `puct_policy_head_e2_ref` large `384:256` DS: `-0.0091`
- Best `last_block_policy` large `384:256` DS: `-0.3698`
- Best `all` large `384:256` DS: `-0.1901`
- Best broader-scope large `1200:1200` DS improvement vs `puct_policy_head_e2_ref`: `+0.2135`
- High-search gate signature was only run for `canonical_ref` and `puct_policy_head_e2_ref`; broader scopes missed the large `384:256` gate entry condition by a wide margin.

Decision: keep policy-head-only training as the safer AlphaZero-lite update path for this replay recipe.

## Inputs

| Item | Path | SHA256 / rows |
|------|------|----------------|
| canonical control checkpoint | `/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz` | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` |
| canonical control artifact | `/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` |
| PR #117 capped PUCT replay | `/tmp/azlite_control_ep2_puct_smoke/control_ep2_puct_selfplay.jsonl` | `045287417b1878662ba51092bf9c770c66f9751a686b2bdcec4456ad4f521393`, rows `12037` |
| PR #117 policy-head e2 artifact | `/tmp/azlite_control_ep2_puct_smoke/artifacts/puct_policy_head_e2` | `40d6db6d9f63074f8e0ac4247bda893c051834cd00f885b14473a5d6c712a35f` |
| PR #117 policy-head e2 checkpoint | `/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e2/checkpoint_epoch2.npz` | `6e13745f102e097d2d7215e4ed642324345b18ec4fcc2dfd87670863fa8b7a9c` |
| generic bootstrap replay | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | rows `9589` |
| random teacher replay | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | rows `2016` |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |

Replay verification:

- PUCT replay hash matched PR #117 exactly.
- Replay duplicate trajectories: `11`
- PR #117 policy-head artifact was reused; it was not retrained.

## Training Lanes

All trained lanes used:

- model type `residual_v3`
- input encoding `kalah_v3`
- hidden sizes `96,3`
- batch size `512`
- seed `42`
- LR `1e-5`
- LR scheduler `none`
- value loss `huber`
- value loss weight `0.3`
- grad clip `1.0`
- policy target mode `sharpened`
- value target mode `sharpened`
- replay weights `4,1,1`

| Candidate | Scope | Epochs | Policy loss | Value loss | Validation loss | Checkpoint SHA256 | Artifact weights SHA256 | Delta norm | Relative delta |
|------|------|------:|------:|------:|------:|------|------|------:|------:|
| `canonical_ref` | `none` | 0 | n/a | n/a | n/a | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` | `0.0000` | `0.0000%` |
| `puct_policy_head_e2_ref` | `policy_head` | 2 | n/a | n/a | n/a | `6e13745f102e097d2d7215e4ed642324345b18ec4fcc2dfd87670863fa8b7a9c` | `40d6db6d9f63074f8e0ac4247bda893c051834cd00f885b14473a5d6c712a35f` | `0.0351` | `0.1305%` |
| `puct_last_block_policy_e1` | `last_block_policy` | 1 | `1.032433` | `0.205208` | `1.137620` | `acc24b0d2fb15d880bcd08a896618951287c12ef4d6aef9ad1cec584822ab939` | `36a7749a2de4e79410edbbeaf686ef4624f9aaa4a7f8f7e46bddcae48d28c4ed` | `0.0272` | `0.1012%` |
| `puct_last_block_policy_e2` | `last_block_policy` | 2 | `1.031663` | `0.205960` | `1.137134` | `7c5e67b4e6df387c20c0d5c22ec41c81b021ef018ea20afbb3dd5edf30bb5410` | `d786649dd578cf7db1c43fe6d554ddbeee002d6b60f8d55f9ce3f2938d2c6a71` | `0.0469` | `0.1744%` |
| `puct_all_e1` | `all` | 1 | `1.030745` | `0.205218` | `1.136123` | `486f464d7eb0ccb9fcdea4f1a2e1728cc2dc52c7c13cb467ea586ca8035553ea` | `d338bf73e874ab9b2ed7944431a288aa234929f0442b2977a17211ce2deb6d0f` | `0.0356` | `0.1323%` |
| `puct_all_e2` | `all` | 2 | `1.027857` | `0.205904` | `1.134685` | `43dc3ca756c389e42953775fe15c563bb911698d11777ad348598ae6eeaff2de` | `2ae58304548650ea80b5a18bcf94649a9ce1b3cf65fa2c526913b5dc01ee95ad` | `0.0576` | `0.2139%` |

## Medium Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories |
|------|------:|------:|------:|------:|------:|------|------:|
| `canonical_ref` | `-0.1719` | `-0.1094` | `-0.2188` | `+0.1133` | `-0.1016` | `0.2734 / 0.4453` | `512` |
| `puct_policy_head_e2_ref` | `+0.0703` | `+0.1328` | `-0.2188` | `+0.0391` | `-0.0625` | `0.2734 / 0.2031` | `512` |
| `puct_last_block_policy_e1` | `-0.4062` | `-0.1484` | `-0.2188` | `+0.3203` | `-0.0625` | `0.2578 / 0.6641` | `512` |
| `puct_last_block_policy_e2` | `-0.4062` | `-0.1094` | `-0.2188` | `+0.3203` | `-0.0625` | `0.2578 / 0.6641` | `512` |
| `puct_all_e1` | `-0.1875` | `-0.2031` | `-0.3281` | `+0.2812` | `-0.0625` | `0.2578 / 0.4453` | `512` |
| `puct_all_e2` | `-0.4219` | `-0.2578` | `-0.3281` | `+0.0625` | `-0.0625` | `0.2578 / 0.6797` | `512` |

## Large Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories | vs `puct_policy_head_e2_ref` 384:256 |
|------|------:|------:|------:|------:|------:|------|------:|------:|
| `canonical_ref` | `-0.1784` | `-0.1120` | `-0.2161` | `+0.0924` | `-0.1901` | `0.2878 / 0.4661` | `1536` | `-0.1693` |
| `puct_policy_head_e2_ref` | `-0.0091` | `+0.0573` | `-0.2161` | `+0.0065` | `-0.1458` | `0.2878 / 0.2969` | `1536` | `0.0000` |
| `puct_last_block_policy_e1` | `-0.3698` | `-0.1562` | `-0.2161` | `+0.2109` | `-0.1458` | `0.2760 / 0.6458` | `1536` | `-0.3607` |
| `puct_last_block_policy_e2` | `-0.3698` | `-0.1120` | `-0.2161` | `+0.2201` | `-0.1458` | `0.2760 / 0.6458` | `1536` | `-0.3607` |
| `puct_all_e1` | `-0.1901` | `-0.2201` | `-0.3438` | `+0.1758` | `-0.1458` | `0.2760 / 0.4661` | `1536` | `-0.1810` |
| `puct_all_e2` | `-0.3815` | `-0.2721` | `-0.3438` | `-0.0039` | `-0.1458` | `0.2760 / 0.6576` | `1536` | `-0.3724` |

Primary large `384:256` ranking:

1. `puct_policy_head_e2_ref` at `-0.0091`
2. `canonical_ref` at `-0.1784`
3. `puct_all_e1` at `-0.1901`
4. `puct_last_block_policy_e1` at `-0.3698`
5. `puct_last_block_policy_e2` at `-0.3698`
6. `puct_all_e2` at `-0.3815`

Notable pattern:

1. Broader scopes did recover large `1200:1200` strength strongly versus the policy-head baseline.
2. That recovery came with a severe collapse on the primary disadvantaged-seat `384:256` metric.
3. `last_block_policy` was the worst tradeoff: it improved high-search `1200:1200` but catastrophically hurt disadvantaged-seat robustness.
4. `all` was also unsafe: `e1` nearly reverted to canonical on large `384:256`, and `e2` regressed much further.

## Default Deterministic Gate

Gate rule for this task: run `canonical_ref`, `puct_policy_head_e2_ref`, and any new candidate within `0.01` DS of `puct_policy_head_e2_ref` or better on large `384:256`.

Only the two references qualified.

| Candidate | Gate status | Classification | high-search signature preserved |
|------|------|------|------|
| `canonical_ref` | run | `high_search_breakthrough` | yes |
| `puct_policy_head_e2_ref` | run | `high_search_breakthrough` | yes |
| `puct_last_block_policy_e1` | not run | missed entry threshold | no claim |
| `puct_last_block_policy_e2` | not run | missed entry threshold | no claim |
| `puct_all_e1` | not run | missed entry threshold | no claim |
| `puct_all_e2` | not run | missed entry threshold | no claim |

## Decision

### `broader_scope_destructive`

No broader-scope candidate met the safe-update rule, and multiple broader-scope candidates lost far more than `0.03` DS versus `puct_policy_head_e2_ref` on large `384:256`.

- `puct_last_block_policy_e1` delta vs policy-head baseline on large `384:256`: `-0.3607`
- `puct_last_block_policy_e2` delta vs policy-head baseline on large `384:256`: `-0.3607`
- `puct_all_e1` delta vs policy-head baseline on large `384:256`: `-0.1810`
- `puct_all_e2` delta vs policy-head baseline on large `384:256`: `-0.3724`

Safe-update checks that failed:

1. No `puct_all` candidate matched `puct_policy_head_e2_ref` within `0.01` DS on large `384:256`.
2. No `last_block_policy` candidate matched `puct_policy_head_e2_ref` within `0.01` DS on large `384:256`.
3. Because the broader-scope candidates were far below the gate-entry threshold, none established preserved high-search gate behavior under the requested rule.

Conclusion: `policy_head` remains the safer PR #117-style update scope for this replay source.

## Promotion

Not run.

Reason: this task explicitly disallows promotion.

## Verification

| Check | Result |
|------|--------|
| `ruff check ml/alphazero_lite script/ai` | PASS |
| `ruff check ml/alphazero_lite/run_control_ep2_puct_scope_ablation.py` | PASS |
| `python -m unittest ml.alphazero_lite.test_run_manifest` | PASS |
| scope ablation experiment | PASS: `/tmp/azlite_control_ep2_puct_scope_ablation/summary_metrics.json` |

Note: the first full run hit the inherited 2-hour benchmark timeout on the large suite. The runner default was raised to `14400` seconds so the suggested command can complete with the same fixed inputs and no new self-play.

## Guardrails

| Guardrail | Status |
|------|------|
| no new self-play generation | PASS |
| reused fixed PR #117 replay | PASS |
| no classic-MCTS added-data continuation | PASS |
| no residual_v4 | PASS |
| no architecture change | PASS |
| no seed sweep | PASS |
| no LR or replay-weight tuning | PASS |
| no promotion | PASS |
| no overwrite of `model-artifact/current` | PASS |

## Artifacts

| Artifact | Path |
|------|------|
| experiment root | `/tmp/azlite_control_ep2_puct_scope_ablation` |
| summary metrics | `/tmp/azlite_control_ep2_puct_scope_ablation/summary_metrics.json` |
| medium eval report | `/tmp/azlite_control_ep2_puct_scope_ablation/eval_medium/temperature_benchmark_report.json` |
| large eval report | `/tmp/azlite_control_ep2_puct_scope_ablation/eval_large/temperature_benchmark_report.json` |
| gate reports | `/tmp/azlite_control_ep2_puct_scope_ablation/eval_gate` |
