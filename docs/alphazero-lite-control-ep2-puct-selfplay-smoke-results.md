# AlphaZero-Lite Control EP2 PUCT Self-Play Smoke Results

**Date**: 2026-06-16
**Classification**: `puct_policy_improvement_signal` + `keep_puct_data_source_for_full_network_followup`
**Schema**: `azlite_control_ep2_puct_selfplay_smoke_v1`

## Summary

Tested the hypothesis that canonical `control_ep2` can generate useful AlphaZero-style PUCT self-play targets when the replay is produced by the current network inside PUCT rather than by classic MCTS.

Result: the smoke test passed the replay audit and both policy-head-only PUCT lanes beat canonical `control_ep2` on the primary large-suite `384:256` disadvantaged-seat score against the fixed `current` opponent.

- Canonical `control_ep2` large `384:256` DS: `-0.1784`
- `puct_policy_head_e1` large `384:256` DS: `-0.0208`
- `puct_policy_head_e2` large `384:256` DS: `-0.0091`
- Best delta vs canonical on large `384:256`: `+0.1693`
- Replay audit classification: `puct_policy_improvement_signal`
- Default gate classification: `high_search_breakthrough` for canonical, `e1`, and `e2`

This does not promote anything, but it does clear the requested smoke-test decision bar for a later full-network follow-up.

## Inputs

| Item | Path | SHA256 / rows |
|------|------|----------------|
| canonical control checkpoint | `/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz` | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` |
| canonical control artifact | `/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` |
| current artifact | `model-artifact/current` | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| generic bootstrap replay | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | rows `9589` |
| random teacher replay | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | rows `2016` |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |

## Replay Generation

Command used:

```bash
.venv/bin/python ml/alphazero_lite/run_control_ep2_puct_selfplay_smoke.py \
  --workdir /tmp/azlite_control_ep2_puct_smoke \
  --control-checkpoint /tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz \
  --control-artifact /tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2 \
  --current model-artifact/current \
  --generic-bootstrap /tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl \
  --random-teacher /tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl \
  --medium-suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --large-suite /tmp/azlite_opening_suite/large_eval.jsonl \
  --games 512 \
  --simulations 384 \
  --max-positions-per-game 24 \
  --workers 24 \
  --seed 42
```

Generation settings:

- player mode `puct`
- simulations `384`
- `c_puct=1.25` (repo default)
- root policy mode `visit_count`
- policy target mode `sharpened`
- value target mode `sharpened`
- input encoding `kalah_v3`
- model type `residual_v3`
- hidden sizes `96,3`

Replay files:

| File | Path | SHA256 | Rows |
|------|------|--------|------:|
| raw replay | `/tmp/azlite_control_ep2_puct_smoke/control_ep2_puct_selfplay_raw.jsonl` | `eed6ec0a48f36871980f6a256c5c7142c38f928d8247e95713ed941b0e5855d2` | `19018` |
| capped replay | `/tmp/azlite_control_ep2_puct_smoke/control_ep2_puct_selfplay.jsonl` | `045287417b1878662ba51092bf9c770c66f9751a686b2bdcec4456ad4f521393` | `12037` |

Cap summary:

- games: `512`
- average uncapped positions/game: `37.1445`
- average kept positions/game: `23.5098`

## Replay Audit

Audit path: `/tmp/azlite_control_ep2_puct_smoke/puct_replay_audit.json`

| Metric | Value |
|------|------:|
| row count | `12037` |
| unique rows | `9403` |
| completed games | `512 / 512` |
| average game length | `37.1445` |
| policy entropy mean / p50 / p90 | `0.4167 / 0.0020 / 1.3704` |
| legal mask validity | `PASS` (`0` invalid rows) |
| value target range | `PASS` (`0` invalid rows) |
| one-hot policy fraction | `0.3792` |
| near-one-hot policy fraction | `0.5766` |
| duplicate state count / rate | `2634 / 0.2188` |
| duplicate trajectory count / rate | `11 / 0.0217` |
| root visit total mean / p50 / p90 | `384 / 384 / 384` |
| search profile hash | `e68fea93ce5ba6755c8b3a57f284e89ccbc07b57ea68d2668e6920b7fd463ae0` |

Top-1 move distribution by pit:

| Pit | Fraction |
|----:|---------:|
| 0 | `0.1080` |
| 1 | `0.1318` |
| 2 | `0.1758` |
| 3 | `0.1313` |
| 4 | `0.2003` |
| 5 | `0.2529` |

Value target summary:

- mean: `0.0304`
- std: `0.5578`
- win/loss/draw: `0.4826 / 0.4121 / 0.1053`

First-1000 raw-policy comparison:

- top-1 agreement rate: `0.6400`
- mean `KL(search_policy || raw_policy)`: `0.6933`
- mean `KL(raw_policy || search_policy)`: `5.9569`
- changed-top-move fraction: `0.3600`
- changed-top-move by phase: opening `0.3099`, mid `0.4933`, late `0.2426`
- classification: `puct_policy_improvement_signal`

Audit verdict: training proceeded. The requested abort conditions were not triggered.

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
- trainable scope `policy_head`
- replay weights `4,1,1`

| Candidate | Epochs | Checkpoint SHA256 | Artifact weights SHA256 | Delta norm vs canonical | Relative delta | Policy loss | Value loss | Validation loss |
|------|------:|------|------|------:|------:|------:|------:|------:|
| `canonical_ref` | 0 | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` | `0.0000` | `0.0000%` | n/a | n/a | n/a |
| `puct_policy_head_e1` | 1 | `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357` | `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece` | `0.0217` | `0.0806%` | `1.033082` | `0.205252` | `1.138135` |
| `puct_policy_head_e2` | 2 | `6e13745f102e097d2d7215e4ed642324345b18ec4fcc2dfd87670863fa8b7a9c` | `40d6db6d9f63074f8e0ac4247bda893c051834cd00f885b14473a5d6c712a35f` | `0.0351` | `0.1305%` | `1.032920` | `0.206054` | `1.137924` |

## Medium Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories |
|------|------:|------:|------:|------:|------:|------|------:|
| `canonical_ref` | `-0.1719` | `-0.1094` | `-0.2188` | `+0.1133` | `-0.1016` | `0.2734 / 0.4453` | `512` |
| `puct_policy_head_e1` | `+0.0547` | `+0.1328` | `-0.2188` | `+0.0938` | `-0.0625` | `0.2578 / 0.2031` | `512` |
| `puct_policy_head_e2` | `+0.0703` | `+0.1328` | `-0.2188` | `+0.0391` | `-0.0625` | `0.2734 / 0.2031` | `512` |

## Large Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories | vs canonical 384:256 |
|------|------:|------:|------:|------:|------:|------|------:|------:|
| `canonical_ref` | `-0.1784` | `-0.1120` | `-0.2161` | `+0.0924` | `-0.1901` | `0.2878 / 0.4661` | `1536` | `0.0000` |
| `puct_policy_head_e1` | `-0.0208` | `+0.0573` | `-0.2161` | `+0.0703` | `-0.1458` | `0.2760 / 0.2969` | `1536` | `+0.1576` |
| `puct_policy_head_e2` | `-0.0091` | `+0.0573` | `-0.2161` | `+0.0065` | `-0.1458` | `0.2878 / 0.2969` | `1536` | `+0.1693` |

Primary ranking on large `384:256`:

1. `puct_policy_head_e2` at `-0.0091`
2. `puct_policy_head_e1` at `-0.0208`
3. `canonical_ref` at `-0.1784`

Notable nuance:

1. The PUCT lanes improved the primary disadvantaged-seat metric very strongly.
2. They also improved `256:768` versus canonical.
3. They did not improve the equal-high `1200:1200` budget versus canonical.

## Default Deterministic Gate

Gate rule for this task: run canonical plus any PUCT candidate that tied or beat canonical on large `384:256`.

All three qualified.

| Candidate | Classification | high-search signature preserved |
|------|------|------|
| `canonical_ref` | `high_search_breakthrough` | yes |
| `puct_policy_head_e1` | `high_search_breakthrough` | yes |
| `puct_policy_head_e2` | `high_search_breakthrough` | yes |

## Decision

### `keep_puct_data_source_for_full_network_followup`

The requested smoke-test acceptance rule was satisfied.

1. `puct_policy_head_e2` large `384:256` DS was `-0.0091`.
2. Canonical `control_ep2` large `384:256` DS was `-0.1784`.
3. The best PUCT lane beat canonical by `+0.1693` on the primary metric.
4. The gated candidate preserved the `high_search_breakthrough` signature.

### `puct_policy_improvement_signal`

The replay audit showed the PUCT targets were meaningfully different from the raw network policy rather than collapsing back to it.

1. Mean `KL(search_policy || raw_policy)` was `0.6933`.
2. `36%` of the first 1,000 rows changed the raw-policy top move.
3. Midgame rows had the largest changed-top-move rate at `0.4933`.

## Promotion

Not run.

Reason: this task is a smoke test only and explicitly disallows promotion.

## Verification

| Check | Result |
|------|--------|
| `ruff check ml/alphazero_lite script/ai` | PASS |
| `ruff check ml/alphazero_lite/self_play.py ml/alphazero_lite/run_control_ep2_puct_selfplay_smoke.py` | PASS |
| `python -m unittest ml.alphazero_lite.test_run_manifest` | PASS |
| smoke experiment | PASS: `/tmp/azlite_control_ep2_puct_smoke/summary_metrics.json` |

## Guardrails

| Guardrail | Status |
|------|------|
| no classic-MCTS added-data continuation | PASS |
| no full-network training | PASS |
| no new architecture | PASS |
| no `residual_v4` | PASS |
| no seed sweep | PASS |
| no promotion | PASS |
| no overwrite of `model-artifact/current` | PASS |
| replay-audit failure not hidden | PASS |

## Artifacts

| Artifact | Path |
|------|------|
| experiment root | `/tmp/azlite_control_ep2_puct_smoke` |
| summary metrics | `/tmp/azlite_control_ep2_puct_smoke/summary_metrics.json` |
| replay audit | `/tmp/azlite_control_ep2_puct_smoke/puct_replay_audit.json` |
| capped replay | `/tmp/azlite_control_ep2_puct_smoke/control_ep2_puct_selfplay.jsonl` |
| medium eval report | `/tmp/azlite_control_ep2_puct_smoke/eval_medium/temperature_benchmark_report.json` |
| large eval report | `/tmp/azlite_control_ep2_puct_smoke/eval_large/temperature_benchmark_report.json` |
| gate reports | `/tmp/azlite_control_ep2_puct_smoke/eval_gate` |
