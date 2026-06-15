# AlphaZero-Lite Control EP2 Multi-Iteration Results

**Date**: 2026-06-15
**Classification**: `control_ep2_still_best` + `extra_data_not_helpful` + `destructive_iteration`
**Schema**: `azlite_control_ep2_multi_iteration_v1`

## Summary

Ran the requested fresh multi-iteration follow-up from canonical `control_ep2`, keeping `residual_v3` and the original iter0 replay anchor, and selecting by the fixed deterministic opening-suite seat-aware benchmark rather than validation loss.

Result: no checkpoint beat canonical `control_ep2` on the primary large-suite `384:256` disadvantaged-seat score.

- Canonical `control_ep2` large `384:256` DS: `-0.1784`
- Best continuation control (`origcont_e1`) large `384:256` DS: `-0.1784` (exact tie)
- Best added-data lane (`selfplay_e2`) large `384:256` DS: `-0.1901`
- All non-canonical lanes regressed `1200:1200` versus canonical `control_ep2` (`+0.0924`)

The added classic-MCTS dataset improved validation loss relative to the original-data continuation control, but worsened opening-suite strength. That satisfies the requested `destructive_iteration` condition.

## Recipe Notes

The repo's historically successful alignment is classic MCTS as the data source. For this experiment, the fresh iteration data was generated with `generate_bootstrap_dataset.py --teacher-mode classic_mcts`.

Important implementation nuance:

1. `generate_bootstrap_dataset.py` has no checkpoint/player input.
2. `self_play.py --player-mode classic_mcts` also ignores the neural checkpoint for move generation.
3. So the meaningful "from control_ep2" part of this run is the training initialization checkpoint, not the classic-MCTS data generator.

This matches the repo's established classic-MCTS-aligned recipe without inventing a new hybrid generator.

## Inputs

| Item | Path | SHA256 |
|------|------|--------|
| canonical control checkpoint used for init | `/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz` | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` |
| canonical control artifact | `/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` |
| current artifact | `model-artifact/current` | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| generic bootstrap replay | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | rows `9589` |
| random teacher replay | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | rows `2016` |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |

## New Dataset

Command used:

```bash
.venv/bin/python ml/alphazero_lite/generate_bootstrap_dataset.py \
  --out /tmp/azlite_control_ep2_multi_iter/control_ep2_classic_mcts_selfplay.jsonl \
  --games 800 \
  --simulations 1200 \
  --seed 60 \
  --max-positions-per-game 16 \
  --input-encoding kalah_v3 \
  --teacher-mode classic_mcts \
  --policy-target-mode sharpened \
  --value-target-mode sharpened \
  --workers 8
```

| Metric | Value |
|-------|-------|
| SHA256 | `74b1aa63c026057828983531ad562009ccc7388f3f66757c69ef5d536429a67b` |
| rows written | `12793` |
| games | `800` |
| simulations | `1200` |
| teacher mode | `classic_mcts` |
| player mode | `n/a` in this generator |
| policy target mode | `sharpened` |
| value target mode | `sharpened` |
| phase distribution | early `4229`, mid `5394`, late `3170` |
| duplicate board count | `1880` |
| teacher top-1 visit share mean | `0.8560` |
| policy entropy mean | `0.3790` |

## Training Lanes

All lanes used `residual_v3`, `kalah_v3`, hidden sizes `96,3`, batch size `512`, `value_loss=huber`, `value_loss_weight=0.3`, `grad_clip=1.0`, `seed=42`, `lr_scheduler=none`, `policy_target_mode=sharpened`, `value_target_mode=sharpened`, and init from canonical `control_ep2`.

### Replay Mixes

| Lane | Replay weights | Effective replay share |
|------|----------------|------------------------|
| `origcont_*` | `4,1` | bootstrap `95.01%`, random teacher `4.99%` |
| `selfplay_*` / `selfplay3e6_*` | `4,1,1` | bootstrap `72.15%`, random teacher `3.79%`, new self-play `24.06%` |

### Checkpoint Metrics

| Lane | Epoch | LR | Checkpoint SHA256 | Delta norm vs control_ep2 | Relative delta | Policy loss | Value loss | Validation loss |
|------|------:|----:|-------------------|---------------------------:|---------------:|------------:|-----------:|----------------:|
| `origcont_e1` | 1 | `1e-5` | `bd788c8879a9d8074a2492c36564fdbe103d6064da0fe03ce188c3be3af09706` | `0.0181` | `0.0671%` | `0.996023` | `0.245113` | `1.180300` |
| `origcont_e2` | 2 | `1e-5` | `be0101343e848b02a8a7ec2cb910e0039926f1e6cde8f258b29a4718a3a28a11` | `0.0313` | `0.1161%` | `0.995134` | `0.245070` | `1.179796` |
| `selfplay_e1` | 1 | `1e-5` | `d2118bdcc86ac6562ddcc712d827a20632ff3fa543a1026f3112640ddbdc82ed` | `0.0284` | `0.1056%` | `1.063943` | `0.241345` | `1.175332` |
| `selfplay_e2` | 2 | `1e-5` | `f6dd72ab55af6e7a1d669f4f5915e625f61da099163c018650af9d81a05fbaf7` | `0.0482` | `0.1791%` | `1.061982` | `0.241364` | `1.173831` |
| `selfplay3e6_e1` | 1 | `3e-6` | `881b21d1409de0e5c6cd9cb57448f24157e8f04f933f3cad0b53d00dc23ea0ee` | `0.0112` | `0.0416%` | `1.064281` | `0.241308` | `1.175673` |
| `selfplay3e6_e2` | 2 | `3e-6` | `d8d63e94c215d2dfb97071efa7cdd7acb7b72bd0f6b4e810f6efbd6ffa3a8e98` | `0.0183` | `0.0679%` | `1.063096` | `0.241345` | `1.174631` |

## Medium Opening-Suite Benchmark

Selection metric remained the fixed deterministic seat-aware opening suite, not validation loss.

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | margin mean / median | duplicate trajectories |
|-----------|------------:|------------:|------------:|--------------:|------------:|------------------|----------------------|-----------------------:|
| `canonical_ref` | `-0.1719` | `-0.1094` | `-0.2188` | `+0.1133` | `-0.1016` | `0.2734 / 0.4453` | `-8.2266 / -14.0` | `512` |
| `origcont_e1` | `-0.1719` | `-0.2031` | `-0.0156` | `-0.0547` | `-0.0625` | `0.2734 / 0.4453` | `-6.7969 / -14.0` | `512` |
| `selfplay_e2` | `-0.1875` | `-0.2031` | `-0.2188` | `-0.1094` | `-0.0625` | `0.2578 / 0.4453` | `-6.9766 / -14.0` | `512` |
| `selfplay_e1` | `-0.2031` | `-0.2031` | `-0.2578` | `-0.0703` | `-0.0625` | `0.2578 / 0.4609` | `-6.3359 / -14.0` | `512` |
| `selfplay3e6_e1` | `-0.3906` | `-0.2031` | `-0.2188` | `-0.2734` | `-0.0625` | `0.2734 / 0.6641` | `-5.9766 / -14.0` | `512` |
| `selfplay3e6_e2` | `-0.3906` | `-0.2031` | `-0.2188` | `-0.1484` | `-0.1016` | `0.2734 / 0.6641` | `-5.9766 / -14.0` | `512` |
| `origcont_e2` | `-0.4062` | `-0.2031` | `+0.0234` | `-0.2734` | `-0.0625` | `0.2578 / 0.6641` | `-5.4453 / -14.0` | `512` |

Medium shortlist carried to large eval: `canonical_ref`, `origcont_e1`, `selfplay_e2`, `selfplay_e1`.

## Large Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | margin mean / median | duplicate trajectories | vs canonical 384:256 |
|-----------|------------:|------------:|------------:|--------------:|------------:|------------------|----------------------|-----------------------:|----------------------:|
| `canonical_ref` | `-0.1784` | `-0.1120` | `-0.2161` | `+0.0924` | `-0.1901` | `0.2878 / 0.4661` | `-7.4688 / -14.0` | `1536` | `0.0000` |
| `origcont_e1` | `-0.1784` | `-0.2083` | `-0.0911` | `-0.1094` | `-0.1458` | `0.2878 / 0.4661` | `-5.7891 / -14.0` | `1536` | `0.0000` |
| `selfplay_e2` | `-0.1901` | `-0.2083` | `-0.2161` | `-0.1029` | `-0.1458` | `0.2760 / 0.4661` | `-6.0365 / -14.0` | `1536` | `-0.0117` |
| `selfplay_e1` | `-0.2018` | `-0.2083` | `-0.2604` | `-0.1068` | `-0.1484` | `0.2760 / 0.4779` | `-5.3828 / -14.0` | `1536` | `-0.0234` |

Primary decision budget ranking on large `384:256`:

1. `canonical_ref` at `-0.1784`
2. `origcont_e1` at `-0.1784` (exact tie on the primary metric, but materially worse `1200:1200`)
3. `selfplay_e2` at `-0.1901`
4. `selfplay_e1` at `-0.2018`

## Default Deterministic Opening Gate

| Candidate | Classification | Standard alt score | 384:256 DS | 1200:1200 DS | 256:768 DS |
|-----------|----------------|-------------------:|------------:|--------------:|------------:|
| `canonical_ref` | `high_search_breakthrough` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `origcont_e1` | `regression_masked_by_seat` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |
| `selfplay_e1` | `high_search_breakthrough` | `0.0000` | `0.0000` | `1.0000` | `0.0000` |
| `selfplay_e2` | `regression_masked_by_seat` | `0.0000` | `0.0000` | `0.0000` | `0.0000` |

The best added-data checkpoint by medium ranking, `selfplay_e2`, did not preserve the default high-search signature.

## Decision

### `control_ep2_still_best`

No checkpoint beat canonical `control_ep2` on the primary large-suite `384:256` disadvantaged-seat score.

### `extra_data_not_helpful`

The original-data continuation control matched or beat every self-play lane on the primary opening-suite ranking:

1. `origcont_e1` tied canonical `control_ep2` on large `384:256`.
2. `selfplay_e2` and `selfplay_e1` were both worse than that tie.
3. Every added-data lane also regressed `1200:1200` versus canonical `control_ep2`.

### `destructive_iteration`

The added-data lane improved validation loss but degraded opening-suite strength:

1. `selfplay_e2` validation loss `1.173831` beat `origcont_e2` `1.179796`.
2. `selfplay_e2` large `384:256` DS `-0.1901` lost to both canonical `control_ep2` and `origcont_e1` (`-0.1784`).
3. `selfplay_e2` also lost the default gate high-search signature (`regression_masked_by_seat`).

This is exactly the failure mode the task asked to detect.

## Local Promotion Gate

Not run.

Reason: no checkpoint clearly beat canonical `control_ep2` on large-suite `384:256`, and the non-canonical checkpoints that were closest all regressed the high-budget/default-opening signature.

## Verification

| Check | Result |
|------|--------|
| dataset generation | PASS: `/tmp/azlite_control_ep2_multi_iter/control_ep2_classic_mcts_selfplay.jsonl` |
| lane training | PASS: completed all requested lanes plus optional `lr=3e-6` lane |
| medium opening-suite benchmark | PASS: `/tmp/azlite_control_ep2_multi_iter/eval_medium/temperature_benchmark_report.json` |
| large opening-suite benchmark | PASS: `/tmp/azlite_control_ep2_multi_iter/eval_large/temperature_benchmark_report.json` |
| default deterministic gate | PASS: `/tmp/azlite_control_ep2_multi_iter/eval_gate/*_default_gate.json` |
| unit tests | `Ran 2338 tests in 104.198s`; `FAILED (failures=1, errors=8, skipped=3)`; matches existing repo baseline from recent AlphaZero-lite docs |
| ruff | PASS: `ruff check ml/alphazero_lite script/ai` |

## Guardrails

| Guardrail | Status |
|-----------|--------|
| residual_v3 only | PASS |
| no residual_v4 | PASS |
| no tablebase overlay | PASS |
| no exact-position patching | PASS |
| no failure/disagreement mining | PASS |
| no agreement filtering | PASS |
| no root-temperature training | PASS |
| no opening-suite curriculum rows | PASS |
| no promotion | PASS |
| no overwrite of `model-artifact/current` | PASS |
| LR not above `1e-5` | PASS |
| no training beyond 2 epochs | PASS |
| selection not driven by validation loss | PASS |

## Artifacts

| Artifact | Path |
|----------|------|
| experiment root | `/tmp/azlite_control_ep2_multi_iter` |
| summary metrics | `/tmp/azlite_control_ep2_multi_iter/summary_metrics.json` |
| new dataset | `/tmp/azlite_control_ep2_multi_iter/control_ep2_classic_mcts_selfplay.jsonl` |
| medium eval report | `/tmp/azlite_control_ep2_multi_iter/eval_medium/temperature_benchmark_report.json` |
| large eval report | `/tmp/azlite_control_ep2_multi_iter/eval_large/temperature_benchmark_report.json` |
| gate reports | `/tmp/azlite_control_ep2_multi_iter/eval_gate` |
