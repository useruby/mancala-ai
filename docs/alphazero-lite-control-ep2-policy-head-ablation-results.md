# AlphaZero-Lite Control EP2 Policy-Head Ablation Results

**Date**: 2026-06-15
**Classification**: `added_data_not_useful_confirmed` + `reject_added_classic_mcts_data`
**Schema**: `azlite_control_ep2_policy_head_ablation_v1`

## Summary

Tested the PR #115 added-data continuation hypothesis while freezing the `control_ep2` trunk/value path and training only the policy head.

Result: policy-head-only fine-tuning did not rescue the added classic-MCTS rows.

- Canonical `control_ep2` large `384:256` DS: `-0.1784`
- Best added-data policy-head lane (`added_policy_head_e1`) large `384:256` DS: `-0.1901`
- Delta vs canonical on large `384:256`: `-0.0117`
- Default high-search signature was re-confirmed for canonical only; no non-canonical lane tied or beat canonical on large `384:256`, so no non-canonical gate rerun was triggered.

That rejects the added classic-MCTS source for now and does not support the trunk/value-drift explanation.

## Inputs

| Item | Path | SHA256 / rows |
|------|------|----------------|
| canonical control checkpoint | `/tmp/azlite_control_continuation_sweep/lr3e5/checkpoint_epoch2.npz` | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` |
| canonical control artifact | `/tmp/azlite_control_ep2_seed_harvest/control_ep2_eval_only/artifact_control_ep2` | `34b3697f95c3c63e092838d10668ba1361dab2a68a0bfe12305a294c45f765ad` |
| current artifact | `model-artifact/current` | `6946aaffb2e32916c1b7ed57437f6c86b94778833954c546af0e78097d9d2781` |
| generic bootstrap replay | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | rows `9589` |
| random teacher replay | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | rows `2016` |
| added classic-MCTS replay | `/tmp/azlite_control_ep2_multi_iter/control_ep2_classic_mcts_selfplay.jsonl` | `74b1aa63c026057828983531ad562009ccc7388f3f66757c69ef5d536429a67b`, rows `12793` |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |

## Run Conditions

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
- init checkpoint `control_ep2`

## Training Lanes

| Lane | Replay weights | Epochs | Checkpoint SHA256 | Delta norm vs canonical | Relative delta | Policy loss | Value loss | Validation loss |
|------|----------------|-------:|-------------------|------------------------:|---------------:|------------:|-----------:|----------------:|
| `canonical_ref` | n/a | 0 | `619376dba87b968cc85bfc401cc230aca64e444dc3570216b0630c2c29a388f9` | `0.0000` | `0.0000%` | n/a | n/a | n/a |
| `orig_policy_head_e1` | `4,1` | 1 | `e685d3a5efb738454db92bf235bc84e90d59913b6c7023cf0e1e5ee99f67c549` | `0.0232` | `0.0861%` | `0.996682` | `0.245148` | `1.180301` |
| `orig_policy_head_e2` | `4,1` | 2 | `946396d77d741515ec78e68570e43c1b7e92144023c171c660750bb45cd549f5` | `0.0351` | `0.1303%` | `0.995951` | `0.245150` | `1.180301` |
| `added_policy_head_e1` | `4,1,1` | 1 | `1d980cf41549e2d71a0860495713b95cbec319ecf0b8ac1deac141aab8bd1343` | `0.0246` | `0.0915%` | `1.065201` | `0.241355` | `1.176323` |
| `added_policy_head_e2` | `4,1,1` | 2 | `87df0739b1104f689756fb5db292d0e38f08317b0ae4d2d0a5bc32968e351523` | `0.0380` | `0.1411%` | `1.064416` | `0.241430` | `1.176240` |

## Medium Opening-Suite Benchmark

Large-eval carry rule: `canonical_ref` plus top 3 non-canonical candidates by medium `384:256` DS.

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories |
|-----------|------------:|------------:|------------:|--------------:|------------:|------------------|-----------------------:|
| `canonical_ref` | `-0.1719` | `-0.1094` | `-0.2188` | `+0.1133` | `-0.1016` | `0.2734 / 0.4453` | `512` |
| `added_policy_head_e1` | `-0.1875` | `-0.1484` | `-0.2188` | `+0.1133` | `-0.0625` | `0.2578 / 0.4453` | `512` |
| `orig_policy_head_e1` | `-0.4062` | `-0.1484` | `-0.2188` | `+0.1328` | `-0.0625` | `0.2578 / 0.6641` | `512` |
| `orig_policy_head_e2` | `-0.4062` | `+0.0938` | `+0.0234` | `-0.1250` | `-0.0625` | `0.2578 / 0.6641` | `512` |
| `added_policy_head_e2` | `-0.4453` | `+0.0938` | `+0.0234` | `+0.0391` | `-0.0625` | `0.2188 / 0.6641` | `512` |

Large-eval shortlist: `canonical_ref`, `added_policy_head_e1`, `orig_policy_head_e1`, `orig_policy_head_e2`.

## Large Opening-Suite Benchmark

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories | vs canonical 384:256 |
|-----------|------------:|------------:|------------:|--------------:|------------:|------------------|-----------------------:|----------------------:|
| `canonical_ref` | `-0.1784` | `-0.1120` | `-0.2161` | `+0.0924` | `-0.1901` | `0.2878 / 0.4661` | `1536` | `0.0000` |
| `added_policy_head_e1` | `-0.1901` | `-0.1562` | `-0.2161` | `+0.0924` | `-0.1458` | `0.2760 / 0.4661` | `1536` | `-0.0117` |
| `orig_policy_head_e1` | `-0.3698` | `-0.1562` | `-0.2161` | `+0.1146` | `-0.1458` | `0.2760 / 0.6458` | `1536` | `-0.1914` |
| `orig_policy_head_e2` | `-0.3698` | `+0.0130` | `-0.0469` | `-0.1706` | `-0.1458` | `0.2760 / 0.6458` | `1536` | `-0.1914` |

Primary ranking on large `384:256`:

1. `canonical_ref` at `-0.1784`
2. `added_policy_head_e1` at `-0.1901`
3. `orig_policy_head_e1` at `-0.3698`
4. `orig_policy_head_e2` at `-0.3698`

## Default Deterministic Gate

Gate rule in this task: rerun canonical plus any candidate that tied or beat canonical on large `384:256`.

Only canonical qualified.

| Candidate | Classification | 384:256 DS | 1200:1200 DS | 256:768 DS | high-search signature preserved |
|-----------|----------------|------------:|--------------:|------------:|---------------------------------|
| `canonical_ref` | `high_search_breakthrough` | `0.0000` | `1.0000` | `0.0000` | yes |
| `added_policy_head_e1` | not run | n/a | n/a | n/a | not evaluated |

## Decision

### `reject_added_classic_mcts_data`

Best added-data policy-head-only candidate still failed the primary acceptance test.

1. `added_policy_head_e1` large `384:256` DS was `-0.1901`.
2. Canonical `control_ep2` large `384:256` DS was `-0.1784`.
3. The gap was `-0.0117`, which is worse than the allowed `-0.01` tolerance.

### `added_data_not_useful_confirmed`

This ablation does not support the idea that PR #115 failed only because full-network training damaged the trunk/value path.

1. PR #115 full added-data baseline (`selfplay_e2`) already regressed to `-0.1901` on large `384:256`.
2. The best policy-head-only added-data lane also landed at `-0.1901` on the same metric.
3. So freezing trunk/value did not recover the canonical control signature on the primary benchmark.

This is not `trunk_value_drift_confirmed`.

## Verification

| Check | Result |
|------|--------|
| runner script | PASS: `ml/alphazero_lite/run_control_ep2_policy_head_ablation.py` |
| experiment summary | PASS: `/tmp/azlite_control_ep2_policy_head_ablation/summary_metrics.json` |
| medium eval report | PASS: `/tmp/azlite_control_ep2_policy_head_ablation/eval_medium/temperature_benchmark_report.json` |
| large eval report | PASS: `/tmp/azlite_control_ep2_policy_head_ablation/eval_large/temperature_benchmark_report.json` |
| gate report | PASS: `/tmp/azlite_control_ep2_policy_head_ablation/eval_gate/canonical_ref_default_gate.json` |
| `ruff check ml/alphazero_lite script/ai` | PASS |
| `ruff check ml/alphazero_lite/run_control_ep2_policy_head_ablation.py` | PASS |
| `python -m unittest ml.alphazero_lite.test_run_manifest` | PASS |

## Guardrails

| Guardrail | Status |
|-----------|--------|
| no new dataset generation | PASS |
| `residual_v3` only | PASS |
| no `residual_v4` | PASS |
| no architecture change | PASS |
| no tablebase overlay | PASS |
| no curriculum mining | PASS |
| no failure/disagreement mining | PASS |
| no seed sweep | PASS |
| no promotion | PASS |
| no overwrite of `model-artifact/current` | PASS |

## Artifacts

| Artifact | Path |
|----------|------|
| experiment root | `/tmp/azlite_control_ep2_policy_head_ablation` |
| summary metrics | `/tmp/azlite_control_ep2_policy_head_ablation/summary_metrics.json` |
| medium eval report | `/tmp/azlite_control_ep2_policy_head_ablation/eval_medium/temperature_benchmark_report.json` |
| large eval report | `/tmp/azlite_control_ep2_policy_head_ablation/eval_large/temperature_benchmark_report.json` |
| gate report | `/tmp/azlite_control_ep2_policy_head_ablation/eval_gate/canonical_ref_default_gate.json` |
