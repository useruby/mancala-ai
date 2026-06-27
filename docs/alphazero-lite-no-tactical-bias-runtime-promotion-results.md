# AlphaZero-Lite No Tactical Bias Runtime Promotion Results

**Date**: 2026-06-27

**Classification**: `promoted_no_tactical_bias_runtime_default`

## Inputs

- Current artifact: `model-artifact/current/weights.json`
- Unchanged artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Expected weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Large suite: `/tmp/azlite_opening_suite/large_eval.jsonl`
- Held-out suites:
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl`
- Budget pairs: `384:256,768:256,768:768,1200:1200,1200:256,256:768`
- Games per opening: `2`
- Workers: `24`
- Seed: `42`

## Search Profiles

- Old default search profile: `{"c_puct": 1.25, "root_policy_mode": "deterministic", "root_prior_transform": null, "search_mode": "full", "tactical_root_bias": 0.1, "value_trust_schedule": null}`
- New default search profile: `{"c_puct": 1.25, "root_policy_mode": "deterministic", "root_prior_transform": null, "search_mode": "full", "tactical_root_bias": 0.0, "value_trust_schedule": null}`

## Files Changed

- `ml/alphazero_lite/self_play.py`
- `ml/alphazero_lite/arena.py`
- `ml/alphazero_lite/benchmark.py`
- `ml/alphazero_lite/test_benchmark.py`
- `ml/alphazero_lite/run_opening_suite_seat_benchmark.py`
- `script/ai/seat_aware_promotion_gate`
- `ml/alphazero_lite/run_no_tactical_bias_runtime_promotion.py`
- `docs/alphazero-lite-no-tactical-bias-runtime-promotion-results.md`

## Effective-Profile Diagnostics

| Path | Default effective profile | Explicit tactical_root_bias=0.1 |
|---|---|---|
| default_runtime_path | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "9ca513f6be4cb4082846d2207f1838059385e6c01dcdfb3f123f065f6801c2d7", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.1}, "simulations": 384, "version": "v1"} |
| opening_suite_benchmark_path | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "9ca513f6be4cb4082846d2207f1838059385e6c01dcdfb3f123f065f6801c2d7", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.1}, "simulations": 384, "version": "v1"} |
| seat_aware_gate_path | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "9ca513f6be4cb4082846d2207f1838059385e6c01dcdfb3f123f065f6801c2d7", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.1}, "simulations": 384, "version": "v1"} |

## Fixed Large Before/After

| Variant | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| old_default_ref | -0.5456 | -0.4648 | -0.2943 | -0.1380 | -0.1797 | -0.4648 |
| promoted_default | -0.3099 | -0.4049 | -0.1146 | +0.2812 | -0.1159 | -0.4049 |
| explicit_no_tactical_bias | -0.3099 | -0.4049 | -0.1146 | +0.2812 | -0.1159 | -0.4049 |

## Held-Out Spot Check

| Suite | old 384:256 | old 768:768 | old 1200:1200 | old 1200:256 | promoted 384:256 | promoted 768:768 | promoted 1200:1200 | promoted 1200:256 | explicit 384:256 | explicit 768:768 | explicit 1200:1200 | explicit 1200:256 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| heldout_seed43_large | -0.5299 | -0.3047 | -0.1719 | -0.1901 | -0.3034 | -0.1302 | +0.2969 | -0.1094 | -0.3034 | -0.1302 | +0.2969 | -0.1094 |
| heldout_seed46_large | -0.5404 | -0.2812 | -0.1406 | -0.1823 | -0.2878 | -0.0938 | +0.2734 | -0.1185 | -0.2878 | -0.0938 | +0.2734 | -0.1185 |

## Gate Result

- Default gate classification: `high_search_breakthrough`
- Standard budget effective profile: `{"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"}`

## Runtime Cost Comparison

Held-out mean move latency at `384:256`.

| Variant | Mean move latency ms | Mean p95 move latency ms |
|---|---|---|
| old_default_ref | +50.85 | +119.16 |
| promoted_default | +49.66 | +118.91 |
| explicit_no_tactical_bias | +50.21 | +119.74 |

## Decision Summary

- Default active in runtime/benchmark/gate paths: `True`
- Explicit `tactical_root_bias=0.1` still reproducible: `True`
- Promoted default equals explicit no-tactical-bias: `True`
- Improvement direction reproduced: `True`
- Gate preserved: `True`
- Final classification: `promoted_no_tactical_bias_runtime_default`
