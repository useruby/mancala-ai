# AlphaZero-Lite c_puct 768:768 0.90 Schedule Promotion Results

**Date**: 2026-06-29

**Classification**: `promoted_cpuct_768eq_090_schedule_default`

## Inputs

- Current artifact: `model-artifact/current/weights.json`
- Unchanged artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Expected weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Large suite: `/tmp/azlite_opening_suite/large_eval.jsonl`
- Held-out suites:
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl`
- Budget pairs: `384:256,768:256,768:768,1200:1200,1200:256,256:768`
- Games per opening: `2`
- Workers: `24`
- Seed: `42`

## Runtime Profiles

- Old runtime profile: `{"c_puct": 1.25, "c_puct_schedule": {}, "root_policy_mode": "deterministic", "root_prior_transform": null, "search_mode": "full", "tactical_root_bias": 0.0}`
- New runtime profile: `{"c_puct": 1.25, "c_puct_schedule": {"768:768": 0.9}, "root_policy_mode": "deterministic", "root_prior_transform": null, "search_mode": "full", "tactical_root_bias": 0.0}`

## Files Changed

- `ml/alphazero_lite/cpuct_schedule.py`
- `ml/alphazero_lite/run_opening_suite_seat_benchmark.py`
- `script/ai/seat_aware_promotion_gate`
- `ml/alphazero_lite/test_run_opening_suite_seat_benchmark.py`
- `ml/alphazero_lite/run_cpuct_768eq_090_schedule_promotion.py`
- `docs/alphazero-lite-cpuct-768eq-090-schedule-promotion-results.md`

## Effective-Profile Diagnostics

| Path | Budget | effective c_puct | profile c_puct | tactical_root_bias | search_profile_hash |
|---|---|---|---|---|---|
| opening_suite_benchmark_path | 384:256 | +1.25 | +1.25 | +0.00 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca |
| opening_suite_benchmark_path | 768:256 | +1.25 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 |
| opening_suite_benchmark_path | 768:768 | +0.90 | +0.90 | +0.00 | ce678807806fda0f938a18715fc02c9f40690b0d1a5a7834ddc7c69a3a8ac5cf |
| opening_suite_benchmark_path | 1200:1200 | +1.25 | +1.25 | +0.00 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d |
| opening_suite_benchmark_path | 1200:256 | +1.25 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 |
| opening_suite_benchmark_path | 256:768 | +1.25 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e |

| Path | Budget | effective c_puct | profile c_puct | tactical_root_bias | search_profile_hash |
|---|---|---|---|---|---|
| seat_aware_gate_path | 384:256 | +1.25 | +1.25 | +0.00 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca |
| seat_aware_gate_path | 768:256 | +1.25 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 |
| seat_aware_gate_path | 768:768 | +0.90 | +0.90 | +0.00 | ce678807806fda0f938a18715fc02c9f40690b0d1a5a7834ddc7c69a3a8ac5cf |
| seat_aware_gate_path | 1200:1200 | +1.25 | +1.25 | +0.00 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d |
| seat_aware_gate_path | 1200:256 | +1.25 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 |
| seat_aware_gate_path | 256:768 | +1.25 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e |

- Runtime/app path applicability: `False`
- Runtime/app path note: `direct single-budget runtime path has no challenger:current schedule input`

## Fixed Large Before/After

| Variant | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| old_no_schedule_default_ref | -0.3099 | -0.4049 | -0.1146 | +0.2812 | -0.1159 | -0.4049 |
| promoted_schedule_default | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |
| explicit_768eq_090_schedule | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |

## Held-Out All-Suite Table

| Suite | Variant | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|---|
| heldout_seed43_large | old_no_schedule_default_ref | -0.3034 | -0.3932 | -0.1302 | +0.2969 | -0.1094 | -0.3932 |
| heldout_seed43_large | promoted_schedule_default | -0.3034 | -0.3932 | +0.6927 | +0.2969 | -0.1094 | -0.3932 |
| heldout_seed43_large | explicit_768eq_090_schedule | -0.3034 | -0.3932 | +0.6927 | +0.2969 | -0.1094 | -0.3932 |
| heldout_seed44_large | old_no_schedule_default_ref | -0.3008 | -0.3971 | -0.1667 | +0.2708 | -0.1094 | -0.3971 |
| heldout_seed44_large | promoted_schedule_default | -0.3008 | -0.3971 | +0.6797 | +0.2708 | -0.1094 | -0.3971 |
| heldout_seed44_large | explicit_768eq_090_schedule | -0.3008 | -0.3971 | +0.6797 | +0.2708 | -0.1094 | -0.3971 |
| heldout_seed45_large | old_no_schedule_default_ref | -0.3112 | -0.3971 | -0.1771 | +0.2526 | -0.1107 | -0.3971 |
| heldout_seed45_large | promoted_schedule_default | -0.3112 | -0.3971 | +0.6693 | +0.2526 | -0.1107 | -0.3971 |
| heldout_seed45_large | explicit_768eq_090_schedule | -0.3112 | -0.3971 | +0.6693 | +0.2526 | -0.1107 | -0.3971 |
| heldout_seed46_large | old_no_schedule_default_ref | -0.2878 | -0.3880 | -0.0938 | +0.2734 | -0.1185 | -0.3880 |
| heldout_seed46_large | promoted_schedule_default | -0.2878 | -0.3880 | +0.6693 | +0.2734 | -0.1185 | -0.3880 |
| heldout_seed46_large | explicit_768eq_090_schedule | -0.2878 | -0.3880 | +0.6693 | +0.2734 | -0.1185 | -0.3880 |
| heldout_seed47_large | old_no_schedule_default_ref | -0.3255 | -0.4036 | -0.1562 | +0.2786 | -0.1081 | -0.4036 |
| heldout_seed47_large | promoted_schedule_default | -0.3255 | -0.4036 | +0.6901 | +0.2786 | -0.1081 | -0.4036 |
| heldout_seed47_large | explicit_768eq_090_schedule | -0.3255 | -0.4036 | +0.6901 | +0.2786 | -0.1081 | -0.4036 |
| heldout_seed48_large | old_no_schedule_default_ref | -0.3216 | -0.4076 | -0.1667 | +0.2604 | -0.1081 | -0.4076 |
| heldout_seed48_large | promoted_schedule_default | -0.3216 | -0.4076 | +0.6589 | +0.2604 | -0.1081 | -0.4076 |
| heldout_seed48_large | explicit_768eq_090_schedule | -0.3216 | -0.4076 | +0.6589 | +0.2604 | -0.1081 | -0.4076 |

## Gate Result

- Default gate classification: `high_search_breakthrough`
- Default gate schedule manifest: `{"default_c_puct": 1.25, "overrides": {"768:768": 0.9}}`
- Standard budget effective profile: `{"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"}`

## Runtime Cost Comparison

| Variant | Mean move latency ms | Mean p95 latency ms | Relative slowdown |
|---|---|---|---|
| old_no_schedule_default_ref | +102.26 | +272.67 | +0.000 |
| promoted_schedule_default | +104.23 | +272.89 | +0.019 |
| explicit_768eq_090_schedule | +104.23 | +272.89 | +0.019 |

## Decision Summary

- Model weights unchanged: `True`
- Default benchmark/gate paths activate the promoted schedule: `True`
- Old no-schedule global 1.25 remains reproducible: `True`
- Promoted default equals explicit 768:768=0.90 schedule: `True`
- 768:768 improvement direction reproduced: `True`
- Gate preserved: `True`
- Final classification: `promoted_cpuct_768eq_090_schedule_default`
