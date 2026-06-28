# AlphaZero-Lite No Tactical Bias c_puct Schedule Preflight Results

**Date**: 2026-06-28

**Classification**: `cpuct_schedule_candidate`

## Inputs

- Current artifact: `model-artifact/current/weights.json`
- Current artifact weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Expected SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- Medium suite: `/tmp/azlite_opening_suite/medium_eval.jsonl`
- Fixed large suite: `/tmp/azlite_opening_suite/large_eval.jsonl`
- Held-out suites:
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl`
- `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed46_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed47_large.jsonl`
- `/tmp/azlite_pr123_weighted_candidate_preflight/suites/heldout_seed48_large.jsonl`
- Tactical root bias: `+0.00`
- Games per opening: `2`
- Workers: `24`
- Seed: `42`

## Lane Schedule Definitions

| Lane | Default c_puct | Per-budget overrides |
|---|---|---|
| default_cpuct_1_25_ref | +1.25 | {} |
| global_cpuct_1_00_ref | +1.00 | {} |
| schedule_768eq_cpuct_1_00 | +1.25 | {"768:768": 1.0} |
| schedule_equal_budget_cpuct_1_00 | +1.25 | {"1200:1200": 1.0, "768:768": 1.0} |
| schedule_low_budget_cpuct_1_00 | +1.25 | {"384:256": 1.0, "768:768": 1.0} |
| schedule_768eq_cpuct_0_90 | +1.25 | {"768:768": 0.9} |
| schedule_768eq_cpuct_1_10 | +1.25 | {"768:768": 1.1} |

## Effective Search Profiles

| Lane | Budget | effective c_puct | tactical_root_bias | search_profile_hash | challenger sims | current sims |
|---|---|---|---|---|---|---|
| default_cpuct_1_25_ref | 384:256 | +1.25 | +0.00 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | 384 | 256 |
| default_cpuct_1_25_ref | 768:256 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 | 768 | 256 |
| default_cpuct_1_25_ref | 768:768 | +1.25 | +0.00 | de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436 | 768 | 768 |
| default_cpuct_1_25_ref | 1200:1200 | +1.25 | +0.00 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | 1200 | 1200 |
| default_cpuct_1_25_ref | 1200:256 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | 1200 | 256 |
| default_cpuct_1_25_ref | 256:768 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e | 256 | 768 |
| global_cpuct_1_00_ref | 384:256 | +1.00 | +0.00 | faf2f586925d01736d376c5212400bda173e65a690e94df3cbe4529e83484b74 | 384 | 256 |
| global_cpuct_1_00_ref | 768:256 | +1.00 | +0.00 | 2210585d78be7fd03030608116ca0de0a29575b80512d92234f6bfb66e46de8d | 768 | 256 |
| global_cpuct_1_00_ref | 768:768 | +1.00 | +0.00 | 427deca4b9d822f8d3b5198b77927a7b9a929bb7d51546bbb7db15da2ca4b7ba | 768 | 768 |
| global_cpuct_1_00_ref | 1200:1200 | +1.00 | +0.00 | b418d04fbe96ef9b5a0e65f561e423948e27e5d0400d8253a97f3b1363d62610 | 1200 | 1200 |
| global_cpuct_1_00_ref | 1200:256 | +1.00 | +0.00 | dda2a5eae80d0afa6e1df56636dcc5adc9c879fdccc4eaa68361370d6e9243f2 | 1200 | 256 |
| global_cpuct_1_00_ref | 256:768 | +1.00 | +0.00 | 540323e0d3124ba48d8a98c3285f05ebda6fb5f4b10800ad57964eb63df51a67 | 256 | 768 |
| schedule_768eq_cpuct_1_00 | 384:256 | +1.25 | +0.00 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | 384 | 256 |
| schedule_768eq_cpuct_1_00 | 768:256 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 | 768 | 256 |
| schedule_768eq_cpuct_1_00 | 768:768 | +1.00 | +0.00 | 427deca4b9d822f8d3b5198b77927a7b9a929bb7d51546bbb7db15da2ca4b7ba | 768 | 768 |
| schedule_768eq_cpuct_1_00 | 1200:1200 | +1.25 | +0.00 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | 1200 | 1200 |
| schedule_768eq_cpuct_1_00 | 1200:256 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | 1200 | 256 |
| schedule_768eq_cpuct_1_00 | 256:768 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e | 256 | 768 |
| schedule_equal_budget_cpuct_1_00 | 384:256 | +1.25 | +0.00 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | 384 | 256 |
| schedule_equal_budget_cpuct_1_00 | 768:256 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 | 768 | 256 |
| schedule_equal_budget_cpuct_1_00 | 768:768 | +1.00 | +0.00 | 427deca4b9d822f8d3b5198b77927a7b9a929bb7d51546bbb7db15da2ca4b7ba | 768 | 768 |
| schedule_equal_budget_cpuct_1_00 | 1200:1200 | +1.00 | +0.00 | b418d04fbe96ef9b5a0e65f561e423948e27e5d0400d8253a97f3b1363d62610 | 1200 | 1200 |
| schedule_equal_budget_cpuct_1_00 | 1200:256 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | 1200 | 256 |
| schedule_equal_budget_cpuct_1_00 | 256:768 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e | 256 | 768 |
| schedule_low_budget_cpuct_1_00 | 384:256 | +1.00 | +0.00 | faf2f586925d01736d376c5212400bda173e65a690e94df3cbe4529e83484b74 | 384 | 256 |
| schedule_low_budget_cpuct_1_00 | 768:256 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 | 768 | 256 |
| schedule_low_budget_cpuct_1_00 | 768:768 | +1.00 | +0.00 | 427deca4b9d822f8d3b5198b77927a7b9a929bb7d51546bbb7db15da2ca4b7ba | 768 | 768 |
| schedule_low_budget_cpuct_1_00 | 1200:1200 | +1.25 | +0.00 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | 1200 | 1200 |
| schedule_low_budget_cpuct_1_00 | 1200:256 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | 1200 | 256 |
| schedule_low_budget_cpuct_1_00 | 256:768 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e | 256 | 768 |
| schedule_768eq_cpuct_0_90 | 384:256 | +1.25 | +0.00 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | 384 | 256 |
| schedule_768eq_cpuct_0_90 | 768:256 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 | 768 | 256 |
| schedule_768eq_cpuct_0_90 | 768:768 | +0.90 | +0.00 | ce678807806fda0f938a18715fc02c9f40690b0d1a5a7834ddc7c69a3a8ac5cf | 768 | 768 |
| schedule_768eq_cpuct_0_90 | 1200:1200 | +1.25 | +0.00 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | 1200 | 1200 |
| schedule_768eq_cpuct_0_90 | 1200:256 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | 1200 | 256 |
| schedule_768eq_cpuct_0_90 | 256:768 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e | 256 | 768 |
| schedule_768eq_cpuct_1_10 | 384:256 | +1.25 | +0.00 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | 384 | 256 |
| schedule_768eq_cpuct_1_10 | 768:256 | +1.25 | +0.00 | 38484cf12cc8360fb400ec1a37d9d98a84de195890fd8b0cf4a35e3e3bb74790 | 768 | 256 |
| schedule_768eq_cpuct_1_10 | 768:768 | +1.10 | +0.00 | e16ae2a424bfe15cdb891ee715903ac695e654d09e7cf3e4d48e307c4d694b3d | 768 | 768 |
| schedule_768eq_cpuct_1_10 | 1200:1200 | +1.25 | +0.00 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | 1200 | 1200 |
| schedule_768eq_cpuct_1_10 | 1200:256 | +1.25 | +0.00 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | 1200 | 256 |
| schedule_768eq_cpuct_1_10 | 256:768 | +1.25 | +0.00 | 6fcc482cd706bb901b43eab74010fa2266a4653cace683300e02fc3f5bcd496e | 256 | 768 |

## Medium DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_cpuct_1_25_ref | -0.2344 | -0.3750 | +0.1250 | +0.3906 | -0.1367 | -0.3750 |
| global_cpuct_1_00_ref | -0.4609 | +0.1055 | +0.3906 | -0.0781 | -0.1641 | +0.1055 |
| schedule_768eq_cpuct_1_00 | -0.2344 | -0.3750 | +0.3906 | +0.3906 | -0.1367 | -0.3750 |
| schedule_equal_budget_cpuct_1_00 | -0.2344 | -0.3750 | +0.3906 | -0.0781 | -0.1367 | -0.3750 |
| schedule_low_budget_cpuct_1_00 | -0.4609 | -0.3750 | +0.3906 | +0.3906 | -0.1367 | -0.3750 |
| schedule_768eq_cpuct_0_90 | -0.2344 | -0.3750 | +0.6484 | +0.3906 | -0.1367 | -0.3750 |
| schedule_768eq_cpuct_1_10 | -0.2344 | -0.3750 | -0.3984 | +0.3906 | -0.1367 | -0.3750 |

## Fixed Large DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_cpuct_1_25_ref | -0.3099 | -0.4049 | -0.1146 | +0.2812 | -0.1159 | -0.4049 |
| global_cpuct_1_00_ref | -0.5143 | +0.0365 | +0.2630 | -0.1120 | -0.1706 | +0.0365 |
| schedule_768eq_cpuct_1_00 | -0.3099 | -0.4049 | +0.2630 | +0.2812 | -0.1159 | -0.4049 |
| schedule_equal_budget_cpuct_1_00 | -0.3099 | -0.4049 | +0.2630 | -0.1120 | -0.1159 | -0.4049 |
| schedule_low_budget_cpuct_1_00 | -0.5143 | -0.4049 | +0.2630 | +0.2812 | -0.1159 | -0.4049 |
| schedule_768eq_cpuct_0_90 | -0.3099 | -0.4049 | +0.6849 | +0.2812 | -0.1159 | -0.4049 |
| schedule_768eq_cpuct_1_10 | -0.3099 | -0.4049 | -0.4974 | +0.2812 | -0.1159 | -0.4049 |

## Held-Out Mean DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_cpuct_1_25_ref | -0.3084 | -0.3978 | -0.1484 | +0.2721 | -0.1107 | -0.3978 |
| global_cpuct_1_00_ref | -0.4978 | +0.0395 | +0.2713 | -0.1068 | -0.1753 | +0.0395 |
| schedule_768eq_cpuct_1_00 | -0.3084 | -0.3978 | +0.2713 | +0.2721 | -0.1107 | -0.3978 |
| schedule_equal_budget_cpuct_1_00 | -0.3084 | -0.3978 | +0.2713 | -0.1068 | -0.1107 | -0.3978 |
| schedule_low_budget_cpuct_1_00 | -0.4978 | -0.3978 | +0.2713 | +0.2721 | -0.1107 | -0.3978 |
| schedule_768eq_cpuct_0_90 | -0.3084 | -0.3978 | +0.6766 | +0.2721 | -0.1107 | -0.3978 |
| schedule_768eq_cpuct_1_10 | -0.3084 | -0.3978 | -0.5304 | +0.2721 | -0.1107 | -0.3978 |

## Held-Out Worst-Suite DS Table

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_cpuct_1_25_ref | -0.3255 | -0.4076 | -0.1771 | +0.2526 | -0.1185 | -0.4076 |
| global_cpuct_1_00_ref | -0.5104 | +0.0247 | +0.2526 | -0.1302 | -0.1888 | +0.0247 |
| schedule_768eq_cpuct_1_00 | -0.3255 | -0.4076 | +0.2526 | +0.2526 | -0.1185 | -0.4076 |
| schedule_equal_budget_cpuct_1_00 | -0.3255 | -0.4076 | +0.2526 | -0.1302 | -0.1185 | -0.4076 |
| schedule_low_budget_cpuct_1_00 | -0.5104 | -0.4076 | +0.2526 | +0.2526 | -0.1185 | -0.4076 |
| schedule_768eq_cpuct_0_90 | -0.3255 | -0.4076 | +0.6589 | +0.2526 | -0.1185 | -0.4076 |
| schedule_768eq_cpuct_1_10 | -0.3255 | -0.4076 | -0.5755 | +0.2526 | -0.1185 | -0.4076 |

## Bootstrap 95% CI

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|
| default_cpuct_1_25_ref_minus_default_384_256 | +0.0000 | +0.0000 | +0.0000 |
| default_cpuct_1_25_ref_minus_default_768_768 | +0.0000 | +0.0000 | +0.0000 |
| default_cpuct_1_25_ref_minus_default_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| default_cpuct_1_25_ref_minus_default_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| global_cpuct_1_00_ref_minus_default_384_256 | -0.1895 | -0.2196 | -0.1589 |
| global_cpuct_1_00_ref_minus_default_768_768 | +0.4197 | +0.3728 | +0.4679 |
| global_cpuct_1_00_ref_minus_default_1200_1200 | -0.3789 | -0.4006 | -0.3576 |
| global_cpuct_1_00_ref_minus_default_1200_256 | -0.0647 | -0.0803 | -0.0490 |
| schedule_768eq_cpuct_1_00_minus_default_384_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_1_00_minus_default_768_768 | +0.4197 | +0.3728 | +0.4679 |
| schedule_768eq_cpuct_1_00_minus_default_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_1_00_minus_default_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_equal_budget_cpuct_1_00_minus_default_384_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_equal_budget_cpuct_1_00_minus_default_768_768 | +0.4197 | +0.3728 | +0.4679 |
| schedule_equal_budget_cpuct_1_00_minus_default_1200_1200 | -0.3789 | -0.4006 | -0.3576 |
| schedule_equal_budget_cpuct_1_00_minus_default_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_low_budget_cpuct_1_00_minus_default_384_256 | -0.1895 | -0.2196 | -0.1589 |
| schedule_low_budget_cpuct_1_00_minus_default_768_768 | +0.4197 | +0.3728 | +0.4679 |
| schedule_low_budget_cpuct_1_00_minus_default_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| schedule_low_budget_cpuct_1_00_minus_default_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_0_90_minus_default_384_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_0_90_minus_default_768_768 | +0.8251 | +0.7695 | +0.8802 |
| schedule_768eq_cpuct_0_90_minus_default_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_0_90_minus_default_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_1_10_minus_default_384_256 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_1_10_minus_default_768_768 | -0.3819 | -0.4128 | -0.3511 |
| schedule_768eq_cpuct_1_10_minus_default_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| schedule_768eq_cpuct_1_10_minus_default_1200_256 | +0.0000 | +0.0000 | +0.0000 |

## P0/P1 Split For 384:256

| Lane | Mean P0 | Mean P1 | Gap | Mean duplicates |
|---|---|---|---|---|
| default_cpuct_1_25_ref | +0.6157 | +0.9240 | +0.3084 | +768.0000 |
| global_cpuct_1_00_ref | +0.3411 | +0.8390 | +0.4978 | +768.0000 |
| schedule_768eq_cpuct_1_00 | +0.6157 | +0.9240 | +0.3084 | +768.0000 |
| schedule_equal_budget_cpuct_1_00 | +0.6157 | +0.9240 | +0.3084 | +768.0000 |
| schedule_low_budget_cpuct_1_00 | +0.3411 | +0.8390 | +0.4978 | +768.0000 |
| schedule_768eq_cpuct_0_90 | +0.6157 | +0.9240 | +0.3084 | +768.0000 |
| schedule_768eq_cpuct_1_10 | +0.6157 | +0.9240 | +0.3084 | +768.0000 |

## Duplicate Trajectory Counts

| Lane | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| default_cpuct_1_25_ref | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 |
| global_cpuct_1_00_ref | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 |
| schedule_768eq_cpuct_1_00 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 |
| schedule_equal_budget_cpuct_1_00 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 |
| schedule_low_budget_cpuct_1_00 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 |
| schedule_768eq_cpuct_0_90 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 |
| schedule_768eq_cpuct_1_10 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 | +768.0000 |

## Runtime Cost Versus Default

| Lane | Mean move latency | P95 move latency | Avg sims/game | Relative slowdown |
|---|---|---|---|---|
| default_cpuct_1_25_ref | +102.91 | +273.17 | +762.67 | +0.000 |
| global_cpuct_1_00_ref | +107.47 | +273.49 | +762.67 | +0.044 |
| schedule_768eq_cpuct_1_00 | +103.27 | +272.39 | +762.67 | +0.003 |
| schedule_equal_budget_cpuct_1_00 | +105.48 | +272.57 | +762.67 | +0.025 |
| schedule_low_budget_cpuct_1_00 | +103.26 | +272.48 | +762.67 | +0.003 |
| schedule_768eq_cpuct_0_90 | +103.84 | +272.76 | +762.67 | +0.009 |
| schedule_768eq_cpuct_1_10 | +102.83 | +272.62 | +762.67 | -0.001 |

## Gate Classification

| Lane | Classification | Gate budgets |
|---|---|---|
| default_cpuct_1_25_ref | high_search_breakthrough | 384:256,1200:1200,1200:256,256:768 |
| schedule_768eq_cpuct_1_00 | high_search_breakthrough | 384:256,1200:1200,1200:256,256:768 |
| schedule_768eq_cpuct_0_90 | high_search_breakthrough | 384:256,1200:1200,1200:256,256:768 |

## Lane Assessments

| Lane | Classification | Delta 768:768 | CI lower 768:768 | Delta 384:256 | Delta 1200:1200 | Delta 1200:256 | Slowdown | Gate |
|---|---|---|---|---|---|---|---|---|
| default_cpuct_1_25_ref | cpuct_schedule_no_better_than_default | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.0000 | +0.000 | high_search_breakthrough |
| global_cpuct_1_00_ref | cpuct_schedule_no_better_than_default | +0.4197 | +0.3728 | -0.1895 | -0.3789 | -0.0647 | +0.044 | not_run |
| schedule_768eq_cpuct_1_00 | cpuct_schedule_candidate | +0.4197 | +0.3728 | +0.0000 | +0.0000 | +0.0000 | +0.003 | high_search_breakthrough |
| schedule_equal_budget_cpuct_1_00 | equal_budget_cpuct_tradeoff | +0.4197 | +0.3728 | +0.0000 | -0.3789 | +0.0000 | +0.025 | not_run |
| schedule_low_budget_cpuct_1_00 | equal_budget_cpuct_tradeoff | +0.4197 | +0.3728 | -0.1895 | +0.0000 | +0.0000 | +0.003 | not_run |
| schedule_768eq_cpuct_0_90 | cpuct_schedule_candidate | +0.8251 | +0.7695 | +0.0000 | +0.0000 | +0.0000 | +0.009 | high_search_breakthrough |
| schedule_768eq_cpuct_1_10 | cpuct_schedule_no_better_than_default | -0.3819 | -0.4128 | +0.0000 | +0.0000 | +0.0000 | -0.001 | not_run |

## Decision Summary

- Gate lanes run: `default_cpuct_1_25_ref,schedule_768eq_cpuct_1_00,schedule_768eq_cpuct_0_90`
- Final classification: `cpuct_schedule_candidate`
