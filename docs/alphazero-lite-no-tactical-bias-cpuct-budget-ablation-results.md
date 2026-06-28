# AlphaZero-Lite No Tactical Bias c_puct Budget Ablation Results

**Date**: 2026-06-28

**Classification**: `cpuct_sensitive_runtime`

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
- Requested c_puct values: `0.75,1,1.25,1.5,1.75,2`
- Requested budget lanes: `384:384,768:768,384:512,512:384,768:1024,1024:768`
- Tactical root bias: `+0.00`
- Games per opening: `2`
- Workers: `24`
- Seed: `42`

## Lane Budget Map

| Lane | c_puct | tactical_root_bias | 384-axis | 768-axis | 1200:1200 | 1200:256 |
|---|---|---|---|---|---|---|
| no_tactical_cpuct_1_25_ref | +1.25 | +0.00 | 384:256 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_cpuct_0_75 | +0.75 | +0.00 | 384:256 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_cpuct_1_00 | +1.00 | +0.00 | 384:256 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_cpuct_1_50 | +1.50 | +0.00 | 384:256 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_cpuct_1_75 | +1.75 | +0.00 | 384:256 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_cpuct_2_00 | +2.00 | +0.00 | 384:256 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_equalized_384_384 | +1.25 | +0.00 | 384:384 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_equalized_768_768 | +1.25 | +0.00 | 384:256 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_high_current_384_512 | +1.25 | +0.00 | 384:512 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_high_challenger_512_384 | +1.25 | +0.00 | 512:384 | 768:768 | 1200:1200 | 1200:256 |
| no_tactical_high_current_768_1024 | +1.25 | +0.00 | 384:256 | 768:1024 | 1200:1200 | 1200:256 |
| no_tactical_high_challenger_1024_768 | +1.25 | +0.00 | 384:256 | 1024:768 | 1200:1200 | 1200:256 |

## Effective Search Profiles

| Lane | Axis | Actual budget | search_profile_hash | search_profile |
|---|---|---|---|---|
| no_tactical_cpuct_0_75 | 384:256 | 384:256 | 0236da1b0848b73158e7307ac2c1350170c594074543e2ab934aaab3b13ceed0 | {"c_puct": 0.75, "challenger_simulations": 384, "current_simulations": 256, "hash": "0236da1b0848b73158e7307ac2c1350170c594074543e2ab934aaab3b13ceed0", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_cpuct_0_75 | 768:768 | 768:768 | ca8c3018d7bee263a5158dd6a9e795412ea6d136ef0ab6a0d29a3bb30157562e | {"c_puct": 0.75, "challenger_simulations": 768, "current_simulations": 768, "hash": "ca8c3018d7bee263a5158dd6a9e795412ea6d136ef0ab6a0d29a3bb30157562e", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_cpuct_0_75 | 1200:1200 | 1200:1200 | 6fad022371de3f949059fc659e55d03ec0284429eaa4932b6ac8f5bb423783ea | {"c_puct": 0.75, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "6fad022371de3f949059fc659e55d03ec0284429eaa4932b6ac8f5bb423783ea", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_0_75 | 1200:256 | 1200:256 | e9ecf74d43b10173ed0d73038bedb7fab08ef0452135c82aa819bff7075106fd | {"c_puct": 0.75, "challenger_simulations": 1200, "current_simulations": 256, "hash": "e9ecf74d43b10173ed0d73038bedb7fab08ef0452135c82aa819bff7075106fd", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_00 | 384:256 | 384:256 | faf2f586925d01736d376c5212400bda173e65a690e94df3cbe4529e83484b74 | {"c_puct": 1.0, "challenger_simulations": 384, "current_simulations": 256, "hash": "faf2f586925d01736d376c5212400bda173e65a690e94df3cbe4529e83484b74", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_cpuct_1_00 | 768:768 | 768:768 | 427deca4b9d822f8d3b5198b77927a7b9a929bb7d51546bbb7db15da2ca4b7ba | {"c_puct": 1.0, "challenger_simulations": 768, "current_simulations": 768, "hash": "427deca4b9d822f8d3b5198b77927a7b9a929bb7d51546bbb7db15da2ca4b7ba", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_cpuct_1_00 | 1200:1200 | 1200:1200 | b418d04fbe96ef9b5a0e65f561e423948e27e5d0400d8253a97f3b1363d62610 | {"c_puct": 1.0, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "b418d04fbe96ef9b5a0e65f561e423948e27e5d0400d8253a97f3b1363d62610", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_00 | 1200:256 | 1200:256 | dda2a5eae80d0afa6e1df56636dcc5adc9c879fdccc4eaa68361370d6e9243f2 | {"c_puct": 1.0, "challenger_simulations": 1200, "current_simulations": 256, "hash": "dda2a5eae80d0afa6e1df56636dcc5adc9c879fdccc4eaa68361370d6e9243f2", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_25_ref | 384:256 | 384:256 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_cpuct_1_25_ref | 768:768 | 768:768 | de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436 | {"c_puct": 1.25, "challenger_simulations": 768, "current_simulations": 768, "hash": "de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_cpuct_1_25_ref | 1200:1200 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_25_ref | 1200:256 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 256, "hash": "f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_50 | 384:256 | 384:256 | 6beabbdb512e3368d71204f1e8b6765fd1145855767893aaf3a7999cf998bd9e | {"c_puct": 1.5, "challenger_simulations": 384, "current_simulations": 256, "hash": "6beabbdb512e3368d71204f1e8b6765fd1145855767893aaf3a7999cf998bd9e", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_cpuct_1_50 | 768:768 | 768:768 | c554f1ee169a318af6b2a034e1c1c69bdab5cddd87f13dbeffcd5d8dfd0556ef | {"c_puct": 1.5, "challenger_simulations": 768, "current_simulations": 768, "hash": "c554f1ee169a318af6b2a034e1c1c69bdab5cddd87f13dbeffcd5d8dfd0556ef", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_cpuct_1_50 | 1200:1200 | 1200:1200 | ee060430e8c9641e1b30fecd40da48d251dcb7248711924f5378e0d4231af7cc | {"c_puct": 1.5, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "ee060430e8c9641e1b30fecd40da48d251dcb7248711924f5378e0d4231af7cc", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_50 | 1200:256 | 1200:256 | 3c9f0ebec4579bed67ef1535e73ac6ab4b1f415e5bbb8b3e0208afceb3218eee | {"c_puct": 1.5, "challenger_simulations": 1200, "current_simulations": 256, "hash": "3c9f0ebec4579bed67ef1535e73ac6ab4b1f415e5bbb8b3e0208afceb3218eee", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_75 | 384:256 | 384:256 | 271921e6eb7ca6333aaa3a14bf0193cb66ef3281f80215b6f2bc467fac943dc7 | {"c_puct": 1.75, "challenger_simulations": 384, "current_simulations": 256, "hash": "271921e6eb7ca6333aaa3a14bf0193cb66ef3281f80215b6f2bc467fac943dc7", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_cpuct_1_75 | 768:768 | 768:768 | e0a31bf097d1a1cbb7f67e91be2b4f738eca9d16f2ef9393795e21a8406b386b | {"c_puct": 1.75, "challenger_simulations": 768, "current_simulations": 768, "hash": "e0a31bf097d1a1cbb7f67e91be2b4f738eca9d16f2ef9393795e21a8406b386b", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_cpuct_1_75 | 1200:1200 | 1200:1200 | 093636a6f82ccae85606029d9ade23d69fd9f2b4057a89644239f345081c1283 | {"c_puct": 1.75, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "093636a6f82ccae85606029d9ade23d69fd9f2b4057a89644239f345081c1283", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_1_75 | 1200:256 | 1200:256 | bea3874f891b1f38766f9fcb67e9ec812bc39fea4fb62d7cf43ac35f4c2f8918 | {"c_puct": 1.75, "challenger_simulations": 1200, "current_simulations": 256, "hash": "bea3874f891b1f38766f9fcb67e9ec812bc39fea4fb62d7cf43ac35f4c2f8918", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_2_00 | 384:256 | 384:256 | 035f25fe16c1a868dd0fd4ac5fb43be8e360a9f68c2d6cfa871f58b04099ac07 | {"c_puct": 2.0, "challenger_simulations": 384, "current_simulations": 256, "hash": "035f25fe16c1a868dd0fd4ac5fb43be8e360a9f68c2d6cfa871f58b04099ac07", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_cpuct_2_00 | 768:768 | 768:768 | 019d5d1d9d2680cd4afe0da13e6339def5fddd5c06f64f8e71f8c07819192b30 | {"c_puct": 2.0, "challenger_simulations": 768, "current_simulations": 768, "hash": "019d5d1d9d2680cd4afe0da13e6339def5fddd5c06f64f8e71f8c07819192b30", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_cpuct_2_00 | 1200:1200 | 1200:1200 | f575208bed11c3281926fae0c0e4e2003b078e9d955860959add93b7d2de7a27 | {"c_puct": 2.0, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "f575208bed11c3281926fae0c0e4e2003b078e9d955860959add93b7d2de7a27", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_cpuct_2_00 | 1200:256 | 1200:256 | 3f6c602728ea5c7bd36cabc88145fd65c23f937065160cc20f6ff14aa29d5b54 | {"c_puct": 2.0, "challenger_simulations": 1200, "current_simulations": 256, "hash": "3f6c602728ea5c7bd36cabc88145fd65c23f937065160cc20f6ff14aa29d5b54", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_equalized_384_384 | 384:256 | 384:384 | 390620b7fd3bf163ad1a23a1076382f681ccf93fce263e225c10b64416635f6a | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 384, "hash": "390620b7fd3bf163ad1a23a1076382f681ccf93fce263e225c10b64416635f6a", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_equalized_384_384 | 768:768 | 768:768 | de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436 | {"c_puct": 1.25, "challenger_simulations": 768, "current_simulations": 768, "hash": "de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_equalized_384_384 | 1200:1200 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_equalized_384_384 | 1200:256 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 256, "hash": "f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_equalized_768_768 | 384:256 | 384:256 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_equalized_768_768 | 768:768 | 768:768 | de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436 | {"c_puct": 1.25, "challenger_simulations": 768, "current_simulations": 768, "hash": "de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_equalized_768_768 | 1200:1200 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_equalized_768_768 | 1200:256 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 256, "hash": "f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_current_384_512 | 384:256 | 384:512 | 79f6904cddd5c4468b60ce7843568744e60154f1b306e7533865d77037fafb21 | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 512, "hash": "79f6904cddd5c4468b60ce7843568744e60154f1b306e7533865d77037fafb21", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 512, "version": "v1"} |
| no_tactical_high_current_384_512 | 768:768 | 768:768 | de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436 | {"c_puct": 1.25, "challenger_simulations": 768, "current_simulations": 768, "hash": "de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_high_current_384_512 | 1200:1200 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_current_384_512 | 1200:256 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 256, "hash": "f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_challenger_512_384 | 384:256 | 512:384 | 2ac0d851e4c1151ef009afca04e6bbce832121c8430ca0ae2bf623385e372d40 | {"c_puct": 1.25, "challenger_simulations": 512, "current_simulations": 384, "hash": "2ac0d851e4c1151ef009afca04e6bbce832121c8430ca0ae2bf623385e372d40", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 512, "version": "v1"} |
| no_tactical_high_challenger_512_384 | 768:768 | 768:768 | de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436 | {"c_puct": 1.25, "challenger_simulations": 768, "current_simulations": 768, "hash": "de13545426faafe02396b419d3ab40c59a587538d0bd48d31599725266894436", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 768, "version": "v1"} |
| no_tactical_high_challenger_512_384 | 1200:1200 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_challenger_512_384 | 1200:256 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 256, "hash": "f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_current_768_1024 | 384:256 | 384:256 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_high_current_768_1024 | 768:768 | 768:1024 | 19978024ca61506ee0137f5bf71da594ef48835443e86d13e4ef05b92b7e4f22 | {"c_puct": 1.25, "challenger_simulations": 768, "current_simulations": 1024, "hash": "19978024ca61506ee0137f5bf71da594ef48835443e86d13e4ef05b92b7e4f22", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1024, "version": "v1"} |
| no_tactical_high_current_768_1024 | 1200:1200 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_current_768_1024 | 1200:256 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 256, "hash": "f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_challenger_1024_768 | 384:256 | 384:256 | bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca | {"c_puct": 1.25, "challenger_simulations": 384, "current_simulations": 256, "hash": "bef9390c802eb626f3481789f3982f195dac6c70ef6676cdf2ddd82e36c67aca", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 384, "version": "v1"} |
| no_tactical_high_challenger_1024_768 | 768:768 | 1024:768 | b2b522ceac56a49c48387a7b13dfd6aa2f772570a8a307f47d85f0ecebdf7ef4 | {"c_puct": 1.25, "challenger_simulations": 1024, "current_simulations": 768, "hash": "b2b522ceac56a49c48387a7b13dfd6aa2f772570a8a307f47d85f0ecebdf7ef4", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1024, "version": "v1"} |
| no_tactical_high_challenger_1024_768 | 1200:1200 | 1200:1200 | 37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 1200, "hash": "37167bcd344771a1dea0ef06d57398a625354201a0be656202734109e6c5020d", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |
| no_tactical_high_challenger_1024_768 | 1200:256 | 1200:256 | f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8 | {"c_puct": 1.25, "challenger_simulations": 1200, "current_simulations": 256, "hash": "f14a2006941c7c7fe2a58916a41cf4d30cc36a826443a837953c36be631807c8", "kind": "arena_eval", "player_mode": "puct", "search_options": {"fpu_mode": "zero", "normalize_values": false, "reuse_subtree": false, "root_policy_mode": "deterministic", "root_temperature": 0.0, "tactical_root_bias": 0.0}, "simulations": 1200, "version": "v1"} |

## Medium DS Table

| Lane | 384-axis | 768-axis | 1200:1200 | 1200:256 | Balanced |
|---|---|---|---|---|---|
| no_tactical_cpuct_0_75 | -0.2812 | -0.1406 | -0.5859 | +0.0000 | -0.3896 |
| no_tactical_cpuct_1_00 | -0.4609 | +0.3906 | -0.0781 | -0.1641 | -0.2178 |
| no_tactical_cpuct_1_25_ref | -0.2344 | +0.1250 | +0.3906 | -0.1367 | +0.0000 |
| no_tactical_cpuct_1_50 | -0.4570 | -0.3984 | -0.0547 | -0.2656 | -0.6279 |
| no_tactical_cpuct_1_75 | -0.4023 | -0.7656 | -0.1094 | -0.4414 | -0.8145 |
| no_tactical_cpuct_2_00 | -0.1914 | -0.6719 | -0.3281 | -0.0352 | -0.5098 |
| no_tactical_equalized_384_384 | -0.3203 | +0.1250 | +0.3906 | -0.1367 | -0.0859 |
| no_tactical_equalized_768_768 | -0.2344 | +0.1250 | +0.3906 | -0.1367 | +0.0000 |
| no_tactical_high_current_384_512 | -0.3789 | +0.1250 | +0.3906 | -0.1367 | -0.1445 |
| no_tactical_high_challenger_512_384 | -0.3789 | +0.1250 | +0.3906 | -0.1367 | -0.1445 |
| no_tactical_high_current_768_1024 | -0.2344 | -0.4023 | +0.3906 | -0.1367 | -0.2637 |
| no_tactical_high_challenger_1024_768 | -0.2344 | -0.4023 | +0.3906 | -0.1367 | -0.2637 |

## Fixed Large DS Table

| Lane | 384-axis | 768-axis | 1200:1200 | 1200:256 | Balanced |
|---|---|---|---|---|---|
| no_tactical_cpuct_1_25_ref | -0.3099 | -0.1146 | +0.2812 | -0.1159 | +0.0000 |
| no_tactical_equalized_384_384 | -0.4688 | -0.1146 | +0.2812 | -0.1159 | -0.1589 |
| no_tactical_high_current_384_512 | -0.4896 | -0.1146 | +0.2812 | -0.1159 | -0.1797 |
| no_tactical_high_challenger_512_384 | -0.4896 | -0.1146 | +0.2812 | -0.1159 | -0.1797 |
| no_tactical_cpuct_1_00 | -0.5143 | +0.2630 | -0.1120 | -0.1706 | -0.1276 |
| no_tactical_high_current_768_1024 | -0.3099 | -0.4297 | +0.2812 | -0.1159 | -0.1576 |
| no_tactical_high_challenger_1024_768 | -0.3099 | -0.4297 | +0.2812 | -0.1159 | -0.1576 |

## Held-Out Mean And Worst-Suite DS Table

| Lane | Mean 384-axis | Worst 384-axis | Mean 768-axis | Worst 768-axis | Mean 1200:1200 | Mean 1200:256 |
|---|---|---|---|---|---|---|
| no_tactical_cpuct_1_25_ref | -0.3084 | -0.3255 | -0.1484 | -0.1771 | +0.2721 | -0.1107 |
| no_tactical_cpuct_1_00 | -0.4978 | -0.5104 | +0.2713 | +0.2526 | -0.1068 | -0.1753 |
| no_tactical_high_current_768_1024 | -0.3084 | -0.3255 | -0.4373 | -0.4622 | +0.2721 | -0.1107 |
| no_tactical_high_challenger_1024_768 | -0.3084 | -0.3255 | -0.4373 | -0.4622 | +0.2721 | -0.1107 |

## Bootstrap CI

| Comparison | Mean | Lower 95% | Upper 95% |
|---|---|---|---|
| no_tactical_cpuct_1_25_ref_minus_baseline_384_256 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_cpuct_1_25_ref_minus_baseline_768_768 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_cpuct_1_25_ref_minus_baseline_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_cpuct_1_25_ref_minus_baseline_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_cpuct_1_00_minus_baseline_384_256 | -0.1895 | -0.2196 | -0.1589 |
| no_tactical_cpuct_1_00_minus_baseline_768_768 | +0.4197 | +0.3728 | +0.4679 |
| no_tactical_cpuct_1_00_minus_baseline_1200_1200 | -0.3789 | -0.4006 | -0.3576 |
| no_tactical_cpuct_1_00_minus_baseline_1200_256 | -0.0647 | -0.0803 | -0.0490 |
| no_tactical_high_current_768_1024_minus_baseline_384_256 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_high_current_768_1024_minus_baseline_768_768 | -0.2888 | -0.3292 | -0.2498 |
| no_tactical_high_current_768_1024_minus_baseline_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_high_current_768_1024_minus_baseline_1200_256 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_high_challenger_1024_768_minus_baseline_384_256 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_high_challenger_1024_768_minus_baseline_768_768 | -0.2888 | -0.3292 | -0.2498 |
| no_tactical_high_challenger_1024_768_minus_baseline_1200_1200 | +0.0000 | +0.0000 | +0.0000 |
| no_tactical_high_challenger_1024_768_minus_baseline_1200_256 | +0.0000 | +0.0000 | +0.0000 |

## P0/P1 Split For 384-axis

| Lane | Mean P0 | Mean P1 | Gap | Mean duplicates |
|---|---|---|---|---|
| no_tactical_cpuct_1_25_ref | +0.6157 | +0.9240 | +0.3084 | +768.0000 |
| no_tactical_cpuct_1_00 | +0.3411 | +0.8390 | +0.4978 | +768.0000 |
| no_tactical_high_current_768_1024 | +0.6157 | +0.9240 | +0.3084 | +768.0000 |
| no_tactical_high_challenger_1024_768 | +0.6157 | +0.9240 | +0.3084 | +768.0000 |

## Runtime Cost

| Lane | Mean move latency | P95 move latency | Avg sims/game | Slowdown vs baseline |
|---|---|---|---|---|
| no_tactical_cpuct_1_25_ref | +49.48 | +118.94 | +384.00 | +0.000 |
| no_tactical_cpuct_1_00 | +49.74 | +118.98 | +384.00 | +0.005 |
| no_tactical_high_current_768_1024 | +49.54 | +119.05 | +384.00 | +0.001 |
| no_tactical_high_challenger_1024_768 | +49.64 | +119.33 | +384.00 | +0.003 |

## Gate Classification

- no_tactical_cpuct_1_25_ref: `high_search_breakthrough` using gate budgets `384:256,1200:1200,1200:256,256:768`

## Decision Summary

- Budget-lane c_puct used after medium c_puct pass: `+1.25`
- Fixed-large carry lanes: `no_tactical_cpuct_1_25_ref,no_tactical_equalized_384_384,no_tactical_high_current_384_512,no_tactical_high_challenger_512_384,no_tactical_cpuct_1_00,no_tactical_high_current_768_1024,no_tactical_high_challenger_1024_768`
- Held-out carry lanes: `no_tactical_cpuct_1_25_ref,no_tactical_cpuct_1_00,no_tactical_high_current_768_1024,no_tactical_high_challenger_1024_768`
- Final classification: `cpuct_sensitive_runtime`
