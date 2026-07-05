# AlphaZero-Lite Current 384:256 Failure Family Trace Results

**Classification**: `diffuse_failure_no_single_intervention`

## Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Promoted Runtime Profile Confirmation

- runtime profile: `{"artifact":"model-artifact/current","budget_pair":"384:256","c_puct_schedule":{"default_c_puct":1.25,"overrides":{"768:768":0.9}},"default_c_puct":1.25,"effective_c_puct_384_256":1.25,"root_policy_mode":"deterministic","root_prior_transform":null,"tactical_root_bias":0.0,"value_transform":null}`

## Suite Hashes

- heldout_seed43_large: `5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9`
- heldout_seed44_large: `323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620`
- heldout_seed45_large: `ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda`
- heldout_seed46_large: `7d8ef94fed48ec58ed11aab9818954915ba5158d8cd90291fb8b6970489b0f04`
- heldout_seed47_large: `2b689b7444e14c7880833570059775184d823d5af0f31cf77ea2f0779dd3d03b`
- heldout_seed48_large: `b3d6a64a3ae86fb46b8859e5b75899dd9703ad57bcd9de330e15ee674a56c262`
- large_eval: `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4`
- medium_eval: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`

## 384:256 Outcome Summary

- games: `5376`
- mean disadvantaged outcome: `-0.2011`
- mean DS contribution: `+0.0349`
- seat context means: `{"challenger_player_0":-0.270833,"challenger_player_1":-0.131324}`

## Root Trace Telemetry Summary

- rows: `89823`
- phase counts: `{"late":25741,"mid":23475,"opening":40607}`
- seat context counts: `{"challenger_player_0_disadvantaged_player_1":43927,"challenger_player_1_disadvantaged_player_0":45896}`
- mean selected visit share: `+0.6646`
- mean top-2 visit margin: `+0.4657`
- selected allows opponent capture rate: `+0.2920`
- selected allows opponent extra-turn rate: `+0.4519`

## Counterfactual Probe Summary

- probed states: `1024`
- states with any improvement >= +0.25: `242`
- mean best improvement: `+0.2516`
- deeper-search mean delta: `+0.3123`
- tactical mean delta: `+0.1152`
- raw-policy mean delta: `-0.0218`

## Failure-Family Taxonomy Table

| Family | States | Openings | Mean imp | Median imp | CI95 | Rate |
|---|---|---|---|---|---|---|
| no_local_better_move_found | 779 | 604 | +0.0000 | +0.0000 | [+0.0000, +0.0000] | +0.0000 |
| opening_family_specific | 562 | 406 | +0.2396 | +0.0000 | [+0.1975, +0.2841] | +0.2242 |
| tactical_capture_miss | 124 | 112 | +1.0914 | +1.0000 | [+0.9919, +1.1909] | +1.0000 |
| value_miscalibrated_root | 117 | 106 | +0.8917 | +0.6667 | [+0.8062, +0.9801] | +1.0000 |
| search_budget_limited | 109 | 102 | +1.3792 | +1.3333 | [+1.2661, +1.4893] | +0.9633 |
| tactical_capture_blunder | 62 | 57 | +0.8763 | +0.6667 | [+0.7527, +1.0054] | +1.0000 |
| policy_prior_trap | 2 | 2 | +0.8333 | +0.8333 | [+0.3333, +1.3333] | +1.0000 |

## Family Overlap Matrix

| Family | no_local_better_move_found | opening_family_specific | policy_prior_trap | search_budget_limited | tactical_capture_blunder | tactical_capture_miss | value_miscalibrated_root |
|---|---|---|---|---|---|---|---|
| no_local_better_move_found | 779 | 435 | 0 | 0 | 0 | 0 | 0 |
| opening_family_specific | 435 | 562 | 2 | 59 | 28 | 67 | 65 |
| policy_prior_trap | 0 | 2 | 2 | 1 | 0 | 0 | 0 |
| search_budget_limited | 0 | 59 | 1 | 109 | 27 | 70 | 44 |
| tactical_capture_blunder | 0 | 28 | 0 | 27 | 62 | 45 | 27 |
| tactical_capture_miss | 0 | 67 | 0 | 70 | 45 | 124 | 60 |
| value_miscalibrated_root | 0 | 65 | 0 | 44 | 27 | 60 | 117 |

## Top 25 Most Costly Root Decisions

| Trace | State | Prefix | Ply | Seat | Selected | Best imp | Families |
|---|---|---|---|---|---|---|---|
| trace-03014 | 90c26a1cdd60 | 2,3,8 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-12684 | 90c26a1cdd60 | 2,3 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-18434 | 90c26a1cdd60 | 2,3,8 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-24896 | 90c26a1cdd60 | 2,3 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-25849 | 90c26a1cdd60 | 2,3,8 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-31909 | 90c26a1cdd60 | 2,3 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-42806 | 90c26a1cdd60 | 2,3,8,7,11 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-79707 | 90c26a1cdd60 | 2,3,8,7 | 11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-45925 | 424eff5c9d6e | 5,6,0,7,1,5 | 14 | 1 | 2 | +2.0000 |  |
| trace-25248 | bef3f629e910 | 1,9,0 | 20 | 0 | 1 | +2.0000 | opening_family_specific,search_budget_limited |
| trace-28271 | bef3f629e910 | 1,9,0,10 | 20 | 0 | 1 | +2.0000 | opening_family_specific,search_budget_limited |
| trace-70201 | a77b93beff23 | 2,3,9,0,11,1 | 20 | 0 | 4 | +2.0000 | search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-30826 | 3aaccc3c3757 | 5,7,8,0,9,3 | 13 | 1 | 2 | +2.0000 | search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-43846 | 7537123d7d9b | 5,6,0 | 13 | 1 | 2 | +2.0000 | value_miscalibrated_root |
| trace-83628 | 835cd492b1f3 | 3,6,0,8,1 | 13 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-76341 | f93530318e82 | 2,1,10,4,11 | 16 | 1 | 2 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-20077 | dce1dd87bf1a | 2,1,8,6 | 19 | 1 | 1 | +2.0000 | search_budget_limited |
| trace-34353 | dce1dd87bf1a | 2,1,8,6,4,9 | 19 | 1 | 1 | +2.0000 | search_budget_limited |
| trace-11274 | cb51cd1ab02a | 1,8,9,1 | 21 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-31359 | cb51cd1ab02a | 1,8,9,1,11 | 21 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-23870 | 60049a3d1ffa | 0,8,7,3 | 22 | 0 | 3 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-82712 | 60049a3d1ffa | 0,8,7,3,10 | 22 | 0 | 3 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-67414 | 36bd9794eabd | 3,11,0,8,10 | 14 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-01746 | 93912eb22469 | 5,10,2 | 16 | 1 | 0 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-81240 | 93912eb22469 | 5,10,2,5 | 16 | 1 | 0 | +2.0000 | search_budget_limited,tactical_capture_miss |

## Top 25 Search-Budget-Limited States

| Trace | State | Prefix | Best imp |
|---|---|---|---|
| trace-30826 | 3aaccc3c3757 | 5,7,8,0,9,3 | +2.0000 |
| trace-81240 | 93912eb22469 | 5,10,2,5 | +2.0000 |
| trace-01746 | 93912eb22469 | 5,10,2 | +2.0000 |
| trace-07614 | 24af8e7738c2 | 4,9,2,0,7 | +2.0000 |
| trace-61430 | 6482467fa851 | 4,6,0,11,0,8 | +2.0000 |
| trace-66873 | 998256567c74 | 4,10,2,0,11,4 | +2.0000 |
| trace-63919 | 5be54d6183e5 | 3,8,10,5,6 | +2.0000 |
| trace-83628 | 835cd492b1f3 | 3,6,0,8,1 | +2.0000 |
| trace-67414 | 36bd9794eabd | 3,11,0,8,10 | +2.0000 |
| trace-82502 | 96b87806a7e0 | 3,10,1,3,9 | +2.0000 |
| trace-68817 | 09375ed43de5 | 2,4,11,5 | +2.0000 |
| trace-70201 | a77b93beff23 | 2,3,9,0,11,1 | +2.0000 |
| trace-42806 | 90c26a1cdd60 | 2,3,8,7,11 | +2.0000 |
| trace-79707 | 90c26a1cdd60 | 2,3,8,7 | +2.0000 |
| trace-25849 | 90c26a1cdd60 | 2,3,8 | +2.0000 |
| trace-18434 | 90c26a1cdd60 | 2,3,8 | +2.0000 |
| trace-03014 | 90c26a1cdd60 | 2,3,8 | +2.0000 |
| trace-28727 | 55a2bbf499af | 2,3,6,0,10 | +2.0000 |
| trace-31909 | 90c26a1cdd60 | 2,3 | +2.0000 |
| trace-24896 | 90c26a1cdd60 | 2,3 | +2.0000 |
| trace-12684 | 90c26a1cdd60 | 2,3 | +2.0000 |
| trace-34353 | dce1dd87bf1a | 2,1,8,6,4,9 | +2.0000 |
| trace-20077 | dce1dd87bf1a | 2,1,8,6 | +2.0000 |
| trace-76341 | f93530318e82 | 2,1,10,4,11 | +2.0000 |
| trace-03479 | adaae9232e92 | 2,0,11,4,9 | +2.0000 |

## Top 25 Tactical Capture/Extra-Turn Failures

| Trace | State | Prefix | Best imp |
|---|---|---|---|
| trace-30826 | 3aaccc3c3757 | 5,7,8,0,9,3 | +2.0000 |
| trace-81240 | 93912eb22469 | 5,10,2,5 | +2.0000 |
| trace-01746 | 93912eb22469 | 5,10,2 | +2.0000 |
| trace-07614 | 24af8e7738c2 | 4,9,2,0,7 | +2.0000 |
| trace-61430 | 6482467fa851 | 4,6,0,11,0,8 | +2.0000 |
| trace-63919 | 5be54d6183e5 | 3,8,10,5,6 | +2.0000 |
| trace-83628 | 835cd492b1f3 | 3,6,0,8,1 | +2.0000 |
| trace-67414 | 36bd9794eabd | 3,11,0,8,10 | +2.0000 |
| trace-82502 | 96b87806a7e0 | 3,10,1,3,9 | +2.0000 |
| trace-68817 | 09375ed43de5 | 2,4,11,5 | +2.0000 |
| trace-70201 | a77b93beff23 | 2,3,9,0,11,1 | +2.0000 |
| trace-42806 | 90c26a1cdd60 | 2,3,8,7,11 | +2.0000 |
| trace-79707 | 90c26a1cdd60 | 2,3,8,7 | +2.0000 |
| trace-25849 | 90c26a1cdd60 | 2,3,8 | +2.0000 |
| trace-18434 | 90c26a1cdd60 | 2,3,8 | +2.0000 |
| trace-03014 | 90c26a1cdd60 | 2,3,8 | +2.0000 |
| trace-61019 | 977a0585f247 | 2,3,6,0,9,3 | +2.0000 |
| trace-28727 | 55a2bbf499af | 2,3,6,0,10 | +2.0000 |
| trace-31909 | 90c26a1cdd60 | 2,3 | +2.0000 |
| trace-24896 | 90c26a1cdd60 | 2,3 | +2.0000 |
| trace-12684 | 90c26a1cdd60 | 2,3 | +2.0000 |
| trace-76341 | f93530318e82 | 2,1,10,4,11 | +2.0000 |
| trace-78528 | 303473999418 | 1,9,5,10,3,11 | +2.0000 |
| trace-31359 | cb51cd1ab02a | 1,8,9,1,11 | +2.0000 |
| trace-11274 | cb51cd1ab02a | 1,8,9,1 | +2.0000 |

## Recommended Next Intervention

- stop local target/intervention work and consider broader architecture or training-objective changes

## Deduplication Addendum

- original probes: `1024`
- unique exact states: `848`
- unique opening prefixes: `760`
- original improvable rows >= +0.25: `242`
- unique exact states improvable >= +0.25: `201`

### Top 25 Unique Exact States

| Trace | State | Prefix | Seat | Selected | Best imp | Families |
|---|---|---|---|---|---|---|
| trace-79707 | 90c26a1cdd60 | 2,3,8,7 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-70201 | a77b93beff23 | 2,3,9,0,11,1 | 0 | 4 | +2.0000 | search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-45925 | 424eff5c9d6e | 5,6,0,7,1,5 | 1 | 2 | +2.0000 |  |
| trace-28271 | bef3f629e910 | 1,9,0,10 | 0 | 1 | +2.0000 | opening_family_specific,search_budget_limited |
| trace-83628 | 835cd492b1f3 | 3,6,0,8,1 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-82712 | 60049a3d1ffa | 0,8,7,3,10 | 0 | 3 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-76341 | f93530318e82 | 2,1,10,4,11 | 1 | 2 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-43846 | 7537123d7d9b | 5,6,0 | 1 | 2 | +2.0000 | value_miscalibrated_root |
| trace-34353 | dce1dd87bf1a | 2,1,8,6,4,9 | 1 | 1 | +2.0000 | search_budget_limited |
| trace-31359 | cb51cd1ab02a | 1,8,9,1,11 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-30826 | 3aaccc3c3757 | 5,7,8,0,9,3 | 1 | 2 | +2.0000 | search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-81240 | 93912eb22469 | 5,10,2,5 | 1 | 0 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-67414 | 36bd9794eabd | 3,11,0,8,10 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-63919 | 5be54d6183e5 | 3,8,10,5,6 | 0 | 0 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-61019 | 977a0585f247 | 2,3,6,0,9,3 | 1 | 2 | +2.0000 | tactical_capture_blunder,tactical_capture_miss |
| trace-20505 | 353d29b8ccf1 | 1,8,10,4,6 | 1 | 1 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_blunder,tactical_capture_miss,value_miscalibrated_root |
| trace-03479 | adaae9232e92 | 2,0,11,4,9 | 1 | 2 | +2.0000 | search_budget_limited |
| trace-61430 | 6482467fa851 | 4,6,0,11,0,8 | 0 | 4 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-39626 | 8e83432e956f | 0,7,2,8,5,9 | 0 | 2 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-07614 | 24af8e7738c2 | 4,9,2,0,7 | 0 | 3 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-82502 | 96b87806a7e0 | 3,10,1,3,9 | 0 | 1 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-81016 | 43720c9a5c08 | 0,11,2 | 0 | 3 | +2.0000 | opening_family_specific,search_budget_limited |
| trace-66873 | 998256567c74 | 4,10,2,0,11,4 | 0 | 5 | +2.0000 | search_budget_limited |
| trace-53508 | 64cfebb243c3 | 5,7,10,4,11,0 | 0 | 3 | +2.0000 |  |
| trace-42353 | d1dd767d3c94 | 4,8,6,5,9,0 | 1 | 1 | +2.0000 | value_miscalibrated_root |

### Top 25 Unique Opening Prefixes

| Trace | State | Prefix | Seat | Selected | Best imp | Families |
|---|---|---|---|---|---|---|
| trace-79707 | 90c26a1cdd60 | 2,3,8,7 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-70201 | a77b93beff23 | 2,3,9,0,11,1 | 0 | 4 | +2.0000 | search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-45925 | 424eff5c9d6e | 5,6,0,7,1,5 | 1 | 2 | +2.0000 |  |
| trace-42806 | 90c26a1cdd60 | 2,3,8,7,11 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-31909 | 90c26a1cdd60 | 2,3 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-28271 | bef3f629e910 | 1,9,0,10 | 0 | 1 | +2.0000 | opening_family_specific,search_budget_limited |
| trace-25849 | 90c26a1cdd60 | 2,3,8 | 0 | 0 | +2.0000 | search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-25248 | bef3f629e910 | 1,9,0 | 0 | 1 | +2.0000 | opening_family_specific,search_budget_limited |
| trace-83628 | 835cd492b1f3 | 3,6,0,8,1 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_blunder,tactical_capture_miss |
| trace-82712 | 60049a3d1ffa | 0,8,7,3,10 | 0 | 3 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-76341 | f93530318e82 | 2,1,10,4,11 | 1 | 2 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-43846 | 7537123d7d9b | 5,6,0 | 1 | 2 | +2.0000 | value_miscalibrated_root |
| trace-34353 | dce1dd87bf1a | 2,1,8,6,4,9 | 1 | 1 | +2.0000 | search_budget_limited |
| trace-31359 | cb51cd1ab02a | 1,8,9,1,11 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-30826 | 3aaccc3c3757 | 5,7,8,0,9,3 | 1 | 2 | +2.0000 | search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-23870 | 60049a3d1ffa | 0,8,7,3 | 0 | 3 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-20077 | dce1dd87bf1a | 2,1,8,6 | 1 | 1 | +2.0000 | search_budget_limited |
| trace-11274 | cb51cd1ab02a | 1,8,9,1 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-81240 | 93912eb22469 | 5,10,2,5 | 1 | 0 | +2.0000 | search_budget_limited,tactical_capture_miss |
| trace-67414 | 36bd9794eabd | 3,11,0,8,10 | 1 | 4 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss |
| trace-63919 | 5be54d6183e5 | 3,8,10,5,6 | 0 | 0 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_miss,value_miscalibrated_root |
| trace-61019 | 977a0585f247 | 2,3,6,0,9,3 | 1 | 2 | +2.0000 | tactical_capture_blunder,tactical_capture_miss |
| trace-20505 | 353d29b8ccf1 | 1,8,10,4,6 | 1 | 1 | +2.0000 | opening_family_specific,search_budget_limited,tactical_capture_blunder,tactical_capture_miss,value_miscalibrated_root |
| trace-03479 | adaae9232e92 | 2,0,11,4,9 | 1 | 2 | +2.0000 | search_budget_limited |
| trace-01746 | 93912eb22469 | 5,10,2 | 1 | 0 | +2.0000 | search_budget_limited,tactical_capture_miss |

### Family Counts After Exact-State Dedup

| Family | State count | Mean imp |
|---|---|---|
| no_local_better_move_found | 644 | +0.0000 |
| opening_family_specific | 463 | +0.2304 |
| tactical_capture_miss | 98 | +1.0374 |
| value_miscalibrated_root | 97 | +0.8866 |
| search_budget_limited | 88 | +1.2727 |
| tactical_capture_blunder | 48 | +0.8542 |
| policy_prior_trap | 2 | +0.8333 |
