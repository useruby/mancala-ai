# AlphaZero-Lite Budget-Anchored Policy-Only Microdistill Results

**Classification**: `micro_update_safe_but_too_weak`

## Current Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## PR #153 Candidate Hashes

- current_init_policy_only_e1_repro: checkpoint_sha256=`960c6df89b25d5587446982d9bfe91e6c8ac83078f636e20776a0dd738221087` weights_sha256=`e80f0e906ff08661d230334df76b37e5497ae14d3e98c2ddb457d95ec868480b`
- current_init_policy_only_e2_repro: checkpoint_sha256=`0d3c0ac67430084c5d5412fdf352f6545ca2a5f4635e56f7c69afbadae5c7cb9` weights_sha256=`55b0a78184d84ce7fd19f0ba2ebea35558d3437044d24baa4d4610251fae1ee6`

## DS Orientation Audit Confirmation

- audit passed: `True`
- reference helper reused: `True`
- bootstrap orientations explicit: `True`

## Dataset Split Audit

- selected rows: `48173`
- target rows: `10675`
- anchor rows: `37498`
- duplicate state count: `18771`
- row counts by budget: `{"1200:1200": 12503, "1200:256": 12495, "384:256": 10675, "768:768": 12500}`
- phase distribution: `{"late": 90, "mid": 31923, "opening": 16160}`
- seat distribution: `{"challenger_player_0": 24246, "challenger_player_1": 23927}`

## Lane Definitions

| Lane | LR | Target weight | Anchor weight | Max steps |
|---|---|---|---|---|
| micro_lr1e-6_anchor4_steps250 | 1e-06 | 1.0 | 4.0 | 250 |
| micro_lr1e-6_anchor8_steps250 | 1e-06 | 1.0 | 8.0 | 250 |
| micro_lr3e-6_anchor8_steps250 | 3e-06 | 1.0 | 8.0 | 250 |
| micro_lr1e-6_anchor8_steps500 | 1e-06 | 1.0 | 8.0 | 500 |
| ultra_micro_lr5e-7_anchor8_steps500 | 5e-07 | 1.0 | 8.0 | 500 |

## Checkpoint/Early-Stop Table

| Lane | Step | Candidate | Weights SHA256 | Probe pass | Early-stop |
|---|---|---|---|---|---|
| micro_lr1e-6_anchor4_steps250 | 50 | micro_lr1e-6_anchor4_steps250_step50 | 2e43fc658200194d27ac47022f29c9e570a44feba5ab62a2dfcb3a12647a8627 | False | continue |
| micro_lr1e-6_anchor4_steps250 | 100 | micro_lr1e-6_anchor4_steps250_step100 | 5b8da6c460340397df3af8612e88a636cfe77f4f0bb9827b9d0de06a2d20d820 | False | continue |
| micro_lr1e-6_anchor4_steps250 | 150 | micro_lr1e-6_anchor4_steps250_step150 | 20cdd48294709c1fd58e04b8c9860272c5d36b8ea3375bb6de6379c8935c990f | False | continue |
| micro_lr1e-6_anchor4_steps250 | 200 | micro_lr1e-6_anchor4_steps250_step200 | 0110a6983206cddbe067539e391ee058c39205782ff9767bed39c5a14827eea1 | False | continue |
| micro_lr1e-6_anchor4_steps250 | 250 | micro_lr1e-6_anchor4_steps250_step250 | e8062c8b3d05da888fc602897a102e71fe7403e124cac36762bbe294abccf99a | False | continue |
| micro_lr1e-6_anchor8_steps250 | 50 | micro_lr1e-6_anchor8_steps250_step50 | 42acf8263f19ed43e8752e2e65699f7bc075f37ef8636f63d6fe6684a3afad5a | False | continue |
| micro_lr1e-6_anchor8_steps250 | 100 | micro_lr1e-6_anchor8_steps250_step100 | 2e431e3d91ec12aec704c303197019e7686f714a37272d834a21a4fb80856842 | False | continue |
| micro_lr1e-6_anchor8_steps250 | 150 | micro_lr1e-6_anchor8_steps250_step150 | 5c79fe38c470807b1d76281f0349eefe12537c8082925fad4fee942ac068a4c3 | False | continue |
| micro_lr1e-6_anchor8_steps250 | 200 | micro_lr1e-6_anchor8_steps250_step200 | dd40c30f5fa7c8abf212c69171d27a2f46dd30341dd27a4c4156a3c069cdfeab | False | continue |
| micro_lr1e-6_anchor8_steps250 | 250 | micro_lr1e-6_anchor8_steps250_step250 | b8090ed3ea8243f316e4d1afca71c6c60108d6531a815e9bef3c34e2042a5539 | False | continue |
| micro_lr3e-6_anchor8_steps250 | 50 | micro_lr3e-6_anchor8_steps250_step50 | a7d79372f19b96615c7d1d5b3e22699ba8517e793fb9f3024f6b715011e136ab | False | continue |
| micro_lr3e-6_anchor8_steps250 | 100 | micro_lr3e-6_anchor8_steps250_step100 | 146e256f56ac2127a34d7b0bb57ad8a775b5a877d49200e5ad02a7b1279cb2f7 | False | continue |
| micro_lr3e-6_anchor8_steps250 | 150 | micro_lr3e-6_anchor8_steps250_step150 | cf1d8cd50b1297322621c95b3bea0f92eb5f8f2084b9a67bb355e76c3207c30a | False | continue |
| micro_lr3e-6_anchor8_steps250 | 200 | micro_lr3e-6_anchor8_steps250_step200 | bf61ede7d86b053d17a1fb123395ff0015bdde473bcd3a9c3defd59892000e31 | False | continue |
| micro_lr3e-6_anchor8_steps250 | 250 | micro_lr3e-6_anchor8_steps250_step250 | 5a4f30ac6e3d59605b3875d9a0a1851c2c4c37306b75b4ba8dbb343bf8bd5894 | False | continue |
| micro_lr1e-6_anchor8_steps500 | 50 | micro_lr1e-6_anchor8_steps500_step50 | 42acf8263f19ed43e8752e2e65699f7bc075f37ef8636f63d6fe6684a3afad5a | False | continue |
| micro_lr1e-6_anchor8_steps500 | 100 | micro_lr1e-6_anchor8_steps500_step100 | 2e431e3d91ec12aec704c303197019e7686f714a37272d834a21a4fb80856842 | False | continue |
| micro_lr1e-6_anchor8_steps500 | 150 | micro_lr1e-6_anchor8_steps500_step150 | 5c79fe38c470807b1d76281f0349eefe12537c8082925fad4fee942ac068a4c3 | False | continue |
| micro_lr1e-6_anchor8_steps500 | 200 | micro_lr1e-6_anchor8_steps500_step200 | dd40c30f5fa7c8abf212c69171d27a2f46dd30341dd27a4c4156a3c069cdfeab | False | continue |
| micro_lr1e-6_anchor8_steps500 | 250 | micro_lr1e-6_anchor8_steps500_step250 | b8090ed3ea8243f316e4d1afca71c6c60108d6531a815e9bef3c34e2042a5539 | False | continue |
| micro_lr1e-6_anchor8_steps500 | 300 | micro_lr1e-6_anchor8_steps500_step300 | a2db6a53e4aa3d40e4a14a84a46d6fcf10e48b8bfb3a0688f85de52e36f2b415 | False | continue |
| micro_lr1e-6_anchor8_steps500 | 350 | micro_lr1e-6_anchor8_steps500_step350 | e4de5e45f70d9ee1e97755718c52d40a702c95e26a16df26d42a4f723193976f | False | continue |
| micro_lr1e-6_anchor8_steps500 | 400 | micro_lr1e-6_anchor8_steps500_step400 | 963d1a1e7b964998a5f316acd185f28a6cd2eccc20d0c4537e1d822a63a36ff3 | False | continue |
| micro_lr1e-6_anchor8_steps500 | 450 | micro_lr1e-6_anchor8_steps500_step450 | c5f0845b60e044591a7b1b9eff40a6a6652f4601f0510c7feb1de8767ba0df61 | False | continue |
| micro_lr1e-6_anchor8_steps500 | 500 | micro_lr1e-6_anchor8_steps500_step500 | 4f571dd116c330d06528b9bf3bfb9df23bd0026633b59dfd79545a13076a6ab8 | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 50 | ultra_micro_lr5e-7_anchor8_steps500_step50 | 6ff65536694b37ac4994f950f36d71ab7a5acebd2e346df5e18d779c779d00a2 | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 100 | ultra_micro_lr5e-7_anchor8_steps500_step100 | d943eddaa34ef7197a63f12153a9e09242bfaa063a481dbdeb6882a7b94bbfb2 | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 150 | ultra_micro_lr5e-7_anchor8_steps500_step150 | 1eb9b818c3a53d121fecad16cdab7335694d78978de7143f397369eaa0cb1c9f | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 200 | ultra_micro_lr5e-7_anchor8_steps500_step200 | a4717eef6c773d117731dea887f0b9d9873904dea14bcbe283af977fa8d1aeae | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 250 | ultra_micro_lr5e-7_anchor8_steps500_step250 | 042442f1cd3b8d2c90127b084d5f432a71852e25919ad08c773a73bb7916ae39 | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 300 | ultra_micro_lr5e-7_anchor8_steps500_step300 | 7935ae7e2e7ab5230658de620e74d167a0768a23deae88e4f1a2249c176503d4 | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 350 | ultra_micro_lr5e-7_anchor8_steps500_step350 | 8d2520625cdc490ca80155d228f1e19d098f69ed0bd326908be5c124b50def4b | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 400 | ultra_micro_lr5e-7_anchor8_steps500_step400 | f4eaf461997430907d7f5060820f3546fab8f0330e3e12b1f30543df7b6fb5fb | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 450 | ultra_micro_lr5e-7_anchor8_steps500_step450 | 5c4a8a6edc13266da8ae3258123b002deaaeb305fcac136b9c9c6fd6990d7712 | False | continue |
| ultra_micro_lr5e-7_anchor8_steps500 | 500 | ultra_micro_lr5e-7_anchor8_steps500_step500 | e3b16b56ec4a22ad5bcce311fd7f87ac0a8d1de981a64ddb9a8aae254e39ac13 | False | continue |

## Policy Probe Table

| Candidate | Teacher top-1 | 384:256 top-1 gain | Raw top-1 change | Entropy | Legal fails |
|---|---|---|---|---|---|
| micro_lr1e-6_anchor4_steps250_step50 | +0.5105 | +0.0007 | +0.0028 | +1.6951 | 0 |
| micro_lr1e-6_anchor4_steps250_step100 | +0.5107 | +0.0008 | +0.0046 | +1.6955 | 0 |
| micro_lr1e-6_anchor4_steps250_step150 | +0.5110 | +0.0013 | +0.0066 | +1.6960 | 0 |
| micro_lr1e-6_anchor4_steps250_step200 | +0.5113 | +0.0017 | +0.0085 | +1.6964 | 0 |
| micro_lr1e-6_anchor4_steps250_step250 | +0.5115 | +0.0024 | +0.0103 | +1.6969 | 0 |
| micro_lr1e-6_anchor8_steps250_step50 | +0.5104 | +0.0007 | +0.0026 | +1.6951 | 0 |
| micro_lr1e-6_anchor8_steps250_step100 | +0.5106 | +0.0007 | +0.0044 | +1.6955 | 0 |
| micro_lr1e-6_anchor8_steps250_step150 | +0.5107 | +0.0008 | +0.0055 | +1.6959 | 0 |
| micro_lr1e-6_anchor8_steps250_step200 | +0.5110 | +0.0011 | +0.0067 | +1.6963 | 0 |
| micro_lr1e-6_anchor8_steps250_step250 | +0.5112 | +0.0018 | +0.0073 | +1.6967 | 0 |
| micro_lr3e-6_anchor8_steps250_step50 | +0.5110 | +0.0013 | +0.0061 | +1.6960 | 0 |
| micro_lr3e-6_anchor8_steps250_step100 | +0.5113 | +0.0018 | +0.0080 | +1.6970 | 0 |
| micro_lr3e-6_anchor8_steps250_step150 | +0.5110 | +0.0013 | +0.0087 | +1.6981 | 0 |
| micro_lr3e-6_anchor8_steps250_step200 | +0.5114 | +0.0018 | +0.0095 | +1.6989 | 0 |
| micro_lr3e-6_anchor8_steps250_step250 | +0.5118 | +0.0025 | +0.0098 | +1.6999 | 0 |
| micro_lr1e-6_anchor8_steps500_step50 | +0.5104 | +0.0007 | +0.0026 | +1.6951 | 0 |
| micro_lr1e-6_anchor8_steps500_step100 | +0.5106 | +0.0007 | +0.0044 | +1.6955 | 0 |
| micro_lr1e-6_anchor8_steps500_step150 | +0.5107 | +0.0008 | +0.0055 | +1.6959 | 0 |
| micro_lr1e-6_anchor8_steps500_step200 | +0.5110 | +0.0011 | +0.0067 | +1.6963 | 0 |
| micro_lr1e-6_anchor8_steps500_step250 | +0.5112 | +0.0018 | +0.0073 | +1.6967 | 0 |
| micro_lr1e-6_anchor8_steps500_step300 | +0.5112 | +0.0016 | +0.0082 | +1.6971 | 0 |
| micro_lr1e-6_anchor8_steps500_step350 | +0.5114 | +0.0017 | +0.0082 | +1.6974 | 0 |
| micro_lr1e-6_anchor8_steps500_step400 | +0.5113 | +0.0016 | +0.0084 | +1.6976 | 0 |
| micro_lr1e-6_anchor8_steps500_step450 | +0.5115 | +0.0019 | +0.0090 | +1.6980 | 0 |
| micro_lr1e-6_anchor8_steps500_step500 | +0.5114 | +0.0020 | +0.0095 | +1.6983 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step50 | +0.5100 | -0.0003 | +0.0015 | +1.6948 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step100 | +0.5103 | +0.0006 | +0.0024 | +1.6950 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step150 | +0.5104 | +0.0006 | +0.0033 | +1.6952 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step200 | +0.5107 | +0.0009 | +0.0043 | +1.6955 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step250 | +0.5108 | +0.0008 | +0.0048 | +1.6957 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step300 | +0.5107 | +0.0007 | +0.0057 | +1.6959 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step350 | +0.5109 | +0.0010 | +0.0061 | +1.6961 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step400 | +0.5110 | +0.0015 | +0.0066 | +1.6963 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step450 | +0.5111 | +0.0015 | +0.0071 | +1.6965 | 0 |
| ultra_micro_lr5e-7_anchor8_steps500_step500 | +0.5112 | +0.0017 | +0.0074 | +1.6966 | 0 |

## Value/Trunk Preservation Table

| Candidate | Preserved | Max abs diff | Changed keys |
|---|---|---|---|
| micro_lr1e-6_anchor4_steps250_step50 | True | +0.0000 | none |
| micro_lr1e-6_anchor4_steps250_step100 | True | +0.0000 | none |
| micro_lr1e-6_anchor4_steps250_step150 | True | +0.0000 | none |
| micro_lr1e-6_anchor4_steps250_step200 | True | +0.0000 | none |
| micro_lr1e-6_anchor4_steps250_step250 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps250_step50 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps250_step100 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps250_step150 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps250_step200 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps250_step250 | True | +0.0000 | none |
| micro_lr3e-6_anchor8_steps250_step50 | True | +0.0000 | none |
| micro_lr3e-6_anchor8_steps250_step100 | True | +0.0000 | none |
| micro_lr3e-6_anchor8_steps250_step150 | True | +0.0000 | none |
| micro_lr3e-6_anchor8_steps250_step200 | True | +0.0000 | none |
| micro_lr3e-6_anchor8_steps250_step250 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step50 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step100 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step150 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step200 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step250 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step300 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step350 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step400 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step450 | True | +0.0000 | none |
| micro_lr1e-6_anchor8_steps500_step500 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step50 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step100 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step150 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step200 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step250 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step300 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step350 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step400 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step450 | True | +0.0000 | none |
| ultra_micro_lr5e-7_anchor8_steps500_step500 | True | +0.0000 | none |

## Search-Aware Probe Table By Budget

| Candidate | Budget | Changed rate | Teacher agree | Root visit KL | Visit-share delta |
|---|---|---|---|---|---|
| micro_lr1e-6_anchor4_steps250_step50 | 384:256 | +0.0156 | +0.9844 | +0.0004 | +0.0002 |
| micro_lr1e-6_anchor4_steps250_step50 | 768:768 | +0.0156 | +0.9844 | +0.0031 | +0.0011 |
| micro_lr1e-6_anchor4_steps250_step50 | 1200:1200 | +0.0312 | +0.9688 | +0.0026 | -0.0051 |
| micro_lr1e-6_anchor4_steps250_step50 | 1200:256 | +0.0312 | +0.9688 | +0.0008 | +0.0010 |
| micro_lr1e-6_anchor4_steps250_step100 | 384:256 | +0.0000 | +1.0000 | +0.0008 | -0.0009 |
| micro_lr1e-6_anchor4_steps250_step100 | 768:768 | +0.0156 | +0.9844 | +0.0034 | -0.0028 |
| micro_lr1e-6_anchor4_steps250_step100 | 1200:1200 | +0.0469 | +0.9531 | +0.0013 | -0.0004 |
| micro_lr1e-6_anchor4_steps250_step100 | 1200:256 | +0.0469 | +0.9531 | +0.0141 | -0.0030 |
| micro_lr1e-6_anchor4_steps250_step150 | 384:256 | +0.0000 | +1.0000 | +0.0011 | -0.0024 |
| micro_lr1e-6_anchor4_steps250_step150 | 768:768 | +0.0312 | +0.9688 | +0.0031 | -0.0030 |
| micro_lr1e-6_anchor4_steps250_step150 | 1200:1200 | +0.0469 | +0.9531 | +0.0032 | -0.0029 |
| micro_lr1e-6_anchor4_steps250_step150 | 1200:256 | +0.0469 | +0.9531 | +0.0025 | -0.0009 |
| micro_lr1e-6_anchor4_steps250_step200 | 384:256 | +0.0156 | +0.9844 | +0.0016 | -0.0005 |
| micro_lr1e-6_anchor4_steps250_step200 | 768:768 | +0.0312 | +0.9688 | +0.0042 | -0.0031 |
| micro_lr1e-6_anchor4_steps250_step200 | 1200:1200 | +0.0469 | +0.9531 | +0.0021 | +0.0015 |
| micro_lr1e-6_anchor4_steps250_step200 | 1200:256 | +0.0312 | +0.9688 | +0.0029 | -0.0037 |
| micro_lr1e-6_anchor4_steps250_step250 | 384:256 | +0.0156 | +0.9844 | +0.0016 | -0.0004 |
| micro_lr1e-6_anchor4_steps250_step250 | 768:768 | +0.0156 | +0.9844 | +0.0061 | -0.0060 |
| micro_lr1e-6_anchor4_steps250_step250 | 1200:1200 | +0.0469 | +0.9531 | +0.0021 | -0.0019 |
| micro_lr1e-6_anchor4_steps250_step250 | 1200:256 | +0.0469 | +0.9531 | +0.0045 | -0.0044 |
| micro_lr1e-6_anchor8_steps250_step50 | 384:256 | +0.0156 | +0.9844 | +0.0004 | +0.0002 |
| micro_lr1e-6_anchor8_steps250_step50 | 768:768 | +0.0156 | +0.9844 | +0.0031 | +0.0015 |
| micro_lr1e-6_anchor8_steps250_step50 | 1200:1200 | +0.0312 | +0.9688 | +0.0027 | -0.0052 |
| micro_lr1e-6_anchor8_steps250_step50 | 1200:256 | +0.0312 | +0.9688 | +0.0009 | +0.0006 |
| micro_lr1e-6_anchor8_steps250_step100 | 384:256 | +0.0000 | +1.0000 | +0.0009 | -0.0009 |
| micro_lr1e-6_anchor8_steps250_step100 | 768:768 | +0.0000 | +1.0000 | +0.0036 | -0.0027 |
| micro_lr1e-6_anchor8_steps250_step100 | 1200:1200 | +0.0469 | +0.9531 | +0.0011 | -0.0008 |
| micro_lr1e-6_anchor8_steps250_step100 | 1200:256 | +0.0469 | +0.9531 | +0.0142 | -0.0022 |
| micro_lr1e-6_anchor8_steps250_step150 | 384:256 | +0.0156 | +0.9844 | +0.0009 | -0.0017 |
| micro_lr1e-6_anchor8_steps250_step150 | 768:768 | +0.0156 | +0.9844 | +0.0033 | -0.0026 |
| micro_lr1e-6_anchor8_steps250_step150 | 1200:1200 | +0.0625 | +0.9375 | +0.0029 | -0.0029 |
| micro_lr1e-6_anchor8_steps250_step150 | 1200:256 | +0.0312 | +0.9688 | +0.0022 | -0.0019 |
| micro_lr1e-6_anchor8_steps250_step200 | 384:256 | +0.0000 | +1.0000 | +0.0009 | -0.0017 |
| micro_lr1e-6_anchor8_steps250_step200 | 768:768 | +0.0469 | +0.9531 | +0.0037 | +0.0001 |
| micro_lr1e-6_anchor8_steps250_step200 | 1200:1200 | +0.0469 | +0.9531 | +0.0020 | -0.0002 |
| micro_lr1e-6_anchor8_steps250_step200 | 1200:256 | +0.0312 | +0.9688 | +0.0027 | -0.0017 |
| micro_lr1e-6_anchor8_steps250_step250 | 384:256 | +0.0000 | +1.0000 | +0.0014 | -0.0001 |
| micro_lr1e-6_anchor8_steps250_step250 | 768:768 | +0.0312 | +0.9688 | +0.0044 | -0.0012 |
| micro_lr1e-6_anchor8_steps250_step250 | 1200:1200 | +0.0781 | +0.9219 | +0.0023 | +0.0018 |
| micro_lr1e-6_anchor8_steps250_step250 | 1200:256 | +0.0312 | +0.9688 | +0.0031 | -0.0020 |
| micro_lr3e-6_anchor8_steps250_step50 | 384:256 | +0.0156 | +0.9844 | +0.0010 | -0.0010 |
| micro_lr3e-6_anchor8_steps250_step50 | 768:768 | +0.0312 | +0.9688 | +0.0045 | -0.0040 |
| micro_lr3e-6_anchor8_steps250_step50 | 1200:1200 | +0.0469 | +0.9531 | +0.0017 | +0.0010 |
| micro_lr3e-6_anchor8_steps250_step50 | 1200:256 | +0.0469 | +0.9531 | +0.0148 | -0.0002 |
| micro_lr3e-6_anchor8_steps250_step100 | 384:256 | +0.0000 | +1.0000 | +0.0013 | +0.0002 |
| micro_lr3e-6_anchor8_steps250_step100 | 768:768 | +0.0156 | +0.9844 | +0.0042 | -0.0032 |
| micro_lr3e-6_anchor8_steps250_step100 | 1200:1200 | +0.0625 | +0.9375 | +0.0017 | +0.0022 |
| micro_lr3e-6_anchor8_steps250_step100 | 1200:256 | +0.0312 | +0.9688 | +0.0036 | -0.0036 |
| micro_lr3e-6_anchor8_steps250_step150 | 384:256 | +0.0156 | +0.9844 | +0.0011 | -0.0000 |
| micro_lr3e-6_anchor8_steps250_step150 | 768:768 | +0.0312 | +0.9688 | +0.0061 | -0.0063 |
| micro_lr3e-6_anchor8_steps250_step150 | 1200:1200 | +0.0625 | +0.9375 | +0.0020 | -0.0014 |
| micro_lr3e-6_anchor8_steps250_step150 | 1200:256 | +0.0312 | +0.9688 | +0.0030 | -0.0033 |
| micro_lr3e-6_anchor8_steps250_step200 | 384:256 | +0.0156 | +0.9844 | +0.0009 | +0.0013 |
| micro_lr3e-6_anchor8_steps250_step200 | 768:768 | +0.0312 | +0.9688 | +0.0059 | -0.0063 |
| micro_lr3e-6_anchor8_steps250_step200 | 1200:1200 | +0.0625 | +0.9375 | +0.0019 | -0.0019 |
| micro_lr3e-6_anchor8_steps250_step200 | 1200:256 | +0.0312 | +0.9688 | +0.0030 | -0.0039 |
| micro_lr3e-6_anchor8_steps250_step250 | 384:256 | +0.0156 | +0.9844 | +0.0011 | +0.0018 |
| micro_lr3e-6_anchor8_steps250_step250 | 768:768 | +0.0312 | +0.9688 | +0.0067 | -0.0027 |
| micro_lr3e-6_anchor8_steps250_step250 | 1200:1200 | +0.0469 | +0.9531 | +0.0017 | -0.0010 |
| micro_lr3e-6_anchor8_steps250_step250 | 1200:256 | +0.0312 | +0.9688 | +0.0033 | -0.0062 |
| micro_lr1e-6_anchor8_steps500_step50 | 384:256 | +0.0156 | +0.9844 | +0.0004 | +0.0002 |
| micro_lr1e-6_anchor8_steps500_step50 | 768:768 | +0.0156 | +0.9844 | +0.0031 | +0.0015 |
| micro_lr1e-6_anchor8_steps500_step50 | 1200:1200 | +0.0312 | +0.9688 | +0.0027 | -0.0052 |
| micro_lr1e-6_anchor8_steps500_step50 | 1200:256 | +0.0312 | +0.9688 | +0.0009 | +0.0006 |
| micro_lr1e-6_anchor8_steps500_step100 | 384:256 | +0.0000 | +1.0000 | +0.0009 | -0.0009 |
| micro_lr1e-6_anchor8_steps500_step100 | 768:768 | +0.0000 | +1.0000 | +0.0036 | -0.0027 |
| micro_lr1e-6_anchor8_steps500_step100 | 1200:1200 | +0.0469 | +0.9531 | +0.0011 | -0.0008 |
| micro_lr1e-6_anchor8_steps500_step100 | 1200:256 | +0.0469 | +0.9531 | +0.0142 | -0.0022 |
| micro_lr1e-6_anchor8_steps500_step150 | 384:256 | +0.0156 | +0.9844 | +0.0009 | -0.0017 |
| micro_lr1e-6_anchor8_steps500_step150 | 768:768 | +0.0156 | +0.9844 | +0.0033 | -0.0026 |
| micro_lr1e-6_anchor8_steps500_step150 | 1200:1200 | +0.0625 | +0.9375 | +0.0029 | -0.0029 |
| micro_lr1e-6_anchor8_steps500_step150 | 1200:256 | +0.0312 | +0.9688 | +0.0022 | -0.0019 |
| micro_lr1e-6_anchor8_steps500_step200 | 384:256 | +0.0000 | +1.0000 | +0.0009 | -0.0017 |
| micro_lr1e-6_anchor8_steps500_step200 | 768:768 | +0.0469 | +0.9531 | +0.0037 | +0.0001 |
| micro_lr1e-6_anchor8_steps500_step200 | 1200:1200 | +0.0469 | +0.9531 | +0.0020 | -0.0002 |
| micro_lr1e-6_anchor8_steps500_step200 | 1200:256 | +0.0312 | +0.9688 | +0.0027 | -0.0017 |
| micro_lr1e-6_anchor8_steps500_step250 | 384:256 | +0.0000 | +1.0000 | +0.0014 | -0.0001 |
| micro_lr1e-6_anchor8_steps500_step250 | 768:768 | +0.0312 | +0.9688 | +0.0044 | -0.0012 |
| micro_lr1e-6_anchor8_steps500_step250 | 1200:1200 | +0.0781 | +0.9219 | +0.0023 | +0.0018 |
| micro_lr1e-6_anchor8_steps500_step250 | 1200:256 | +0.0312 | +0.9688 | +0.0031 | -0.0020 |
| micro_lr1e-6_anchor8_steps500_step300 | 384:256 | +0.0156 | +0.9844 | +0.0013 | +0.0002 |
| micro_lr1e-6_anchor8_steps500_step300 | 768:768 | +0.0156 | +0.9844 | +0.0045 | -0.0035 |
| micro_lr1e-6_anchor8_steps500_step300 | 1200:1200 | +0.0625 | +0.9375 | +0.0018 | +0.0009 |
| micro_lr1e-6_anchor8_steps500_step300 | 1200:256 | +0.0312 | +0.9688 | +0.0036 | -0.0027 |
| micro_lr1e-6_anchor8_steps500_step350 | 384:256 | +0.0156 | +0.9844 | +0.0015 | +0.0009 |
| micro_lr1e-6_anchor8_steps500_step350 | 768:768 | +0.0469 | +0.9531 | +0.0155 | +0.0017 |
| micro_lr1e-6_anchor8_steps500_step350 | 1200:1200 | +0.0625 | +0.9375 | +0.0018 | +0.0007 |
| micro_lr1e-6_anchor8_steps500_step350 | 1200:256 | +0.0312 | +0.9688 | +0.0038 | -0.0039 |
| micro_lr1e-6_anchor8_steps500_step400 | 384:256 | +0.0156 | +0.9844 | +0.0011 | +0.0016 |
| micro_lr1e-6_anchor8_steps500_step400 | 768:768 | +0.0156 | +0.9844 | +0.0057 | -0.0039 |
| micro_lr1e-6_anchor8_steps500_step400 | 1200:1200 | +0.0781 | +0.9219 | +0.0017 | -0.0005 |
| micro_lr1e-6_anchor8_steps500_step400 | 1200:256 | +0.0312 | +0.9688 | +0.0029 | -0.0036 |
| micro_lr1e-6_anchor8_steps500_step450 | 384:256 | +0.0156 | +0.9844 | +0.0009 | +0.0020 |
| micro_lr1e-6_anchor8_steps500_step450 | 768:768 | +0.0156 | +0.9844 | +0.0057 | -0.0034 |
| micro_lr1e-6_anchor8_steps500_step450 | 1200:1200 | +0.0781 | +0.9219 | +0.0018 | +0.0011 |
| micro_lr1e-6_anchor8_steps500_step450 | 1200:256 | +0.0312 | +0.9688 | +0.0030 | -0.0043 |
| micro_lr1e-6_anchor8_steps500_step500 | 384:256 | +0.0156 | +0.9844 | +0.0010 | +0.0022 |
| micro_lr1e-6_anchor8_steps500_step500 | 768:768 | +0.0156 | +0.9844 | +0.0059 | -0.0033 |
| micro_lr1e-6_anchor8_steps500_step500 | 1200:1200 | +0.0625 | +0.9375 | +0.0019 | -0.0014 |
| micro_lr1e-6_anchor8_steps500_step500 | 1200:256 | +0.0312 | +0.9688 | +0.0030 | -0.0047 |
| ultra_micro_lr5e-7_anchor8_steps500_step50 | 384:256 | +0.0156 | +0.9844 | +0.0003 | -0.0008 |
| ultra_micro_lr5e-7_anchor8_steps500_step50 | 768:768 | +0.0156 | +0.9844 | +0.0008 | +0.0031 |
| ultra_micro_lr5e-7_anchor8_steps500_step50 | 1200:1200 | +0.0156 | +0.9844 | +0.0008 | -0.0013 |
| ultra_micro_lr5e-7_anchor8_steps500_step50 | 1200:256 | +0.0312 | +0.9688 | +0.0005 | +0.0012 |
| ultra_micro_lr5e-7_anchor8_steps500_step100 | 384:256 | +0.0156 | +0.9844 | +0.0004 | -0.0001 |
| ultra_micro_lr5e-7_anchor8_steps500_step100 | 768:768 | +0.0156 | +0.9844 | +0.0029 | +0.0019 |
| ultra_micro_lr5e-7_anchor8_steps500_step100 | 1200:1200 | +0.0469 | +0.9531 | +0.0014 | -0.0029 |
| ultra_micro_lr5e-7_anchor8_steps500_step100 | 1200:256 | +0.0469 | +0.9531 | +0.0131 | +0.0004 |
| ultra_micro_lr5e-7_anchor8_steps500_step150 | 384:256 | +0.0000 | +1.0000 | +0.0005 | +0.0006 |
| ultra_micro_lr5e-7_anchor8_steps500_step150 | 768:768 | +0.0000 | +1.0000 | +0.0036 | -0.0023 |
| ultra_micro_lr5e-7_anchor8_steps500_step150 | 1200:1200 | +0.0625 | +0.9375 | +0.0028 | -0.0044 |
| ultra_micro_lr5e-7_anchor8_steps500_step150 | 1200:256 | +0.0469 | +0.9531 | +0.0131 | +0.0002 |
| ultra_micro_lr5e-7_anchor8_steps500_step200 | 384:256 | +0.0000 | +1.0000 | +0.0007 | +0.0001 |
| ultra_micro_lr5e-7_anchor8_steps500_step200 | 768:768 | +0.0156 | +0.9844 | +0.0036 | -0.0021 |
| ultra_micro_lr5e-7_anchor8_steps500_step200 | 1200:1200 | +0.0469 | +0.9531 | +0.0011 | -0.0008 |
| ultra_micro_lr5e-7_anchor8_steps500_step200 | 1200:256 | +0.0469 | +0.9531 | +0.0131 | -0.0001 |
| ultra_micro_lr5e-7_anchor8_steps500_step250 | 384:256 | +0.0156 | +0.9844 | +0.0009 | -0.0015 |
| ultra_micro_lr5e-7_anchor8_steps500_step250 | 768:768 | +0.0156 | +0.9844 | +0.0034 | -0.0019 |
| ultra_micro_lr5e-7_anchor8_steps500_step250 | 1200:1200 | +0.0469 | +0.9531 | +0.0015 | -0.0007 |
| ultra_micro_lr5e-7_anchor8_steps500_step250 | 1200:256 | +0.0625 | +0.9375 | +0.0144 | -0.0021 |
| ultra_micro_lr5e-7_anchor8_steps500_step300 | 384:256 | +0.0000 | +1.0000 | +0.0009 | -0.0013 |
| ultra_micro_lr5e-7_anchor8_steps500_step300 | 768:768 | +0.0469 | +0.9531 | +0.0036 | -0.0029 |
| ultra_micro_lr5e-7_anchor8_steps500_step300 | 1200:1200 | +0.0469 | +0.9531 | +0.0017 | +0.0008 |
| ultra_micro_lr5e-7_anchor8_steps500_step300 | 1200:256 | +0.0625 | +0.9375 | +0.0146 | -0.0012 |
| ultra_micro_lr5e-7_anchor8_steps500_step350 | 384:256 | +0.0156 | +0.9844 | +0.0009 | -0.0006 |
| ultra_micro_lr5e-7_anchor8_steps500_step350 | 768:768 | +0.0312 | +0.9688 | +0.0038 | -0.0035 |
| ultra_micro_lr5e-7_anchor8_steps500_step350 | 1200:1200 | +0.0469 | +0.9531 | +0.0018 | +0.0013 |
| ultra_micro_lr5e-7_anchor8_steps500_step350 | 1200:256 | +0.0469 | +0.9531 | +0.0026 | -0.0035 |
| ultra_micro_lr5e-7_anchor8_steps500_step400 | 384:256 | +0.0000 | +1.0000 | +0.0009 | -0.0010 |
| ultra_micro_lr5e-7_anchor8_steps500_step400 | 768:768 | +0.0312 | +0.9688 | +0.0036 | -0.0008 |
| ultra_micro_lr5e-7_anchor8_steps500_step400 | 1200:1200 | +0.0469 | +0.9531 | +0.0019 | +0.0007 |
| ultra_micro_lr5e-7_anchor8_steps500_step400 | 1200:256 | +0.0312 | +0.9688 | +0.0025 | -0.0030 |
| ultra_micro_lr5e-7_anchor8_steps500_step450 | 384:256 | +0.0000 | +1.0000 | +0.0011 | -0.0009 |
| ultra_micro_lr5e-7_anchor8_steps500_step450 | 768:768 | +0.0469 | +0.9531 | +0.0036 | -0.0010 |
| ultra_micro_lr5e-7_anchor8_steps500_step450 | 1200:1200 | +0.0625 | +0.9375 | +0.0022 | +0.0007 |
| ultra_micro_lr5e-7_anchor8_steps500_step450 | 1200:256 | +0.0312 | +0.9688 | +0.0035 | -0.0037 |
| ultra_micro_lr5e-7_anchor8_steps500_step500 | 384:256 | +0.0000 | +1.0000 | +0.0012 | -0.0001 |
| ultra_micro_lr5e-7_anchor8_steps500_step500 | 768:768 | +0.0469 | +0.9531 | +0.0041 | -0.0017 |
| ultra_micro_lr5e-7_anchor8_steps500_step500 | 1200:1200 | +0.0469 | +0.9531 | +0.0020 | +0.0002 |
| ultra_micro_lr5e-7_anchor8_steps500_step500 | 1200:256 | +0.0312 | +0.9688 | +0.0038 | -0.0040 |

## Aborted-Checkpoint Table

| Lane | Reasons |
|---|---|
| none |  |

## Medium DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| not_run |  |  |  |  |  |  |

## Fixed-Large DS Table

| Candidate | 384:256 | 768:256 | 768:768 | 1200:1200 | 1200:256 | 256:768 |
|---|---|---|---|---|---|---|
| not_run |  |  |  |  |  |  |

## Held-Out Table

| Candidate | Mean 384:256 | Worst 384:256 | Mean 768:768 | Mean 1200:1200 | Mean 1200:256 |
|---|---|---|---|---|---|
| not_run |  |  |  |  |  |

## Bootstrap CIs

| Candidate | Budget | Orientation | Mean | Lower 95% | Upper 95% |
|---|---|---|---|---|---|
| not_run |  |  |  |  |  |

## P0/P1 Split For 384:256

| Candidate | Mean P0 | Mean P1 | Gap |
|---|---|---|---|
| not_run |  |  |  |

## Duplicate Trajectory Count

| Candidate | Mean duplicates |
|---|---|
| not_run |  |

## Runtime Cost

| Candidate | Mean move latency ms | Mean p95 latency ms |
|---|---|---|
| not_run |  |  |

## Gate Result

- gate not run

## Final Classification

- result: `micro_update_safe_but_too_weak`
