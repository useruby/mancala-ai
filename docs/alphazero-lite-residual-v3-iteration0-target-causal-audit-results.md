# AlphaZero-Lite Residual_v3 Iteration-0 Target Causal Audit Results

**Classification**: `target_rows_not_causally_helpful`

## Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Promoted Search Schedule Confirmation

- default c_puct: `1.25`
- overrides: `{"768:768": 0.9}`
- root_policy_mode: `deterministic`
- tactical_root_bias: `0.0`

## Input Hashes

- current_weights: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- fixed_large_suite: `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4`
- manifest: `f9c80affef4be9385e3e44f1a2c2629002f41d51db5d4cabbe4aeada1c64e3d2`
- medium_suite: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`
- selected_rows: `60555445f5f6b2f8aadcb96f338a79fdca0b113c92a53c3588bfd000e8d0b88d`

## Target-Row Validation Table

| Lane | Rows | State Order Fingerprint |
|---|---|---|
| sim768_equal_override | 256 | 62cd1185b77b3f79a9f7ae3f459acc3a185c79f10b3f350dc9c2e9b8b9fd50d9 |
| sim384_default | 256 | 62cd1185b77b3f79a9f7ae3f459acc3a185c79f10b3f350dc9c2e9b8b9fd50d9 |
| sim768_default | 256 | 62cd1185b77b3f79a9f7ae3f459acc3a185c79f10b3f350dc9c2e9b8b9fd50d9 |
| sim1200_default | 256 | 62cd1185b77b3f79a9f7ae3f459acc3a185c79f10b3f350dc9c2e9b8b9fd50d9 |

## Target-Lane Agreement Matrix

| Lane | sim768_equal_override | sim384_default | sim768_default | sim1200_default |
|---|---|---|---|---|
| sim768_equal_override | 256 | 47 | 137 | 148 |
| sim384_default | 47 | 256 | 77 | 46 |
| sim768_default | 137 | 77 | 256 | 126 |
| sim1200_default | 148 | 46 | 126 | 256 |

## Selected-Row Forced Outcome Table

| Lane | 384 Mean | 384 CI95 | 768 Mean | 768 CI95 | 1200 Mean | 1200 CI95 |
|---|---|---|---|---|---|---|
| sim768_equal_override | +0.0898 | [-0.0586, +0.2383] | +0.1289 | [+0.0000, +0.2578] | +0.1328 | [-0.0117, +0.2773] |
| sim384_default | -0.0039 | [-0.0117, +0.0000] | +0.0039 | [-0.0117, +0.0234] | -0.0117 | [-0.0312, +0.0000] |
| sim768_default | +0.0938 | [-0.0391, +0.2266] | +0.2305 | [+0.1133, +0.3477] | +0.0352 | [-0.0898, +0.1562] |
| sim1200_default | +0.1406 | [-0.0078, +0.2891] | +0.2500 | [+0.1211, +0.3789] | +0.1602 | [+0.0195, +0.3008] |

## Held-Out Nearby Forced Outcome Table

- held-out rows: `320` (`ply 5-7=243`, `fallback ply 4/8=77`)

| Lane | 384 Mean | 384 CI95 | 768 Mean | 768 CI95 | 1200 Mean | 1200 CI95 |
|---|---|---|---|---|---|---|
| sim768_equal_override | +0.0437 | [-0.0125, +0.1031] | +0.0563 | [+0.0000, +0.1156] | +0.0594 | [+0.0000, +0.1219] |
| sim384_default | +0.0000 | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] | +0.0000 | [+0.0000, +0.0000] |
| sim768_default | +0.0375 | [-0.0094, +0.0845] | +0.0250 | [-0.0156, +0.0688] | +0.0094 | [-0.0375, +0.0563] |
| sim1200_default | +0.0625 | [+0.0125, +0.1156] | +0.0406 | [-0.0063, +0.0906] | +0.0437 | [-0.0094, +0.0969] |

## Bootstrap 95% CI For Target Minus Current Outcome Delta

Selected rows and held-out rows use the same bootstrap procedure over per-row deltas.

## Forced Outcome By Seat Context

| Group | Lane | 384 Mean | 384 CI95 |
|---|---|---|---|
| selected seat 0 | sim768_equal_override | +0.0936 | [-0.0936, +0.2749] |
| selected seat 0 | sim384_default | -0.0058 | [-0.0175, +0.0000] |
| selected seat 0 | sim768_default | +0.0468 | [-0.1228, +0.2164] |
| selected seat 0 | sim1200_default | +0.1404 | [-0.0469, +0.3275] |
| selected seat 1 | sim768_equal_override | +0.0824 | [-0.1765, +0.3294] |
| selected seat 1 | sim384_default | +0.0000 | [+0.0000, +0.0000] |
| selected seat 1 | sim768_default | +0.1882 | [-0.0471, +0.4235] |
| selected seat 1 | sim1200_default | +0.1412 | [-0.0941, +0.3765] |
| heldout seat 0 | sim768_equal_override | +0.0452 | [-0.0323, +0.1226] |
| heldout seat 0 | sim384_default | +0.0000 | [+0.0000, +0.0000] |
| heldout seat 0 | sim768_default | +0.0323 | [-0.0258, +0.0903] |
| heldout seat 0 | sim1200_default | +0.0581 | [+0.0000, +0.1290] |
| heldout seat 1 | sim768_equal_override | +0.0424 | [-0.0485, +0.1333] |
| heldout seat 1 | sim384_default | +0.0000 | [+0.0000, +0.0000] |
| heldout seat 1 | sim768_default | +0.0424 | [-0.0244, +0.1152] |
| heldout seat 1 | sim1200_default | +0.0667 | [-0.0121, +0.1515] |

## Forced Outcome By Opening Prefix Ply

| Group | Lane | 384 Mean |
|---|---|---|
| selected ply 3 | sim768_equal_override | +0.0000 |
| selected ply 3 | sim384_default | +0.0000 |
| selected ply 3 | sim768_default | +0.0000 |
| selected ply 3 | sim1200_default | +0.0000 |
| selected ply 4 | sim768_equal_override | +0.0000 |
| selected ply 4 | sim384_default | +0.0000 |
| selected ply 4 | sim768_default | +0.0000 |
| selected ply 4 | sim1200_default | +0.0000 |
| selected ply 5 | sim768_equal_override | +0.1667 |
| selected ply 5 | sim384_default | +0.0000 |
| selected ply 5 | sim768_default | -0.1111 |
| selected ply 5 | sim1200_default | +0.2222 |
| selected ply 6 | sim768_equal_override | +0.0794 |
| selected ply 6 | sim384_default | -0.0047 |
| selected ply 6 | sim768_default | +0.1308 |
| selected ply 6 | sim1200_default | +0.1308 |

## Forced Outcome By Target Entropy Bucket

| Bucket | Lane | 384 Mean |
|---|---|---|
| <1.80 | sim768_equal_override | +0.1429 |
| <1.80 | sim384_default | +0.0000 |
| <1.80 | sim768_default | +0.0000 |
| <1.80 | sim1200_default | +0.0000 |
| >=2.40 | sim768_equal_override | +0.1395 |
| >=2.40 | sim384_default | +0.0000 |
| >=2.40 | sim768_default | +0.0678 |
| >=2.40 | sim1200_default | +0.3182 |
| [1.80,2.10) | sim768_equal_override | +0.1915 |
| [1.80,2.10) | sim384_default | +0.0000 |
| [1.80,2.10) | sim768_default | +0.0000 |
| [1.80,2.10) | sim1200_default | +0.3333 |
| [2.10,2.40) | sim768_equal_override | +0.0440 |
| [2.10,2.40) | sim384_default | -0.0057 |
| [2.10,2.40) | sim768_default | +0.1105 |
| [2.10,2.40) | sim1200_default | +0.0533 |

## Forced Outcome By Target Value Bucket

| Bucket | Lane | 384 Mean |
|---|---|---|
| <-0.05 | sim768_equal_override | +0.1224 |
| <-0.05 | sim384_default | +0.0000 |
| <-0.05 | sim768_default | -0.0408 |
| <-0.05 | sim1200_default | +0.2653 |
| >=0.20 | sim768_equal_override | +0.3750 |
| >=0.20 | sim384_default | +0.0000 |
| >=0.20 | sim768_default | +0.3750 |
| >=0.20 | sim1200_default | -0.0625 |
| [-0.05,0.00) | sim768_equal_override | +0.3269 |
| [-0.05,0.00) | sim384_default | +0.0000 |
| [-0.05,0.00) | sim768_default | +0.2308 |
| [-0.05,0.00) | sim1200_default | +0.2885 |
| [0.00,0.10) | sim768_equal_override | -0.0215 |
| [0.00,0.10) | sim384_default | +0.0000 |
| [0.00,0.10) | sim768_default | +0.1720 |
| [0.00,0.10) | sim1200_default | +0.0645 |
| [0.10,0.20) | sim768_equal_override | -0.0870 |
| [0.10,0.20) | sim384_default | -0.0217 |
| [0.10,0.20) | sim768_default | -0.1739 |
| [0.10,0.20) | sim1200_default | +0.0652 |

## First 25 Most Harmful Target Rows

| Rank | Lane | State Hash | Delta | Current | Target |
|---|---|---|---|---|---|
| 3 | sim768_default | 26c10f12211e7f83b8a361f9b3e453bbc4af2d921faa3f61ec360b7c3382dc40 | -2.0000 | 5 | 2 |
| 15 | sim768_equal_override | 946054cea04da481dee60803981d5ac923cb9de0e716df2e090624165059f6a2 | -2.0000 | 0 | 3 |
| 68 | sim768_equal_override | f102e6c9022b10776f9bc4e70ec96ed078396a2d983ca40c58cd64dd060c2032 | -2.0000 | 2 | 5 |
| 3 | sim768_equal_override | 26c10f12211e7f83b8a361f9b3e453bbc4af2d921faa3f61ec360b7c3382dc40 | -2.0000 | 5 | 3 |
| 15 | sim1200_default | 946054cea04da481dee60803981d5ac923cb9de0e716df2e090624165059f6a2 | -2.0000 | 0 | 2 |
| 68 | sim1200_default | f102e6c9022b10776f9bc4e70ec96ed078396a2d983ca40c58cd64dd060c2032 | -2.0000 | 2 | 1 |
| 58 | sim1200_default | 6f60a5a9c709dd9f4f9c387f6516a944d27a8eeee58cc3a2faef6a6e94f0b3aa | -2.0000 | 3 | 4 |
| 239 | sim1200_default | e0b0281735efa552784e67c3f0e74091149774f0775923bf6791bb5e25e888c1 | -2.0000 | 0 | 4 |
| 14 | sim768_default | 7d984626e1195e40da760f06b00886201f7cf8642ad245b995894e05150fa08e | -2.0000 | 2 | 0 |
| 14 | sim768_equal_override | 7d984626e1195e40da760f06b00886201f7cf8642ad245b995894e05150fa08e | -2.0000 | 2 | 0 |
| 63 | sim768_default | ea0b6b662c3a7ef4eaf9100ff59dda41f29bd072b35a5ab6e62ea3a46ef0d428 | -2.0000 | 5 | 2 |
| 15 | sim768_default | 946054cea04da481dee60803981d5ac923cb9de0e716df2e090624165059f6a2 | -2.0000 | 0 | 3 |
| 28 | sim768_default | 1b7098f4bb480dc96a7988b89b2db0b4778a63c4ffeca21d9255cb0ff87ff22a | -2.0000 | 5 | 3 |
| 26 | sim768_default | 3840517b470a44e9e4604160c3b8588442e67c07f44aaf0b41011fb2c1ad2ee0 | -2.0000 | 3 | 2 |
| 16 | sim768_default | 0fb69105e255a069091991af700e3923992056537835b7f8c31824ecbfc1d5fa | -2.0000 | 1 | 4 |
| 16 | sim768_equal_override | 0fb69105e255a069091991af700e3923992056537835b7f8c31824ecbfc1d5fa | -2.0000 | 1 | 5 |
| 16 | sim1200_default | 0fb69105e255a069091991af700e3923992056537835b7f8c31824ecbfc1d5fa | -2.0000 | 1 | 5 |
| 12 | sim768_default | c083c28041104fbff9b3cdf5b042cd8ab83f98ad3b11521d8723fd0c3c238436 | -2.0000 | 5 | 2 |
| 243 | sim768_default | 01e2cb9dc0f272544434640c28476706ec1142db60952361376a09af0f39fad2 | -2.0000 | 1 | 5 |
| 14 | sim1200_default | 7d984626e1195e40da760f06b00886201f7cf8642ad245b995894e05150fa08e | -2.0000 | 2 | 0 |
| 26 | sim768_equal_override | 3840517b470a44e9e4604160c3b8588442e67c07f44aaf0b41011fb2c1ad2ee0 | -2.0000 | 3 | 2 |
| 49 | sim1200_default | a1aa877edd5b721949a925349e10f33861c81edf77e329678a47fc6eff95d609 | -2.0000 | 3 | 4 |
| 26 | sim1200_default | 3840517b470a44e9e4604160c3b8588442e67c07f44aaf0b41011fb2c1ad2ee0 | -2.0000 | 3 | 2 |
| 8 | sim768_default | 1b22a54e1bc909711f519abeb83f4a8b76574765133190ee73a1e67f1e59cbe5 | -2.0000 | 1 | 4 |
| 38 | sim768_equal_override | 855e060f851168db27aa685baa48c84f84930db6c44f3a497b5f5322ccd28682 | -2.0000 | 0 | 4 |

## First 25 Most Helpful Target Rows

| Rank | Lane | State Hash | Delta | Current | Target |
|---|---|---|---|---|---|
| 182 | sim1200_default | f8212eefabc161254ba7bcd80bf4c53250f2430f39b022097955da2653c4bdc7 | +2.0000 | 5 | 0 |
| 182 | sim768_equal_override | f8212eefabc161254ba7bcd80bf4c53250f2430f39b022097955da2653c4bdc7 | +2.0000 | 5 | 0 |
| 236 | sim768_equal_override | ffb2be7af18c86ba13c3acc3bc72ed9029f92edea4f9bd4368e5f58b9b627e30 | +2.0000 | 0 | 1 |
| 201 | sim768_equal_override | b91ca4fc46a70293b026bcb7f25f4f1bbe9f49588e6d89209efd8008757751e3 | +2.0000 | 4 | 0 |
| 182 | sim768_default | f8212eefabc161254ba7bcd80bf4c53250f2430f39b022097955da2653c4bdc7 | +2.0000 | 5 | 0 |
| 51 | sim1200_default | a049bca8a770041038bb175ea11e9415d1756ea1fa15b2d6a80aa2d36dcce7d7 | +2.0000 | 1 | 0 |
| 39 | sim768_equal_override | 6f1f47afc46bb0728e76f07f7b429e6318f9de152c99d7be69c9547ca6c94810 | +2.0000 | 1 | 4 |
| 92 | sim768_equal_override | 472546d2c2cefd2c15351e2fdebf3537dc5e9c930a2b8d95a840bb8872dd0754 | +2.0000 | 0 | 1 |
| 201 | sim1200_default | b91ca4fc46a70293b026bcb7f25f4f1bbe9f49588e6d89209efd8008757751e3 | +2.0000 | 4 | 0 |
| 82 | sim1200_default | 398a5c21b6aef7dc2e96c0d8ae5ff3a27a3bf2fc6f6a4e5bd4b3173bd255221d | +2.0000 | 1 | 0 |
| 221 | sim768_equal_override | 596999e193ce465b6c606792ee419d83edc1b77d89af5fa56c53495d2d475659 | +2.0000 | 2 | 1 |
| 57 | sim768_equal_override | ae620de85cfec74503872393cf1c1d8ebd38cf5563efc22ca644571bcc871315 | +2.0000 | 5 | 0 |
| 175 | sim1200_default | 276031a26f59e383c77abbb402e706fd6ee6979915f5d2e5570a8e9a1d3c8594 | +2.0000 | 2 | 3 |
| 147 | sim1200_default | dc7705dc0a37175ece03074da0c879cd82345af02c55c600f2bfc335fc6f4e19 | +2.0000 | 3 | 4 |
| 128 | sim768_equal_override | 7fd7f97e5a24e54351082f2a31622e336acda19f1673a5b34d369cdbeb2af9c9 | +2.0000 | 2 | 4 |
| 51 | sim768_equal_override | a049bca8a770041038bb175ea11e9415d1756ea1fa15b2d6a80aa2d36dcce7d7 | +2.0000 | 1 | 0 |
| 6 | sim768_equal_override | cfe19af403d66f8068934ec5732ae0fd82b804f587422ad0e140aaec31617b7f | +2.0000 | 2 | 3 |
| 221 | sim1200_default | 596999e193ce465b6c606792ee419d83edc1b77d89af5fa56c53495d2d475659 | +2.0000 | 2 | 1 |
| 118 | sim768_equal_override | f316f382be66b60e1512cad5e81c578487912f44f07714deded3a321eff34121 | +2.0000 | 0 | 5 |
| 185 | sim1200_default | 566b5289a238ec02d194833e952931685eb493c2735f288f69034201af84c3a7 | +2.0000 | 3 | 1 |
| 161 | sim1200_default | 9139d0ad011310a557efeca64e727495e083114d463709e5400909a006b10407 | +2.0000 | 3 | 4 |
| 102 | sim768_equal_override | eec5fdf8554f2e5936cdd8b988083416603ce05ea16ff664d661663470773384 | +2.0000 | 1 | 3 |
| 92 | sim1200_default | 472546d2c2cefd2c15351e2fdebf3537dc5e9c930a2b8d95a840bb8872dd0754 | +2.0000 | 0 | 1 |
| 134 | sim768_equal_override | ee10a09cd0d1b3dc6e2b3c268d0aed172fda3435fc448f39b91759407ab72b5c | +2.0000 | 3 | 2 |
| 128 | sim768_default | 7fd7f97e5a24e54351082f2a31622e336acda19f1673a5b34d369cdbeb2af9c9 | +2.0000 | 2 | 4 |

## Final Classification

- classification: `target_rows_not_causally_helpful`
- rationale: No target lane clears the required selected-row causal bar across continuation budgets with consistently positive confidence intervals.
