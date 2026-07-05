# AlphaZero-Lite Residual_v3 Causal Target Filter Results

**Classification**: `no_predictive_causal_filter`

## Artifact Hash

- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`

## Promoted Search Schedule Confirmation

- default c_puct: `1.25`
- overrides: `{"768:768": 0.9}`
- root_policy_mode: `deterministic`
- tactical_root_bias: `0.0`

## PR #147 Input Hashes

- canonical_suite: `135fe2c813f972d1a030f5756d57f915713cf325b1fe2945c289758c9237f595`
- fixed_large_suite: `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4`
- medium_suite: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`
- pr147_summary: `83de62f8bd5f0ac3d4a532254f2a6fb91d07cf18c2cc4593892ad14f2368d40e`
- selected_forced_outcomes: `3a29f193d8a11a54aeedfdb476480067c08250016ade9c61fc3dac07005089fa`
- selected_rows: `60555445f5f6b2f8aadcb96f338a79fdca0b113c92a53c3588bfd000e8d0b88d`
- target_row_table: `c5c722c6fd12b40cf55f12e9bb37207de684f0f0f1739e2fcabf309c45ff06f3`

## Causal Row Dataset Summary

- rows: `1024`
- unique states: `256`
- harmful_any rate: `0.303711`
- robust_helpful rate: `0.21875`

## Filter Discovery Table

| Filter | Rows | States | 384 Mean | Worst Mean | Harmful | 384 CI95 | Status |
|---|---|---|---|---|---|---|---|
| sim1200_default | 256 | 256 | +0.1406 | +0.1406 | 40.6% | [-0.0117, +0.2891] | harmful_any rate > 10%; CI too wide to interpret |
| sim1200_value_lt_neg005 | 49 | 49 | +0.2653 | +0.1429 | 36.7% | [-0.1020, +0.6122] | selected rows < 64; harmful_any rate > 10%; CI too wide to interpret |
| sim1200_entropy_ge_240 | 44 | 44 | +0.3182 | +0.0227 | 36.4% | [-0.0455, +0.6818] | selected rows < 64; harmful_any rate > 10%; CI too wide to interpret |
| sim1200_ply6 | 214 | 214 | +0.1308 | +0.1121 | 43.0% | [-0.0235, +0.2944] | harmful_any rate > 10%; CI too wide to interpret |
| sim1200_disagree_raw_agree_puct | 34 | 34 | +0.0000 | +0.0000 | 0.0% | [+0.0000, +0.0000] | selected rows < 64; worst-budget mean <= 0 |
| lane_agreement_ge_2 | 724 | 250 | +0.1050 | +0.1050 | 31.1% | [+0.0304, +0.1809] | harmful_any rate > 10% |
| exclude_state_any_minus_two | 664 | 166 | +0.3599 | +0.3042 | 15.1% | [+0.2982, +0.4232] | harmful_any rate > 10% |
| target_value_nonpositive_bucket | 404 | 101 | +0.1510 | +0.1213 | 28.2% | [+0.0519, +0.2500] | harmful_any rate > 10% |
| positive_entropy_bucket_slice | 0 | 0 | +0.0000 | +0.0000 | 0.0% | [+0.0000, +0.0000] | selected rows < 64; worst-budget mean <= 0 |
| sim1200_seat0 | 171 | 171 | +0.1404 | +0.1404 | 42.1% | [-0.0468, +0.3275] | harmful_any rate > 10%; CI too wide to interpret |
| sim1200_seat1 | 85 | 85 | +0.1412 | +0.0824 | 37.6% | [-0.0941, +0.3765] | harmful_any rate > 10%; CI too wide to interpret |
| sim1200_seat0_ply6 | 148 | 148 | +0.1486 | +0.1486 | 43.9% | [-0.0541, +0.3514] | harmful_any rate > 10%; CI too wide to interpret |
| sim1200_seat1_ply6 | 66 | 66 | +0.0909 | -0.0152 | 40.9% | [-0.1970, +0.3636] | harmful_any rate > 10%; worst-budget mean <= 0; CI too wide to interpret |

## Held-Out Filter Validation Table

| Filter | Rows | 384 Mean | 384 CI95 | Worst Mean | Harmful | Status |
|---|---|---|---|---|---|---|
| target_value_nonpositive_bucket | 1760 | -0.0040 | [-0.0222, +0.0142] | -0.0040 | 5.0% | validated |
| sim1200_seat0_ply6 | 600 | +0.0267 | [-0.0167, +0.0683] | +0.0267 | 8.2% | validated |
| sim1200_seat1 | 339 | +0.0590 | [+0.0118, +0.1091] | +0.0295 | 8.3% | validated |

## Harmful-Row Analysis

| Lane | State Hash | 384 Delta | Target | Current |
|---|---|---|---|---|
| sim1200_default | 01af21939a11f742b6650d1515fbe215ac01d4a4e7b9f96618ae31cf471f5f0b | -2.0000 | 5 | 1 |
| sim768_equal_override | 01af21939a11f742b6650d1515fbe215ac01d4a4e7b9f96618ae31cf471f5f0b | -2.0000 | 5 | 1 |
| sim1200_default | 01e2cb9dc0f272544434640c28476706ec1142db60952361376a09af0f39fad2 | -2.0000 | 4 | 1 |
| sim768_default | 01e2cb9dc0f272544434640c28476706ec1142db60952361376a09af0f39fad2 | -2.0000 | 5 | 1 |
| sim768_equal_override | 01e2cb9dc0f272544434640c28476706ec1142db60952361376a09af0f39fad2 | -2.0000 | 4 | 1 |
| sim1200_default | 0fb69105e255a069091991af700e3923992056537835b7f8c31824ecbfc1d5fa | -2.0000 | 5 | 1 |
| sim768_default | 0fb69105e255a069091991af700e3923992056537835b7f8c31824ecbfc1d5fa | -2.0000 | 4 | 1 |
| sim768_equal_override | 0fb69105e255a069091991af700e3923992056537835b7f8c31824ecbfc1d5fa | -2.0000 | 5 | 1 |
| sim1200_default | 1443963097f36df2b7684d78c1d8422e14fd74721fd7d561cb09d9af74370334 | -2.0000 | 4 | 2 |
| sim768_default | 1443963097f36df2b7684d78c1d8422e14fd74721fd7d561cb09d9af74370334 | -2.0000 | 4 | 2 |

## Helpful-Row Analysis

| Lane | State Hash | 384 Delta | Target | Current |
|---|---|---|---|---|
| sim768_default | 039180acf5d7c369ddfb0f559270f38dcb5213e184b25ab6ee4b94df07134cfe | +2.0000 | 1 | 4 |
| sim768_equal_override | 039180acf5d7c369ddfb0f559270f38dcb5213e184b25ab6ee4b94df07134cfe | +2.0000 | 1 | 4 |
| sim1200_default | 0429e5405164261116d7b15b27bf5a6c07d49f50941d3f37702d14a8e83f62c0 | +2.0000 | 5 | 0 |
| sim768_default | 0429e5405164261116d7b15b27bf5a6c07d49f50941d3f37702d14a8e83f62c0 | +2.0000 | 5 | 0 |
| sim768_equal_override | 0429e5405164261116d7b15b27bf5a6c07d49f50941d3f37702d14a8e83f62c0 | +2.0000 | 5 | 0 |
| sim1200_default | 11f35439de358cd0e66834e2905da8dbee6ed8e459f98c44f4dba1f77a7b5035 | +2.0000 | 0 | 4 |
| sim768_default | 11f35439de358cd0e66834e2905da8dbee6ed8e459f98c44f4dba1f77a7b5035 | +2.0000 | 0 | 4 |
| sim768_equal_override | 11f35439de358cd0e66834e2905da8dbee6ed8e459f98c44f4dba1f77a7b5035 | +2.0000 | 0 | 4 |
| sim1200_default | 14692f29a2805a9d3910e6bde93ba4bbf18b5ab7c3bde17b4a61571f33ee89b9 | +2.0000 | 5 | 2 |
| sim768_default | 14692f29a2805a9d3910e6bde93ba4bbf18b5ab7c3bde17b4a61571f33ee89b9 | +2.0000 | 5 | 2 |

## Recommended Next Action

- stop residual-v3 opening iteration-0 target training
