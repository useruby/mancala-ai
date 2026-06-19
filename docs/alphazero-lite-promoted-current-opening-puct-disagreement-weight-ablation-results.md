# AlphaZero-Lite Promoted-Current Opening PUCT Disagreement Weight Ablation Results

**Date**: 2026-06-19

**Classification**: `replay_weight_underpowered`

## Summary

Reusing the exact PR #122 mined replay and increasing only its replay weight did
change the result. The weight-1 reference (`4,1,1`) still matched the prior
regression, but heavier mined replay materially improved the primary opening
suite and held-out suites.

- effective mined replay fraction rose from `4.72%` at weight `1` to `16.54%`
  at weight `4`, `28.38%` at weight `8`, and `44.22%` at weight `16`
- PR #122 weight-1 reference moved mined-state top-1 on only `4.20%` of rows
  and regressed fixed large `384:256` to `-0.5508`
- best fixed/held-out lane was `disagreement_w8_policy_head_e1`
  with fixed large `384:256 = -0.1563` and held-out mean `384:256 = -0.1445`
- the lane that satisfied the task decision rule was `disagreement_w16_policy_head_e2`:
  mined-state top-1 changed `13.10%`, fixed large `384:256 = -0.3424`
  (`+0.0508` vs promoted current), held-out mean `384:256 = -0.3485`
  (`+0.0638`), and no `1200:1200` / `1200:256` regression beyond `-0.03`

This supports the original hypothesis: PR #122 failed primarily because the
mined replay was underweighted in the training mix.

## Inputs

| Item | Path | SHA256 / rows |
|---|---|---|
| promoted current weights | `model-artifact/current/weights.json` | `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece` |
| promoted e1 checkpoint | `/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz` | `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357` |
| generic bootstrap | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | `5d01a60e9dfc8756ffa47f2c8222e496a8ce8bd661457658de4884198344e147` / `9589` |
| random teacher | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | `7ca93389d1be93bd1cf09d23ddfb9f040bb402a718cd991ac49e082bd7e2f69a` / `2016` |
| PR #122 mined replay | `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_puct_disagreement_replay.jsonl` | `59187c271babbd8bcfec5f621a819d60c545188358e41fdd554724960eb55b02` / `2000` |
| PR #122 disagreement audit | `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_state_disagreement_audit.json` | `5bd9780e5e8b7cf638377569240db565a16ef5f8b3f96d00d0c7c6c178cdecf4` |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| fixed large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |
| heldout seed 43 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl` | `5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9` |
| heldout seed 44 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl` | `323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620` |
| heldout seed 45 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl` | `ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda` |

The mined replay row count was verified and the run fails if it is not exactly
`2000` rows.

## Replay Fractions

| Mined weight | Replay weights | Effective mined fraction |
|---|---|---:|
| `1` | `4,1,1` | `4.72%` |
| `4` | `4,1,4` | `16.54%` |
| `8` | `4,1,8` | `28.38%` |
| `16` | `4,1,16` | `44.22%` |

## Training And Mined-State Movement

| Candidate | Epochs | Replay weights | Policy loss | Value loss | Validation loss | Top-1 changed | Top-1 matches search | KL(search || raw) | KL(raw || search) | Delta norm | Relative delta |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pr122_weight1_e2_ref` | `2` | `4,1,1` | `1.027136` | `0.238889` | `1.206340` | `0.0420` | `0.0315` | `0.6979` | `0.8714` | `0.0308` | `0.1145%` |
| `disagreement_w4_policy_head_e1` | `1` | `4,1,4` | `1.106222` | `0.222907` | `1.264773` | `0.0485` | `0.0345` | `0.6933` | `0.8680` | `0.0323` | `0.1200%` |
| `disagreement_w4_policy_head_e2` | `2` | `4,1,4` | `1.102316` | `0.222854` | `1.264773` | `0.0695` | `0.0510` | `0.6869` | `0.8628` | `0.0493` | `0.1831%` |
| `disagreement_w8_policy_head_e1` | `1` | `4,1,8` | `1.185187` | `0.207000` | `1.323612` | `0.0710` | `0.0520` | `0.6879` | `0.8638` | `0.0452` | `0.1678%` |
| `disagreement_w8_policy_head_e2` | `2` | `4,1,8` | `1.181643` | `0.206960` | `1.323612` | `0.0925` | `0.0655` | `0.6785` | `0.8562` | `0.0709` | `0.2634%` |
| `disagreement_w16_policy_head_e1` | `1` | `4,1,16` | `1.287893` | `0.185773` | `1.402266` | `0.0900` | `0.0645` | `0.6808` | `0.8580` | `0.0653` | `0.2426%` |
| `disagreement_w16_policy_head_e2` | `2` | `4,1,16` | `1.280239` | `0.185719` | `1.401941` | `0.1310` | `0.0940` | `0.6676` | `0.8476` | `0.1066` | `0.3959%` |

Checkpoint and artifact hashes:

| Candidate | Checkpoint SHA256 | Artifact weights SHA256 |
|---|---|---|
| `pr122_weight1_e2_ref` | `7174fc20c68d3215b68c069a0d291e872a234e6c1d467f4d27898407d5282f4a` | `c04422f971bd8865956bd2c6e9fe26b370857113ac5ace035457d405118990f7` |
| `disagreement_w4_policy_head_e1` | `8d8cebf5c3c3b242d0f64f2e074096cf0c77cf29df1482ee0beb60e63efe24f3` | `07de9d18663fb2f127f9b41abf49d88f971dd674bad342cf74be5a5b7c56f647` |
| `disagreement_w4_policy_head_e2` | `04ec8f9420459359b21b7f2d3b5ab5c6f42bc054711c3e13d9538dc8456ab616` | `65b70a676c87593a22d46333272f2fb6e98d0cdedcd8dbc36237ba8a75184177` |
| `disagreement_w8_policy_head_e1` | `cd9ef83902516283d680fec0d3986cea832bf467042aaabd814ecd5026ec1e0e` | `2f1e1420c1cb02517faa702d4aa4a97f6d51e1bd0c87f782abad1183a76ce9ad` |
| `disagreement_w8_policy_head_e2` | `1964e474f14c218910b7f64341c24ba75357098bd27dac81697135f07179ecb1` | `f931744a57ffb7166f9bf18718c09d061f12fdbef966374d26fcce42ca63025a` |
| `disagreement_w16_policy_head_e1` | `dc0843cd6f0fc4278e9a2ae04c79e50847ac9e374d0af00ae0e69154efc49865` | `ccae3a7db0e2154104fee2c1489c3dc0dfbd02bc329c8a90980ae9493f913d27` |
| `disagreement_w16_policy_head_e2` | `c4b74cf357e67798b24b3ed49a593034b6c30ea9619ef5fd0bc3d36a7bcfac9f` | `51a191bbf417969f522024cc46d7fa4cb23e8f714d9a7d5e4c6eac40462c13be` |

## Medium Suite

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS |
|---|---:|---:|---:|---:|---:|---:|
| `promoted_current_ref` | `-0.3203` | `-0.3828` | `+0.0859` | `-0.3359` | `-0.1797` | `-0.3828` |
| `pr122_weight1_e2_ref` | `-0.5469` | `-0.3125` | `-0.5078` | `-0.1836` | `-0.1797` | `-0.2344` |
| `disagreement_w4_policy_head_e1` | `-0.2031` | `-0.3125` | `-0.0039` | `-0.1992` | `-0.1797` | `-0.3398` |
| `disagreement_w4_policy_head_e2` | `-0.1484` | `-0.3125` | `-0.2461` | `-0.0938` | `-0.1797` | `-0.2812` |
| `disagreement_w8_policy_head_e1` | `-0.1133` | `-0.3672` | `-0.4883` | `-0.1992` | `-0.1797` | `-0.2656` |
| `disagreement_w8_policy_head_e2` | `-0.3047` | `-0.3672` | `-0.5078` | `-0.1328` | `-0.1797` | `-0.2656` |
| `disagreement_w16_policy_head_e1` | `-0.1914` | `-0.3125` | `-0.3984` | `-0.1133` | `-0.0586` | `-0.2109` |
| `disagreement_w16_policy_head_e2` | `-0.3359` | `-0.1914` | `-0.4805` | `+0.0000` | `-0.1797` | `-0.3555` |

## Fixed Large Suite

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `promoted_current_ref` | `-0.3932` | `-0.3984` | `-0.1589` | `-0.4167` | `-0.1706` | `-0.3984` | `0.4115 / 0.8047` | `1536` |
| `pr122_weight1_e2_ref` | `-0.5508` | `-0.3216` | `-0.5872` | `-0.3060` | `-0.1706` | `-0.3477` | `0.2539 / 0.8047` | `1536` |
| `disagreement_w4_policy_head_e1` | `-0.2617` | `-0.3216` | `-0.2266` | `-0.3034` | `-0.1706` | `-0.4193` | `0.4154 / 0.6771` | `1536` |
| `disagreement_w4_policy_head_e2` | `-0.1979` | `-0.3216` | `-0.3958` | `-0.2474` | `-0.1706` | `-0.2812` | `0.5430 / 0.7409` | `1536` |
| `disagreement_w8_policy_head_e1` | `-0.1563` | `-0.3854` | `-0.5651` | `-0.3034` | `-0.1706` | `-0.2695` | `0.5208 / 0.6771` | `1536` |
| `disagreement_w8_policy_head_e2` | `-0.3177` | `-0.3854` | `-0.5872` | `-0.2435` | `-0.1706` | `-0.2695` | `0.3594 / 0.6771` | `1536` |
| `disagreement_w16_policy_head_e1` | `-0.2604` | `-0.3216` | `-0.4596` | `-0.2214` | `-0.0859` | `-0.2174` | `0.4167 / 0.6771` | `1536` |
| `disagreement_w16_policy_head_e2` | `-0.3424` | `-0.2370` | `-0.5508` | `-0.1706` | `-0.1706` | `-0.3750` | `0.4622 / 0.8047` | `1536` |

Primary `384:256` delta vs promoted current:

- `pr122_weight1_e2_ref`: `-0.1576`
- `disagreement_w4_policy_head_e1`: `+0.1315`
- `disagreement_w4_policy_head_e2`: `+0.1953`
- `disagreement_w8_policy_head_e1`: `+0.2370`
- `disagreement_w8_policy_head_e2`: `+0.0755`
- `disagreement_w16_policy_head_e1`: `+0.1328`
- `disagreement_w16_policy_head_e2`: `+0.0508`

## Held-Out Large Suites

| Candidate | Mean 384:256 DS | Worst 384:256 DS |
|---|---:|---:|
| `promoted_current_ref` | `-0.4123` | `-0.4349` |
| `pr122_weight1_e2_ref` | `-0.5595` | `-0.5807` |
| `disagreement_w4_policy_head_e1` | `-0.2665` | `-0.2878` |
| `disagreement_w4_policy_head_e2` | `-0.1940` | `-0.2161` |
| `disagreement_w8_policy_head_e1` | `-0.1445` | `-0.1654` |
| `disagreement_w8_policy_head_e2` | `-0.3168` | `-0.3346` |
| `disagreement_w16_policy_head_e1` | `-0.2678` | `-0.2826` |
| `disagreement_w16_policy_head_e2` | `-0.3485` | `-0.3711` |

## Movement Breakdown

`disagreement_w16_policy_head_e2` was the first lane to clear the `10%`
mined-state top-1 movement threshold.

By raw-margin bucket:

| Bucket | States | Top-1 changed | Changed rate | Top-1 matches search |
|---|---:|---:|---:|---:|
| `< 0.02` | `152` | `79` | `0.5197` | `0.3816` |
| `0.02 <= margin < 0.05` | `227` | `70` | `0.3084` | `0.2203` |
| `0.05 <= margin < 0.10` | `321` | `66` | `0.2056` | `0.1464` |
| `>= 0.10` | `1300` | `47` | `0.0362` | `0.0254` |

By phase:

| Phase | States | Top-1 changed | Changed rate | Top-1 matches search |
|---|---:|---:|---:|---:|
| `opening` | `1200` | `185` | `0.1542` | `0.1133` |
| `mid` | `600` | `55` | `0.0917` | `0.0533` |
| `late` | `200` | `22` | `0.1100` | `0.1000` |

The extra movement stayed concentrated in the low-margin and opening-heavy part
of the replay, which is where this ablation was intended to push harder.

## Gate

Gate rule for this task: run the default deterministic gate for promoted current
and any weighted candidate that beats promoted current by at least `+0.01` DS on
fixed large `384:256`.

- `promoted_current_ref`: gate run, classification `high_search_breakthrough`
- `disagreement_w4_policy_head_e1`: gate run, classification `high_search_breakthrough`
- `disagreement_w4_policy_head_e2`: gate run, classification `high_search_breakthrough`
- `disagreement_w8_policy_head_e1`: gate run, classification `high_search_breakthrough`
- `disagreement_w8_policy_head_e2`: gate run, classification `high_search_breakthrough`
- `disagreement_w16_policy_head_e1`: gate run, classification `high_search_breakthrough`
- `disagreement_w16_policy_head_e2`: gate run, classification `high_search_breakthrough`

## Decision

This run matches `replay_weight_underpowered`.

- higher mined replay weights did increase mined-state movement substantially:
  `4.20%` at weight `1`, up to `13.10%` at weight `16` / epoch `2`
- multiple weighted lanes improved the fixed large `384:256` score by well over
  `+0.03`, with the best at `+0.2370`
- held-out mean `384:256` also improved strongly, best at `+0.2678`
- the qualifying `w16_e2` lane crossed the `10%` movement threshold while still
  improving fixed large and held-out `384:256` and avoiding the forbidden
  `1200:1200` / `1200:256` regressions

The mined targets are not obviously harmful under stronger weighting. The PR #122
failure mode was that the mined disagreement replay was too diluted inside the
`4,1,1` mix.

## Artifacts

- workdir: `/tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation`
- summary: `/tmp/azlite_promoted_current_opening_puct_disagreement_weight_ablation/summary_metrics.json`
- reused disagreement audit: `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_state_disagreement_audit.json`
- reused mined replay: `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_puct_disagreement_replay.jsonl`

Note: the held-out seed 44 and 45 benchmark reports in the ablation workdir were
completed by merging the already-cached deterministic `current` / PR #122
reference candidate reports with newly-run weighted-candidate deterministic
reports after the original all-candidate benchmark exceeded the per-run timeout.
The suite files, seed, root-policy mode, budget pairs, and game counts remained
identical.
