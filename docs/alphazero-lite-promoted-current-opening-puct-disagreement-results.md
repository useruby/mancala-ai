# AlphaZero-Lite Promoted-Current Opening PUCT Disagreement Results

**Date**: 2026-06-18

**Classification**: `training_update_too_weak`

## Summary

Opening-suite evaluation states were not search-saturated. Promoted current PUCT
changed the raw top-1 move on `23,409 / 65,506` unique evaluation states
(`35.74%`), with `13,363` high-confidence disagreements. That was enough signal
to mine a balanced `2,000`-row replay, but a 1-epoch or 2-epoch policy-head-only
 update from promoted e1 still moved the raw top-1 on only `3.10%` and `4.20%`
of mined states and both trained candidates regressed badly on the primary fixed
large `384:256` opening-suite score.

- promoted current fixed large `384:256` DS: `-0.3932`
- iter2 self-play e2 fixed large `384:256` DS: `-0.3932`
- opening disagreement e1 fixed large `384:256` DS: `-0.4727`
- opening disagreement e2 fixed large `384:256` DS: `-0.5508`
- held-out mean `384:256` DS: promoted current `-0.4123`, e1 `-0.4844`, e2 `-0.5595`
- decision: disagreement exists, but the policy-head-only update is too weak for this replay recipe

Next work should test update strength or weighting, not more same-distribution self-play.

## Inputs

| Item | Path | SHA256 / rows |
|---|---|---|
| promoted current weights | `model-artifact/current/weights.json` | `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece` |
| promoted e1 checkpoint | `/tmp/azlite_control_ep2_puct_smoke/puct_policy_head_e1/checkpoint_epoch1.npz` | `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357` |
| generic bootstrap | `/tmp/azlite_random_agreement_replay/generic_bootstrap.jsonl` | `5d01a60e9dfc8756ffa47f2c8222e496a8ce8bd661457658de4884198344e147` / `9589` |
| random teacher | `/tmp/azlite_random_teacher_quality/random_teacher_1200_train.jsonl` | `7ca93389d1be93bd1cf09d23ddfb9f040bb402a718cd991ac49e082bd7e2f69a` / `2016` |
| medium suite | `/tmp/azlite_opening_suite/medium_eval.jsonl` | `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04` |
| fixed large suite | `/tmp/azlite_opening_suite/large_eval.jsonl` | `ebc86f053a8a4c12f3e937b5acfe37df7b0d2f1f0002969dcac077a9f99d65a4` |
| heldout seed 43 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed43_large.jsonl` | `5e0ed96ba56f99318c32a139692309759659da16a0a98c590e35f7e2496cade9` |
| heldout seed 44 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed44_large.jsonl` | `323f7abec32fb00d3b7b6b153ddd5692435755a3bc5340d8aa5aec32ca639620` |
| heldout seed 45 | `/tmp/azlite_control_ep2_puct_head_preflight/suites/heldout_seed45_large.jsonl` | `ca72c8b7fe1adf7229f81183321475033f8548dcdeb1c71fd16c85c35ce89cda` |
| iter2 self-play e2 ref checkpoint | `/tmp/azlite_promoted_current_puct_iter2/iter2_puct_policy_head_e2/checkpoint_epoch2.npz` | `cb3c7005432a53dea684e5d742e738413437af37e9cf1eb1bccb253f77ca9a0d` |
| iter2 self-play e2 ref weights | `/tmp/azlite_promoted_current_puct_iter2/iter2_puct_policy_head_e2/artifact_iter2_puct_policy_head_e2/weights.json` | `0bd93993361c37a79648265b77ddf0c6c31d911f96b551014c6c1d48866b5c68` |

All expected promoted-current and init-checkpoint hashes matched.

## Audit

Audit path: `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_state_disagreement_audit.json`

Collection settings:

- replay distribution: deterministic suite continuations on fixed large plus three held-out large suites
- benchmark seat pattern: `384:256` challenger/current, both sides `model-artifact/current`
- disagreement teacher search: PUCT `384` sims, `c_puct=1.25`, `root_policy_mode=visit_count`, `tactical_root_bias=0.0`

Key audit metrics:

- unique evaluation states: `65,506`
- overall disagreement rate: `35.74%`
- disagreement by phase: opening `30.75%`, mid `44.54%`, late `26.22%`
- disagreement by seat context: challenger `36.00%`, current `35.54%`
- disagreement on poor fixed-large `384:256` P0 games: `34.54%` across `5,116` states
- mean KL(search || raw): opening challenger `0.1878`, opening current `0.1441`, mid challenger `0.3641`, mid current `0.3175`, late challenger `0.2692`, late current `0.2504`
- duplicate trajectory count in suite replay: `816 / 3072` games (`26.56%`)
- high-confidence disagreements: `13,363`

Raw-margin buckets:

| Bucket | States | Changed top-1 | Rate | Mean KL(search || raw) |
|---|---:|---:|---:|---:|
| `< 0.02` | `3026` | `1945` | `0.6428` | `0.3403` |
| `0.02 <= margin < 0.05` | `4322` | `2607` | `0.6032` | `0.3353` |
| `0.05 <= margin < 0.10` | `6450` | `3715` | `0.5760` | `0.3424` |
| `>= 0.10` | `51708` | `15142` | `0.2928` | `0.2752` |

Replay mining summary:

- selected rows after dedupe: `2000`
- preferred disagreements: `1270`
- fallback KL rows: `730`
- fallback sharpen rows: `0`
- phase caps hit exactly: opening `1200`, mid `600`, late `200`
- top-move cap distribution: `0:268, 1:420, 2:304, 3:420, 4:420, 5:168`

The audit rejects `eval_search_saturated`. Search disagrees often on evaluation-relevant states.

## Training

Replay path: `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_puct_disagreement_replay.jsonl`

- mined states: `2000`
- training rows: `2000`
- replay mix: `generic_bootstrap,random_teacher,opening_puct_disagreement_replay`
- replay weights: `4,1,1`
- recipe: `residual_v3`, `kalah_v3`, `96,3`, batch `512`, LR `1e-5`, `value_loss_weight=0.3`, `grad_clip=1.0`, `trainable_scope=policy_head`

The loss numbers below were reproduced from the saved checkpoints with the same command line and matched the saved checkpoint hashes exactly.

| Candidate | Epochs | Checkpoint SHA256 | Artifact weights SHA256 | Delta norm vs e1 | Relative delta | Policy loss | Value loss | Validation loss | Top-1 changed on mined states | Top-1 matches search |
|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
| `promoted_current_ref` | `0` | `a793f32565b0c706c4228e4de3bc00aea5c471089ec940c4fe85e726fe4f9357` | `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece` | `0.0000` | `0.0000%` | n/a | n/a | n/a | `0.0000` | `0.0000` |
| `iter2_selfplay_e2_ref` | `0` | `cb3c7005432a53dea684e5d742e738413437af37e9cf1eb1bccb253f77ca9a0d` | `0bd93993361c37a79648265b77ddf0c6c31d911f96b551014c6c1d48866b5c68` | `0.0289` | `0.1074%` | n/a | n/a | n/a | `0.0265` | `0.0210` |
| `opening_disagreement_policy_head_e1` | `1` | `c5153093ebfeed446295b6de975c5baf3141fda27572ab2dafb21df766b5bb8e` | `d4ce8512f58cb108ea7827f1924f63a15f550683fe458a2c9b6666fa24c07d28` | `0.0213` | `0.0791%` | `1.028135` | `0.238719` | `1.206340` | `0.0310` | `0.0245` |
| `opening_disagreement_policy_head_e2` | `2` | `7174fc20c68d3215b68c069a0d291e872a234e6c1d467f4d27898407d5282f4a` | `c04422f971bd8865956bd2c6e9fe26b370857113ac5ace035457d405118990f7` | `0.0308` | `0.1145%` | `1.027136` | `0.238889` | `1.206340` | `0.0420` | `0.0315` |

The trained lanes did move the raw policy, but not enough. The best lane changed top-1 on only `4.20%` of mined states, below the `5%` rule.

## Evaluation

All opening-suite tables below use current/opponent artifact hash `6ac71425ef4c69574a8943f42c4b3e036ad272967ae264050a287dda206a3ece`.

### Medium Suite

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS |
|---|---:|---:|---:|---:|---:|---:|
| `promoted_current_ref` | `-0.3203` | `-0.3828` | `+0.0859` | `-0.3359` | `-0.1797` | `-0.3828` |
| `iter2_selfplay_e2_ref` | `-0.3203` | `-0.3281` | `+0.0664` | `-0.3359` | `-0.1797` | `-0.3672` |
| `opening_disagreement_policy_head_e1` | `-0.4531` | `-0.3281` | `-0.0234` | `-0.2148` | `-0.1797` | `-0.2656` |
| `opening_disagreement_policy_head_e2` | `-0.5469` | `-0.3125` | `-0.5078` | `-0.1836` | `-0.1797` | `-0.2344` |

### Fixed Large Suite

| Candidate | 384:256 DS | 768:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | 256:768 DS | 384:256 P0 / P1 | duplicate trajectories |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| `promoted_current_ref` | `-0.3932` | `-0.3984` | `-0.1589` | `-0.4167` | `-0.1706` | `-0.3984` | `0.4115 / 0.8047` | `1536` |
| `iter2_selfplay_e2_ref` | `-0.3932` | `-0.3346` | `-0.1810` | `-0.4167` | `-0.1706` | `-0.3698` | `0.4115 / 0.8047` | `1536` |
| `opening_disagreement_policy_head_e1` | `-0.4727` | `-0.3346` | `-0.2487` | `-0.3320` | `-0.1706` | `-0.3828` | `0.3320 / 0.8047` | `1536` |
| `opening_disagreement_policy_head_e2` | `-0.5508` | `-0.3216` | `-0.5872` | `-0.3060` | `-0.1706` | `-0.3477` | `0.2539 / 0.8047` | `1536` |

Large-suite deltas vs promoted current on the primary `384:256` budget:

- `iter2_selfplay_e2_ref`: `+0.0000`
- `opening_disagreement_policy_head_e1`: `-0.0794`
- `opening_disagreement_policy_head_e2`: `-0.1576`

### Held-Out Large Suites

| Candidate | Mean 384:256 DS | Worst 384:256 DS |
|---|---:|---:|
| `promoted_current_ref` | `-0.4123` | `-0.4349` |
| `iter2_selfplay_e2_ref` | `-0.4123` | `-0.4349` |
| `opening_disagreement_policy_head_e1` | `-0.4844` | `-0.5078` |
| `opening_disagreement_policy_head_e2` | `-0.5595` | `-0.5807` |

## Gate

Gate rule for this task: run the default deterministic gate for promoted current and any opening-disagreement candidate that beats promoted current by at least `+0.01` DS on fixed large `384:256`.

- `promoted_current_ref`: gate run, classification `high_search_breakthrough`
- `opening_disagreement_policy_head_e1`: did not qualify
- `opening_disagreement_policy_head_e2`: did not qualify

## Decision

This run matches `training_update_too_weak`.

- audit disagreement rate `35.74%` is well above the `10%` threshold
- high-confidence disagreements `13,363` show there is still evaluation-relevant search signal
- best trained lane changed top-1 on only `4.20%` of mined states
- no trained lane improved fixed large `384:256`

The bottleneck is not lack of disagreement states. The bottleneck is that this
policy-head-only update, at this LR and replay weighting, is too small to move
the model where the search is disagreeing.

## Artifacts

- workdir: `/tmp/azlite_promoted_current_opening_puct_disagreement`
- summary: `/tmp/azlite_promoted_current_opening_puct_disagreement/summary_metrics.json`
- disagreement audit: `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_state_disagreement_audit.json`
- mined replay: `/tmp/azlite_promoted_current_opening_puct_disagreement/opening_puct_disagreement_replay.jsonl`
