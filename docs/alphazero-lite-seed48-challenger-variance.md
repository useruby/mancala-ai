# Seed48 Challenger Variance Audit

Classification: `selfplay_variance_dominant`.

Phase 0 passed. Training and export from the frozen valid six-worker seed443
self-play reproduced both the original checkpoint SHA256
`f65f466ebbfc807a7e561ba383c101dd02f83a40cfc55ab621692ee44dcf2cf5` and
weights SHA256 `2e517e21097dba818b362aa43b7c6917c17438e1ad0c8a16136ba10a20500a8b`.

The preregistered self-play-data selection is seed401, seed407, seed413, and
seed419: the lowest four available PR #341 workers6 uniform1200 datasets with
validated hashes. The fixed-data training seeds are 443, 1001, 1003, 1009,
and 1013. The plan records the parent, all replay hashes, and the exact
training contract.

The external PR #341 opening suite disappeared after its initial validation.
`build_opening_suite.py --opening-plies 2,4,6 --suite-sizes 32,128,384
--seed 49` reconstructed the historical corpus at
`.tmp/seed48-challenger-variance/pr341-suite-reconstruction/medium_eval.jsonl`.
Its SHA256 exactly matches the required
`57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`.

The primary diagnostic arena was equal 384-vs-384 against seed48 with the
same deterministic, seat-balanced 128-opening suite used by PR #341. Each
reported score combines 256 games, 128 in each challenger-seat condition.

| Group | Cells | Mean | SD | Min/max | Range |
| --- | ---: | ---: | ---: | --- | ---: |
| Different self-play, train seed443 | 4 | 0.5151 | 0.1620 | 0.3750/0.6641 | 0.2891 |
| Seed443 self-play, different training seeds | 5 | 0.5016 | 0.0735 | 0.4160/0.5664 | 0.1504 |

`selfplay_strength_sd / training_strength_sd = 2.2025`. The self-play range
exceeds 0.05 and self-play SD exceeds 1.5 times the training-seed SD, meeting
the preregistered `selfplay_variance_dominant` rule. This is a variance
diagnostic, not a significance claim.

| Supporting metric | Self-play mean, SD, range | Training-seed mean, SD, range |
| --- | --- | --- |
| Raw optimal mass | 0.6230, 0.0098, 0.0211 | 0.6262, 0.0076, 0.0179 |
| Raw expected regret | 2.3106, 0.0833, 0.1697 | 2.2749, 0.0641, 0.1686 |
| MCTS-384 optimal mass | 0.6974, 0.0061, 0.0144 | 0.6975, 0.0114, 0.0298 |
| MCTS-384 expected regret | 1.6255, 0.0598, 0.1437 | 1.6071, 0.0890, 0.2168 |

The next experiment should improve self-play volume or diversity, or pool
validated self-play before retraining while holding the training recipe fixed.

No self-play was generated. No canonical gate was run, no promotion was
performed, and no model selection was performed.
