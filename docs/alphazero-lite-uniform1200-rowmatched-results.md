# Uniform1200 Replay-Mass Confound Results

Classification: `uniform1200_replay_mass_confound_rejected`.

PR #287's parent, fixed replays, dynamic replays, and checkpoints were verified before use. `train.load_jsonl_replay()` loads every source once and tiles its compact indexes by its positive integer replay weight; consequently, source exposure is `raw rows x weight`. No self-play, MCTS setting, fixed replay, training setting, or production artifact was changed. No promotion was attempted.

## Replay Exposure

| seed | lane | dynamic rows x weight | selected rows x weight | guard rows x weight | effective total | dynamic fraction |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 44 | control | 63,352 x 1 | 20 x 1 | 5 x 2 | 63,382 | 99.9527% |
| 44 | uniform1200 | 71,166 x 1 | 20 x 1 | 5 x 2 | 71,196 | 99.9579% |
| 44 | rowmatched | 63,352 x 1 | 20 x 1 | 5 x 2 | 63,382 | 99.9527% |
| 45 | control | 64,108 x 1 | 20 x 1 | 5 x 2 | 64,138 | 99.9532% |
| 45 | uniform1200 | 70,975 x 1 | 20 x 1 | 5 x 2 | 71,005 | 99.9577% |
| 45 | rowmatched | 64,108 x 1 | 20 x 1 | 5 x 2 | 64,138 | 99.9532% |
| 46 | control | 63,562 x 1 | 20 x 1 | 5 x 2 | 63,592 | 99.9528% |
| 46 | uniform1200 | 70,584 x 1 | 20 x 1 | 5 x 2 | 70,614 | 99.9575% |
| 46 | rowmatched | 63,562 x 1 | 20 x 1 | 5 x 2 | 63,592 | 99.9528% |

Row-matched dynamic exposure exactly equals control in every seed; preflight rejects any mismatch.

## Distribution Preservation

Rows were proportionally allocated over `(value_target_bucket_for_move_index(move_index), player, winner)` with largest-remainder quotas, then chosen by SHA-256 ordering of canonical row JSON. The maximum phase/player/outcome percentage-point deviation from full uniform was 0.0021, below the registered 1.0 point tolerance in all seeds. Mean policy entropy was full/rowmatched: seed 44 `0.268533/0.268446`, seed 45 `0.271743/0.272626`, seed 46 `0.272766/0.272987`. Canonical-state counts were `51,115/45,872`, `51,183/46,583`, and `50,680/46,030`; trajectory starts were `1,600/1,401`, `1,600/1,445`, and `1,600/1,423`. Sampling was row-neutral: it did not use target confidence, values, tactical labels, capture, or extra-turn information.

## Training And Arenas

All lanes used PR #287's parent checkpoint, seed, four epochs, batch size 512, residual_v3 `96,3`, Huber value loss, sharpened targets, and `1,1,2` replay weights. Frozen opening-suite SHA was `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`; evaluation seed was 42.

| seed | rowmatched vs control, 1200:1200 | rowmatched vs full uniform1200, 1200:1200 |
| ---: | --- | ---: |
| 44 | +0.1406 [0.1055, 0.1758] | -0.1621 |
| 45 | +0.2207 | -0.0078 |
| 46 | +0.2695 | -0.0938 |

Seed 44 also completed PR #287's full budget list: `384:256 +0.2676 [0.2031, 0.3281]`, `1200:1200 +0.1406 [0.1055, 0.1758]`, `1200:256 +0.1523 [0.0977, 0.2070]`, and `256:768 +0.1680 [0.1230, 0.2129]`. The exact benchmark artifacts are retained under `/home/alex/Mancala/rowmatched-work`.

## Production Gate And Forensics

| seed | arena | failure reasons | overall top-1/regret/blunder delta | capture top-1/regret/blunder delta | sparse top-1/regret/blunder delta |
| ---: | ---: | --- | --- | --- | --- |
| 44 | 1.000 | regression_check_failed; forensic_overall_regressed | 0.0000 / +0.0190 / +0.0580 | 0.0000 / -0.0490 / -0.1250 | 0.0000 / 0.0000 / 0.0000 |
| 45 | 1.000 | forensic_overall_regressed | 0.0000 / +0.0147 / +0.0580 | 0.0000 / -0.0463 / -0.1666 | 0.0000 / +0.0123 / 0.0000 |
| 46 | 0.500 | arena_score_below_threshold | not run after arena prefilter | not run after arena prefilter | not run after arena prefilter |

The direct arena advantage remains positive in all three seeds, but no row-matched candidate passes the unchanged gate. The decisive repeated regression is overall forensic blunder rate for seeds 44 and 45; seed 46 retains the same arena failure as full uniform1200. This does not support replay mass as the production-gate cause.

Recommended next experiment: bucket-level replay-distribution audit.
