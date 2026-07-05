# AlphaZero-Lite Residual_v3 Opening Iteration-0 Target-Lane Ablation Results

**Classification**: `residual_v3_opening_iteration0_target_lane_rejected`

## Scope

This experiment reused the full iteration-0 preflight artifact from PR #145 and reran the same residual_v3 selected-opening training recipe with alternate target lanes only.

- do not promote
- do not overwrite `model-artifact/current`
- architecture: `residual_v3`
- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- reused preflight manifest SHA256: `f9c80affef4be9385e3e44f1a2c2629002f41d51db5d4cabbe4aeada1c64e3d2`
- reused selected positions SHA256: `60555445f5f6b2f8aadcb96f338a79fdca0b113c92a53c3588bfd000e8d0b88d`
- selected rows: `256`
- medium suite SHA256: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`
- epochs: `8`
- batch size: `128`
- seed: `42`
- no value transform
- no root-prior transform
- no tablebase overlay
- no seed-lottery promotion

## Commands

Reused preflight artifacts from PR #145:

```bash
sha256sum \
  /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_training_manifest.json \
  /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl \
  model-artifact/current/weights.json \
  /tmp/azlite_opening_suite/medium_eval.jsonl
```

`sim384_default` train + selected-suite eval:

```bash
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 \
.venv/bin/python ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default \
  --manifest /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_training_manifest.json \
  --selected-rows /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl \
  --current model-artifact/current \
  --eval-suite /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl \
  --budget-pairs 384:256,768:768,1200:1200,1200:256 \
  --games-per-opening 2 \
  --workers 24 \
  --epochs 8 \
  --batch-size 128 \
  --seed 42 \
  --timeout 21600 \
  --target-lane-label sim384_default
```

`sim384_default` medium-suite eval:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default/evaluation_medium \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default/artifacts/low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default/artifacts/very_low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default/artifacts/very_low_lr_policy_head \
  --budget-pairs 384:256,768:768,1200:1200,1200:256 \
  --games-per-opening 2 \
  --seed 42 \
  --root-policy-mode deterministic \
  --workers 24 \
  --timeout 21600 \
  --c-puct 1.25 \
  --c-puct-schedule-json '{"768:768":0.9}' \
  --tactical-root-bias 0.0
```

`sim768_default` train + selected-suite eval:

```bash
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 \
.venv/bin/python ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default \
  --manifest /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_training_manifest.json \
  --selected-rows /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl \
  --current model-artifact/current \
  --eval-suite /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl \
  --budget-pairs 384:256,768:768,1200:1200,1200:256 \
  --games-per-opening 2 \
  --workers 24 \
  --epochs 8 \
  --batch-size 128 \
  --seed 42 \
  --timeout 21600 \
  --target-lane-label sim768_default
```

`sim768_default` medium-suite eval:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default/evaluation_medium \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default/artifacts/low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default/artifacts/very_low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default/artifacts/very_low_lr_policy_head \
  --budget-pairs 384:256,768:768,1200:1200,1200:256 \
  --games-per-opening 2 \
  --seed 42 \
  --root-policy-mode deterministic \
  --workers 24 \
  --timeout 21600 \
  --c-puct 1.25 \
  --c-puct-schedule-json '{"768:768":0.9}' \
  --tactical-root-bias 0.0
```

Optional `sim1200_default` train + selected-suite eval:

```bash
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 \
.venv/bin/python ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default \
  --manifest /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_training_manifest.json \
  --selected-rows /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl \
  --current model-artifact/current \
  --eval-suite /tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl \
  --budget-pairs 384:256,768:768,1200:1200,1200:256 \
  --games-per-opening 2 \
  --workers 24 \
  --epochs 8 \
  --batch-size 128 \
  --seed 42 \
  --timeout 21600 \
  --target-lane-label sim1200_default
```

Optional `sim1200_default` medium-suite eval:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default/evaluation_medium \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default/artifacts/low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default/artifacts/very_low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default/artifacts/very_low_lr_policy_head \
  --budget-pairs 384:256,768:768,1200:1200,1200:256 \
  --games-per-opening 2 \
  --seed 42 \
  --root-policy-mode deterministic \
  --workers 24 \
  --timeout 21600 \
  --c-puct 1.25 \
  --c-puct-schedule-json '{"768:768":0.9}' \
  --tactical-root-bias 0.0
```

## Target Datasets

| Target lane | Workdir | Target manifest SHA256 | Target dataset SHA256 | Target lane distribution |
|---|---|---|---|---|
| `sim768_equal_override` | `/tmp/opencode/residual_v3_opening_iteration0_full_train` | `1be30dc63d5922336b9420045b589d6092dd0cba3bf5a82412645400f72ee3bd` | `b5f4315759df18eb24a4d98c17a105971fd2199850c6418085eb62406e77d042` | `sim768_equal_override=256` |
| `sim384_default` | `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default` | `23a59fb1f056edfb67cd8cc861c37afaa41c6c89b0371f639c7cc2b66e8979a7` | `a4d35ce30725e5764aab627a99979ac8468d9a5f78e7c3505f7c38f07515d6e4` | `sim384_default=256` |
| `sim768_default` | `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default` | `094c11c145f6d4648b3c362cce3f6b61dfed6b1f71bc9033e5b21bb039424e5f` | `da04e26543716975d138bb8f55dde022733f8773eb34564aea65d321b378177c` | `sim768_default=256` |
| `sim1200_default` | `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default` | `438767dc4c3f9cfd26530f7ef392442a2efec47c2873b19e38d696a1c54fbd9e` | `890dc7c9627a030fa49966ab21505ffbd5642cd3a57b7fd8dee5f96eb5a48fc1` | `sim1200_default=256` |

## Candidate Artifact Hashes

| Target lane | Candidate | Checkpoint SHA256 | Artifact weights SHA256 |
|---|---|---|---|
| `sim768_equal_override` | `low_lr_all` | `5e8c4b247c53aca84f82cecfc92195ecfb3abd4818e9c0217fc6b9e6df54ca6c` | `2cfe0d0ae8a18539d15f18fc94756d7680262116a39460c6166ee1e813fdc00b` |
| `sim768_equal_override` | `very_low_lr_all` | `4383d6ad8a84ca0bf4e8c3a3ea605f1d406c2b91f2c7a48c3422d78b5c99b3cd` | `0960d7e346756ce5cc746e2271bcaf2c233e58077e55f4ed780cc569192a300f` |
| `sim768_equal_override` | `very_low_lr_policy_head` | `5c562928f0bca2a7f61627cd92f19a43f96e59376dde30ed471c16934693af89` | `4b1f81ec4c84ac29afcf88622948c4f4534006d2b35531bce6b2afac9d59f171` |
| `sim384_default` | `low_lr_all` | `e0021d7e6c78bae70aaf68e8587f28aec1d2ca5c15ed06a59ee6a6f49cc4c62c` | `ce2f0a5d1fea18a69ee7a9d232b4f4acc0c38b45a852f6609950b70dd8e4f4ee` |
| `sim384_default` | `very_low_lr_all` | `9ed2e418adaa07df5ea42f037317db5839ce4adf54d321f88ac7130bee3fad89` | `c8514f2e15b193c136aa7fef85e23097ec36af8cb45467d8ae015484b10f723d` |
| `sim384_default` | `very_low_lr_policy_head` | `d83994db10b187f7d8293c1d598e573237ea7a7969679e7428ebc52766edbe4c` | `721a20aea9c08d67ec4c25ac217c231ca8080463aaa07b08329b46127a173003` |
| `sim768_default` | `low_lr_all` | `f02366c7ead12817887082c557a5a8a4b438e12cf3601da35836bee240e24234` | `2579424449b301fb90b650d82c273e5469e2138484e0ada46e716f63859a2b2c` |
| `sim768_default` | `very_low_lr_all` | `ce3997e866bc420e10994ec5527dda8a5b7ca47a8b00603e52b34d0b62a48d4a` | `923e558303b3fa8a2e615ce48d66d32f6c954067d183009779a1706717de306e` |
| `sim768_default` | `very_low_lr_policy_head` | `d5e5bec4f50fbd69cbc6bd31e4d2afae81f551632f3f15212c5f00d35c409c96` | `25088d55bafa47d98d36d0af8391bc5e27ef3923331ab87fe3414cc3dbdcf083` |
| `sim1200_default` | `low_lr_all` | `d1559f23fd7b94feff29f9af623fd4f340bc07012bef4e98fd34d9f2f6ff7432` | `a31cd3a9983614b03cced2e40c15607cf73ea228bff638ac53f04dfcd2062708` |
| `sim1200_default` | `very_low_lr_all` | `adf2b1735d9d566bd431728abf289863d5e63df95f9c90e2ad9170bad04d7564` | `57d0a6a9579ef8afe01dc4d0d7ebbb35d87e84b7a51304856ca89f5617c5fdeb` |
| `sim1200_default` | `very_low_lr_policy_head` | `c3591175bd3b2be381254b86b0ae5633887472a4b33ebe1d6bee2b031890c560` | `03d0abecaa5f3b4e591e47061039952db57648fab9d6fd385b960b8ff61b07ae` |

## Selected Iteration-0 Suite

- baseline report: `/tmp/opencode/residual_v3_opening_iteration0_full_train/evaluation/temperature_benchmark_report.json`
- `sim384_default` report: `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default/evaluation/temperature_benchmark_report.json`
- `sim768_default` report: `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default/evaluation/temperature_benchmark_report.json`
- `sim1200_default` report: `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default/evaluation/temperature_benchmark_report.json`
- suite SHA256: `60555445f5f6b2f8aadcb96f338a79fdca0b113c92a53c3588bfd000e8d0b88d`
- suite size: `256`
- games per opening: `2`

| Target lane | Candidate | 384:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS |
|---|---|---:|---:|---:|---:|
| `sim768_equal_override` | `low_lr_all` | `-0.355469` | `-0.015625` | `0.009766` | `0.025391` |
| `sim768_equal_override` | `very_low_lr_all` | `-0.621094` | `0.123047` | `0.304688` | `-0.251953` |
| `sim768_equal_override` | `very_low_lr_policy_head` | `-0.257812` | `0.341797` | `0.275391` | `-0.125000` |
| `sim384_default` | `low_lr_all` | `-0.644531` | `0.054688` | `0.277344` | `-0.181641` |
| `sim384_default` | `very_low_lr_all` | `-0.621094` | `0.347656` | `0.269531` | `-0.251953` |
| `sim384_default` | `very_low_lr_policy_head` | `-0.392578` | `0.154297` | `0.125000` | `-0.101562` |
| `sim768_default` | `low_lr_all` | `-0.628906` | `0.074219` | `0.130859` | `-0.125000` |
| `sim768_default` | `very_low_lr_all` | `-0.621094` | `0.355469` | `0.312500` | `-0.101562` |
| `sim768_default` | `very_low_lr_policy_head` | `-0.392578` | `0.341797` | `0.425781` | `-0.101562` |
| `sim1200_default` | `low_lr_all` | `-0.634766` | `-0.177734` | `0.015625` | `0.025391` |
| `sim1200_default` | `very_low_lr_all` | `-0.320312` | `0.060547` | `0.304688` | `-0.251953` |
| `sim1200_default` | `very_low_lr_policy_head` | `-0.267578` | `0.248047` | `0.125000` | `0.048828` |

## Medium Suite

- baseline report: `/tmp/opencode/residual_v3_opening_iteration0_full_train/evaluation_medium/temperature_benchmark_report.json`
- `sim384_default` report: `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim384_default/evaluation_medium/temperature_benchmark_report.json`
- `sim768_default` report: `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim768_default/evaluation_medium/temperature_benchmark_report.json`
- `sim1200_default` report: `/tmp/opencode/residual_v3_opening_iteration0_ablation_sim1200_default/evaluation_medium/temperature_benchmark_report.json`
- suite SHA256: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`
- suite size: `128`
- games per opening: `2`

| Target lane | Candidate | 384:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS |
|---|---|---:|---:|---:|---:|
| `sim768_equal_override` | `low_lr_all` | `-0.394531` | `0.156250` | `0.125000` | `-0.042969` |
| `sim768_equal_override` | `very_low_lr_all` | `-0.601562` | `0.257812` | `0.386719` | `-0.265625` |
| `sim768_equal_override` | `very_low_lr_policy_head` | `-0.210938` | `0.472656` | `0.378906` | `-0.164062` |
| `sim384_default` | `low_lr_all` | `-0.621094` | `0.191406` | `0.355469` | `-0.175781` |
| `sim384_default` | `very_low_lr_all` | `-0.601562` | `0.433594` | `0.328125` | `-0.265625` |
| `sim384_default` | `very_low_lr_policy_head` | `-0.335938` | `0.253906` | `0.257812` | `-0.144531` |
| `sim768_default` | `low_lr_all` | `-0.593750` | `0.160156` | `0.171875` | `-0.164062` |
| `sim768_default` | `very_low_lr_all` | `-0.601562` | `0.382812` | `0.378906` | `-0.144531` |
| `sim768_default` | `very_low_lr_policy_head` | `-0.335938` | `0.472656` | `0.500000` | `-0.144531` |
| `sim1200_default` | `low_lr_all` | `-0.605469` | `-0.035156` | `0.132812` | `-0.042969` |
| `sim1200_default` | `very_low_lr_all` | `-0.359375` | `0.117188` | `0.386719` | `-0.265625` |
| `sim1200_default` | `very_low_lr_policy_head` | `-0.226562` | `0.363281` | `0.257812` | `-0.023438` |

## Comparison Against PR #145 `sim768_equal_override`

Best `384:256` DS by target lane:

| Target lane | Best selected-suite 384:256 DS | Delta vs PR #145 | Best medium-suite 384:256 DS | Delta vs PR #145 |
|---|---:|---:|---:|---:|
| `sim768_equal_override` | `-0.257812` | `0.000000` | `-0.210938` | `0.000000` |
| `sim384_default` | `-0.392578` | `-0.134766` | `-0.335938` | `-0.125000` |
| `sim768_default` | `-0.392578` | `-0.134766` | `-0.335938` | `-0.125000` |
| `sim1200_default` | `-0.267578` | `-0.009766` | `-0.226562` | `-0.015625` |

Interpretation:

- `sim384_default` did not recover the primary `384:256` gate. It was materially worse than PR #145 on both suites.
- `sim768_default` did not recover the primary `384:256` gate. Removing the equal-budget `c_puct` override from the target lane did not fix the low-budget failure.
- optional `sim1200_default` came closest to PR #145 on `384:256`, but still remained negative on both suites and even regressed `768:768` for `low_lr_all`.
- across all alternate target lanes, no candidate turned the primary `384:256` gate positive on either evaluated suite.

## Conclusion

Requested gate interpretation:

- if `sim384_default` improves `384:256` but regresses high-budget checks, classify mixed and propose a preservation or blended-target follow-up
- if no lane improves `384:256`, stop this selected-opening supervised-target recipe
- if a lane improves `384:256` without regressing `768:768`, `1200:1200`, or `1200:256`, propose a separate deterministic promotion-gate PR

Result:

- no alternate target lane improved `384:256` on the selected iteration-0 suite
- no alternate target lane improved `384:256` on the broader medium suite
- the failure is therefore not specific to `sim768_equal_override`
- this selected-opening supervised-target recipe should stop in its current form
- explicit classification: `residual_v3_opening_iteration0_target_lane_rejected`
- promotion decision: **do not promote**
- `model-artifact/current` remained unchanged

## Verification

```bash
.venv/bin/ruff check \
  ml/alphazero_lite/run_residual_v3_opening_iteration0_preflight.py \
  ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py \
  ml/alphazero_lite/test_run_residual_v3_opening_iteration0_preflight.py \
  ml/alphazero_lite/test_run_residual_v3_opening_iteration0_train.py

.venv/bin/python -m unittest \
  ml.alphazero_lite.test_run_residual_v3_opening_iteration0_preflight \
  ml.alphazero_lite.test_run_residual_v3_opening_iteration0_train
```
