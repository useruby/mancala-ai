# AlphaZero-Lite Residual_v3 Opening Iteration-0 Full Results

**Classification**: `residual_v3_opening_iteration0_full_rejected`

## Scope

Ran the full iteration-0 experiment on the merged PR #143/#144 pipeline without promotion and without overwriting `model-artifact/current`.

- canonical current artifact unchanged: `model-artifact/current`
- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- target lane: `sim768_equal_override`
- model family: `residual_v3` only
- seed: `42`
- no value transform
- no root-prior transform
- no tablebase overlay
- no seed-lottery promotion

## Full Commands

Preflight:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python ml/alphazero_lite/run_residual_v3_opening_iteration0_preflight.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_full_preflight \
  --current model-artifact/current \
  --expected-current-sha256 8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a \
  --selection-limit 256 \
  --seed 42 \
  --workers 24
```

Train + selected-suite eval:

```bash
OMP_NUM_THREADS=24 OPENBLAS_NUM_THREADS=24 MKL_NUM_THREADS=24 \
.venv/bin/python ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_full_train \
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
  --timeout 21600
```

Medium-suite eval:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_full_train/evaluation_medium \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/opencode/residual_v3_opening_iteration0_full_train/artifacts/low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_full_train/artifacts/very_low_lr_all,/tmp/opencode/residual_v3_opening_iteration0_full_train/artifacts/very_low_lr_policy_head \
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

## Full Preflight Manifest

- manifest: `/tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_training_manifest.json`
- manifest SHA256: `f9c80affef4be9385e3e44f1a2c2629002f41d51db5d4cabbe4aeada1c64e3d2`
- canonical generated suite: `/tmp/opencode/residual_v3_opening_iteration0_full_preflight/inputs/opening_suite_canonical.jsonl`
- canonical suite SHA256: `135fe2c813f972d1a030f5756d57f915713cf325b1fe2945c289758c9237f595`
- canonical suite rows: `28961`
- selected positions: `/tmp/opencode/residual_v3_opening_iteration0_full_preflight/manifests/iteration0_selected_positions.jsonl`
- selected positions SHA256: `60555445f5f6b2f8aadcb96f338a79fdca0b113c92a53c3588bfd000e8d0b88d`
- selected rows: `256`
- search profile hash: `42b3a19a4d85679c0ef219d494238b76170ff183e2922095940a034ad5c2093c`

Selected-row distributions:

| Field | Distribution |
|---|---|
| phase | `early=256` |
| seat | `0=171`, `1=85` |
| opening prefix ply | `3=4`, `4=2`, `5=36`, `6=214` |
| first move family | `0=35`, `1=77`, `2=29`, `3=35`, `4=48`, `5=32` |
| selection tags | `unstable_search=256` |

## Target Dataset

- target manifest: `/tmp/opencode/residual_v3_opening_iteration0_full_train/targets/iteration0_target_manifest.json`
- target manifest SHA256: `1be30dc63d5922336b9420045b589d6092dd0cba3bf5a82412645400f72ee3bd`
- target dataset: `/tmp/opencode/residual_v3_opening_iteration0_full_train/targets/iteration0_train_targets.jsonl`
- target dataset SHA256: `b5f4315759df18eb24a4d98c17a105971fd2199850c6418085eb62406e77d042`
- target rows: `256`
- train.py compatibility: `feature_count=27`, `policy_size=6`, `value_count=256`
- value summary: `min=-0.165795`, `max=0.392061`, `mean=0.036005`
- target lane distribution: `sim768_equal_override=256`

## Train Commands

`low_lr_all`

```bash
/home/alex/Mancala/ai/.venv/bin/python /home/alex/Mancala/ai/ml/alphazero_lite/train.py --data /tmp/opencode/residual_v3_opening_iteration0_full_train/targets/iteration0_train_targets.jsonl --out /tmp/opencode/residual_v3_opening_iteration0_full_train/candidates/low_lr_all/checkpoint_epoch8.npz --epochs 8 --batch-size 128 --lr 3e-05 --seed 42 --device auto --value-loss-weight 0.3 --value-loss huber --val-split 0.0 --grad-clip 1.0 --hidden-sizes 96,3 --model-type residual_v3 --input-encoding kalah_v3 --policy-target-mode default --value-target-mode default --save-top-k 0 --save-epochs 8 --lr-scheduler none --trainable-scope all --init-checkpoint /tmp/opencode/residual_v3_opening_iteration0_full_train/inputs/current_from_weights_json.npz
```

`very_low_lr_all`

```bash
/home/alex/Mancala/ai/.venv/bin/python /home/alex/Mancala/ai/ml/alphazero_lite/train.py --data /tmp/opencode/residual_v3_opening_iteration0_full_train/targets/iteration0_train_targets.jsonl --out /tmp/opencode/residual_v3_opening_iteration0_full_train/candidates/very_low_lr_all/checkpoint_epoch8.npz --epochs 8 --batch-size 128 --lr 1e-05 --seed 42 --device auto --value-loss-weight 0.3 --value-loss huber --val-split 0.0 --grad-clip 1.0 --hidden-sizes 96,3 --model-type residual_v3 --input-encoding kalah_v3 --policy-target-mode default --value-target-mode default --save-top-k 0 --save-epochs 8 --lr-scheduler none --trainable-scope all --init-checkpoint /tmp/opencode/residual_v3_opening_iteration0_full_train/inputs/current_from_weights_json.npz
```

`very_low_lr_policy_head`

```bash
/home/alex/Mancala/ai/.venv/bin/python /home/alex/Mancala/ai/ml/alphazero_lite/train.py --data /tmp/opencode/residual_v3_opening_iteration0_full_train/targets/iteration0_train_targets.jsonl --out /tmp/opencode/residual_v3_opening_iteration0_full_train/candidates/very_low_lr_policy_head/checkpoint_epoch8.npz --epochs 8 --batch-size 128 --lr 1e-05 --seed 42 --device auto --value-loss-weight 0.3 --value-loss huber --val-split 0.0 --grad-clip 1.0 --hidden-sizes 96,3 --model-type residual_v3 --input-encoding kalah_v3 --policy-target-mode default --value-target-mode default --save-top-k 0 --save-epochs 8 --lr-scheduler none --trainable-scope policy_head --init-checkpoint /tmp/opencode/residual_v3_opening_iteration0_full_train/inputs/current_from_weights_json.npz
```

## Candidate Artifacts

| Candidate | Scope | Checkpoint SHA256 | Artifact weights SHA256 |
|---|---|---|---|
| `low_lr_all` | `all` | `5e8c4b247c53aca84f82cecfc92195ecfb3abd4818e9c0217fc6b9e6df54ca6c` | `2cfe0d0ae8a18539d15f18fc94756d7680262116a39460c6166ee1e813fdc00b` |
| `very_low_lr_all` | `all` | `4383d6ad8a84ca0bf4e8c3a3ea605f1d406c2b91f2c7a48c3422d78b5c99b3cd` | `0960d7e346756ce5cc746e2271bcaf2c233e58077e55f4ed780cc569192a300f` |
| `very_low_lr_policy_head` | `policy_head` | `5c562928f0bca2a7f61627cd92f19a43f96e59376dde30ed471c16934693af89` | `4b1f81ec4c84ac29afcf88622948c4f4534006d2b35531bce6b2afac9d59f171` |

## Evaluation

Selected iteration-0 suite:

- report: `/tmp/opencode/residual_v3_opening_iteration0_full_train/evaluation/temperature_benchmark_report.json`
- suite SHA256: `60555445f5f6b2f8aadcb96f338a79fdca0b113c92a53c3588bfd000e8d0b88d`
- suite size: `256`
- games per opening: `2`

| Candidate | 384:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS |
|---|---:|---:|---:|---:|
| `low_lr_all` | `-0.355469` | `-0.015625` | `0.009766` | `0.025391` |
| `very_low_lr_all` | `-0.621094` | `0.123047` | `0.304688` | `-0.251953` |
| `very_low_lr_policy_head` | `-0.257812` | `0.341797` | `0.275391` | `-0.125000` |

Broader deterministic medium suite:

- report: `/tmp/opencode/residual_v3_opening_iteration0_full_train/evaluation_medium/temperature_benchmark_report.json`
- suite SHA256: `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`
- suite size: `128`
- games per opening: `2`

| Candidate | 384:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS |
|---|---:|---:|---:|---:|
| `low_lr_all` | `-0.394531` | `0.156250` | `0.125000` | `-0.042969` |
| `very_low_lr_all` | `-0.601562` | `0.257812` | `0.386719` | `-0.265625` |
| `very_low_lr_policy_head` | `-0.210938` | `0.472656` | `0.378906` | `-0.164062` |

Notes:

- The generated canonical suite has `28961` openings, so a full deterministic benchmark there would be materially larger than this PR’s full run. The broader follow-up eval here therefore used the canonical existing medium suite.
- No candidate improved the primary `384:256` budget on either evaluated suite.

## Comparison Against Smoke

Smoke report: `docs/alphazero-lite-residual-v3-opening-iteration0-training-results.md`

- smoke classification: `residual_v3_opening_iteration0_candidate_mixed`
- smoke target rows: `2`
- smoke `games-per-opening`: `1`
- smoke artifact hashes:
  - `low_lr_all`: `66bdced700ac473316f91f72a6e7f9d36520f47a776b71f7933dcd206167bfa9`
  - `very_low_lr_all`: `591c278f9341bd5f61c06eee024530d58ce87aefce129771a4cc4d42172645f1`
  - `very_low_lr_policy_head`: `2043a7c47f1931b0fda2d84ba7afbf2239d6e35b380032bdbe6abe1b1081b4d4`
- smoke DS highlights:
  - `384:256`: all lanes `0.0`
  - `768:768`: `0.5`, `0.5`, `1.0`
  - `1200:1200`: all lanes `0.0`
  - `1200:256`: `0.5`, `0.5`, `0.0`

Full-scale comparison:

- the smoke run’s non-negative `384:256` signal did not survive the full 256-row target set
- `768:768` still improved for the two very-low-LR lanes on both full suites, and for `low_lr_all` on the medium suite only
- `1200:1200` stayed non-regressive across the full suites
- `1200:256` regressed for two lanes on the selected suite and all lanes on the medium suite except the selected-suite `low_lr_all` case

## Conclusion

Requested gate interpretation:

- if no candidate improves `384:256`, reject this exact iteration-0 target recipe
- if `384:256` improves but `1200:1200` regresses, classify mixed and do not promote
- only if primary improves and high-budget checks do not regress, propose a follow-up deterministic promotion gate

Result:

- no candidate improved `384:256` on the selected suite
- no candidate improved `384:256` on the broader medium suite
- therefore this exact iteration-0 target recipe is **rejected**
- explicit conclusion: `residual_v3_opening_iteration0_full_rejected`
- promotion decision: **do not promote**
- `model-artifact/current` remained unchanged

## Verification

```bash
.venv/bin/ruff check \
  ml/alphazero_lite/run_residual_v3_opening_iteration0_preflight.py \
  ml/alphazero_lite/test_run_residual_v3_opening_iteration0_preflight.py \
  ml/alphazero_lite/test_run_residual_v3_opening_iteration0_train.py

.venv/bin/python -m unittest \
  ml.alphazero_lite.test_run_residual_v3_opening_iteration0_preflight \
  ml.alphazero_lite.test_run_residual_v3_opening_iteration0_train
```
