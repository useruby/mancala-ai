# AlphaZero-Lite Residual_v3 Opening Iteration-0 Training Results

**Classification**: `residual_v3_opening_iteration0_candidate_mixed`

## Scope

This PR adds deterministic iteration-0 target materialization and a guarded residual_v3-only train/eval runner:

- `ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py`
- `ml/alphazero_lite/test_run_residual_v3_opening_iteration0_train.py`

The smoke run below used the frozen preflight smoke manifest at `/tmp/opencode/residual_v3_opening_iteration0_smoke` to verify the full path end to end without promotion.

## Guardrails

- `model-artifact/current` unchanged
- current weights SHA256: `8d70e90a684caf946ab3f3e5d81a24e65be939b5be932930c389945fd9bb4e7a`
- architecture locked to `residual_v3`
- `default_c_puct = 1.25`
- override `768:768 -> 0.90`
- `root_policy_mode = deterministic`
- `root_prior_transform = null`
- `tactical_root_bias = 0.0`
- `value_transform = null`
- no tablebase overlay
- no promotion

## Frozen Inputs

- preflight manifest: `/tmp/opencode/residual_v3_opening_iteration0_smoke/manifests/iteration0_training_manifest.json`
- preflight manifest SHA256: `ef8a6aa33d869e8099c863380989af050d1be9a1827f064a8cb9c092212c96dc`
- selected positions: `/tmp/opencode/residual_v3_opening_iteration0_smoke/manifests/iteration0_selected_positions.jsonl`
- selected positions SHA256: `93cb39dbd99814ad177aa9f27848585fe636e335775f0e0f480657315e0140c3`

## Target Materialization

- target lane: `sim768_equal_override`
- target policy source: `selected_rows.search_results[].search_policy`
- target value source: `selected_rows.search_results[].root_value`
- target dataset: `/tmp/opencode/residual_v3_opening_iteration0_train_smoke/targets/iteration0_train_targets.jsonl`
- target dataset SHA256: `245559fb71a4bef76ee9e8c650672dcd480905c0469da166691106eb7b4f6ae9`
- target manifest SHA256: `49b28fc52a541ca1cf2696a0163db835ada522027433f0107e40dc8c7b51d87a`
- rows: `2`
- compatibility: `train.py` accepted `feature_count=27`, `policy_size=6`, `value_count=2`

## Smoke Command

```bash
.venv/bin/python ml/alphazero_lite/run_residual_v3_opening_iteration0_train.py \
  --workdir /tmp/opencode/residual_v3_opening_iteration0_train_smoke \
  --manifest /tmp/opencode/residual_v3_opening_iteration0_smoke/manifests/iteration0_training_manifest.json \
  --selected-rows /tmp/opencode/residual_v3_opening_iteration0_smoke/manifests/iteration0_selected_positions.jsonl \
  --current model-artifact/current \
  --eval-suite /tmp/opencode/residual_v3_opening_iteration0_smoke/manifests/iteration0_selected_positions.jsonl \
  --games-per-opening 1 \
  --workers 1 \
  --epochs 1 \
  --batch-size 16 \
  --timeout 1800
```

## Train Commands

`low_lr_all`

```bash
/home/alex/Mancala/ai/.venv/bin/python /home/alex/Mancala/ai/ml/alphazero_lite/train.py --data /tmp/opencode/residual_v3_opening_iteration0_train_smoke/targets/iteration0_train_targets.jsonl --out /tmp/opencode/residual_v3_opening_iteration0_train_smoke/candidates/low_lr_all/checkpoint_epoch1.npz --epochs 1 --batch-size 16 --lr 3e-05 --seed 42 --device auto --value-loss-weight 0.3 --value-loss huber --val-split 0.0 --grad-clip 1.0 --hidden-sizes 96,3 --model-type residual_v3 --input-encoding kalah_v3 --policy-target-mode default --value-target-mode default --save-top-k 0 --save-epochs 1 --lr-scheduler none --trainable-scope all --init-checkpoint /tmp/opencode/residual_v3_opening_iteration0_train_smoke/inputs/current_from_weights_json.npz
```

`very_low_lr_all`

```bash
/home/alex/Mancala/ai/.venv/bin/python /home/alex/Mancala/ai/ml/alphazero_lite/train.py --data /tmp/opencode/residual_v3_opening_iteration0_train_smoke/targets/iteration0_train_targets.jsonl --out /tmp/opencode/residual_v3_opening_iteration0_train_smoke/candidates/very_low_lr_all/checkpoint_epoch1.npz --epochs 1 --batch-size 16 --lr 1e-05 --seed 42 --device auto --value-loss-weight 0.3 --value-loss huber --val-split 0.0 --grad-clip 1.0 --hidden-sizes 96,3 --model-type residual_v3 --input-encoding kalah_v3 --policy-target-mode default --value-target-mode default --save-top-k 0 --save-epochs 1 --lr-scheduler none --trainable-scope all --init-checkpoint /tmp/opencode/residual_v3_opening_iteration0_train_smoke/inputs/current_from_weights_json.npz
```

`very_low_lr_policy_head`

```bash
/home/alex/Mancala/ai/.venv/bin/python /home/alex/Mancala/ai/ml/alphazero_lite/train.py --data /tmp/opencode/residual_v3_opening_iteration0_train_smoke/targets/iteration0_train_targets.jsonl --out /tmp/opencode/residual_v3_opening_iteration0_train_smoke/candidates/very_low_lr_policy_head/checkpoint_epoch1.npz --epochs 1 --batch-size 16 --lr 1e-05 --seed 42 --device auto --value-loss-weight 0.3 --value-loss huber --val-split 0.0 --grad-clip 1.0 --hidden-sizes 96,3 --model-type residual_v3 --input-encoding kalah_v3 --policy-target-mode default --value-target-mode default --save-top-k 0 --save-epochs 1 --lr-scheduler none --trainable-scope policy_head --init-checkpoint /tmp/opencode/residual_v3_opening_iteration0_train_smoke/inputs/current_from_weights_json.npz
```

## Candidate Artifacts

| Candidate | Artifact weights SHA256 | Trainable scope |
|---|---|---|
| `low_lr_all` | `66bdced700ac473316f91f72a6e7f9d36520f47a776b71f7933dcd206167bfa9` | `all` |
| `very_low_lr_all` | `591c278f9341bd5f61c06eee024530d58ce87aefce129771a4cc4d42172645f1` | `all` |
| `very_low_lr_policy_head` | `2043a7c47f1931b0fda2d84ba7afbf2239d6e35b380032bdbe6abe1b1081b4d4` | `policy_head` |

## Evaluation

Seat-aware deterministic opening-suite smoke benchmark:

- eval suite path: `/tmp/opencode/residual_v3_opening_iteration0_smoke/manifests/iteration0_selected_positions.jsonl`
- eval suite SHA256: `93cb39dbd99814ad177aa9f27848585fe636e335775f0e0f480657315e0140c3`
- benchmark report: `/tmp/opencode/residual_v3_opening_iteration0_train_smoke/evaluation/temperature_benchmark_report.json`

| Candidate | 384:256 DS | 768:768 DS | 1200:1200 DS | 1200:256 DS | Follow-up gate? |
|---|---:|---:|---:|---:|---|
| `low_lr_all` | `0.0` | `0.5` | `0.0` | `0.5` | no |
| `very_low_lr_all` | `0.0` | `0.5` | `0.0` | `0.5` | no |
| `very_low_lr_policy_head` | `0.0` | `1.0` | `0.0` | `0.0` | no |

## Outcome

- Primary `384:256` did not improve for any lane.
- `768:768` did not regress and improved on the smoke suite.
- `1200:1200` and `1200:256` stayed non-negative on the smoke suite.
- No lane met the full follow-up gate because none cleared the `384:256` improvement requirement.
- Follow-up promotion-gate PR worthiness: `no`

## Notes

- The new runner writes reproducible target and training manifests with hashes, distributions, target-generation settings, candidate commands, and evaluation summaries.
- Target generation is deterministic and transform-free.
- The smoke classification stays `mixed` rather than `positive` because the main opening-sensitive gate remains unmet.
