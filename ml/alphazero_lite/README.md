# AlphaZero-lite Manual Training Module

For repo-wide setup, shared dev tooling, linting, and `pre-commit` usage, start with the root [`README.md`](../../README.md). This guide stays focused on AlphaZero-lite training and operations.

This folder contains a manual, language-agnostic training workflow for Mancala
(`kalah`) model artifacts consumed by Rails runtime code.

The workflow is intentionally simple:

1. Generate MCTS-supervised training records from real Kalah games.
2. Train a tiny policy-value model checkpoint.
3. Export artifacts (`model.npz` + `weights.json` + `metadata.json`) into Rails storage.

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r ml/alphazero_lite/requirements.txt

.venv/bin/python ml/alphazero_lite/generate_bootstrap_dataset.py \
  --out /tmp/azlite/self_play.jsonl \
  --games 2000 \
  --simulations 5000 \
  --seed 42 \
  --max-positions-per-game 24 \
  --workers 6 \
  --tree-reuse-enabled
python ml/alphazero_lite/train.py --data /tmp/azlite/self_play.jsonl --out /tmp/azlite/checkpoint.npz --epochs 12 --batch-size 512 --device auto --hidden-sizes 128,128 --value-loss huber --huber-delta 1.0 --value-loss-weight 0.5 --val-split 0.1 --grad-clip 1.0 --save-top-k 3
python ml/alphazero_lite/export_artifact.py \
  --checkpoint /tmp/azlite/checkpoint.npz \
  --out-dir storage/ai/alphazero_lite/versions/azlite-local-001 \
  --version azlite-local-001
```

Recommended stronger local run (still CPU-friendly):

```bash
.venv/bin/python ml/alphazero_lite/generate_bootstrap_dataset.py \
  --out /tmp/azlite/self_play_stronger.jsonl \
  --games 2000 \
  --simulations 5000 \
  --seed 42 \
  --max-positions-per-game 24 \
  --workers 6 \
  --tree-reuse-enabled
python ml/alphazero_lite/train.py --data /tmp/azlite/self_play_stronger.jsonl --out /tmp/azlite/checkpoint_stronger.npz --epochs 40 --batch-size 512 --device auto --hidden-sizes 128,128 --value-loss huber --huber-delta 1.0 --value-loss-weight 0.5 --val-split 0.1 --grad-clip 1.0 --save-top-k 3
```

Large batch (chunked, resumable, parallel workers):

```bash
.venv/bin/python ml/alphazero_lite/generate_bootstrap_dataset.py \
  --out /tmp/azlite/mega5000x10000.jsonl \
  --games 5000 \
  --simulations 10000 \
  --seed 42 \
  --max-positions-per-game 24 \
  --workers 6 \
  --tree-reuse-enabled
```

To make an exported checkpoint active in Rails before promotion-gate reporting:

```bash
mkdir -p model-artifact/current
cp storage/ai/alphazero_lite/versions/azlite-local-001/weights.json model-artifact/current/weights.json
cp storage/ai/alphazero_lite/versions/azlite-local-001/metadata.json model-artifact/current/metadata.json
```

`arena_report.json` is not written by `export_artifact.py`. Copy or promote that report only after `script/ai/local_promotion_gate` or the RunPod wrapper produces it.

## Output contract

The export script writes:

- `model.npz`: MLP weights and biases (`float32`)
- `weights.json`: JSON-serialized model weights for Ruby evaluator inference
- `metadata.json`: contract metadata compatible with Rails loader

Important metadata fields:

- `schema_version = "azlite_model_v1"`
- `input_encoding = "kalah_v3"`
- `feature_count = 27`
- `policy_size = 6`
- `architecture.policy_size = 6`

## RunPod notes

Use the dedicated wrapper when you want remote compute to handle the heavier pipeline run and promotion gate.

```bash
script/ai/runpod_training_experiment \
  --config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_local.json \
  --results-path storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap \
  --local-results-path /tmp/runpod-results
```

The wrapper:

- bundles `ml/alphazero_lite`, `script/ai`, the selected config, and `model-artifact/current`
- runs `pipeline.py` remotely
- runs `script/ai/local_promotion_gate` remotely against the produced candidate
- downloads the remote results directory back under `--local-results-path` using the remote results directory name as the leaf folder
- exits `0` when remote execution completed and results were downloaded, even if the candidate failed the promotion gate
- uses the downloaded `local_promotion_gate.json` to report experiment success or failure

By default it uses the `cpu3c-16-32` pod profile so heavier training and evaluation runs stop monopolizing the local laptop while still leaving room to scale up later if needed.

### Superhuman quick commands

Launch a superhuman training/evaluation run on RunPod with the lossless gate
(400 arena games, 0 losses required):

```bash
script/ai/runpod_superhuman_experiment \
  --config-path ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase1.json \
  --phase2-config ml/alphazero_lite/configs/aggressive_v3_clone_extend_phase2.json
```

After results download and gate pass, publish the candidate into the runtime
`current` artifact path:

```bash
script/ai/promote_superhuman_candidate \
  tmp/runpod_results/aggressive-v3-clone-extend-iter2/aggressive-v3-clone-extend-iter2
```

Deploy and rollback happen from the main application repository using that repository's standard deploy flow.

## Local promotion gate

After exporting a local candidate artifact, run the local promotion gate before any stricter RunPod validation:

```bash
script/ai/local_promotion_gate \
  --candidate-path storage/ai/alphazero_lite/versions/azlite-local-001 \
  --out /tmp/azlite/local_promotion_report.json
```

This runs three local checks using the existing evaluators, writes sibling subreports next to the summary JSON, and only passes when the candidate clears the arena threshold and matches or beats the current model against `MCTS1200`.

For the `v2` lane, keep the same screening flow. First use a dry run to confirm the planned iteration path/name for `aggressive_v2`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v2.yaml \
  --dry-run
```

That dry run only confirms the planned iteration path/name. It does not produce a screenable candidate. The real pipeline must run through export so `storage/ai/alphazero_lite/versions/aggressive-v2-iter1` contains the artifact files before `local_promotion_gate` can evaluate it.

Then screen the exported `v2` iteration directory directly with no extra encoding or model-family flags:

```bash
script/ai/local_promotion_gate \
  --candidate-path storage/ai/alphazero_lite/versions/aggressive-v2-iter1 \
  --out /tmp/azlite/aggressive-v2-iter1-local-promotion.json
```

`script/ai/local_promotion_gate` already accepts generic candidate and current artifact paths, so `aggressive-v2-iter1` uses the same contract as the existing lane.

For the search-quality ablation lane, use the dedicated local config so the pipeline renders the upgraded search stack while keeping the same artifact contract:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v2_search_quality_local_versions/aggressive-v2-search-quality-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v2_search_quality_local.json \
  --out /tmp/azlite/aggressive-v2-search-quality-local-iter1-local-promotion.json
```

That local ablation config keeps the `aggressive_v2` train/export shape, but the rendered search commands now carry `--fpu-mode parent_q`, `--reuse-subtree`, `--normalize-values`, `--root-policy-mode deterministic`, and `--tactical-root-bias 0.1` through the search-quality arena sweep, `mcts1200_baseline_report`, `current_mcts1200_baseline_report`, and benchmark reporting path.

For the first `kalah_v3` tactical-encoding lane, use the dedicated local config to keep the same `residual_v2` student family while switching self-play, training, and export to `kalah_v3`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_tactical_encoding_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_tactical_encoding_local_versions/aggressive-v3-tactical-encoding-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_tactical_encoding_local.json \
  --out /tmp/azlite/aggressive-v3-tactical-encoding-local-iter1-local-promotion.json
```

This local lane stays close to `aggressive_v2`: it keeps the existing local promotion gate and evaluation path, but the rendered self-play, train, and export commands now use `kalah_v3` with the same `residual_v2` train/export family.

For the first `kalah_v3` specialized-heads lane, use the dedicated local config to keep the same local `kalah_v3` pipeline shape while switching train and export to `residual_v3`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_specialized_heads_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_specialized_heads_local_versions/aggressive-v3-specialized-heads-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_specialized_heads_local.json \
  --out /tmp/azlite/aggressive-v3-specialized-heads-local-iter1-local-promotion.json
```

This local lane stays close to the existing `kalah_v3` tactical-encoding run, but the rendered train and export commands now use the `residual_v3` specialized-heads family while keeping `kalah_v3` fixed.

### Policy-target lane

For the sharpened policy-target follow-up lane, use the dedicated local config to keep the same `kalah_v3` and `residual_v3` setup while changing only the policy-target path. Run the pipeline config below before screening the exported iteration directory:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_policy_target_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_policy_target_local_versions/aggressive-v3-policy-target-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_policy_target_local.json \
  --out /tmp/azlite/aggressive-v3-policy-target-local-iter1-local-promotion.json
```

This local lane keeps the existing `kalah_v3` self-play shape and `residual_v3` train/export family, but the rendered self-play and train commands now add `--policy-target-mode sharpened`, and the bootstrap dataset command sets the same sharpened policy-target mode before screening `aggressive-v3-policy-target-local-iter1` through the usual local promotion gate.

### Widened specialized-heads lane

For the widened `residual_v3` follow-up lane, use the dedicated local config to keep the same `kalah_v3` specialized-heads flow while widening the train hidden sizes. Run `.venv/bin/python ml/alphazero_lite/pipeline.py --config ml/alphazero_lite/configs/aggressive_v3_specialized_heads_wide_local.json` before screening the exported iteration directory:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_specialized_heads_wide_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_specialized_heads_wide_local_versions/aggressive-v3-specialized-heads-wide-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_specialized_heads_wide_local.json \
  --out /tmp/azlite/aggressive-v3-specialized-heads-wide-local-iter1-local-promotion.json
```

This widened local lane keeps the existing `kalah_v3` self-play and local promotion screen, but the rendered train step uses the wider `residual_v3` hidden sizes before exporting `aggressive-v3-specialized-heads-wide-local-iter1` for the same gate.

### Value-target lane

For the sharpened value-target follow-up lane, use the dedicated local config to keep the same `kalah_v3`, `residual_v3`, and sharpened policy-target setup while changing only the value-target path:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_value_target_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_value_target_local_versions/aggressive-v3-value-target-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_value_target_local.json \
  --out /tmp/azlite/aggressive-v3-value-target-local-iter1-local-promotion.json
```

This local lane keeps the existing `kalah_v3` self-play shape, `residual_v3` train/export family, and `--policy-target-mode sharpened` contract, but the rendered self-play, train, and bootstrap commands now also add `--value-target-mode sharpened` before screening `aggressive-v3-value-target-local-iter1` through the usual local promotion gate.

### Aligned sharpened value-target lane

For the aligned rerun of the sharpened value-target experiment, use the dedicated local config to keep the same `kalah_v3`, `residual_v3`, `--policy-target-mode sharpened`, and `--value-target-mode sharpened` contract while changing only the lane identity and output path:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_value_target_aligned_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_value_target_aligned_local_versions/aggressive-v3-value-target-aligned-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_value_target_aligned_local.json \
  --out /tmp/azlite/aggressive-v3-value-target-aligned-local-iter1-local-promotion.json
```

This aligned rerun keeps the existing sharpened value-target lane shape intact so the experiment measures semantic alignment rather than a new target mode before screening `aggressive-v3-value-target-aligned-local-iter1` through the usual local promotion gate.

### Capacity lane

For the first pure capacity follow-up lane, use the dedicated local config to keep the same `kalah_v3`, `residual_v3`, `--policy-target-mode sharpened`, and `--value-target-mode sharpened` setup while changing only the train architecture from `64,2` to `96,3`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_capacity_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_capacity_local_versions/aggressive-v3-capacity-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_capacity_local.json \
  --out /tmp/azlite/aggressive-v3-capacity-local-iter1-local-promotion.json
```

This capacity lane keeps the existing `kalah_v3` arena gate and sharpened target contract fixed, runs from a cold start, and uses the historical Ruby-default `MCTS1200` baseline policy (`--root-policy-mode visit_count`, `--tactical-root-bias 0.0`) through the Python baseline script, so the experiment measures extra model capacity without changing encoding or promotion gates.

### Larger capacity lane

For the next pure capacity follow-up, use the dedicated local config to keep the same cold-start `kalah_v3`, `residual_v3`, sharpened policy target, and sharpened value-target flow while increasing train architecture capacity from `96,3` to `128,3`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_capacity_large_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_capacity_large_local_versions/aggressive-v3-capacity-large-local-iter1 \
  --current-path /tmp/azlite_v3_value_target_local_versions/aggressive-v3-value-target-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_capacity_large_local.json \
  --out /tmp/azlite/aggressive-v3-capacity-large-local-iter1-local-promotion.json
```

This larger capacity lane keeps the same arena gate, encoding, and historical `MCTS1200` baseline contract fixed, but pushes the `residual_v3` student to `128,3` so the follow-up can test whether one larger step in capacity breaks the current `MCTS1200` plateau.

### Stronger bootstrap teacher lane

For the next teacher-strength follow-up, use the dedicated local config to keep the same cold-start `kalah_v3`, `residual_v3`, sharpened targets, and `96,3` train architecture while increasing only the bootstrap teacher from `1200` to `2400` simulations:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_stronger_bootstrap_local_versions/aggressive-v3-stronger-bootstrap-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_local.json \
  --out /tmp/azlite/aggressive-v3-stronger-bootstrap-local-iter1-local-promotion.json
```

This lane keeps the current `96,3` capacity setup fixed, but raises the bootstrap teacher to `2400` so the experiment tests whether stronger offline supervision improves `MCTS1200` results without changing self-play, encoding, or promotion gates.

### Stronger bootstrap confirmation lane

For the first robustness rerun, keep the stronger-bootstrap recipe fixed and change only the seed schedule:

- `seed = 84`
- `--seed-sweep 81,82,83`

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_confirm_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_stronger_bootstrap_confirm_local_versions/aggressive-v3-stronger-bootstrap-confirm-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_confirm_local.json \
  --out /tmp/azlite/aggressive-v3-stronger-bootstrap-confirm-local-iter1-local-promotion.json
```

This confirmation lane keeps the stronger bootstrap teacher and `96,3` architecture unchanged, and only reruns the experiment with a different seed schedule so we can measure robustness rather than stack another recipe change.

### Stronger bootstrap confirmation sweep

For the next two robustness reruns, keep the stronger bootstrap recipe fixed and vary only the seed schedule:

- B lane: `seed = 126`, `--seed-sweep 121,122,123`
- C lane: `seed = 168`, `--seed-sweep 161,162,163`

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_confirm_b_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_stronger_bootstrap_confirm_b_local_versions/aggressive-v3-stronger-bootstrap-confirm-b-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_confirm_b_local.json \
  --out /tmp/azlite/aggressive-v3-stronger-bootstrap-confirm-b-local-iter1-local-promotion.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_confirm_c_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_stronger_bootstrap_confirm_c_local_versions/aggressive-v3-stronger-bootstrap-confirm-c-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_confirm_c_local.json \
  --out /tmp/azlite/aggressive-v3-stronger-bootstrap-confirm-c-local-iter1-local-promotion.json
```

These two lanes keep the stronger bootstrap teacher and `96,3` architecture unchanged, and extend the confirmation sweep so we can judge recipe robustness by repeatability across multiple seed schedules rather than a single rerun.

### Stronger bootstrap more-data lane

For the first direct robustness follow-up, keep the stronger bootstrap teacher at `2400` and the `96,3` architecture fixed, but increase bootstrap games from `600` to `900`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_more_data_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_stronger_bootstrap_more_data_local_versions/aggressive-v3-stronger-bootstrap-more-data-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_more_data_local.json \
  --out /tmp/azlite/aggressive-v3-stronger-bootstrap-more-data-local-iter1-local-promotion.json
```

This lane keeps the stronger bootstrap recipe intact, but raises bootstrap volume to `900` games so the next experiment tests whether a modest increase in teacher-generated data reduces variance without changing search, encoding, or model size.

### Hybrid value-target lane

For the dedicated hybrid value-target follow-up lane, use the dedicated local config to keep the same `kalah_v3`, `residual_v3`, and `--policy-target-mode sharpened` setup while changing only the value-target path:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_hybrid_value_target_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_hybrid_value_target_local_versions/aggressive-v3-hybrid-value-target-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_hybrid_value_target_local.json \
  --out /tmp/azlite/aggressive-v3-hybrid-value-target-local-iter1-local-promotion.json
```

This local lane keeps the existing `kalah_v3` self-play shape, `residual_v3` train/export family, and `--policy-target-mode sharpened` contract, but the rendered self-play, train, and bootstrap commands now use `--value-target-mode hybrid` before screening `aggressive-v3-hybrid-value-target-local-iter1` through the usual local promotion gate.

### Phase-aware value-target lane

For the phase-aware value-target follow-up lane, use the dedicated local config to keep the same `kalah_v3`, `residual_v3`, and `--policy-target-mode sharpened` setup while changing only the value-target path:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_phase_aware_value_target_local.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_v3_phase_aware_value_target_local_versions/aggressive-v3-phase-aware-value-target-local-iter1 \
  --config-path ml/alphazero_lite/configs/aggressive_v3_phase_aware_value_target_local.json \
  --out /tmp/azlite/aggressive-v3-phase-aware-value-target-local-iter1-local-promotion.json
```

This local lane keeps the existing `kalah_v3` self-play shape, `residual_v3` train/export family, and `--policy-target-mode sharpened` contract, but the rendered self-play, train, and bootstrap commands now use `--value-target-mode phase_aware_sharpened` before screening `aggressive-v3-phase-aware-value-target-local-iter1` through the usual local promotion gate.
