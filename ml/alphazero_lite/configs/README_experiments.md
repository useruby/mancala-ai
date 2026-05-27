# Replay Bootstrap Weight Ablation

Hypothesis: changing only the final bootstrap replay weight in the AlphaZero-lite v3 clone/extend phase-1 lane will show whether the candidate benefits more from stronger bootstrap supervision or from keeping the replay mix closer to fresh self-play.

Changed variable: the `train` step `--replay-weights` final bootstrap weight only.

- `exp_v3_replay_bootstrap_w1.json`: `"{replay_weights},1"`
- `exp_v3_replay_bootstrap_w2.json`: `"{replay_weights},2"`
- `exp_v3_replay_bootstrap_w8.json`: `"{replay_weights},8"`

Run each config with `pipeline.py`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w1.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w2.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w8.json
```

Run `local_promotion_gate` on each produced candidate:

```bash
script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_exp_v3_replay_bootstrap_w1_versions/exp-v3-replay-bootstrap-w1-iter1 \
  --config-path ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w1.json \
  --out /tmp/azlite/exp-v3-replay-bootstrap-w1-local-promotion.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_exp_v3_replay_bootstrap_w2_versions/exp-v3-replay-bootstrap-w2-iter1 \
  --config-path ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w2.json \
  --out /tmp/azlite/exp-v3-replay-bootstrap-w2-local-promotion.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_exp_v3_replay_bootstrap_w8_versions/exp-v3-replay-bootstrap-w8-iter1 \
  --config-path ml/alphazero_lite/configs/exp_v3_replay_bootstrap_w8.json \
  --out /tmp/azlite/exp-v3-replay-bootstrap-w8-local-promotion.json
```

Compare these metrics across runs:

- `arena_report`
- `mcts1200_report`
- `benchmark_report`
- `perspective_audit`
- `best_val_loss`

# Self-Play Search-Depth Ablation

Hypothesis: changing only the self-play search budget in the AlphaZero-lite v3 clone/extend phase-1 lane will isolate search target quality versus throughput.

Changed variables:

- `exp_v3_selfplay_sims384.json`: self-play `--simulations 384` at the same `1600` games
- `exp_v3_selfplay_sims768_small.json`: self-play `--simulations 768` with `--games 800` for cost control

Interpretation note: the `384` run isolates deeper self-play search targets at fixed data volume, while the `768-small` run trades throughput for search depth and is not directly comparable on data volume.

Run each config with `pipeline.py`:

```bash
.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_selfplay_sims384.json

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/exp_v3_selfplay_sims768_small.json
```

Run `local_promotion_gate` on each produced candidate:

```bash
script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_exp_v3_selfplay_sims384_versions/exp-v3-selfplay-sims384-iter1 \
  --config-path ml/alphazero_lite/configs/exp_v3_selfplay_sims384.json \
  --out /tmp/azlite/exp-v3-selfplay-sims384-local-promotion.json

script/ai/local_promotion_gate \
  --candidate-path /tmp/azlite_exp_v3_selfplay_sims768_small_versions/exp-v3-selfplay-sims768-small-iter1 \
  --config-path ml/alphazero_lite/configs/exp_v3_selfplay_sims768_small.json \
  --out /tmp/azlite/exp-v3-selfplay-sims768-small-local-promotion.json
```

Optional `arena.py` baseline-style confirm with the usual lane-B evaluation budget:

```bash
.venv/bin/python ml/alphazero_lite/arena.py \
  --challenger /tmp/azlite_exp_v3_selfplay_sims384_versions/exp-v3-selfplay-sims384-iter1 \
  --current storage/ai/alphazero_lite/current \
  --games 120 \
  --challenger-simulations 640 \
  --current-simulations 256 \
  --workers 6 \
  --min-score 0.55 \
  --out /tmp/azlite/exp-v3-selfplay-sims384-arena-confirm.json

.venv/bin/python ml/alphazero_lite/arena.py \
  --challenger /tmp/azlite_exp_v3_selfplay_sims768_small_versions/exp-v3-selfplay-sims768-small-iter1 \
  --current storage/ai/alphazero_lite/current \
  --games 120 \
  --challenger-simulations 640 \
  --current-simulations 256 \
  --workers 6 \
  --min-score 0.55 \
  --out /tmp/azlite/exp-v3-selfplay-sims768-small-arena-confirm.json
```

Validate each config before running:

```bash
.venv/bin/python -m json.tool ml/alphazero_lite/configs/exp_v3_selfplay_sims384.json
.venv/bin/python -m json.tool ml/alphazero_lite/configs/exp_v3_selfplay_sims768_small.json
```
