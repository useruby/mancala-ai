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
