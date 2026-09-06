# Exact-Lane Promotion-Gated Replication (Full Schedule)

Classification: `exact_promotion_replication_gate_not_met`.

Promotion-gated replication of the exact lane at a full production
schedule. Two seeds, finetuned from `model-artifact/current`
(`residual_v3` 96×3, kalah_v3), 46 epochs, batch 512, lr 1e-5,
value-loss weight 0.3, huber, val-split 0.1, `lr-scheduler none`,
`trainable-scope all`. No model was promoted; `model-artifact/current`
is untouched.

## Training

- Data: frozen 8,000 exact train rows (SHA
  `bd383b9f281066f3711e066a64e05cf3311548d9d505dc45391502252509d978`).
- Init: current weights materialized via
  `pipeline.materialize_weights_json_checkpoint` and loaded with
  `train.load_checkpoint_into_model` (verified loadable before launch).
- Holdout (frozen 2,000 exact states): seed42 51.50% top-in-set /
  0.711 MAE; seed43 51.40% / 0.709. Consistent across seeds; below the
  from-scratch residual_v3 v2 point (53.25% / 0.599) — finetuning a
  converged incumbent on 8k off-distribution states moves policy only
  modestly.

## Sealed benchmark (medium_eval, 128 openings × 2 games × 2 seats)

Production gate rule (`candidate_gate_summary`): primary 384:256 ds > 0,
768:768 ds ≥ 0, 1200:1200 ds ≥ 0, 1200:256 ds ≥ 0.

| seed   | 384:256 (std) | 768:768 | 1200:1200 | 1200:256 | gate        |
| ------ | ------------- | ------- | --------- | -------- | ----------- |
| seed42 | +0.277         | −0.023  | +0.117    | −0.133    | FAIL (2 budgets) |
| seed43 | −0.113         | +0.281  | +0.125    | −0.211    | FAIL (2 budgets) |

Scores: seed42 standard starts_0 1.0000 / starts_1 0.7227; seed43
standard 0.7031 / 0.8164. Neither seed passes the gate; failures are
complementary (each seed fails the two budgets the other passes),
indicating seed noise rather than a systematic exact-label defect.

## Seat-aware gate attempt (informative failure)

`script/ai/seat_aware_promotion_gate` was attempted with null runtime
profiles but its arena backend plays mirror games from the standard
initial position with no opening suite (256/256 W/L split by seat —
first-player win in all 512 games), so it cannot discriminate
challengers. Its `runtime_profile_gate_invalid` output is recorded only
to document that it is the wrong instrument here; the opening-suite
benchmark above is the evaluative leg. No promotion decision is taken
from the seat-aware gate.

## Artifacts

- `docs/data/alphazero-lite-exact-promotion-replication-summary.json`
- `docs/data/alphazero-lite-exact-promotion-benchmark-report.json`
- Checkpoints/benchmark workdir (not committed):
  `/tmp/azlite_exact_promotion/lanes/`,
  `/tmp/azlite_exact_promotion/bench/`

## Reproduction

```bash
PYTHONPATH=. python - <<'EOF'
from pathlib import Path
from ml.alphazero_lite.pipeline import materialize_weights_json_checkpoint
materialize_weights_json_checkpoint(
    weights_path=Path('model-artifact/current/weights.json'),
    out_path=Path('/tmp/azlite_exact_promotion/inputs/current_from_weights_json.npz'),
)
EOF
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_training_ablation \
  --exact-train /tmp/azlite_exact_teacher_production/train.jsonl \
  --mcts-train /tmp/azlite_exact_ablation/mcts_train.jsonl \
  --exact-holdout /tmp/azlite_exact_teacher_production/holdout.jsonl \
  --source-states /tmp/azlite_exact_teacher_production/source_states.jsonl \
  --workdir /tmp/azlite_exact_promotion/lanes \
  --out-summary /tmp/azlite_exact_promotion/promotion_summary.json \
  --lanes exact --seeds 42,43 --arena-seeds 1 \
  --model-type residual_v3 --hidden-sizes 96,3 --epochs 46 \
  --batch-size 512 --lr 1e-5 --value-loss-weight 0.3 \
  --lr-scheduler none --trainable-scope all \
  --init-checkpoint /tmp/azlite_exact_promotion/inputs/current_from_weights_json.npz \
  --current model-artifact/current --arena-workers 8
python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/azlite_exact_promotion/bench \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates /tmp/azlite_exact_promotion/lanes/exact/seed42/artifact,/tmp/azlite_exact_promotion/lanes/exact/seed43/artifact \
  --budget-pairs "384:256,768:768,1200:1200,1200:256" \
  --games-per-opening 2 --seed 42 --workers 8
```

## Verdict and recommended next experiment

`exact_promotion_replication_gate_not_met`: do not promote. The exact
lane at full schedule is measurable on the frozen holdout but fails the
production multi-budget gate at both seeds. Recommended follow-up is a
diagnostic, not a larger train: (1) score the v2 from-scratch residual
checkpoints on the same four-budget benchmark to separate "exact labels"
from "finetune dynamics" as the failure source; (2) if from-scratch
passes where finetune fails, retry finetuning with the production
policy-head-only scope and 1–2 epoch schedule instead of full-trunk
46-epoch finetuning.
