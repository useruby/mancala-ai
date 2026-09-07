# Scale-Up: 26k-State From-Scratch Exact Lane at 46 Epochs

Classification: `exact_scale26k_gate_not_met`.

Scale test recommended by the attribution experiment: does the
from-scratch exact-lane gate pass rate rise with more states and a full
schedule? No model was promoted; `model-artifact/current` is untouched.

## Cohort

- 26,000 new states frozen with the same pipeline (seed 20260907, 6,500
  per active-stone bucket), disjoint from the v1 10k cohort by canonical
  key (0 overlap), plus identical leakage exclusions (96 feasibility +
  224 forensic + 384 opening-suite keys). 840 games scanned, 38,353
  positions visited.
- Native hybrid labeling: 25,999/26,000 solved (99.996%); one explicit
  timeout retained (`exact-prod-frozen-33-40-04475`, 33–40 bucket),
  no fallback. Deterministic 80/20 split: 20,799 train / 5,200 holdout,
  canonical overlap 0.
- Source SHA `85655f8a0c81f757d67d0b5aca24cd5c62c6cdbbf830e79d571642a0e12b16b0`;
  train SHA `0d4c768795328eca09ed8b5dfc3d03c4020f18f39932f3c73cd81a31ddbbc898`.

## Training (from scratch, residual_v3 96×3, 46 epochs, lr 1e-3, cosine)

| seed | holdout top-in-set | value MAE | policy CE |
| ---- | ------------------ | --------- | --------- |
| 42   | 56.88%             | 0.481     | 1.382     |
| 43   | 54.87%             | 0.478     | 1.388     |
| 44   | 58.31%             | 0.476     | 1.373     |

Mean 56.69% / 0.478 — strictly better than the 8k/16-epoch v2 point
(53.25% / 0.599) on both heads at all seeds. Holdout improves with
scale; seed spread narrows to ±1.7pp.

## Gate (medium_eval, 512 games/budget)

| candidate | 384:256 | 768:768 | 1200:1200 | 1200:256 | gate |
| --------- | ------- | ------- | --------- | -------- | ---- |
| seed42    | +0.219  | +0.297  | +0.102    | −0.086    | fail |
| seed43    | +0.195  | +0.117  | −0.043    | +0.227    | fail |
| seed44    | −0.066  | +0.051  | −0.098    | −0.320    | fail |

Pass rate 0/3 — down from 1/3 at 8k states. Holdout gains did not
transfer to the gate: seed44 has the best holdout (58.31%) and the
worst gate (all four budgets red); the 1200:256 budget fails two of
three lanes. The gate failure mode is not cured by 2.6× data and a
full schedule.

## Interpretation

The variance question is now answered against scaling this recipe:
pass rate fell 1/3 → 0/3 while holdout improved, so frozen-holdout
fidelity and opening-suite strength are decoupled here. The exact lane
learns the teacher distribution faithfully (holdout ↑ with data) but
that distribution does not beat the incumbent where the gate measures.
Plausible causes: (1) the 8–40-stone cohort is off the opening
distribution the suite plays; (2) ±1 value targets lose the margin
ordering that PUCT search exploits at high budgets; (3) 46 epochs still
underfit value (MAE ~0.48). The exact-label program should pivot from
"more of the same states" to one of these mechanisms.

## Artifacts

- `docs/data/alphazero-lite-exact-scale26k-gate.json`
- `docs/data/alphazero-lite-exact-scale26k-training-summary.json`
- `docs/data/alphazero-lite-exact-scale26k-production-summary.json`
- `docs/data/alphazero-lite-exact-scale26k-source-cohort.json`
- Full rows/checkpoints/benchmark (not committed):
  `/tmp/azlite_exact_scale/`, `/tmp/azlite_exact_scale_train/`,
  `/tmp/azlite_exact_scale_bench/`

## Reproduction

```bash
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_training_ablation \
  --exact-train /tmp/azlite_exact_scale/train.jsonl \
  --mcts-train /tmp/azlite_exact_scale/train.jsonl \
  --exact-holdout /tmp/azlite_exact_scale/holdout.jsonl \
  --source-states /tmp/azlite_exact_scale/source_states.jsonl \
  --workdir /tmp/azlite_exact_scale_train \
  --out-summary /tmp/azlite_exact_scale_train/summary.json \
  --lanes exact --seeds 42,43,44 \
  --model-type residual_v3 --hidden-sizes 96,3 --epochs 46 \
  --current model-artifact/current --skip-arena
python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/azlite_exact_scale_bench \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates <seed42,seed43,seed44 artifacts, comma-separated> \
  --budget-pairs "384:256,768:768,1200:1200,1200:256" \
  --games-per-opening 2 --seed 42 --workers 8
```

## Recommended next experiment

Stop scaling states. Run a mechanism probe on the existing 26k/10k
artifacts: (a) margin-ordered value targets (normalized exact margins
instead of ±1 signs) to test the value-ordering hypothesis without new
labels; (b) an opening-distribution exact cohort (low-stone states from
real opening prefixes) to test the distribution-mismatch hypothesis.
Whichever probe moves the 1200:256 budget determines the next data
investment.
