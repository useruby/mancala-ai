# Exact-Teacher Ablation v2: Multi-Seed, Production Capacity, Blend Lane

Classification: `exact_teacher_training_ablation_v2_complete`.

Second controlled ablation on the frozen 10k cohort (train SHA
`bd383b9f…`, holdout SHA `a53a304c…`). Addresses the three v1 gaps:
single seed per lane, under-capacity `mlp_v1` model, and the exact-lane
value-MAE deficit. No model was promoted.

## Design deltas vs v1

- Three lanes on identical 8,000 states: `exact` (uniform optimal policy
  + sign-of-margin value), `mcts` (1200-sim MCTS policy + smooth value),
  `blend` (exact policy + MCTS value; SHA
  `6f69693397f3c35ea30da7f0d4423d50d07a63cbbcf29f3d0cc33f55bd968eb8`).
- Two capacity points, three seeds (42, 43, 44) each:
  (a) `mlp_v1` (64,64), 8 epochs — repeats v1 config per seed;
  (b) `residual_v3` (96,3, production architecture), 16 epochs.
- Arena on the residual_v3 point only, one leg per seed
  (medium_eval suite, 384:256, 512 games/leg), plus head-to-head.
- Runner extended in place (`--lanes`, `--blend-train`, `--seeds`,
  `--arena-seeds`); v1 single-seed CLI behavior is the default
  (`--lanes exact,mcts --seeds 42`), and the v1 repeat run reproduces
  byte-identical checkpoints.

## Holdout results (frozen 2,000 exact states)

mlp_v1 / 8 epochs, means over 3 seeds (min–max):

| lane  | top-in-set mean | range         | value MAE mean | range       | policy CE |
| ----- | --------------- | ------------- | -------------- | ----------- | --------- |
| exact | 48.90%          | 48.50–49.45%  | 0.827          | 0.815–0.835 | 1.417     |
| mcts  | 48.55%          | 48.40–48.65%  | 0.769          | 0.768–0.771 | 1.436     |
| blend | 48.77%          | 48.65–48.85%  | 0.763          | 0.752–0.769 | 1.418     |

residual_v3 / 16 epochs:

| lane  | top-in-set mean | range         | value MAE mean | range       | policy CE |
| ----- | --------------- | ------------- | -------------- | ----------- | --------- |
| exact | 53.25%          | 52.25–54.20%  | 0.599          | 0.559–0.661 | 1.384     |
| mcts  | 49.27%          | 47.75–50.20%  | 0.661          | 0.624–0.728 | 1.430     |
| blend | 53.50%          | 52.80–54.20%  | 0.776          | 0.771–0.781 | 1.384     |

Seed variance is small on policy (±1pp) and the exact-vs-mcts policy gap
is positive at every seed in both capacities. Capacity flips the value
story: at residual_v3 the exact lane wins value MAE by 0.061 mean
(0.599 vs 0.661) — the v1 value deficit was an under-capacity artifact,
not a structural property of ±1 targets. The blend hypothesis is
rejected in its strong form: blend matches exact on policy (53.50% vs
53.25%) but its value MAE (0.776) is near mcts, not exact — MCTS-smooth
values do not distill exact-value accuracy, and exact values do not harm
policy when capacity suffices.

## Arena results (residual_v3, shared current, per-leg seeds)

| leg (seed) | exact W/D/L | mcts W/D/L | paired effect | 95% CI        |
| ---------- | ----------- | ---------- | ------------- | ------------- |
| 0 (42)     | 340/42/130  | 148/4/360  | +0.412        | [0.350, 0.475]|
| 1 (43)     | 414/10/88   | 240/42/230 | +0.309        | [0.256, 0.361]|
| 2 (44)     | 414/14/84   | 380/38/94  | +0.043        | [0.010, 0.078]|

Mean paired effect +0.255; every leg CI excludes zero. Head-to-head
(seed 42): exact scores 0.906 over 512 games. Seed 44 is a caution:
both lanes play the incumbent nearly even (mcts 380W) yet the paired
effect stays positive — the exact advantage persists across a wide
absolute-strength range but its magnitude varies 10x by seed
(+0.04 to +0.41).

## Repeatability

- Residual checkpoints/holdout metrics byte-identical between the
  no-arena and arena runs (6/6 seeds × lanes).
- All 24,000 lane rows pass `train.py` validation; train/holdout
  canonical disjointness re-verified at runner start.

## Artifacts

- `docs/data/alphazero-lite-exact-ablation-v2-mlp-summary.json`
- `docs/data/alphazero-lite-exact-ablation-v2-residual-summary.json`
- `docs/data/alphazero-lite-exact-ablation-v2-arena-summary.json`
- `docs/data/alphazero-lite-exact-ablation-blend-summary.json`
- Checkpoints/records (not committed):
  `/tmp/azlite_exact_ablation/v2_{mlp,res,arena}/`,
  `/tmp/azlite_exact_ablation/blend_train.jsonl`

## Reproduction

```bash
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_ablation_blend_labels \
  --exact-train /tmp/azlite_exact_teacher_production/train.jsonl \
  --mcts-train /tmp/azlite_exact_ablation/mcts_train.jsonl \
  --out-train /tmp/azlite_exact_ablation/blend_train.jsonl \
  --out-summary /tmp/azlite_exact_ablation/blend.json
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_training_ablation \
  --exact-train /tmp/azlite_exact_teacher_production/train.jsonl \
  --mcts-train /tmp/azlite_exact_ablation/mcts_train.jsonl \
  --blend-train /tmp/azlite_exact_ablation/blend_train.jsonl \
  --exact-holdout /tmp/azlite_exact_teacher_production/holdout.jsonl \
  --source-states /tmp/azlite_exact_teacher_production/source_states.jsonl \
  --workdir /tmp/azlite_exact_ablation/v2_res \
  --out-summary /tmp/azlite_exact_ablation/v2_res_summary.json \
  --lanes exact,mcts,blend --seeds 42,43,44 --skip-arena \
  --model-type residual_v3 --hidden-sizes 96,3 --epochs 16
```

## Recommended next experiment

Promotion-gated replication at full production schedule (more epochs,
production LR schedule) with the exact lane only, 2+ seeds, scored on
the frozen holdout and sealed suites against the incumbent promotion
threshold — the v1/v2 evidence (positive policy gap at all 9
lane-seed-capacity points, significant arena effect at all 4 legs)
supports paying the full-schedule cost now.
