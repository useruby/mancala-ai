# Exact-vs-MCTS Controlled Training Ablation

Classification: `exact_teacher_training_ablation_complete`.

Follow-up to the production-ready exact-teacher experiment
(`docs/native-hybrid-exact-teacher-production-results.md`). No model was
promoted; both lanes are measurement-only challengers.

## Design

- Identical 8,000 frozen train states in both lanes (byte-identical
  `source_id` order, state vectors, and canonical keys; verified
  row-for-row at runner start).
- `exact` lane: native-hybrid exact labels (uniform optimal policy,
  sign-of-margin value). Train SHA
  `bd383b9f281066f3711e066a64e05cf3311548d9d505dc45391502252509d978`.
- `mcts` lane: classic-MCTS 1200-simulation relabels of the same states
  (default policy target, temperature 1.0, deterministic per-state seeds).
  Train SHA `014ac99af8df40815ba3942658f4f57902952fa184e169923efb2fc7e00a0a5f`.
- Fixed across lanes: `mlp_v1` (64,64), kalah_v3, 8 epochs, batch 512,
  lr 1e-3, seed 42, value-loss weight 0.5, huber, val-split 0.1.
- Frozen 2,000-state exact holdout (SHA
  `a53a304c66cf84fc461df3b6248c328b521b776e6296fcfad13690958b3e0d16`),
  disjoint from train by canonical key.
- Sealed arena: medium_eval suite (SHA
  `57ea2f461b0cfb63be0b0fed9e3f818f47cd775b5970a105e61524c000c57e04`),
  context 384:256, 512 games per leg, shared current control
  (`storage/ai/alphazero_lite/current`).

## Holdout results (frozen exact labels as ground truth)

| lane  | top-in-optimal-set | single-opt agreement | value MAE | policy CE |
| ----- | ------------------ | -------------------- | --------- | --------- |
| exact | 975/2000 (48.75%)  | 40.41% (n=1381)      | 0.8316    | 1.4178    |
| mcts  | 972/2000 (48.60%)  | 40.70% (n=1381)      | 0.7712    | 1.4358    |

Contrast (exact minus mcts): top-in-set +0.15pp, value MAE +0.060
(mcts better). By stone bucket the lanes track within ~1pp on policy and
mcts leads value MAE in every bucket. Train losses: exact policy 1.2025 /
value 0.4190; mcts policy 1.2751 / value 0.1337 — the mcts lane fits its
smooth value targets more easily, as expected for ±1 vs continuous targets.

## Arena results (sealed, paired)

- Shared-current legs: exact-challenger 62W/4D/446L (12.1%); mcts
  challenger 0W/0D/512L (0.0%). Neither lane is competitive with the
  production incumbent — expected for 8-epoch `mlp_v1` models trained on
  8k mid/late-game exact states.
- Paired exact-minus-mcts candidate effect: **+0.125**, 95% opening
  bootstrap CI **[0.090, 0.162]** (128 openings, excludes zero).
- Head-to-head exact-challenger vs mcts-challenger as current: exact
  scores **0.912** over 512 games (446W/42D/24L).
- Seat split of the paired effect: P0 +0.250, P1 +0.000 — the entire
  advantage accrues as the first player.

## Repeatability

- Re-ran both lanes end-to-end (`--skip-arena`): checkpoint SHAs and all
  holdout metrics byte-identical across runs
  (exact `f68a5cb7…`, mcts `74f4f22c…`).
- All 8,000 + 8,000 + 2,000 rows pass `train.py` validation.

## Interpretation

Exact labels replace noisy MCTS labels *measurably but narrowly*: a
statistically significant paired arena advantage (+0.125, CI excludes
zero) and a decisive head-to-head (0.912), but essentially no holdout
policy edge (+0.15pp) and worse holdout value MAE (+0.060) at this
low-capacity operating point. The value-MAE deficit is structural: ±1
targets are harder to fit in 8 epochs than the smooth MCTS values, and
the single shared seed means internal val-splits coincide — no seed
variance is measured.

## Artifacts

- `docs/data/alphazero-lite-exact-teacher-training-ablation-summary.json`
- `docs/data/alphazero-lite-exact-ablation-mcts-relabel-summary.json`
- Checkpoints/records (not committed):
  `/tmp/azlite_exact_ablation/full/{exact,mcts}/checkpoint.npz`,
  `/tmp/azlite_exact_ablation/full/arena/*/records.json`

## Reproduction

```bash
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_ablation_mcts_relabel \
  --source-states /tmp/azlite_exact_teacher_production/source_states.jsonl \
  --exact-train /tmp/azlite_exact_teacher_production/train.jsonl \
  --out-train /tmp/azlite_exact_ablation/mcts_train.jsonl \
  --out-summary /tmp/azlite_exact_ablation/mcts_relabel.json
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_training_ablation \
  --exact-train /tmp/azlite_exact_teacher_production/train.jsonl \
  --mcts-train /tmp/azlite_exact_ablation/mcts_train.jsonl \
  --exact-holdout /tmp/azlite_exact_teacher_production/holdout.jsonl \
  --source-states /tmp/azlite_exact_teacher_production/source_states.jsonl \
  --workdir /tmp/azlite_exact_ablation/full \
  --out-summary /tmp/azlite_exact_ablation/ablation_summary.json
```

## Recommended next experiment

Widen the ablation before any promotion decision: (1) repeat at 2-3
seeds to measure lane variance (current n=1 seed each); (2) add a
higher-capacity/production-like architecture and longer schedule, since
both lanes underfit value here (MAE ~0.8); (3) test whether blending
exact policy targets with MCTS-smooth values closes the value-MAE gap
while keeping the policy signal.
