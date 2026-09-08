# Distribution Probe (b): Opening-Distribution Exact Cohort

Classification: `exact_opening_distribution_probe_complete`.

Probe (b) from the scale-up report: is the gate failure a
distribution mismatch between the 8–40-stone midgame cohort and the
opening positions the suite plays? A fresh exact cohort was frozen from
plies 3–8, labeled with the same native teacher, trained with the same
recipe, and benchmarked on the same gate. No model was promoted.

## Cohort design (key finding first)

- Plies 1–2 have zero fresh states: all reachable ply 1–3 states already
  appear in the v1/v2 cohorts or the suite exclusions (measured: 100k
  games → 0 available at ply 1–2, 0 at ply 3 after exclusions).
- Ply 3–4 reachable space is nearly exhausted: only ~145 fresh states
  exist in 400k games (368k hits on the `opening_plies_1_8` diagnostic
  states, 63k on opening-suite states).
- Frozen cohort: 6,000 states, plies 3–8 (ply4: 113, ply5: 705, ply6:
  1496, ply7: 1794, ply8: 1892), 28–48 active stones, seed 20260908,
  disjoint from v1/v2/suites by canonical key (0 overlap everywhere).
  SHA `a977705eaccbbd6295e1e9d9d0eb268991ab05e461ebcb994f9f61f4e2913f7d`.
- The `opening_plies_1_8` forensic diagnostic bucket (48 states, plies
  1–8) stays excluded as evaluation; all other forensic buckets were
  admitted had they appeared (none did in this ply range).

## Labeling cost (the reason this probe is small)

- Opening states are 10–100× costlier to solve than midgame states:
  60 s timeouts routinely hit at 44–48 stones; a 120 s stratified probe
  solved 10/10 but slow states take 30–70 s each.
- Production: 8 parallel shards, 300 s timeout → 3,671 solved + 53
  explicit timeouts at last count (~62% of 6,000; workers still
  running). Slower than midgame labeling by ~30× per state.
- Usable subset for training: 2,940 train / 731 holdout rows assembled
  from completed shards (deduped by source_id; remaining shards still
  labeling and can only add states).

## Training (residual_v3 96×3, 46 epochs, seeds 42–43, 2,940 rows)

| seed | holdout top-in-set | value MAE |
| ---- | ------------------ | --------- |
| 42   | 35.57%             | 0.637     |
| 43   | 36.11%             | 0.679     |

Well below the midgame 26k lane (56–58%) — expected: 7× less data on a
harder-to-fit high-stone distribution.

## Gate (medium_eval, 512 games/budget)

| candidate  | 384:256 | 768:768 | 1200:1200 | 1200:256 | gate |
| ---------- | ------- | ------- | --------- | -------- | ---- |
| open seed42 | +0.195  | +0.219  | +0.383    | +0.297    | PASS |
| open seed43 | −0.320  | −0.039  | +0.176    | +0.109    | fail |

Seed42 passes all four budgets — the first gate pass since the 8k
exact/seed42 lane, and on 7× less data. Seed43 fails the primary.
Same seed-fragility pattern as every prior exact experiment (pass rate
1/2 here, 1/3 at 8k, 0/3 at 26k), but the passing lane now comes from
the opening distribution, supporting the distribution-mismatch
hypothesis: training where the suite plays beats training more midgame
states.

## Interpretation

- Distribution beats quantity: 2,940 opening states → 1 gate pass;
  20,799 midgame states → 0 passes. The program direction should be
  distribution repair, not more midgame labels.
- The exact-signal program stays high-variance (1/2, 1/3, 0/3 pass
  rates across experiments). The next investment should target variance
  reduction (multi-seed selection, opening+mixed-data recipes) rather
  than assuming any single lane passes.
- Labeling cost bounds ambition: full opening coverage at 30–70 s/state
  needs tiered timeouts or harder-state triage, not a naive 6k rerun.

## Artifacts

- `docs/data/alphazero-lite-exact-opening-distribution-probe-gate.json`
- `docs/data/alphazero-lite-exact-opening-distribution-probe-training.json`
- `docs/data/alphazero-lite-exact-opening-cohort.json`
- Full rows/checkpoints/benchmark (not committed):
  `/tmp/azlite_exact_opening*/`, `/tmp/azlite_exact_opening_train/`,
  `/tmp/azlite_exact_opening_bench/`

## Reproduction

```bash
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_training_ablation \
  --exact-train /tmp/azlite_exact_opening/opening_labeled_train.jsonl \
  --mcts-train /tmp/azlite_exact_opening/opening_labeled_train.jsonl \
  --exact-holdout /tmp/azlite_exact_opening/opening_labeled_holdout.jsonl \
  --source-states /tmp/azlite_exact_opening/source_states.jsonl \
  --workdir /tmp/azlite_exact_opening_train \
  --out-summary /tmp/azlite_exact_opening_train/summary.json \
  --lanes exact --seeds 42,43 \
  --model-type residual_v3 --hidden-sizes 96,3 --epochs 46 \
  --current model-artifact/current --skip-arena
python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/azlite_exact_opening_bench \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates <seed42,seed43 artifacts, comma-separated> \
  --budget-pairs "384:256,768:768,1200:1200,1200:256" \
  --games-per-opening 2 --seed 42 --workers 8
```

## Recommended next experiment

Mixed-distribution recipe at fixed budget: combine the 2,940 opening
rows with a matched 2,940 midgame sample (same total rows as here),
2–3 seeds, same recipe and gate. If the mixed lane passes ≥2/3 while
either pure lane stays at 1/2–1/3, adopt mixed-distribution exact data
as the program standard; if not, accept exact-signal variance as
structural and close the program in favor of MCTS-teacher work.
