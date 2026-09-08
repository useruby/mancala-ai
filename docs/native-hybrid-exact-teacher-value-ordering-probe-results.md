# Value-Ordering Probe: Margin-Ordered Targets vs ±1 Signs

Classification: `exact_value_ordering_probe_complete`.

Mechanism probe (a) from the scale-up report: does restoring exact
margin ordering to the value target move the gate, without labeling a
single new state? All variants train on identical 20,799 states with
identical exact policies; only `value` differs. No model was promoted.

## Variants (26k train rows, rewritten values only)

- `sign` (baseline): ±1/0 root-perspective sign — the production exact lane.
- `margin48`: root-perspective margin / 48 (observed range ±0.79).
- `margin_tanh`: tanh(margin / 12) (range ±0.996, compressed ordering).
- Order/state/policy identity verified row-for-row; all files pass
  `train.py` validation.

## Training (residual_v3 96×3, 46 epochs, lr 1e-3, seeds 42–43)

Frozen 5,200-holdout, scored against ±1 signs:

| lane      | seed42 top / MAE | seed43 top / MAE |
| --------- | ---------------- | ---------------- |
| sign      | 56.88% / 0.481   | 54.87% / 0.478   |
| margin48  | 55.08% / 0.831   | 55.46% / 0.829   |
| margin_tanh | 57.25% / 0.670 | 55.65% / 0.646   |

Policy is invariant to value scale (55–57% everywhere) — the policy
head learns the same exact-optimal mapping regardless. Value MAE
against ±1 is off-scale by construction for margin lanes (they predict
a different quantity); the informative comparison is the gate.

## Gate (seed42 checkpoints, 4 budgets, 512 games each)

| lane        | 384:256 | 768:768 | 1200:1200 | 1200:256 | gate |
| ----------- | ------- | ------- | --------- | -------- | ---- |
| sign        | +0.219  | +0.297  | +0.102    | −0.086    | fail |
| margin48    | +0.613  | +0.000  | +0.168    | −0.270    | fail |
| margin_tanh | +0.156  | +0.289  | −0.453    | −0.188    | fail |

The value-ordering hypothesis is rejected in both directions:
margin48 wins the primary budget decisively (+0.613 vs +0.219) yet
fails 1200:256 worse (−0.270 vs −0.086); margin_tanh collapses at
equal_high (−0.453). Smoother, ordering-preserving values do not fix
the high-budget non-regression failure — they relocate it. PUCT search
at 1200 simulations does not exploit the learned margin ordering
against this incumbent.

## Artifacts

- `docs/data/alphazero-lite-exact-value-ordering-probe-gate.json`
- `docs/data/alphazero-lite-exact-value-ordering-probe-training.json`
- `docs/data/alphazero-lite-exact-value-variant-margin48.json`
- `docs/data/alphazero-lite-exact-value-variant-margin-tanh.json`
- Checkpoints/benchmark (not committed):
  `/tmp/azlite_exact_valueprobe/{train,bench}/`

## Reproduction

```bash
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_value_variant_labels \
  --exact-train /tmp/azlite_exact_scale/train.jsonl --variant margin48 \
  --out-train /tmp/azlite_exact_valueprobe/margin48_train.jsonl \
  --out-summary /tmp/azlite_exact_valueprobe/margin48.json
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_training_ablation \
  --exact-train /tmp/azlite_exact_scale/train.jsonl \
  --mcts-train /tmp/azlite_exact_valueprobe/margin48_train.jsonl \
  --blend-train /tmp/azlite_exact_valueprobe/margin_tanh_train.jsonl \
  --exact-holdout /tmp/azlite_exact_scale/holdout.jsonl \
  --source-states /tmp/azlite_exact_scale/source_states.jsonl \
  --workdir /tmp/azlite_exact_valueprobe/train \
  --out-summary /tmp/azlite_exact_valueprobe/probe_summary.json \
  --lanes exact,mcts,blend --seeds 42,43 \
  --model-type residual_v3 --hidden-sizes 96,3 --epochs 46 \
  --current model-artifact/current --skip-arena
```

## Recommended next experiment

Probe (b), the distribution hypothesis, is now the lead: label an
opening-distribution exact cohort (low-stone states from real opening
prefixes, same native teacher) and benchmark it against the midgame
cohort. If opening states pass the gate, the program becomes
distribution repair; if they fail identically, the exact-signal program
should close in favor of MCTS-teacher work.
