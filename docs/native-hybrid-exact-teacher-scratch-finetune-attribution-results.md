# Scratch-vs-Finetune Attribution: Exact Labels on the 4-Budget Gate

Classification: `exact_scratch_finetune_attribution_complete`.

Diagnostic follow-up to the promotion replication
(`docs/native-hybrid-exact-teacher-promotion-replication-results.md`),
which recommended separating "exact labels" from "finetune dynamics" as
the gate-failure source. No model was promoted; `model-artifact/current`
is untouched.

## Part 1: v2 from-scratch residual checkpoints on the gate

The six v2-arena checkpoints (`residual_v3` 96×3, 16 epochs, lr 1e-3,
from scratch) were benchmarked on the production four-budget gate
(medium_eval, 128 openings × 2 games × 2 seats = 512 games/budget).

| candidate  | 384:256 | 768:768 | 1200:1200 | 1200:256 | gate |
| ---------- | ------- | ------- | --------- | -------- | ---- |
| exact s42  | +0.324  | +0.117  | +0.391    | +0.082   | PASS |
| exact s43  | −0.121  | +0.129  | +0.027    | −0.250    | fail |
| exact s44  | −0.207  | −0.039  | −0.137    | −0.285    | fail |
| mcts s42   | +0.086  | +0.008  | −0.191    | −0.141    | fail |
| mcts s43   | −0.457  | −0.297  | −0.094    | +0.031    | fail |
| mcts s44   | −0.082  | −0.281  | −0.098    | +0.062    | fail |

Gate rule: 384:256 ds > 0, 768:768 / 1200:1200 / 1200:256 ds ≥ 0.
Exactly one of six passes: exact/seed42. All three mcts lanes fail,
two of them on all four budgets simultaneously. The exact-label effect
is therefore visible at the gate but seed-fragile from scratch: the
passing lane beats the best mcts lane by +0.24 on the primary budget.

## Part 2: policy-head-only finetune (production scope, 1–2 epochs)

Exact lane finetuned from current, `trainable-scope policy_head`,
lr 1e-5, seeds 42–43. Frozen-holdout movement is negligible
(37.6%→37.9% top-in-set vs 51.4% for full-trunk 46-epoch finetune),
and the 4-budget gate fails at both epochs:

| candidate | 384:256 | 768:768 | 1200:1200 | 1200:256 | gate |
| --------- | ------- | ------- | --------- | -------- | ---- |
| ph e1 s42 | −0.375  | +0.555  | +0.391    | −0.145    | fail |
| ph e2 s42 | −0.266  | +0.324  | +0.371    | −0.145    | fail |

The production scope/schedule does not transfer exact policy signal to
the incumbent: the policy head barely moves (holdout +0.3pp over two
epochs) while the 1200:256 non-regression budget stays red.

## Interpretation

- Failure source is substantially "finetune dynamics", not "exact
  labels": from-scratch exact training produces a gate-passing lane;
  no finetune variant (full-trunk 46-epoch or policy-head 1–2 epoch)
  does.
- But "exact labels" alone do not clear the gate either: from-scratch
  exact passes 1/3 seeds. The label signal is real (all mcts lanes
  fail; paired arena effects stay positive) yet high-variance at 8k
  states / 16 epochs.
- Policy-head finetuning is a dead end for this data: two epochs move
  nothing measurable; longer head-only schedules are unlikely to beat
  full-trunk finetuning, which already fails the gate.

## Artifacts

- `docs/data/alphazero-lite-exact-scratch-finetune-attribution-gate.json`
  (compact per-candidate budgets + gate verdicts)
- `docs/data/alphazero-lite-exact-policyhead-finetune-gate.json`
- `docs/data/alphazero-lite-exact-policyhead-ph1-summary.json`
- `docs/data/alphazero-lite-exact-policyhead-ph2-summary.json`
- Full benchmark reports (not committed):
  `/tmp/azlite_exact_scratch_bench/temperature_benchmark_report.json`,
  `/tmp/azlite_exact_ph/bench/temperature_benchmark_report.json`

## Reproduction

```bash
python ml/alphazero_lite/run_opening_suite_seat_benchmark.py \
  --workdir /tmp/azlite_exact_scratch_bench \
  --suite /tmp/azlite_opening_suite/medium_eval.jsonl \
  --current model-artifact/current \
  --candidates <v2_arena exact/mcts seed42-44 artifacts, comma-separated> \
  --budget-pairs "384:256,768:768,1200:1200,1200:256" \
  --games-per-opening 2 --seed 42 --workers 8
```

## Recommended next experiment

Scale the from-scratch exact lane, not the finetune: more states
(≥25k, same frozen pipeline), 2–3 seeds, residual_v3 at ≥46 epochs,
then the four-budget gate. If the pass rate stays 1/3, the variance
itself becomes the research question (value-target smoothing or
multi-optimum policy sharpening); if it rises, proceed to promotion
replication of the winning recipe.
