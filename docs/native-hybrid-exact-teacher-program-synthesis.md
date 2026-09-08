# Native Hybrid Exact Teacher: Program Synthesis

Program classification: `exact_teacher_program_mixed_do_not_promote`.

This is the index document for the exact-teacher program (PR #281
through the distribution probes). It combines eight experiments into
one verdict. No model was promoted by any experiment in this program;
`model-artifact/current` is untouched throughout.

## Experiment map

| # | Experiment | Report | Verdict |
| - | ---------- | ------ | ------- |
| 0 | Feasibility (PR #281) | `docs/native-hybrid-exact-teacher-feasibility-results.md` | feasible |
| 1 | Label production (10k) | `docs/native-hybrid-exact-teacher-production-results.md` | `exact_teacher_production_ready` |
| 2 | Training ablation v1 | `docs/native-hybrid-exact-teacher-training-ablation-results.md` | `exact_teacher_training_ablation_complete` |
| 3 | Ablation v2 (multi-seed, capacity, blend) | `docs/native-hybrid-exact-teacher-ablation-v2-results.md` | `exact_teacher_training_ablation_v2_complete` |
| 4 | Promotion replication | `docs/native-hybrid-exact-teacher-promotion-replication-results.md` | `exact_promotion_replication_gate_not_met` |
| 5 | Scratch-vs-finetune attribution | `docs/native-hybrid-exact-teacher-scratch-finetune-attribution-results.md` | `exact_scratch_finetune_attribution_complete` |
| 6 | Scale to 26k states | `docs/native-hybrid-exact-teacher-scale26k-results.md` | `exact_scale26k_gate_not_met` |
| 7 | Value-ordering probe | `docs/native-hybrid-exact-teacher-value-ordering-probe-results.md` | `exact_value_ordering_probe_complete` |
| 8 | Opening-distribution probe | `docs/native-hybrid-exact-teacher-opening-distribution-probe-results.md` | `exact_opening_distribution_probe_complete` |

## What held up at every step

- **Label production is solved.** 10k cohort at 100% solve rate, 26k
  cohort at 99.996% (one explicit timeout retained, no fallback).
  Deterministic relabels agree 256/256; resume is idempotent;
  train/holdout canonical overlap is 0 everywhere; leakage exclusions
  (feasibility corpus, forensic suite, opening suites) verified at
  every freeze.
- **Exact policy signal is real.** The exact lane beats the
  same-state MCTS lane on holdout top-in-optimal-set at all 9
  lane-seed-capacity points in v2, all 3 scale-up seeds, and both
  value-variant probes. Paired arena effects favor exact at all 4 v2
  legs (mean +0.255, every CI excludes zero) with head-to-head 0.906.
- **Value semantics are validated, not invented.** Native margins are
  player-zero final stone differences; training values are
  root-perspective signs. Every row passes `train.py` validation.

## What did not hold up

- **The production gate.** Four-budget rule (384:256 ds > 0; 768:768,
  1200:1200, 1200:256 ds ≥ 0) results across all from-scratch exact
  lanes: 8k v2 1/3 pass, 26k scale-up 0/3, opening probe 1/2. Only two
  exact lanes in the entire program ever passed: 8k exact/seed42 and
  opening/seed42.
- **Finetuning.** Full-trunk 46-epoch and policy-head 1–2-epoch
  finetunes of the incumbent on exact data both fail the gate. The
  policy-head variant barely moves holdout (+0.3pp) — a dead end.
- **Value-ordering rescue.** Margin-ordered targets (margin/48,
  tanh(margin/12)) relocate the high-budget failure instead of fixing
  it. Rejected in both directions.
- **Scaling midgame states.** 8k → 26k improved holdout (53.25% →
  56.69%, MAE 0.599 → 0.478) while the gate pass rate fell (1/3 →
  0/3). Holdout fidelity and suite strength are decoupled here.

## The one positive directional signal

Distribution beats quantity: 2,940 opening-distribution states produced
a gate pass where 20,799 midgame states produced none. The suite plays
openings; training where it plays wins. But the opening result is also
1/2 by seed, and opening labels cost ~30× per state (30–70 s at 44–48
stones), so this is a direction, not a victory.

## Verdict

Do not promote any exact-lane checkpoint. Do not scale midgame exact
labeling further. The exact teacher is production-ready as a *label
source* and rejected as a *promotion vehicle* in its current form.

## Recommended next experiment

Mixed-distribution recipe at fixed budget: 2,940 opening rows + 2,940
midgame rows, 2–3 seeds, same recipe and gate. Mixed ≥2/3 while pure
lanes stay ~1/2 → adopt mixed exact data as the program standard; else
accept variance as structural and close the program in favor of
MCTS-teacher work.

## Provenance

- Code: `ml/alphazero_lite/exact_teacher_labeling.py`,
  `run_exact_teacher_label_production.py`,
  `run_exact_teacher_mcts_audit.py`,
  `run_exact_teacher_training_ablation.py` (+ `--lanes`, `--seeds`,
  `--blend-train`, `--init-checkpoint`, `--trainable-scope` extensions),
  `run_exact_ablation_mcts_relabel.py`,
  `run_exact_ablation_blend_labels.py`,
  `run_exact_value_variant_labels.py`, plus both test files (50 tests).
- Data: `docs/data/alphazero-lite-exact-*` (compact gate/training
  summaries; full rows/checkpoints/benchmarks live in
  `/tmp/azlite_exact_{teacher_production,ablation,scale,valueprobe,opening*,promotion,ph,scratch_bench}/`
  and are not committed).
- Full SHAs, per-budget tables, and reproduction commands are in each
  linked report.
