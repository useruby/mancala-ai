# Native Hybrid Exact-Teacher Label Production

Classification: `exact_teacher_production_ready`.

Data-production/measurement experiment only. No model was trained or promoted.

## Frozen source cohort

- 10,000 unique training-eligible states, frozen before labeling (seed
  `20260906`, 2,500 per active-stone bucket `8-16`/`17-24`/`25-32`/`33-40`).
- Source states SHA-256
  `f5d6bb3256bf31ae3fe8089cdd8e1ebd602e731f46f874251a1b9374bc5f49cb`
  (`docs/data/alphazero-lite-exact-teacher-source-cohort.json`).
- Found in 318 random-play games (14,498 positions visited, 1,148
  duplicates skipped, 3,059 out-of-scope 41+ stone positions skipped).
- Leakage protection: 75 forensic-suite states and 53 opening-suite states
  (medium 128 + large 384 + small-smoke 32) excluded at freeze time; zero
  overlap with the 96-state PR #281 feasibility corpus and zero overlap
  with the 224-state forensic suite; train/holdout canonical overlap is 0
  (train hash `75286d64…`, holdout hash `5385cff1…`).
- Deterministic 80/20 canonical-key split: 8,000 train / 2,000 holdout.

## Exact label semantics (validated, not invented)

- Native `label` returns player-zero final stone margins; root extremum is
  `max` for player 0 / `min` for player 1; `optimal_actions` are exactly the
  legal actions attaining it (checked per label; mismatches fail loudly).
- Training policy: uniform over the exact optimal-action set, zero
  elsewhere (passes `train.py` policy validation on all 10,000 rows).
- Training value: sign of the root-perspective margin (`+1` win / `0`
  draw / `-1` loss), matching the repository win-rate convention
  (`2 * win_rate - 1`). Raw integer margins are preserved verbatim
  (`exact_root_margin`, `exact_action_margins`); `exact_value_normalized_48`
  is diagnostic only.
- Full-cohort label mix: 5,817 wins / 3,400 losses / 783 draws;
  2,923 multi-optimum states (29.23%). Phase mix: 152 early / 4,513 mid /
  5,335 late.
- No MCTS fallback exists anywhere in the path: timeouts/errors persist as
  explicit failed/unsolved rows. None occurred.

## Production metrics (10,000 states, 30 s timeout, warm process)

- Requested 10,000; solved 10,000; failed 0; solve rate 100%.
- Wall time 2,700.7 s; label latency (wall) p50 0.6 ms / p90 467 ms /
  p95 1.48 s / p99 5.44 s / max 32.27 s (max exceeds the 30 s request
  timeout only because it includes two sequential deterministic-repeat
  requests plus validation per state; no request timed out).
- CPU accounting: the 2.7 s process-time counter covers only the Python
  driver; the native solver burns CPU in-process. PR #281 measured
  16,415 successful labels/CPU-hour on the same teacher; at that rate this
  cohort costs ~0.61 CPU-hours (~36 min single-core).
- Deterministic repeat: 256/256 identical value/action-set labels across
  fresh-process relabels, and 256/256 agreement with production rows.
- Resume: completed source ids in existing train/holdout/failures outputs
  are skipped without relabeling (unit-tested; production completed in one
  run with no resume needed).

## Paired MCTS-vs-exact audit (1,200 frozen states, 1,200-sim teacher)

- MCTS top action in exact optimal set: 795/1,200 (66.25%).
- Exact single-optimum states: 689; MCTS top-1 agreement 54.72%.
- Exact multi-optimum states: 511/1,200 (42.58%).
- Mean absolute value disagreement (aligned root-perspective): 0.441;
  max 1.913.
- By stone bucket (in-set rate / mean |Δv|): 8-16: 75.86% / 0.342 (n=584);
  17-24: 64.38% / 0.502 (n=292); 25-32: 50.96% / 0.545 (n=208);
  33-40: 50.00% / 0.602 (n=116).
- By phase: early 50.00% / 0.685 (n=14); mid 54.67% / 0.510 (n=300);
  late 70.43% / 0.414 (n=886).
- MCTS top visit share is non-predictive of exact correctness here: 0.501
  when correct vs 0.520 when incorrect.

## Artifacts

- `docs/data/alphazero-lite-exact-teacher-production-summary.json`
- `docs/data/alphazero-lite-exact-teacher-mcts-audit-summary.json`
- `docs/data/alphazero-lite-exact-teacher-source-cohort.json`
- Full rows (train 8,000 / holdout 2,000, not committed):
  `/tmp/azlite_exact_teacher_production/{source_states,train,holdout}.jsonl`

## Reproduction

```bash
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_label_production \
  --source-states /tmp/azlite_exact_teacher_production/source_states.jsonl \
  --out-train /tmp/azlite_exact_teacher_production/train.jsonl \
  --out-holdout /tmp/azlite_exact_teacher_production/holdout.jsonl \
  --out-failures /tmp/azlite_exact_teacher_production/failures.jsonl \
  --out-summary /tmp/azlite_exact_teacher_production/summary.json \
  --native-probe /tmp/girving-check/native_probe \
  --artifact /tmp/kalah_v1_tier18_mmiqjum_/kalah_v1_18.kvtb --timeout 30
PYTHONPATH=. python -m ml.alphazero_lite.run_exact_teacher_mcts_audit \
  --source-states /tmp/azlite_exact_teacher_production/source_states.jsonl \
  --exact-train /tmp/azlite_exact_teacher_production/train.jsonl \
  --exact-holdout /tmp/azlite_exact_teacher_production/holdout.jsonl \
  --out-pairs /tmp/azlite_exact_teacher_production/audit_pairs.jsonl \
  --out-summary /tmp/azlite_exact_teacher_production/audit_summary.json \
  --audit-size 1200
```

## Recommended next experiment

Controlled training ablation: train one run on the 8,000 exact-labeled
train rows vs one run on the identical 8,000 states labeled by the
1,200-simulation classic-MCTS teacher, holding architecture, optimizer,
and epochs fixed; score both on the frozen 2,000-state exact holdout
(top-in-optimal-set rate + value MAE) and the sealed arena suites.
