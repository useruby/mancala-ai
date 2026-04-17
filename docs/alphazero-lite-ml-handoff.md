# AlphaZero-lite ML Handoff

This is the practical baseline for the next ML engineer iteration.

## Current Baseline

- Runtime target in this handoff is local AlphaZero-lite for `hard` (the same infrastructure also supports the stricter `superhuman` lane); `easy` and `medium` remain pure MCTS.
- Current promoted artifact path is `model-artifact/current` (`weights.json`, `metadata.json`, `arena_report.json`).
- Note: some pipeline configs reference `storage/ai/alphazero_lite/current` for historical compatibility. Before each lane, verify which path exists in your environment and set `current_path` consistently.
- Search/eval gate is baseline-relative against `MCTS1200` (candidate must score at least as well as current).
- Promotion gate uses `script/ai/local_promotion_gate` with defaults: arena `120` games, MCTS baseline `40` games, min arena score `0.55`, and candidate-vs-MCTS score must be >= current-vs-MCTS score.
- Key strategy that unlocked progress: train from `classic_mcts` teacher/player mode (teacher-student alignment).

## Immediate Goals

- Improve candidate strength vs current without regressing `MCTS1200` baseline-relative score.
- Reduce run-to-run variance by validating improvements across multiple seed schedules.
- Keep artifact contract stable for Rails runtime (`weights.json` + metadata schema compatibility).
- Preserve iteration speed: prefer reproducible config-driven runs over ad-hoc command edits.

## Known Pain Points

- Historical false positives: strong AZ-vs-AZ arena results can still fail vs `MCTS1200`.
- Teacher mismatch risk: changing teacher/search semantics without aligned evaluation can stall progress.
- Variance at small sample sizes: short runs can suggest gains that disappear at confirmation scale.
- Throughput constraints: long local runs are expensive; use RunPod wrapper for heavier experiments.

## Throughput Flags

Use spare RAM to reduce repeated evaluator/search work before scaling up pod size.

- For checkpoint-guided self-play, enable `--evaluator-cache-size` when repeated state evaluation is likely.
- For bootstrap runs using `--position-selection-mode hybrid_teacher`, enable `--teacher-search-reuse`.
- When using a checkpoint, make `--input-encoding` match the checkpoint metadata `input_encoding` field. Example: the current stronger bootstrap artifact at `/tmp/azlite_v3_superhuman_versions/aggressive-v3-superhuman-iter1/metadata.json` expects `kalah_v3`.

Focused benchmark results from this branch:

- Self-play benchmark:
  - command shape: `ml/alphazero_lite/self_play.py --checkpoint .../model.npz --input-encoding kalah_v3 --games 8 --workers 1 --simulations 96`
  - baseline: `6.020s`
  - with `--evaluator-cache-size 50000`: `4.203s`
  - result: about `30.2%` faster with `cache_hit_rate=0.713468` and identical row count (`297`)
- Bootstrap benchmark:
  - command shape: `ml/alphazero_lite/generate_bootstrap_dataset.py --games 8 --workers 1 --simulations 96 --max-positions-per-game 12 --position-selection-mode hybrid_teacher`
  - baseline: `5.304s`
  - with `--teacher-search-reuse`: `4.475s`
  - result: about `15.6%` faster, with identical retained rows (`70`) and fewer total simulations (`81792` down to `54528`)

Recommendation:

- Default these flags on for heavy experiment lanes where output semantics stay acceptable:
  - self-play with checkpoint: `--evaluator-cache-size 50000`
  - hybrid-teacher bootstrap: `--teacher-search-reuse`

## Standard Experiment Command

Use this as the default reproducible run entrypoint:

```bash
script/ai/runpod_training_experiment \
  --config-path ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_local.json \
  --results-path storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap \
  --local-results-path /tmp/runpod-results
```

Notes:

- Swap only `--config-path` (and optionally paths) per lane.
- Wrapper runs pipeline + `local_promotion_gate` remotely and downloads outputs.

## Robustness Confirmation On RunPod

Use the dedicated wrapper for multi-seed confirmation runs that are too expensive locally:

```bash
script/ai/runpod_model_robustness_confirmation
```

Default behavior:

- launches `script/ai/model_robustness_confirmation` remotely on RunPod
- downloads the full confirmation output tree to `/tmp/runpod-robustness-confirmation-results`
- preserves results even when the sweep exits non-zero because `passed=false`
- reports final pass/fail from downloaded `aggregate_summary.json`

Important:

- `passed=false` is a completed experiment result, not an infrastructure failure
- the top-level result file for confirmation runs is `aggregate_summary.json`
- if `aggregate_summary.json` is malformed or missing, fall back to `run_manifest.json` triage to distinguish real run failure from transport/setup failure
- long confirmation sweeps should prefer this wrapper over local execution

## Minimum Artifacts Per Run

Treat a run as usable only when all files below are present in the downloaded results leaf:

- `<results>/<candidate_dir>/weights.json`
- `<results>/<candidate_dir>/metadata.json`
- `<results>/<candidate_dir>/arena_report.json`
- `<results>/local_promotion_gate.json`
- `<results>/candidate_vs_current_arena.json`
- `<results>/candidate_vs_mcts1200.json`
- `<results>/current_vs_mcts1200.json`

If any item is missing, do not compare or promote; rerun/fix pipeline output first.

## Run Triage (Copy/Paste Checklist)

1. Open `<results>/local_promotion_gate.json`.
2. Check `passed`.
3. If `passed=false`, inspect `failure_reasons` and classify:
   - `arena_score_below_threshold` or `candidate_not_stronger_than_hard`: model strength issue.
   - `candidate_mcts_below_current`: baseline-relative MCTS regression.
   - `*_games_below_minimum`: run quality issue (not enough games).
   - `arena_losses_above_threshold` (lossless lane): fails strict superhuman contract.
   - `arena_move_time_*`: latency/performance regression.
4. Action policy:
   - run-quality failures -> rerun same config with stable infrastructure;
   - strength failures -> keep artifacts for analysis, do not promote;
   - performance failures -> profile search/inference before next lane.

## Next 3 Experiments

1) Confidence-gate-only lane (keep training/search fixed, tighten promotion confidence)

```bash
RUN_ID="confidence-gate-$(date +%Y%m%d-%H%M%S)"
TMP_CONFIG="/tmp/${RUN_ID}-phase2-confidence.json"

RUN_ID="$RUN_ID" TMP_CONFIG="$TMP_CONFIG" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

src = Path("ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json")
cfg = json.loads(src.read_text(encoding="utf-8"))
for step in cfg.get("steps", []):
    if step.get("name") == "benchmark_contract":
        cmd = step.get("command", [])
        if "--min-confidence-lower-bound" not in cmd:
            cmd.extend(["--min-confidence-lower-bound", "0.55"])
        step["command"] = cmd
Path(os.environ["TMP_CONFIG"]).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

.venv/bin/python ml/alphazero_lite/pipeline.py --config "$TMP_CONFIG"
```

2) Opponent-pool lane (introduce checkpoint-pool diversity in self-play)

```bash
RUN_ID="opponent-pool-$(date +%Y%m%d-%H%M%S)"
POOL_JSON="/tmp/${RUN_ID}-opponent-pool.json"
TMP_CONFIG="/tmp/${RUN_ID}-phase1-opponent-pool.json"

POOL_JSON="$POOL_JSON" TMP_CONFIG="$TMP_CONFIG" .venv/bin/python - <<'PY'
import json
import os
from pathlib import Path

pool_path = Path(os.environ["POOL_JSON"])
candidate_pool = [
    "model-artifact/current/model.npz",
    "storage/ai/alphazero_lite/current/model.npz",
]
checkpoints = [str(Path(path).resolve()) for path in candidate_pool if Path(path).exists()]
if not checkpoints:
    raise SystemExit("no model.npz found for opponent pool; export one checkpoint first")
pool_path.write_text(json.dumps({"checkpoints": checkpoints}, indent=2), encoding="utf-8")

src = Path("ml/alphazero_lite/configs/aggressive_v3_superhuman_phase1.json")
cfg = json.loads(src.read_text(encoding="utf-8"))
for step in cfg.get("steps", []):
    if step.get("name") == "self_play":
        cmd = step.get("command", [])
        if "--opponent-pool-config" not in cmd:
            cmd.extend(["--opponent-pool-config", os.environ["POOL_JSON"]])
        step["command"] = cmd
Path(os.environ["TMP_CONFIG"]).write_text(json.dumps(cfg, indent=2), encoding="utf-8")
PY

.venv/bin/python ml/alphazero_lite/pipeline.py --config "$TMP_CONFIG"
```

3) Tablebase-assisted lane (validate integration path and tactical upside)

```bash
.venv/bin/python -m unittest \
  ml.alphazero_lite.test_endgame_tablebase \
  ml.alphazero_lite.test_classic_mcts

.venv/bin/python ml/alphazero_lite/pipeline.py \
  --config ml/alphazero_lite/configs/aggressive_v3_superhuman_phase2.json \
  --dry-run
```

Notes:

- For lanes 1 and 2, compare against baseline with `script/ai/local_promotion_gate` before any promotion.
- Keep `weights.json` / `metadata.json` schema unchanged unless runtime contract changes are explicitly planned.

## Source Of Truth

- Treat script behavior and defaults as source-of-truth over this document:
  - `script/ai/local_promotion_gate`
  - `script/ai/runpod_training_experiment`
- If this document and script defaults diverge, follow scripts and update this handoff file in the same change.

## Review Checklist (High-Risk Areas)

Use this checklist before approving ML changes that touch promotion/evaluation or self-play behavior.

### 1) Benchmark Contract Validation

- [ ] `benchmark.py` rejects malformed MCTS reports (missing keys, wrong types, negative counts, inconsistent totals).
- [ ] `report_validation.py` enforces strict arena integer fields (no bool/string coercion).
- [ ] Confidence gate is opt-in (`--min-confidence-lower-bound`) and default behavior stays backward compatible.
- [ ] `promotion_decision.passed` in arena validation is checked against score-threshold semantics (not confidence gate).
- [ ] New benchmark checks are reflected in `local_promotion_gate.json` and are explainable from report fields.

### 2) Self-Play Profile Semantics

- [ ] `search_profile_hash` is deterministic for identical semantic inputs.
- [ ] Changing one semantic search field changes the hash.
- [ ] Profile semantics match player mode:
  - `puct` profile includes PUCT fields,
  - `classic_mcts` profile uses classic fields and does not pretend to be PUCT.
- [ ] When opponent pool is enabled, profile includes a stable pool fingerprint so runs are reproducible/auditable.
- [ ] Arena and MCTS baseline reports include compatible search-profile metadata for cross-run comparability.

### 3) Data Integrity And Leakage Checks

- [ ] Replay train/val split happens by unique source rows, not by expanded weighted positions.
- [ ] A single source replay row cannot appear in both train and validation sets.
- [ ] Seeded runs produce deterministic split behavior.

### 4) Release Safety

- [ ] Required artifacts exist (`weights.json`, `metadata.json`, `arena_report.json`, gate reports).
- [ ] No schema-breaking changes to runtime contract unless explicitly coordinated with Rails runtime.
- [ ] Promotion decision is supported by both arena and baseline-relative MCTS checks.
