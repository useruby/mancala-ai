# AlphaZero-lite ML Handoff

This is the practical baseline for the next ML engineer iteration.

## Current Baseline

- Runtime target in this handoff is local AlphaZero-lite for `hard` (the same infrastructure also supports the stricter `superhuman` lane); `easy` and `medium` remain pure MCTS.
- Current promoted runtime artifact path is `model-artifact/current` (`weights.json`, `metadata.json`, `arena_report.json`). Downloaded RunPod candidate trees now have a wider stronger-lane contract and must be promoted with `script/ai/promote_runpod_candidate` instead of manual copies.
- Note: some pipeline configs reference `storage/ai/alphazero_lite/current` for historical compatibility. Before each lane, verify which path exists in your environment and set `current_path` consistently.
- Search/eval gate is baseline-relative against `MCTS1200` (candidate must score at least as well as current).
- Promotion gate uses `script/ai/local_promotion_gate` with defaults: arena `120` games, MCTS baseline `40` games, min arena score `0.55`, and candidate-vs-MCTS score must be >= current-vs-MCTS score.
- Key strategy that unlocked progress: train from `classic_mcts` teacher/player mode (teacher-student alignment).

## Immediate Goals

- Improve candidate strength vs current without regressing `MCTS1200` baseline-relative score.
- Reduce run-to-run variance by validating improvements across multiple seed schedules.
- Keep artifact contract stable for the deployed runtime (`weights.json` + metadata schema compatibility).
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

## Stronger-Bootstrap More-Data RunPod Wrapper

Use the dedicated issue-#1 wrapper when you want the stronger-bootstrap more-data lane with a predictable downloaded result tree and a concise recommendation summary.

Default command:

```bash
script/ai/runpod_stronger_bootstrap_more_data_experiment
```

Default behavior:

- uses `ml/alphazero_lite/configs/aggressive_v3_stronger_bootstrap_more_data_local.json`
- writes remote results under `storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data`
- downloads results under `/tmp/runpod-stronger-bootstrap-more-data-results`
- writes a lane summary to `<downloaded-results-root>/<remote-results-dir-name>/issue1_summary.json`
- prints the same summary JSON to stdout, including the effective paths for the candidate artifact and key reports

Override the result directory when you need reruns or alternate seeds without editing the wrapper:

```bash
script/ai/runpod_stronger_bootstrap_more_data_experiment \
  --results-path storage/ai/alphazero_lite/versions/runpod-stronger-bootstrap-more-data-seed43
```

Summary interpretation:

- `confirm`: candidate passed the current issue checks, but alternate-seed confirmation is still required before treating it as promotion-ready
- `pivot`: candidate beat current in arena play but regressed against `MCTS1200`, so the next task should target tactical or hard-state follow-up
- `reject`: candidate failed the lane decision rules and should not be promoted

Important:

- the wrapper is orchestration plus reporting only; it does not change training or promotion semantics
- do not manually copy files into `model-artifact/current`
- use repo promotion tooling only after a candidate is validated and ready for promotion review

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

## Stronger-Lane Promotion Contract

Treat a downloaded RunPod result tree as usable only when the candidate leaf and root-level reports are complete.

Candidate leaf (`<results>/<candidate_dir>/`):

- `weights.json`
- `metadata.json`
- `arena_report.json`
- `run_manifest.json`

Root-level reports (`<results>/`):

- `local_promotion_gate.json`
- `candidate_vs_current_arena.json`
- `candidate_vs_mcts1200.json`
- `current_vs_mcts1200.json`
- `candidate_regression_suite.json`

If any item is missing, do not compare or promote; rerun/fix pipeline output first.

## Promoting A Downloaded RunPod Candidate

Do not manually copy files into `model-artifact/current`. Use the promotion helper so the gate report and candidate directory are validated before the local runtime artifact is swapped.

```bash
script/ai/promote_runpod_candidate /path/to/downloaded-results
```

Promotion flow:

1. Confirm `<results>/local_promotion_gate.json` exists and has `passed=true`.
2. Detect exactly one candidate artifact directory under the downloaded root.
3. Validate that the gate report `candidate_path` matches the detected candidate directory name.
4. Atomically replace `model-artifact/current` with the candidate runtime files (`weights.json`, `metadata.json`, `arena_report.json`).

Important:

- `run_manifest.json` and the root-level comparison/regression reports stay in the downloaded result tree for audit/debugging; they are not copied into `model-artifact/current`.
- Do not touch `model-artifact/current` from an unvalidated RunPod download.
- If the promotion helper rejects the tree, fix the result contract or rerun the lane; do not bypass the helper.

## Hard-Bot Promotion Backlog Tracker

Use the repo-owned backlog tracker when you want one tracked readiness campaign for a hard-bot candidate artifact.

Initialize or advance a campaign stage with:

```bash
script/ai/run_hard_bot_promotion_backlog \
  --manifest-path /tmp/hard-bot-campaign.json \
  --campaign-id candidate-iter3 \
  --candidate-path /path/to/candidate \
  --stage promotion_gate \
  --dry-run
```

Report the current readiness state with:

```bash
script/ai/report_hard_bot_promotion_readiness \
  --manifest-path /tmp/hard-bot-campaign.json
```

Current implementation note:

- `script/ai/run_hard_bot_promotion_backlog` does not execute the full promotion workflow end-to-end.
- It initializes or updates the manifest, prints the stage command in `--dry-run`, and can record stubbed stage outputs for the documented stages.
- Use it as a campaign tracker and command generator, then run or supply the underlying stage outputs separately.

The manifest is the audit trail for the workflow: it records the candidate artifact, generated reports, multi-seed confirmation evidence, and the final readiness state used to decide whether the candidate is promotion-ready.

## Phase 2 Ablation Workflow

Use the ablation runner against a required parent artifact from a completed stronger-lane result tree. The parent artifact input is the baseline candidate directory that each variant is compared against.

Runnable example:

```bash
script/ai/run_local_superhuman_phase2_ablation --variant replay_balanced --parent-artifact tmp/runpod_results_partial/aggressive-v3-superhuman-iter1
```

Initial variants run in this order:

1. `replay_balanced`
2. `selfplay_only`
3. `phase1_arch`

Why this order:

- `replay_balanced` is the smallest phase 2 change, so it gives the fastest read on whether replay mix alone explains the gain.
- `selfplay_only` isolates the self-play contribution after that, without also changing architecture.
- `phase1_arch` runs last because it is the broadest fallback comparison and is least specific to phase 2 behavior.

After each run, review these outputs before deciding whether the variant is stronger than its parent:

- `tmp/local_superhuman_phase2_ablation/<variant>/local_promotion_gate.json`
- `tmp/local_superhuman_phase2_ablation/<variant>/current_regressions.json`
- `tmp/local_superhuman_phase2_ablation/<variant>/parent_regressions.json`

The runner also prints the exact generated output paths in its final JSON payload.

Promotion stays blocked unless `script/ai/local_promotion_gate` passes. No ablation artifact should be promoted unless `script/ai/local_promotion_gate` passes.

## Opponent-Pool Phase-1 Comparison Lane

Use the existing script-based launcher for the phase-1 opponent-pool comparison lane. Do not invent a separate runner or edit pipeline commands by hand.

Runnable dry-run example:

```bash
RUN_ID="opponent-pool-$(date +%Y%m%d-%H%M%S)"

script/ai/run_local_superhuman_strength_experiment \
  --run-id "$RUN_ID" \
  --dry-run
```

What the launcher prepares:

- `opponent_pool_path`: generated checkpoint pool JSON used by lane B phase-1 self-play via `--opponent-pool-config`
- `stage1_config_path`: generated one-iteration phase-1 config for the balanced stage-1 run
- `stage2_config_path`: generated one-iteration phase-2 config for the balanced stage-2 follow-up run

The script prints a JSON summary. Inspect these fields before running the real lane:

- `opponent_pool_path`
- `stage1_config_path`
- `stage2_config_path`
- `lane_b_report_path`
- `lane_b_stage1_command`
- `lane_b_stage2_command`
- `lane_b_gate_command`

Lane B evaluation flow:

1. Run `script/ai/local_promotion_gate` using the generated lane-B gate command to write `lane_b_report_path`.
2. If you want a direct regression diff against the current artifact, run `script/ai/compare_superhuman_regressions` with `--baseline-artifact model-artifact/current` and `--candidate-artifact <lane-b-candidate-dir>`.

Notes:

- `script/ai/local_promotion_gate` remains the promotion decision source of truth.
- `script/ai/compare_superhuman_regressions` is a Python entrypoint with a stable `script/ai/...` command name; it is for inspecting regression deltas and does not replace the promotion gate.
- Keep the documented flow aligned with `script/ai/run_local_superhuman_strength_experiment` output fields and generated commands.

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

## Incumbent Forensic Suite

Run the position-level forensic report before starting any new promotion lane:

```bash
.venv/bin/python ml/alphazero_lite/run_forensic_suite.py \
  --suite ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json \
  --current-artifact model-artifact/current \
  --challenger-artifact <artifact_path> \
  --mcts-simulations 1200 \
  --teacher-simulations 1800 \
  --out artifacts/incumbent_forensics.json
```

Review these fields first:

- `systems.current.overall.top1_agreement`
- `systems.current.overall.average_regret`
- `systems.challenger.overall.top1_agreement`
- `systems.challenger.overall.average_regret`
- `buckets.sparse_endgame.systems.current`
- `buckets.capture_available.systems.current`
- `buckets.incumbent_proxy_disagreement.systems.current`
- `buckets.incumbent_proxy_disagreement.systems.challenger`

Interpretation notes:

- Use `top1_agreement` as the first policy signal against the classic-MCTS teacher.
- Use `average_regret` to distinguish close misses from catastrophic move choice errors.
- Use `blunder_rate` as the fastest tactical-risk read: if it rises in a hard bucket, treat that as a real move-quality regression even when aggregate strength still looks good.
- Use `value_calibration_mae` to spot value-head drift even when top-1 policy agreement looks acceptable.
- Start with `incumbent_proxy_disagreement` to inspect incumbent-style proxy disagreement states, then check `sparse_endgame` and `capture_available` for broader tactical and endgame regressions.

## Interpreting `local_promotion_gate.json` Forensic Outputs

The promotion report now includes a forensic quality gate in addition to arena and MCTS checks.

- `forensic_quality`: top-level summary of whether the candidate cleared the hard-coded v1 forensic quality gate.
- `forensic_quality.critical_buckets`: per-bucket forensic gate results for the critical forensic buckets that currently block promotion.
- `hard_suite_buckets`: hard arena bucket summaries from the phase-based hard-opponent run (`opening`, `midgame`, `late`). This is separate from the forensic gate.

How to read them:

- `forensic_quality.passed=true` means the candidate cleared the overall forensic quality gate.
- `forensic_quality.passed=false` means the candidate showed position-level quality regressions severe enough to block promotion even if arena and baseline-relative MCTS passed.
- `forensic_quality.critical_buckets` is the fastest way to see where a forensic failure came from. Each bucket-level pass/fail means "did the candidate stay inside the allowed forensic envelope for this class of positions?"
- `hard_suite_buckets` is still useful, but only for the hard arena phase split. It does **not** carry forensic pass/fail state.
- A failed bucket is a targeted regression signal, not just noise in aggregate win rate.

Current hard buckets to inspect first:

- `capture_available`: tactical conversion / obvious capture handling bucket.
- `sparse_endgame`: low-material endgame bucket.

Action policy:

- If a bucket fails because `blunder_rate` jumps, treat it as a move-selection regression and inspect the bucket positions before running another promotion attempt.
- If `capture_available` fails, prioritize tactical-policy debugging and teacher-target review.
- If `sparse_endgame` fails, prioritize endgame/value calibration review.
- If the forensic gate fails but arena/MCTS pass, do **not** promote. Keep the artifact, inspect `forensic_quality.critical_buckets`, and treat the result as "strength looks promising but quality is not release-safe yet." Use the bucket deltas to choose the next lane or ablation.

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

2) Opponent-pool lane (use the script-based phase-1 comparison flow above)

```bash
RUN_ID="opponent-pool-$(date +%Y%m%d-%H%M%S)"

script/ai/run_local_superhuman_strength_experiment \
  --run-id "$RUN_ID" \
  --dry-run
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

## Tactical Hard-State Replay Lane

The next hard-bot strength lane is documented in this section and in the tactical-lane implementation scripts under `ml/alphazero_lite/` and `script/ai/`.

Use `script/ai/run_local_tactical_replay_experiment` as the default local entrypoint for the local tactical replay run tree. The wrapper exists to run and dry-run the local mine -> label -> train -> select flow, generate the run-local runtime config automatically, and write the run summary/selection artifacts in one predictable tree.

Required inputs for a normal local run:

- `--run-id` for the run directory name;
- `--output-root` for the parent run tree;
- `--current-path` for the incumbent runtime artifact;
- `--base-config ml/alphazero_lite/configs/aggressive_v3_tactical_replay_local.json` for the tactical replay training lane;
- `--forensic-suite ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json` for the forensic inputs.

Dry-run the wrapper first to confirm the planned local stage order, generated commands, and output paths before starting a real local run:

```bash
RUN_ID="tactical-replay-$(date +%Y%m%d-%H%M%S)"

script/ai/run_local_tactical_replay_experiment \
  --run-id "$RUN_ID" \
  --output-root /tmp/tactical-replay-runs \
  --current-path model-artifact/current \
  --base-config ml/alphazero_lite/configs/aggressive_v3_tactical_replay_local.json \
  --forensic-suite ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json \
  --dry-run
```

Inspect the dry-run JSON first:

- `summary_path` points at `<output-root>/<run-id>/summary.json`;
- `paths.runtime_config_path` points at the generated run-local config under `<output-root>/<run-id>/inputs/runtime_config.json`;
- `paths.selection_dir` points at `<output-root>/<run-id>/selection`, where the wrapper materializes `selection/artifact` plus `selection/selection_manifest.json` after candidate selection.

Exploratory confirmation and the final holdout/decision workflow stay downstream of the default launcher flow. Treat multi-seed exploratory confirmation as an external prerequisite before final promotion review, then use the documented holdout commands below against the selected artifact tree.

Rerun guidance:

- use the default local wrapper flow when you want the lane to mine, label, train, and select a new candidate from scratch;
- use `--start-stage final_holdout` with `--selected-artifact /path/to/artifact` only when you are re-entering the downstream holdout portion with an already chosen artifact and want the launcher to rebuild the local run tree paths for that stage;
- use `--selected-artifact` again when you need to point the downstream stage at a manually chosen candidate artifact instead of auto-selecting from the generated `versions/` tree;
- do not expect the default wrapper invocation to produce the exploratory confirmation artifact or tactical lane decision on its own.

Implementation flow:

- generate a completed forensic report first with `ml/alphazero_lite/run_forensic_suite.py --suite --current-artifact --challenger-artifact --out`, then mine hard states from that report with `ml/alphazero_lite/mine_hard_states.py --inputs --out-jsonl --out-report`;
- label mined states with `ml/alphazero_lite/label_tactical_states.py --input --policy-simulations 1200 --value-simulations 1200 --seed 42 --policy-target-mode sharpened --value-target-mode sharpened --input-encoding kalah_v3 --out-labeled --out-tactical-train --out-preservation-train`;
- train the local tactical replay config against the generated tactical replay dataset via the wrapper-generated runtime config;
- export and select one candidate artifact under the wrapper run tree;
- run final holdout forensics, bucket gate, arena aggregation, MCTS aggregation, regression report, local promotion gate, and tactical lane decision report only after exploratory confirmation exists.

Hard-bot promotion uses base `384` AlphaZero-lite simulations for both candidate and current arena arms. Runs using base `640` simulations are superhuman-strength checks and do not substitute for hard-bot promotion.

Final holdout workflow shape:

~~~bash
# Multi-seed exploratory confirmation must complete before final holdout.
# Do not use bare defaults here for a tactical candidate: the wrapper defaults target
# the superhuman phase-2 config and parent artifact.
# Point the wrapper at the exact tactical candidate/parent/config you want confirmed,
# or provide an already-generated aggregate_summary.json from a matching confirmation run.
script/ai/runpod_model_robustness_confirmation \
  --base-config ml/alphazero_lite/configs/aggressive_v3_tactical_replay_local.json \
  --parent-artifact <tactical_parent_artifact> \
  --current-path model-artifact/current \
  --results-path storage/ai/alphazero_lite/versions/tactical-robustness-confirmation \
  --local-results-path /tmp/tactical-robustness-confirmation-results

.venv/bin/python ml/alphazero_lite/run_forensic_suite.py \
  --suite ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json \
  --current-artifact model-artifact/current \
  --challenger-artifact artifacts/tactical_lane/selected/artifact \
  --mcts-simulations 1200 \
  --teacher-simulations 1800 \
  --seed 3041 \
  --out artifacts/tactical_lane/final/selected_candidate_forensics.json

.venv/bin/python ml/alphazero_lite/arena.py \
  --challenger artifacts/tactical_lane/selected/artifact \
  --current model-artifact/current \
  --games 120 \
  --seed 1041 \
  --challenger-simulations 384 \
  --current-simulations 384 \
  --reuse-subtree \
  --root-policy-mode deterministic \
  --tactical-root-bias 0.1 \
  --out artifacts/tactical_lane/final/arena_seed_1041.json

.venv/bin/python ml/alphazero_lite/run_forensic_suite.py \
  --suite ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json \
  --current-artifact model-artifact/current \
  --challenger-artifact model-artifact/current \
  --mcts-simulations 1200 \
  --teacher-simulations 1800 \
  --seed 2040 \
  --out artifacts/tactical_lane/final/baseline_candidate_forensics.json

.venv/bin/python ml/alphazero_lite/mcts1200_baseline.py \
  --challenger-path artifacts/tactical_lane/selected/artifact \
  --games 40 \
  --seed 2041 \
  --az-base-simulations 384 \
  --mcts-simulations 1200 \
  --reuse-subtree \
  --root-policy-mode deterministic \
  --tactical-root-bias 0.1 \
  --out artifacts/tactical_lane/final/candidate_mcts_seed_2041.json

.venv/bin/python ml/alphazero_lite/mcts1200_baseline.py \
  --challenger-path model-artifact/current \
  --games 40 \
  --seed 2041 \
  --az-base-simulations 384 \
  --mcts-simulations 1200 \
  --reuse-subtree \
  --root-policy-mode deterministic \
  --tactical-root-bias 0.1 \
  --out artifacts/tactical_lane/final/current_mcts_seed_2041.json

script/ai/check_superhuman_regressions \
  --positions test/fixtures/ai/superhuman_regression_positions.json \
  --artifact artifacts/tactical_lane/selected/artifact \
  --simulations 384 \
  --reuse-subtree \
  --root-policy-mode deterministic \
  --tactical-root-bias 0.1 \
  --out artifacts/tactical_lane/final/candidate_regression_suite.json

.venv/bin/python ml/alphazero_lite/check_bucket_promotion_gate.py \
  --baseline-forensics artifacts/tactical_lane/final/baseline_candidate_forensics.json \
  --candidate-forensics artifacts/tactical_lane/final/selected_candidate_forensics.json \
  --out artifacts/tactical_lane/final/bucket_gate.json

.venv/bin/python ml/alphazero_lite/aggregate_holdout_reports.py \
  --arena-inputs artifacts/tactical_lane/final/arena_seed_1041.json \
  --candidate-mcts-inputs artifacts/tactical_lane/final/candidate_mcts_seed_2041.json \
  --current-mcts-inputs artifacts/tactical_lane/final/current_mcts_seed_2041.json \
  --out-arena artifacts/tactical_lane/final/arena_holdout.json \
  --out-candidate-mcts artifacts/tactical_lane/final/candidate_mcts_holdout.json \
  --out-current-mcts artifacts/tactical_lane/final/current_mcts_holdout.json

script/ai/local_promotion_gate \
  --candidate-path artifacts/tactical_lane/selected/artifact \
  --current-path model-artifact/current \
  --arena-games 120 \
  --mcts-games 40 \
  --min-mcts-games 40 \
  --min-arena-score 0.55 \
  --max-arena-move-time-mean-ms 200 \
  --max-arena-move-time-p95-ms 350 \
  --stub-arena-report artifacts/tactical_lane/final/arena_holdout.json \
  --stub-candidate-mcts-report artifacts/tactical_lane/final/candidate_mcts_holdout.json \
  --stub-current-mcts-report artifacts/tactical_lane/final/current_mcts_holdout.json \
  --stub-regression-report artifacts/tactical_lane/final/candidate_regression_suite.json \
  --stub-forensic-report artifacts/tactical_lane/final/selected_candidate_forensics.json \
  --out artifacts/tactical_lane/final/local_promotion_gate.json

.venv/bin/python ml/alphazero_lite/write_tactical_lane_decision.py \
  --bucket-gate artifacts/tactical_lane/final/bucket_gate.json \
  --promotion-gate artifacts/tactical_lane/final/local_promotion_gate.json \
  --exploratory-summary /tmp/tactical-robustness-confirmation-results/tactical-robustness-confirmation/aggregate_summary.json \
  --out artifacts/tactical_lane/final/tactical_lane_decision.json
~~~

Notes:

- For tactical final holdout, only use `script/ai/runpod_model_robustness_confirmation` when its `--base-config`, `--parent-artifact`, and result paths are set for the tactical candidate being reviewed. The wrapper defaults are for the superhuman phase-2 lane and are not the right exploratory artifact for tactical holdout by themselves.
- Final holdout requires both the exploratory confirmation artifact and the holdout gate artifacts above; `write_tactical_lane_decision.py` consumes both.
- When re-entering the wrapper at `--start-stage decision`, pass the exact `aggregate_summary.json` path from the matching exploratory confirmation run if that artifact lives outside the run-local tree.

## Source Of Truth

- Treat script behavior and defaults as source-of-truth over this document:
  - `script/ai/local_promotion_gate`
  - `script/ai/runpod_training_experiment`
  - `script/ai/run_local_superhuman_recovery`
  - `script/ai/superhuman_budget_sweep`
- If this document and script defaults diverge, follow scripts and update this handoff file in the same change.

## Recovery Note

- The regenerated `aggressive-v3-superhuman-iter2` recovery on `mancala-ai`
  failed its native phase-2 prefilter at `896 vs 256` with arena score `0.5`
  (`15W 15L 0D`).
- The first `local_superhuman_budget_sweep` result tree was not authoritative:
  the original sweep script labeled multiple budgets but still ran the fixed
  phase-2 challenger budget. Ignore that artifact when interpreting strength.
- The corrected sweep rerun, produced by `script/ai/superhuman_budget_sweep`
  after the per-lane config rewrite fix, showed the same `0.5` arena score and
  lossless failure reasons at every tested challenger budget `384..896`.
- Treat the corrected rerun output as the authoritative conclusion: this
  regenerated candidate did not recover promotion viability by changing the
  challenger search budget alone.

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
- [ ] No schema-breaking changes to runtime contract unless explicitly coordinated with the deployed runtime owner.
- [ ] Promotion decision is supported by both arena and baseline-relative MCTS checks.
