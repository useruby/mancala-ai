#!/usr/bin/env python3
"""Coach-style pipeline scaffold for AlphaZero-lite aggressive plan."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.input_encodings import feature_count_for
from ml.alphazero_lite.report_validation import ArenaReportValidationError, validate_arena_report
from ml.alphazero_lite.run_manifest import build_manifest, write_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--skip-step", action="append", default=[])
    parser.add_argument("--start-iteration", type=int, default=None)
    return parser.parse_args()


def load_config(path: Path) -> dict:
    raw = path.read_text(encoding="utf-8")
    return json.loads(raw)


def checkpoint_feature_count(checkpoint_path: Path) -> int | None:
    if not checkpoint_path.exists():
        return None

    import numpy as np

    try:
        with np.load(checkpoint_path) as npz:
            for key in ("w_input", "w_hidden_1", "w1"):
                if key in npz:
                    return int(npz[key].shape[0])
    except (OSError, ValueError):
        return None

    return None


def drop_incompatible_parent_checkpoint(command: list[str], *, parent_checkpoint: Path) -> list[str]:
    if "--checkpoint" not in command or "--input-encoding" not in command:
        return command

    checkpoint_index = command.index("--checkpoint")
    if checkpoint_index + 1 >= len(command) or command[checkpoint_index + 1] != str(parent_checkpoint):
        return command

    expected_feature_count = feature_count_for(command[command.index("--input-encoding") + 1])
    actual_feature_count = checkpoint_feature_count(parent_checkpoint)
    if actual_feature_count is None or actual_feature_count == expected_feature_count:
        return command

    return command[:checkpoint_index] + command[checkpoint_index + 2 :]


def render_command(
    command: list[str],
    *,
    iteration: int,
    iter_dir: Path,
    run_id: str,
    versions_dir: Path,
    current_path: str,
    parent_model_dir: Path,
    parent_checkpoint: Path,
    replay_data: str,
    replay_weights: str,
) -> list[str]:
    rendered: list[str] = []
    for token in command:
        value = token
        value = value.replace("{iteration}", str(iteration))
        value = value.replace("{iter_dir}", str(iter_dir))
        value = value.replace("{run_id}", run_id)
        value = value.replace("{versions_dir}", str(versions_dir))
        value = value.replace("{current_path}", current_path)
        value = value.replace("{parent_model_dir}", str(parent_model_dir))
        value = value.replace("{parent_checkpoint}", str(parent_checkpoint))
        value = value.replace("{replay_data}", replay_data)
        value = value.replace("{replay_weights}", replay_weights)
        rendered.append(value)
    return drop_incompatible_parent_checkpoint(rendered, parent_checkpoint=parent_checkpoint)


def render_path(
    path: str,
    *,
    iteration: int,
    iter_dir: Path,
    run_id: str,
    versions_dir: Path,
    current_path: str,
    parent_model_dir: Path,
    parent_checkpoint: Path,
    replay_data: str,
    replay_weights: str,
) -> Path:
    rendered = path
    rendered = rendered.replace("{iteration}", str(iteration))
    rendered = rendered.replace("{iter_dir}", str(iter_dir))
    rendered = rendered.replace("{run_id}", run_id)
    rendered = rendered.replace("{versions_dir}", str(versions_dir))
    rendered = rendered.replace("{current_path}", current_path)
    rendered = rendered.replace("{parent_model_dir}", str(parent_model_dir))
    rendered = rendered.replace("{parent_checkpoint}", str(parent_checkpoint))
    rendered = rendered.replace("{replay_data}", replay_data)
    rendered = rendered.replace("{replay_weights}", replay_weights)
    return Path(rendered)


def build_replay_context(*, iteration: int, iter_dir: Path, versions_dir: Path, run_id: str, replay_window: int) -> tuple[str, str]:
    replay_paths: list[Path] = [iter_dir / "self_play.jsonl"]

    for previous_iteration in range(iteration - 1, max(0, iteration - replay_window), -1):
        previous_path = versions_dir / f"{run_id}-iter{previous_iteration}" / "self_play.jsonl"
        if previous_path.exists():
            replay_paths.append(previous_path)

    replay_data = ",".join(str(path) for path in replay_paths)
    replay_weights = ",".join(str(weight) for weight in range(len(replay_paths), 0, -1))
    return replay_data, replay_weights


def run_step(
    step: dict,
    *,
    iteration: int,
    iter_dir: Path,
    run_id: str,
    versions_dir: Path,
    repo_root: Path,
    current_path: str,
    parent_model_dir: Path,
    parent_checkpoint: Path,
    replay_data: str,
    replay_weights: str,
) -> dict:
    name = step.get("name", "unnamed_step")
    command = step.get("command", [])
    if not isinstance(command, list) or not command:
        return {"name": name, "status": "failed", "error": "step command must be a non-empty list"}

    rendered = render_command(
        command,
        iteration=iteration,
        iter_dir=iter_dir,
        run_id=run_id,
        versions_dir=versions_dir,
        current_path=current_path,
        parent_model_dir=parent_model_dir,
        parent_checkpoint=parent_checkpoint,
        replay_data=replay_data,
        replay_weights=replay_weights,
    )
    started = time.time()
    result = subprocess.run(rendered, cwd=repo_root, capture_output=True, text=True, check=False)
    duration = round(time.time() - started, 4)

    log_path = iter_dir / f"{name}.log"
    log_path.write_text(
        f"command={json.dumps(rendered)}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}",
        encoding="utf-8",
    )

    return {
        "name": name,
        "status": "completed" if result.returncode == 0 else "failed",
        "command": rendered,
        "returncode": result.returncode,
        "duration_s": duration,
        "log_path": str(log_path),
    }


def gate_failures(
    gates: dict,
    *,
    iteration: int,
    iter_dir: Path,
    run_id: str,
    versions_dir: Path,
    current_path: str,
    parent_model_dir: Path,
    parent_checkpoint: Path,
    replay_data: str,
    replay_weights: str,
    skipped_steps: set[str],
) -> list[dict]:
    failures: list[dict] = []

    rules_parity_report = gates.get("rules_parity_report")
    if rules_parity_report and "rules_parity_fuzz" not in skipped_steps:
        parity_path = render_path(
            rules_parity_report,
            iteration=iteration,
            iter_dir=iter_dir,
            run_id=run_id,
            versions_dir=versions_dir,
            current_path=current_path,
            parent_model_dir=parent_model_dir,
            parent_checkpoint=parent_checkpoint,
            replay_data=replay_data,
            replay_weights=replay_weights,
        )
        if not parity_path.exists():
            failures.append({"code": "rules_parity_report_missing", "message": f"missing rules parity report: {parity_path}"})
        else:
            payload = json.loads(parity_path.read_text(encoding="utf-8"))
            if not payload.get("parity_passed", False):
                failures.append({"code": "rules_parity_failed", "message": "rules parity check did not pass"})

    perspective_audit_report = gates.get("perspective_audit_report")
    if perspective_audit_report:
        audit_path = render_path(
            perspective_audit_report,
            iteration=iteration,
            iter_dir=iter_dir,
            run_id=run_id,
            versions_dir=versions_dir,
            current_path=current_path,
            parent_model_dir=parent_model_dir,
            parent_checkpoint=parent_checkpoint,
            replay_data=replay_data,
            replay_weights=replay_weights,
        )
        if not audit_path.exists():
            failures.append({"code": "perspective_audit_missing", "message": f"missing perspective audit report: {audit_path}"})
        else:
            payload = json.loads(audit_path.read_text(encoding="utf-8"))
            if not payload.get("passed", False):
                failures.append({"code": "perspective_audit_failed", "message": "perspective audit did not pass"})

    arena_report = gates.get("arena_report")
    if arena_report:
        arena_path = render_path(
            arena_report,
            iteration=iteration,
            iter_dir=iter_dir,
            run_id=run_id,
            versions_dir=versions_dir,
            current_path=current_path,
            parent_model_dir=parent_model_dir,
            parent_checkpoint=parent_checkpoint,
            replay_data=replay_data,
            replay_weights=replay_weights,
        )
        min_arena_score = float(gates.get("min_arena_score", 0.55))
        if not arena_path.exists():
            failures.append({"code": "arena_report_missing", "message": f"missing arena report: {arena_path}"})
        else:
            payload = json.loads(arena_path.read_text(encoding="utf-8"))
            try:
                result = validate_arena_report(report=payload, min_score=min_arena_score)
            except ArenaReportValidationError as error:
                failures.append({"code": error.code.lower(), "message": str(error)})
            else:
                score = float(result["score"])
                if score < min_arena_score:
                    failures.append({"code": "arena_score_below_threshold", "message": f"arena score {score:.4f} < {min_arena_score:.4f}"})

    benchmark_report = gates.get("benchmark_report")
    if benchmark_report:
        benchmark_path = render_path(
            benchmark_report,
            iteration=iteration,
            iter_dir=iter_dir,
            run_id=run_id,
            versions_dir=versions_dir,
            current_path=current_path,
            parent_model_dir=parent_model_dir,
            parent_checkpoint=parent_checkpoint,
            replay_data=replay_data,
            replay_weights=replay_weights,
        )
        if not benchmark_path.exists():
            failures.append({"code": "benchmark_report_missing", "message": f"missing benchmark report: {benchmark_path}"})
        else:
            payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
            checks = payload.get("checks", [])
            if not isinstance(checks, list) or not checks:
                failures.append({"code": "benchmark_checks_missing", "message": "benchmark report missing checks results"})
            else:
                checks_with_status = [check for check in checks if isinstance(check, dict) and "passed" in check]
                if not checks_with_status:
                    failures.append({"code": "benchmark_checks_unscored", "message": "benchmark checks do not include pass/fail values"})
                else:
                    failing_checks = [check.get("id", "unknown") for check in checks_with_status if not bool(check.get("passed"))]
                    if failing_checks:
                        failures.append({"code": "benchmark_checks_failed", "message": f"benchmark checks failed: {', '.join(failing_checks)}"})

                    runtime_check = next((check for check in checks_with_status if check.get("id") == "runtime_parity"), None)
                    if runtime_check:
                        chooser_score = runtime_check.get("chooser_score")
                        preloaded_score = runtime_check.get("preloaded_score")
                        max_delta = float(gates.get("max_runtime_parity_delta", 0.10))
                        if chooser_score is not None and preloaded_score is not None:
                            delta = abs(float(chooser_score) - float(preloaded_score))
                            if not math.isfinite(delta) or delta > max_delta:
                                failures.append({"code": "runtime_parity_delta_exceeded", "message": f"runtime parity delta {delta:.4f} > {max_delta:.4f}"})

    return failures


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[2]
    config_path = Path(args.config)
    config = load_config(config_path)

    run_id = config.get("run_id", "aggressive-v1")
    seed = int(config.get("seed", 42))
    total_iterations = int(args.iterations or config.get("iterations", 1))
    start_iteration = max(1, int(args.start_iteration or config.get("start_iteration", 1)))
    final_iteration = start_iteration + total_iterations - 1
    versions_dir = Path(config.get("versions_dir", "storage/ai/alphazero_lite/versions"))
    current_path = config.get("current_path", "storage/ai/alphazero_lite/current")
    steps = config.get("steps", [])
    gates = config.get("gates", {})
    replay_window = max(1, int(config.get("replay_window", 1)))

    for iteration in range(start_iteration, start_iteration + total_iterations):
        iter_dir = versions_dir / f"{run_id}-iter{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        parent_model_dir = Path(current_path) if iteration == start_iteration else versions_dir / f"{run_id}-iter{iteration - 1}"
        checkpoint_candidate = parent_model_dir / "checkpoint.npz"
        fallback_model = parent_model_dir / "model.npz"
        parent_checkpoint = checkpoint_candidate if checkpoint_candidate.exists() else fallback_model

        manifest = build_manifest(
            run_id=run_id,
            iteration=iteration,
            seed=seed,
            config_path=str(config_path),
            parent_version=current_path if iteration == start_iteration else f"{run_id}-iter{iteration - 1}",
            status="planned" if args.dry_run else "running",
            notes={"phase": "phase_1_scaffold", "dry_run": bool(args.dry_run)},
        )

        manifest["steps"] = []
        replay_data, replay_weights = build_replay_context(
            iteration=iteration,
            iter_dir=iter_dir,
            versions_dir=versions_dir,
            run_id=run_id,
            replay_window=replay_window,
        )

        skipped_steps = set(args.skip_step)

        if not args.dry_run:
            for step in steps:
                if step.get("skip_before_final_iteration") and iteration < final_iteration:
                    skipped_result = {
                        "name": step.get("name", "unnamed_step"),
                        "status": "skipped",
                        "reason": "skip_before_final_iteration",
                    }
                    manifest["steps"].append(skipped_result)
                    write_manifest(iter_dir / "run_manifest.json", manifest)
                    continue
                if step.get("name") in skipped_steps:
                    continue
                replay_data, replay_weights = build_replay_context(
                    iteration=iteration,
                    iter_dir=iter_dir,
                    versions_dir=versions_dir,
                    run_id=run_id,
                    replay_window=replay_window,
                )
                step_result = run_step(
                    step,
                    iteration=iteration,
                    iter_dir=iter_dir,
                    run_id=run_id,
                    versions_dir=versions_dir,
                    repo_root=repo_root,
                    current_path=current_path,
                    parent_model_dir=parent_model_dir,
                    parent_checkpoint=parent_checkpoint,
                    replay_data=replay_data,
                    replay_weights=replay_weights,
                )
                manifest["steps"].append(step_result)
                if step_result["status"] == "failed":
                    manifest["status"] = "failed"
                    write_manifest(iter_dir / "run_manifest.json", manifest)
                    raise SystemExit(1)

            failures = gate_failures(
                gates,
                iteration=iteration,
                iter_dir=iter_dir,
                run_id=run_id,
                versions_dir=versions_dir,
                current_path=current_path,
                parent_model_dir=parent_model_dir,
                parent_checkpoint=parent_checkpoint,
                replay_data=replay_data,
                replay_weights=replay_weights,
                skipped_steps=skipped_steps,
            )
            manifest["gate_failures"] = failures
            if failures:
                manifest["status"] = "failed"
                write_manifest(iter_dir / "run_manifest.json", manifest)
                raise SystemExit(1)

            manifest["status"] = "completed"

        write_manifest(iter_dir / "run_manifest.json", manifest)

    if args.dry_run:
        print("pipeline_dry_run_complete")
    else:
        print("pipeline_scaffold_complete")


if __name__ == "__main__":
    main()
