#!/usr/bin/env python3
"""Coach-style pipeline scaffold for AlphaZero-lite aggressive plan."""

from __future__ import annotations

import argparse
import ast
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
import zipfile
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

    try:
        with zipfile.ZipFile(checkpoint_path) as archive:
            for key in ("w_input.npy", "w_hidden_1.npy", "w1.npy"):
                if key in archive.namelist():
                    with archive.open(key) as handle:
                        if handle.read(6) != b"\x93NUMPY":
                            continue
                        major = handle.read(1)[0]
                        minor = handle.read(1)[0]
                        if major == 1:
                            header_len = struct.unpack("<H", handle.read(2))[0]
                        else:
                            header_len = struct.unpack("<I", handle.read(4))[0]
                        header = handle.read(header_len).decode("latin1")
                        metadata = ast.literal_eval(header.strip())
                        shape = metadata.get("shape")
                        if shape:
                            return int(shape[0])
    except (OSError, ValueError, zipfile.BadZipFile, SyntaxError, struct.error):
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
    hard_state_validation_path: str = "",
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
        value = value.replace("{hard_state_validation_path}", hard_state_validation_path)
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
    hard_state_validation_path: str = "",
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
    rendered = rendered.replace("{hard_state_validation_path}", hard_state_validation_path)
    return Path(rendered)


def resolve_step_command(command: list[str], *, repo_root: Path) -> list[str]:
    if not command:
        return command

    executable = command[0]
    if executable != ".venv/bin/python":
        return command

    local_venv = repo_root / executable
    if local_venv.exists():
        return command

    for ancestor in repo_root.parents:
        candidate = ancestor / executable
        if candidate.exists():
            return [str(candidate), *command[1:]]

    return [sys.executable, *command[1:]]


def load_fixed_replay_sources(config: dict, *, repo_root: Path) -> list[tuple[Path, int]]:
    raw_sources = config.get("fixed_replay_sources", [])
    if raw_sources is None:
        return []
    if not isinstance(raw_sources, list):
        raise SystemExit("fixed_replay_sources must be a list")

    fixed_sources: list[tuple[Path, int]] = []
    for source in raw_sources:
        if not isinstance(source, dict):
            raise SystemExit("fixed_replay_sources entries must be objects")

        path = source.get("path")
        weight = source.get("weight")

        if not isinstance(path, str) or not path.strip():
            raise SystemExit("fixed replay source path must be a non-empty string")
        if not isinstance(weight, int) or isinstance(weight, bool) or weight <= 0:
            raise SystemExit("fixed replay source weight must be a positive integer")

        replay_path = Path(path)
        if not replay_path.is_absolute():
            replay_path = repo_root / replay_path
        if not replay_path.exists():
            raise SystemExit(f"missing fixed replay source: {replay_path}")
        fixed_sources.append((replay_path.resolve(), weight))

    return fixed_sources


def resolve_config_path(path: str, *, repo_root: Path) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = repo_root / resolved
    return resolved


def validate_startup_paths(
    *,
    repo_root: Path,
    resolved_parent_artifact_path: Path | None,
    hard_state_validation_path: str,
    config: dict,
    dry_run: bool,
) -> None:
    if dry_run:
        return

    if "parent_artifact_path" in config:
        parent_path = resolved_parent_artifact_path
        if parent_path is None:
            raise SystemExit("parent_artifact_path must resolve before validation")
        if not parent_path.exists():
            raise SystemExit(f"missing parent_artifact_path: {parent_path}")

        checkpoint_path = parent_path / "checkpoint.npz"
        model_path = parent_path / "model.npz"
        if not checkpoint_path.exists() and not model_path.exists():
            raise SystemExit(
                f"parent_artifact_path must contain checkpoint.npz or model.npz: {parent_path}"
            )

    if "hard_state_validation_path" in config:
        validation_path = resolve_config_path(hard_state_validation_path, repo_root=repo_root)
        if not validation_path.exists():
            raise SystemExit(f"missing hard_state_validation_path: {validation_path}")


def has_self_play_step(steps: list[dict]) -> bool:
    return any(step.get("name") == "self_play" for step in steps if isinstance(step, dict))


def build_replay_context(
    *,
    iteration: int,
    iter_dir: Path,
    versions_dir: Path,
    run_id: str,
    replay_window: int,
    include_current_iteration: bool,
) -> tuple[list[Path], list[int]]:
    replay_paths: list[Path] = []

    if include_current_iteration:
        replay_paths.append(iter_dir / "self_play.jsonl")

    for previous_iteration in range(iteration - 1, max(0, iteration - replay_window), -1):
        previous_path = versions_dir / f"{run_id}-iter{previous_iteration}" / "self_play.jsonl"
        if previous_path.exists():
            replay_paths.append(previous_path)

    replay_weights = list(range(len(replay_paths), 0, -1))
    return replay_paths, replay_weights


def merge_replay_context(
    dynamic_paths: list[Path],
    dynamic_weights: list[int],
    fixed_sources: list[tuple[Path, int]] | None = None,
) -> tuple[str, str]:
    replay_paths = list(dynamic_paths)
    replay_weights = list(dynamic_weights)

    for path, weight in fixed_sources or []:
        replay_paths.append(path)
        replay_weights.append(weight)

    replay_data = ",".join(str(path) for path in replay_paths)
    replay_weights_text = ",".join(str(weight) for weight in replay_weights)
    return replay_data, replay_weights_text


def resolve_replay_context(
    *,
    iteration: int,
    iter_dir: Path,
    versions_dir: Path,
    run_id: str,
    replay_window: int,
    include_current_iteration: bool,
    fixed_replay_sources: list[tuple[Path, int]] | None = None,
) -> tuple[str, str]:
    replay_paths, replay_weights = build_replay_context(
        iteration=iteration,
        iter_dir=iter_dir,
        versions_dir=versions_dir,
        run_id=run_id,
        replay_window=replay_window,
        include_current_iteration=include_current_iteration,
    )
    return merge_replay_context(replay_paths, replay_weights, fixed_replay_sources)


def json_safe_config(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(key): json_safe_config(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe_config(item) for item in value]
    return repr(value)


def numpy_build_summary(numpy_module) -> dict:
    config = getattr(numpy_module, "__config__", None)
    if config is None:
        return {}

    config_dict = getattr(config, "CONFIG", None)
    if isinstance(config_dict, dict):
        return json_safe_config(config_dict)

    get_info = getattr(config, "get_info", None)
    if callable(get_info):
        summary = {}
        for name in ("blas_opt", "lapack_opt", "openblas", "blas_ilp64_opt", "lapack_ilp64_opt"):
            info = get_info(name)
            if info:
                summary[name] = json_safe_config(info)
        return summary

    return {}


def environment_report() -> dict:
    uname = platform.uname()
    env_keys = [
        "CUDA_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "PYTHONHASHSEED",
    ]

    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "implementation": platform.python_implementation(),
        },
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "system": uname.system,
            "release": uname.release,
            "version": uname.version,
            "node": uname.node,
        },
        "env": {key: os.environ.get(key) for key in env_keys},
    }

    def safe_call(function):
        if not callable(function):
            return None

        try:
            return function()
        except Exception:
            return None

    torch_fallback = {
        "version": None,
        "cuda": {
            "available": None,
            "version": None,
            "device_count": None,
            "cudnn_version": None,
        },
    }

    try:
        import numpy  # type: ignore

        report["numpy"] = {
            "version": numpy.__version__,
            "build": numpy_build_summary(numpy),
        }
    except Exception:
        report["numpy"] = {"version": None, "build": {}}

    try:
        import torch  # type: ignore

        cuda_module = getattr(torch, "cuda", None)
        torch_version = getattr(torch, "version", None)
        cudnn_module = getattr(getattr(torch, "backends", None), "cudnn", None)
        cuda_available = safe_call(getattr(cuda_module, "is_available", None))
        report["torch"] = {
            "version": getattr(torch, "__version__", None),
            "cuda": {
                "available": cuda_available,
                "version": getattr(torch_version, "cuda", None),
                "device_count": safe_call(getattr(cuda_module, "device_count", None)) if cuda_available is True else None,
                "cudnn_version": safe_call(getattr(cudnn_module, "version", None)) if cuda_available is True else None,
            },
        }
    except Exception:
        report["torch"] = torch_fallback

    return report


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
    hard_state_validation_path: str = "",
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
        hard_state_validation_path=hard_state_validation_path,
    )
    rendered = resolve_step_command(rendered, repo_root=repo_root)
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
    hard_state_validation_path: str = "",
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
            hard_state_validation_path=hard_state_validation_path,
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
            hard_state_validation_path=hard_state_validation_path,
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
            hard_state_validation_path=hard_state_validation_path,
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
            hard_state_validation_path=hard_state_validation_path,
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
    current_path = config.get("current_path", "model-artifact/current")
    parent_artifact_path = config.get("parent_artifact_path", current_path)
    resolved_first_iteration_parent_path = resolve_config_path(parent_artifact_path, repo_root=repo_root)
    resolved_parent_artifact_path = (
        resolved_first_iteration_parent_path if "parent_artifact_path" in config else None
    )
    steps = config.get("steps", [])
    gates = config.get("gates", {})
    replay_window = max(1, int(config.get("replay_window", 1)))
    include_current_iteration = has_self_play_step(steps)
    fixed_replay_sources = load_fixed_replay_sources(config, repo_root=repo_root)
    hard_state_validation_path = str(config.get("hard_state_validation_path", ""))
    validate_startup_paths(
        repo_root=repo_root,
        resolved_parent_artifact_path=resolved_parent_artifact_path,
        hard_state_validation_path=hard_state_validation_path,
        config=config,
        dry_run=args.dry_run,
    )

    for iteration in range(start_iteration, start_iteration + total_iterations):
        iter_dir = versions_dir / f"{run_id}-iter{iteration}"
        iter_dir.mkdir(parents=True, exist_ok=True)
        (iter_dir / "environment.json").write_text(
            json.dumps(environment_report(), indent=2, sort_keys=True),
            encoding="utf-8",
        )

        parent_model_dir = (
            resolved_first_iteration_parent_path
            if iteration == start_iteration
            else versions_dir / f"{run_id}-iter{iteration - 1}"
        )
        checkpoint_candidate = parent_model_dir / "checkpoint.npz"
        fallback_model = parent_model_dir / "model.npz"
        parent_checkpoint = checkpoint_candidate if checkpoint_candidate.exists() else fallback_model

        manifest = build_manifest(
            run_id=run_id,
            iteration=iteration,
            seed=seed,
            config_path=str(config_path),
            parent_version=parent_artifact_path if iteration == start_iteration else f"{run_id}-iter{iteration - 1}",
            status="planned" if args.dry_run else "running",
            notes={
                "phase": "phase_1_scaffold",
                "dry_run": bool(args.dry_run),
                "parent_artifact_path": parent_artifact_path if iteration == start_iteration else None,
            },
        )

        manifest["steps"] = []
        replay_data, replay_weights = resolve_replay_context(
            iteration=iteration,
            iter_dir=iter_dir,
            versions_dir=versions_dir,
            run_id=run_id,
            replay_window=replay_window,
            include_current_iteration=include_current_iteration,
            fixed_replay_sources=fixed_replay_sources,
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
                replay_data, replay_weights = resolve_replay_context(
                    iteration=iteration,
                    iter_dir=iter_dir,
                    versions_dir=versions_dir,
                    run_id=run_id,
                    replay_window=replay_window,
                    include_current_iteration=include_current_iteration,
                    fixed_replay_sources=fixed_replay_sources,
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
                    hard_state_validation_path=hard_state_validation_path,
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
                hard_state_validation_path=hard_state_validation_path,
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
