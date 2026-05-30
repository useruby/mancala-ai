#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_REFERENCE_ARTIFACT = (
    "ml/alphazero_lite/fixtures/incumbent_train_only_forensic_references_v1.json"
)
DEFAULT_BASE_RUNTIME_CONFIG = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/runtime_config.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_guarded_w2_prior_calibration_setup"
DEFAULT_RUN_ID = "guarded-w2-prior-calibration"
CALIBRATION_ARTIFACT_WEIGHT = 1


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _rewrite_command_seed_flags(
    command: list[object], *, original_seed: int, new_seed: int
) -> list[object]:
    rewritten = list(command)
    if "--seed" in rewritten:
        seed_index = rewritten.index("--seed")
        if seed_index + 1 < len(rewritten):
            rewritten[seed_index + 1] = str(new_seed)

    if "--seed-sweep" in rewritten:
        sweep_index = rewritten.index("--seed-sweep")
        if sweep_index + 1 < len(rewritten):
            sweep_values = [
                int(value.strip())
                for value in str(rewritten[sweep_index + 1]).split(",")
                if value.strip()
            ]
            if sweep_values:
                rewritten[sweep_index + 1] = ",".join(
                    str(new_seed + (value - original_seed)) for value in sweep_values
                )
    return rewritten


def apply_runtime_seed(config: dict, *, seed: int) -> dict:
    rewritten = json.loads(json.dumps(config))
    original_seed = int(rewritten.get("seed", seed))
    rewritten["seed"] = int(seed)
    for step in rewritten.get("steps", []):
        if not isinstance(step, dict):
            continue
        command = step.get("command")
        if not isinstance(command, list):
            continue
        step["command"] = _rewrite_command_seed_flags(
            command,
            original_seed=original_seed,
            new_seed=int(seed),
        )
    return rewritten


def replace_flag_value(command: list[object], *, flag: str, value: str) -> list[object]:
    rewritten = list(command)
    if flag in rewritten:
        index = rewritten.index(flag)
        if index + 1 < len(rewritten):
            rewritten[index + 1] = value
            return rewritten
    rewritten.extend([flag, value])
    return rewritten


def align_target_modes(config: dict) -> dict:
    rewritten = json.loads(json.dumps(config))
    for step in rewritten.get("steps", []):
        if not isinstance(step, dict):
            continue
        command = step.get("command")
        if not isinstance(command, list):
            continue
        if step.get("name") in {"self_play", "train"}:
            command = replace_flag_value(
                command,
                flag="--policy-target-mode",
                value="sharpened",
            )
            command = replace_flag_value(
                command,
                flag="--value-target-mode",
                value="sharpened",
            )
            step["command"] = command
    return rewritten


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--reference-artifact", default=DEFAULT_REFERENCE_ARTIFACT)
    parser.add_argument("--base-runtime-config", default=DEFAULT_BASE_RUNTIME_CONFIG)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def build_runtime_config(
    *,
    base_runtime_config: dict,
    run_root: Path,
    seed: int,
    calibration_artifact_path: Path,
) -> dict:
    runtime_config = align_target_modes(
        apply_runtime_seed(base_runtime_config, seed=seed)
    )
    runtime_config["run_id"] = f"{runtime_config['run_id']}-prior-calibration"
    runtime_config["versions_dir"] = str(run_root / "versions")
    fixed_replay_sources = list(runtime_config.get("fixed_replay_sources", []))
    fixed_replay_sources.append(
        {"path": str(calibration_artifact_path), "weight": CALIBRATION_ARTIFACT_WEIGHT}
    )
    runtime_config["fixed_replay_sources"] = fixed_replay_sources
    runtime_config["seed"] = int(seed)
    return runtime_config


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    python = python_bin(root)
    run_root = resolve_path(root, args.output_root) / args.run_id
    run_root.mkdir(parents=True, exist_ok=True)

    reference_artifact_path = resolve_path(root, args.reference_artifact)
    base_runtime_config_path = resolve_path(root, args.base_runtime_config)
    calibration_artifact_path = (
        run_root / "capture_002_003_guarded_w2_prior_calibration_artifact.jsonl"
    )
    calibration_summary_path = (
        run_root / "capture_002_003_guarded_w2_prior_calibration_artifact_summary.json"
    )

    subprocess.run(
        [
            python,
            "-m",
            "ml.alphazero_lite.build_capture_002_003_guarded_w2_prior_calibration_artifact",
            "--reference-artifact",
            str(reference_artifact_path),
            "--out",
            str(calibration_artifact_path),
            "--out-summary",
            str(calibration_summary_path),
        ],
        cwd=root,
        check=True,
        capture_output=False,
        text=True,
    )

    base_runtime_config = load_json(base_runtime_config_path)
    calibration_summary = load_json(calibration_summary_path)
    runtime_config = build_runtime_config(
        base_runtime_config=base_runtime_config,
        run_root=run_root,
        seed=args.seed,
        calibration_artifact_path=calibration_artifact_path,
    )
    runtime_config_path = run_root / "runtime_config.json"
    write_json(runtime_config_path, runtime_config)

    summary = {
        "run_id": args.run_id,
        "base_runtime_config_path": str(base_runtime_config_path),
        "reference_artifact_path": str(reference_artifact_path),
        "calibration_artifact_path": str(calibration_artifact_path),
        "calibration_summary_path": str(calibration_summary_path),
        "runtime_config_path": str(runtime_config_path),
        "calibration_artifact_weight": CALIBRATION_ARTIFACT_WEIGHT,
        "calibration_summary": calibration_summary,
        "fixed_replay_sources": runtime_config.get("fixed_replay_sources", []),
    }
    summary_path = run_root / "setup_summary.json"
    write_json(summary_path, summary)
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "runtime_config_path": str(runtime_config_path),
                "calibration_artifact_path": str(calibration_artifact_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
