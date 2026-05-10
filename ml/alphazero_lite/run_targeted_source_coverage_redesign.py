#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from ml.alphazero_lite.build_tactical_balanced_replay import build_balanced_replay_dataset


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCAL_EXPERIMENT_WRAPPER = REPO_ROOT / "script/ai/run_local_tactical_replay_experiment"


VARIANT_RUN_IDS = {
    "capped_11": "targeted-source-coverage-033-capped-11",
    "expanded_12_guard_reinforced": "targeted-source-coverage-033-expanded-12-guard-002",
}


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(*, path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _variant_replay_artifact_path(*, output_root: Path, run_id: str) -> Path:
    return output_root / run_id / "final" / "tactical_balanced_replay.jsonl"


def _variant_runtime_config_path(*, output_root: Path, run_id: str) -> Path:
    return output_root / run_id / "inputs" / "runtime_config.json"


def _python_bin() -> Path:
    if os.environ.get("AZLITE_EXPERIMENT_PYTHON"):
        return Path(os.environ["AZLITE_EXPERIMENT_PYTHON"])

    candidates = [REPO_ROOT / ".venv/bin/python"]
    candidates.extend(parent / ".venv/bin/python" for parent in REPO_ROOT.parents)
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate

    return Path(sys.executable)


def _write_variant_runtime_config(*, run_id: str, output_root: Path, base_config_path: Path) -> Path:
    config = _load_json(base_config_path)
    replay_artifact_path = _variant_replay_artifact_path(output_root=output_root, run_id=run_id)
    fixed_replay_sources = list(config.get("fixed_replay_sources") or [])
    if not fixed_replay_sources:
        raise ValueError("base config must define fixed_replay_sources")
    rewritten_sources = []
    for index, source in enumerate(fixed_replay_sources):
        if index == 0:
            rewritten_sources.append({**source, "path": str(replay_artifact_path)})
        else:
            rewritten_sources.append(dict(source))
    config["fixed_replay_sources"] = rewritten_sources
    runtime_config_path = _variant_runtime_config_path(output_root=output_root, run_id=run_id)
    _write_json(path=runtime_config_path, payload=config)
    return runtime_config_path


def build_variant_artifact_summaries(
    *, output_root: Path, regression_positions_path: Path, tactical_replay_path: Path
) -> dict[str, dict]:
    summaries = {}
    for variant, run_id in VARIANT_RUN_IDS.items():
        replay_artifact_path = _variant_replay_artifact_path(output_root=output_root, run_id=run_id)
        _, summary = build_balanced_replay_dataset(
            regression_positions_path=regression_positions_path,
            tactical_replay_path=tactical_replay_path,
            out_path=replay_artifact_path,
            variant=variant,
        )
        summaries[variant] = summary
    return summaries


def run_variant_training(
    *,
    variant: str,
    run_id: str,
    output_root: Path,
    base_config_path: Path,
    current_path: str,
    forensic_suite_path: Path,
) -> dict:
    runtime_config_path = _write_variant_runtime_config(
        run_id=run_id,
        output_root=output_root,
        base_config_path=base_config_path,
    )
    completed = subprocess.run(
        args=[
            str(_python_bin()),
            str(LOCAL_EXPERIMENT_WRAPPER),
            "--run-id",
            run_id,
            "--output-root",
            str(output_root),
            "--current-path",
            current_path,
            "--base-config",
            str(runtime_config_path),
            "--forensic-suite",
            str(forensic_suite_path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    stdout_lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not stdout_lines:
        raise ValueError(f"local experiment wrapper produced no stdout for {variant}")
    return json.loads(stdout_lines[-1])


def run_redesign(*, output_root: Path, base_config_path: Path, current_path: str, forensic_suite_path: Path) -> dict:
    artifact_summaries = build_variant_artifact_summaries(
        output_root=output_root,
        regression_positions_path=REPO_ROOT / "test/fixtures/ai/superhuman_regression_positions.json",
        tactical_replay_path=REPO_ROOT / "ml/alphazero_lite/tactical_balanced_replay_source.jsonl",
    )
    expected_variants = set(VARIANT_RUN_IDS)
    if set(artifact_summaries) != expected_variants or not all(
        artifact_summaries[variant].get("pass_flags", {}).get("structurally_valid", False)
        for variant in VARIANT_RUN_IDS
    ):
        return {
            "training_started": False,
            "artifact_summaries": artifact_summaries,
        }
    return {
        "training_started": True,
        "artifact_summaries": artifact_summaries,
        "training_results": {
            variant: run_variant_training(
                variant=variant,
                run_id=VARIANT_RUN_IDS[variant],
                output_root=output_root,
                base_config_path=base_config_path,
                current_path=current_path,
                forensic_suite_path=forensic_suite_path,
            )
            for variant in VARIANT_RUN_IDS
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--current-path", required=True)
    parser.add_argument("--forensic-suite", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_redesign(
        output_root=Path(args.output_root),
        base_config_path=Path(args.base_config),
        current_path=args.current_path,
        forensic_suite_path=Path(args.forensic_suite),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
