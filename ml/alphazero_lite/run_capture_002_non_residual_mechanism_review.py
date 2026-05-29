#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))


DEFAULT_SOURCE_SELECTION_PRESSURE_ABLATION_PATH = (
    "/tmp/azlite_capture_002_selection_pressure_ablation/"
    "capture-002-selection-pressure-ablation/"
    "capture_002_selection_pressure_ablation_summary.json"
)
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_capture_002_non_residual_mechanism_review"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--source-selection-pressure-ablation-artifact",
        default=DEFAULT_SOURCE_SELECTION_PRESSURE_ABLATION_PATH,
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_command(command: list[str], *, cwd: Path) -> None:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True)
    if completed.stdout:
        sys.stdout.write(completed.stdout)
    if completed.stderr:
        sys.stderr.write(completed.stderr)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    source_artifact_path = resolve_path(root, args.source_selection_pressure_ablation_artifact)
    out_root = resolve_path(root, args.output_root) / args.run_id
    out_root.mkdir(parents=True, exist_ok=True)

    artifact_path = out_root / "capture_002_non_residual_mechanism_review.json"
    run_command(
        [
            sys.executable,
            "-m",
            "ml.alphazero_lite.capture_002_non_residual_mechanism_review",
            "--source-selection-pressure-ablation-artifact",
            str(source_artifact_path),
            "--out",
            str(artifact_path),
        ],
        cwd=root,
    )

    artifact = load_json(artifact_path)
    summary = {
        "run_id": args.run_id,
        "source_selection_pressure_ablation_artifact": str(source_artifact_path),
        "artifact_path": str(artifact_path),
        "schema": artifact.get("schema"),
        "row_id": artifact.get("row_id"),
        "candidate_path": artifact.get("candidate_path"),
        "classification": artifact.get("classification"),
        "decision": artifact.get("decision"),
        "variant_summary": artifact.get("variant_summary"),
        "source_snapshot": artifact.get("source_snapshot"),
    }
    summary_path = out_root / "capture_002_non_residual_mechanism_review_summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"summary_path": str(summary_path)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
