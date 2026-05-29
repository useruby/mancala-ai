#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.run_rule_conditioned_opening_full_guarded_experiment import (
    load_json,
    repo_root,
    write_json,
)


DEFAULT_ARTIFACT = (
    "/tmp/azlite_rule_conditioned_opening_full_guarded/"
    "rule-conditioned-opening-full-guarded/w2/versions/"
    "aggressive-v3-targeted-hard-state-replay-rule-conditioned-opening-full-guarded-w2-iter1"
)
DEFAULT_VALIDATION_PATH = "ml/alphazero_lite/fixtures/incumbent_forensic_suite_v1.json"
DEFAULT_OUTPUT_ROOT = "/tmp/azlite_puct1200_root_prior_transform_validation"
DEFAULT_TRANSFORM = "seed4_extra_turn_damp_010_when_4_legal_two_captures_noncapture5"
SCHEMA = "azlite_puct1200_root_prior_transform_validation_v1"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", default=DEFAULT_ARTIFACT)
    parser.add_argument("--validation-path", default=DEFAULT_VALIDATION_PATH)
    parser.add_argument("--root-prior-transform", default=DEFAULT_TRANSFORM)
    parser.add_argument("--teacher-simulations", type=int, default=1200)
    parser.add_argument("--artifact-simulations", type=int, default=1200)
    parser.add_argument("--c-puct", type=float, default=1.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args(argv)


def resolve_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def python_bin(root: Path) -> str:
    candidate = root / ".venv/bin/python"
    if candidate.is_file():
        return str(candidate)
    return sys.executable


def build_markdown(summary: dict) -> str:
    report = summary["validation_report"]
    overall = report.get("overall") or {}
    return "\n".join(
        [
            "# PUCT1200 Root-Prior Transform Validation",
            "",
            "## Context",
            "",
            "This run validates the narrow root-prior transform on the broader forensic hard-state suite using direct artifact PUCT search at 1200 simulations.",
            "",
            f"- artifact: `{summary['artifact_path']}`",
            f"- validation path: `{summary['validation_path']}`",
            f"- root prior transform: `{summary['root_prior_transform']}`",
            f"- validation report: `{summary['validation_report_path']}`",
            "",
            "## Overall",
            "",
            f"- position count: `{report.get('position_count')}`",
            f"- policy top1 agreement: `{report.get('policy_top1_agreement')}`",
            f"- average regret: `{report.get('average_regret')}`",
            f"- value calibration mae: `{report.get('value_calibration_mae')}`",
            f"- bucket count: `{len(report.get('buckets') or {})}`",
            "",
            "## Notes",
            "",
            f"- overall summary: `{overall}`",
        ]
    ) + "\n"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    root = repo_root()
    artifact_path = resolve_path(root, args.artifact)
    validation_path = resolve_path(root, args.validation_path)
    output_root = resolve_path(root, args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    validation_report_path = output_root / "hard_state_validation_puct1200.json"
    summary_path = output_root / "puct1200_root_prior_transform_validation_summary.json"
    report_path = root / "docs/alphazero-lite-puct1200-root-prior-transform-validation.md"

    command = [
        python_bin(root),
        "-m",
        "ml.alphazero_lite.hard_state_validation",
        "--artifact-path",
        str(artifact_path),
        "--validation-path",
        str(validation_path),
        "--teacher-simulations",
        str(args.teacher_simulations),
        "--artifact-simulations",
        str(args.artifact_simulations),
        "--c-puct",
        str(args.c_puct),
        "--seed",
        str(args.seed),
        "--root-prior-transform",
        str(args.root_prior_transform),
        "--out",
        str(validation_report_path),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    validation_report = load_json(validation_report_path)
    summary = {
        "schema": SCHEMA,
        "artifact_path": str(artifact_path),
        "validation_path": str(validation_path),
        "root_prior_transform": str(args.root_prior_transform),
        "validation_report_path": str(validation_report_path),
        "summary_path": str(summary_path),
        "report_path": str(report_path),
        "command": command,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "validation_report": validation_report,
    }
    write_json(summary_path, summary)
    report_path.write_text(build_markdown(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "report_path": str(report_path),
                "validation_report_path": str(validation_report_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
