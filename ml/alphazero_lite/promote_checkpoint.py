#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.report_validation import (
    ArenaReportValidationError,
    validate_arena_report,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be a non-negative integer")
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument(
        "--target",
        action="append",
        default=None,
        metavar="TARGET",
        help="Destination directory (repeatable). Defaults to model-artifact/current",
    )
    parser.add_argument("--gate-report")
    parser.add_argument("--require-lossless", action="store_true")
    parser.add_argument("--max-losses", type=non_negative_int, default=0)
    return parser.parse_args()


def resolve_repo_path(path: str) -> Path:
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    return resolved


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    report_path = checkpoint_path / "arena_report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing arena report: {report_path}")

    if args.gate_report:
        gate_report_path = resolve_repo_path(args.gate_report)
        if not gate_report_path.exists():
            raise SystemExit(f"Missing gate report: {gate_report_path}")
        try:
            gate_report = json.loads(gate_report_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SystemExit(
                f"Malformed gate report JSON: {gate_report_path}: {error}"
            ) from error
        if not gate_report.get("passed", False):
            raise SystemExit(f"Gate report did not pass: {gate_report_path}")

    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise SystemExit(
            f"Malformed arena report JSON: {report_path}: {error}"
        ) from error
    try:
        result = validate_arena_report(report=report, min_score=args.min_score)
    except ArenaReportValidationError as error:
        raise SystemExit(str(error)) from error

    if not result["passed"]:
        raise SystemExit(
            f"Checkpoint did not meet threshold score={result['score']} min_score={result['min_score']}"
        )

    if args.require_lossless:
        losses = int(report.get("losses", 0))
        if losses > args.max_losses:
            raise SystemExit(
                f"lossless requirement failed: losses={losses} max_losses={args.max_losses}"
            )

    targets_raw = args.target if args.target else ["model-artifact/current"]
    targets = []
    for raw in targets_raw:
        t = resolve_repo_path(raw)
        t.mkdir(parents=True, exist_ok=True)
        targets.append(t)

    required_files = ["metadata.json", "arena_report.json", "weights.json"]
    missing_files = [
        str(checkpoint_path / filename)
        for filename in required_files
        if not (checkpoint_path / filename).exists()
    ]
    if missing_files:
        raise SystemExit(f"Missing required file: {missing_files[0]}")

    for target in targets:
        for entry in target.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()

        for filename in required_files:
            source = checkpoint_path / filename
            shutil.copy2(source, target / filename)

        print(f"Promoted checkpoint from {checkpoint_path} to {target}")
    print(f"Score={result['score']} MinScore={result['min_score']}")


if __name__ == "__main__":
    main()
