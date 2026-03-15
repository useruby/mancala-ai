#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.report_validation import ArenaReportValidationError, validate_arena_report


REPO_ROOT = Path(__file__).resolve().parents[2]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path")
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--target", default="storage/ai/alphazero_lite/current")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path)
    report_path = checkpoint_path / "arena_report.json"
    if not report_path.exists():
        raise SystemExit(f"Missing arena report: {report_path}")

    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        result = validate_arena_report(report=report, min_score=args.min_score)
    except ArenaReportValidationError as error:
        raise SystemExit(str(error)) from error

    if not result["passed"]:
        raise SystemExit(
            f"Checkpoint did not meet threshold score={result['score']} min_score={result['min_score']}"
        )

    target = Path(args.target)
    if not target.is_absolute():
        target = REPO_ROOT / target
    target.mkdir(parents=True, exist_ok=True)

    required_files = ["metadata.json", "arena_report.json", "weights.json"]
    missing_files = [str(checkpoint_path / filename) for filename in required_files if not (checkpoint_path / filename).exists()]
    if missing_files:
        raise SystemExit(f"Missing required file: {missing_files[0]}")

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
