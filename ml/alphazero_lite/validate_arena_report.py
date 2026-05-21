#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite.report_validation import (
    ArenaReportValidationError,
    validate_arena_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True)
    parser.add_argument("--min-score", type=float, default=0.55)
    parser.add_argument("--require-pass", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = Path(args.report)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    try:
        result = validate_arena_report(report=report, min_score=args.min_score)
    except ArenaReportValidationError as error:
        raise SystemExit(str(error)) from error

    if args.require_pass and not bool(result["passed"]):
        raise SystemExit("arena_prefilter_failed")

    if args.require_pass:
        print("arena_prefilter_passed")
    else:
        print("arena_report_valid")


if __name__ == "__main__":
    main()
