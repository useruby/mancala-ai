#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import hard_state_teacher_labeling as labeling


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-jsonl", required=True, type=Path)
    parser.add_argument("--out-report", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = labeling.load_labeled_rows(args.input_jsonl)
        report = labeling.build_comparison_report(rows)
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    args.out_report.parent.mkdir(parents=True, exist_ok=True)
    args.out_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {"out_report": str(args.out_report), "pair_count": report["pair_count"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
