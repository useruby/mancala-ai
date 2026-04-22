#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[2]))

from ml.alphazero_lite import hard_state_mining


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--out-jsonl", required=True)
    parser.add_argument("--out-report", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows, summary = hard_state_mining.run_pipeline(
            [value.strip() for value in args.inputs.split(",") if value.strip()]
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    hard_state_mining.write_jsonl(Path(args.out_jsonl), rows)
    hard_state_mining.write_summary_report(Path(args.out_report), summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
