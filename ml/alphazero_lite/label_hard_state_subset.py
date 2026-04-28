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
    parser.add_argument("--mined-jsonl", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--top-n", required=True, type=int)
    parser.add_argument("--canonical-budget", required=True, type=int)
    parser.add_argument("--stronger-budget", required=True, type=int)
    parser.add_argument("--teacher-mode", default="classic_mcts")
    parser.add_argument("--input-encoding", default="kalah_v3")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        rows = labeling.load_hard_state_rows(args.mined_jsonl)
        selected_rows = labeling.select_top_ranked_rows(rows, top_n=args.top_n)
        labeled_rows = labeling.build_dual_budget_rows(
            selected_rows,
            canonical_budget=args.canonical_budget,
            stronger_budget=args.stronger_budget,
            teacher_mode=args.teacher_mode,
            input_encoding=args.input_encoding,
            seed=args.seed,
        )
    except ValueError as error:
        print(str(error), file=sys.stderr)
        return 1

    labeling.write_jsonl(args.out, labeled_rows)
    print(json.dumps({"out": str(args.out), "rows": len(labeled_rows)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
