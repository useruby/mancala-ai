"""Merge completed exact-solver bucket reports into one qualification report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ml.alphazero_lite.run_exact_solver_feasibility_preflight import qualification


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.input]
    if not all(report.get("complete") for report in reports):
        raise ValueError("all bucket reports must be complete")
    parity = reports[0]["rules_parity"]
    exactness = reports[0]["exactness_validation"]
    rows = [row for report in reports for row in report["corpus"]]
    if len({row["id"] for row in rows}) != 96:
        raise ValueError(
            "merged reports must contain exactly 96 distinct corpus states"
        )
    report = {
        "schema": "exact_kalah_solver_feasibility_v1",
        "seed": reports[0]["seed"],
        "complete": True,
        "rules_parity": parity,
        "exactness_validation": exactness,
        "corpus": rows,
        "qualification": qualification(rows, parity, exactness),
    }
    args.output.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return 0 if report["qualification"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
