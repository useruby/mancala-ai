"""Append-only semantic correction checks for the seed515 verification."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ml.alphazero_lite import verify_seed515  # noqa: E402

PUBLICATION = verify_seed515.PUBLICATION
PROPOSAL = ROOT / "docs/seed512-deployment-evaluation-proposal.md"


def _finite(value: Any, reason: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(reason) from exc
    if not math.isfinite(number):
        raise ValueError(reason)
    return number


def _validate_numeric_report() -> None:
    evidence = json.loads((PUBLICATION / "evidence.json").read_text())
    summary = evidence["summary"]
    for config in ("baseline", "P"):
        stats = summary["configs"][config]
        for field in ("p50_seconds", "p95_seconds", "throughput_searches_per_second"):
            _finite(stats[field], f"nonfinite_aggregate:{config}:{field}")
    for field in ("paired_ratio_p50", "paired_ratio_p95"):
        _finite(summary[field], f"nonfinite_paired:{field}")
    rows = summary["per_state_latency_ratios"]
    seen: set[str] = set()
    for row in rows:
        identity = row["state_sha256"]
        if identity in seen:
            raise ValueError("duplicate_per_state_record")
        seen.add(identity)
        _finite(row["P_over_baseline"], "nonfinite_per_state_ratio")

    def validate_tree(value: Any, location: str = "summary") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                validate_tree(child, f"{location}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                validate_tree(child, f"{location}[{index}]")
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            _finite(value, f"nonfinite_statistic:{location}")

    validate_tree(summary)


def _markdown_rows(text: str) -> list[list[str]]:
    rows = []
    for line in text.splitlines():
        if not line.strip().startswith("|"):
            continue
        cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
        if all(re.fullmatch(r":?-+:?", cell) for cell in cells):
            continue
        rows.append(cells)
    return rows


def _expected_table(summary: dict[str, Any]) -> list[list[str]]:
    # These are the documented seed514 aggregate and stratum table cells, with
    # configuration and unit kept in the cell identity rather than searched globally.
    stats = summary["bucket_statistics"]
    overall = summary["summary"]["configs"]
    definitions = [
        ("All", "baseline", overall["baseline"]),
        ("All", "P", overall["P"]),
    ]
    for label, bucket in (
        (">32 stones", "gt32"),
        ("17–32 stones", "17to32"),
        ("≤16 stones, exact-root", "le16"),
    ):
        for config in ("baseline", "P"):
            definitions.append((label, config, stats[bucket]["configs"][config]))
    expected = []
    for label, config, values in definitions:
        p95 = (
            f"{values['p95_seconds']:.2f} s"
            if label.startswith("≤16")
            else (
                f"{values['p95_seconds']:.3f} s"
                if label == "All"
                else f"{values['p95_seconds'] * 1000:.1f} ms"
            )
        )
        throughput = (
            f"{values['throughput_searches_per_second']:.4f}"
            if label.startswith("≤16")
            else f"{values['throughput_searches_per_second']:.3f}"
            if label == "All"
            else f"{values['throughput_searches_per_second']:.2f}"
        )
        expected.append(
            [
                label,
                "Baseline" if config == "baseline" else "P",
                str(values["searches"]),
                f"{values['p50_seconds'] * 1000:.1f} ms",
                p95,
                f"{throughput} searches/s",
            ]
        )
    return expected


def _check_table(
    document: Path, expected: list[list[str]], proposal: bool = False
) -> None:
    rows = _markdown_rows(document.read_text())
    target_header = (
        [
            "Cohort / configuration",
            "Searches",
            "p50 latency",
            "p95 latency",
            "throughput",
        ]
        if proposal
        else ["Cohort", "Configuration", "Searches", "p50", "p95", "Throughput"]
    )
    try:
        start = next(i for i, row in enumerate(rows) if row == target_header)
    except StopIteration as exc:
        raise ValueError(f"table_header:{document.name}") from exc
    actual = rows[start + 1 : start + 1 + len(expected)]
    normalized = []
    for row in actual:
        if proposal and len(row) == 5:
            cohort_config = row[0].split(" — ", 1)
            if len(cohort_config) != 2:
                raise ValueError(f"table_structure:{document.name}")
            normalized.append(
                [
                    cohort_config[0],
                    "Baseline" if cohort_config[1].lower() == "baseline" else "P",
                    *row[1:],
                ]
            )
        else:
            normalized.append(row)
    if normalized != expected:
        raise ValueError(f"table_cell_or_unit:{document.name}")


def verify(execution_record: Path | None = None) -> dict[str, Any]:
    _validate_numeric_report()
    report = verify_seed515.verify(PUBLICATION)
    # Re-key the replay output for display checks and validate each value in its
    # labeled row/column. Reuse its exact independent arithmetic, not seed515's
    # substring searches.
    replay = {
        "bucket_statistics": report["bucket_statistics"],
        "summary": json.loads((PUBLICATION / "evidence.json").read_text())["summary"],
    }
    expected = _expected_table(replay)
    _check_table(PUBLICATION / "results.md", expected)
    # Deployment table groups baseline/P per stratum in the same units.
    # Its header and data schema have five cells; evaluate directly with labels.
    proposal_rows = _markdown_rows(PROPOSAL.read_text())
    header = [
        "Cohort / configuration",
        "Searches",
        "p50 latency",
        "p95 latency",
        "throughput",
    ]
    try:
        start = next(i for i, row in enumerate(proposal_rows) if row == header)
    except StopIteration as exc:
        raise ValueError("table_header:proposal") from exc
    actual = proposal_rows[start + 1 : start + 1 + len(expected)]
    proposal_labels = {
        "All": "All states",
        "≤16 stones, exact-root": "≤16 stones, exact-root dispatch",
    }
    target = [
        [
            f"{proposal_labels.get(r[0], r[0])} — {'baseline' if r[1] == 'Baseline' else 'P'}",
            *r[2:],
        ]
        for r in expected
    ]
    if actual != target:
        for index, (found, wanted) in enumerate(zip(actual, target, strict=False)):
            if found != wanted:
                raise ValueError(
                    f"table_cell_or_unit:proposal:row_{index}:{found!r}:{wanted!r}"
                )
        raise ValueError("table_cell_or_unit:proposal:row_count")
    physical = None
    if execution_record is not None:
        physical = json.loads(execution_record.read_text())
        for key in (
            "command",
            "exit_status",
            "checkout_identity",
            "inventory_before",
            "inventory_after",
        ):
            if key not in physical:
                raise ValueError(f"execution_record_missing:{key}")
        if physical["exit_status"] != 0:
            raise ValueError("execution_record_nonzero_exit")
        if physical["inventory_before"] != physical["inventory_after"]:
            raise ValueError("execution_record_inventory_changed")
    return {
        "schema": "seed516-semantic-correction-v1",
        "status": "verified",
        "seed515_receipt_sha256": verify_seed515.digest(
            verify_seed515.OUTPUT / "seed515-verification-receipt.json"
        ),
        "finite_numeric_statistics": "passed",
        "labeled_table_cells_and_units": "passed",
        "portability_execution": physical,
        "runtime_dependency": "The registered seed512 exact-root tablebase is authenticated from local bytes; if the checkout contains its LFS pointer, the registered external materialization is separately authenticated.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="write the report to a new file")
    parser.add_argument("--execution-record", type=Path)
    args = parser.parse_args()
    result = verify(args.execution_record)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        with args.output.open("x") as output:
            output.write(rendered)
    print(rendered, end="")
