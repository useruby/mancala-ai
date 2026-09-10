#!/usr/bin/env python3
"""Prove legacy KVTB1 generator and native-probe compatibility before tier 21."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.run_kalah_v1_native_tablebase_preflight import unrank  # noqa: E402


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def corpus() -> list[dict[str, Any]]:
    rows = []
    index = 0
    while len(rows) < 256:
        tier = 1 + index % 20
        pits = unrank(
            tier, (index * 7919 + 17) % __import__("math").comb(tier + 11, 11)
        )
        player = index % 2
        if pits[player * 6 : player * 6 + 6].count(0) == 6:
            index += 1
            continue
        rows.append(
            {
                "operation": "label",
                "pits": list(pits),
                "stores": [0, 0],
                "player": player,
            }
        )
        index += 1
    return rows


def labels(
    probe: Path, tablebase: Path, rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    result = subprocess.run(
        [probe],
        input="".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        env=os.environ | {"NATIVE_CANONICAL_KVTB": str(tablebase)},
        check=True,
        capture_output=True,
        text=True,
    )
    return [json.loads(line) for line in result.stdout.splitlines()]


def comparable(label: dict[str, Any]) -> dict[str, Any]:
    return {
        "exact_value": label["exact_value"],
        "action_values": label["action_values"],
        "optimal_actions": label["optimal_actions"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-probe", type=Path, required=True)
    parser.add_argument("--new-probe", type=Path, required=True)
    parser.add_argument("--tier18", type=Path, required=True)
    parser.add_argument("--tier19", type=Path, required=True)
    parser.add_argument("--tier20", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = corpus()
    report: dict[str, Any] = {
        "schema": "kvtb21_legacy_probe_parity_v1",
        "corpus_sha256": hashlib.sha256(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "corpus_states": len(rows),
        "old_probe_sha256": sha256(args.old_probe),
        "new_probe_sha256": sha256(args.new_probe),
        "artifacts": {},
    }
    for tier, tablebase in ((18, args.tier18), (19, args.tier19), (20, args.tier20)):
        old, new = (
            labels(args.old_probe, tablebase, rows),
            labels(args.new_probe, tablebase, rows),
        )
        equal = [comparable(item) for item in old] == [comparable(item) for item in new]
        report["artifacts"][str(tier)] = {
            "tablebase_sha256": sha256(tablebase),
            "tablebase_bytes": tablebase.stat().st_size,
            "labels_equal": equal,
            "multi_optimal_rows": sum(len(item["optimal_actions"]) > 1 for item in new),
            "legal_action_counts": sorted({len(item["action_values"]) for item in new}),
            "active_stone_tiers": sorted({sum(item["pits"]) for item in rows}),
        }
    report["passed"] = all(
        item["labels_equal"] for item in report["artifacts"].values()
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
