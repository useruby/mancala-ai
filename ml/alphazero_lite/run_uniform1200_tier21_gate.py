#!/usr/bin/env python3
"""Run canonical tier 21 on only unresolved radius-0/1 PR #291 states."""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if __package__ in (None, ""):
    sys.path.append(str(ROOT))

from ml.alphazero_lite.kalah_rules import KalahGame  # noqa: E402
from ml.alphazero_lite.run_uniform1200_exact_neighborhood_audit import (  # noqa: E402
    COHORT_SHA256,
    COMPLETION_RETRY_TIMEOUT_SECONDS,
    NativeHybridProcess,
    active_stones,
    native_label_payload,
    original_coverage_gate,
    root_training_value,
    sha256_file,
    stable_sha,
    validate_native_response,
)
from ml.alphazero_lite.run_uniform1200_tier20_feasibility import persist  # noqa: E402

BASELINE = ROOT / "docs/data/alphazero-lite-uniform1200-exact-neighborhood-audit.json"
TIER19 = (
    ROOT / "docs/data/alphazero-lite-uniform1200-tier19-unresolved-feasibility.json"
)
TIER20 = (
    ROOT / "docs/data/alphazero-lite-uniform1200-tier20-unresolved-feasibility.json"
)
TIER21_SHA256 = "f126f64be2010abae6bd5f3b369b40a1cb7497b5f6903914c7ba9be0037a41a7"


def blockers(
    baseline: list[dict], tier19: list[dict], tier20: list[dict]
) -> list[dict]:
    prior = [
        {row["canonical_state_key"]: row for row in rows} for rows in (tier19, tier20)
    ]
    rows = []
    for row in baseline:
        key = row["canonical_state_key"]
        status19, status20 = (
            item.get(key, {}).get("status", "not_attempted") for item in prior
        )
        if (
            row["radius"] in (0, 1)
            and not row["terminal"]
            and "exact_solved" not in (row["oracle_status"], status19, status20)
        ):
            rows.append(row | {"tier19_status": status19, "tier20_status": status20})
    rows.sort(key=lambda row: (row["radius"], stable_sha(row["canonical_state_key"])))
    if Counter(row["radius"] for row in rows) != {0: 3, 1: 6}:
        raise ValueError("expected exactly three radius-0 and six radius-1 blockers")
    return rows


def cohort_sha(rows: list[dict]) -> str:
    frozen = [
        {
            key: row[key]
            for key in (
                "canonical_state_key",
                "state",
                "radius",
                "provenance",
                "oracle_status",
                "tier19_status",
                "tier20_status",
            )
        }
        | {
            "active_stones": active_stones(row),
            "prior_attempt_history": row.get("oracle_attempt_history", []),
        }
        for row in rows
    ]
    return stable_sha(json.dumps(frozen, sort_keys=True, separators=(",", ":")))


def validate_inputs(probe: Path, tablebase: Path) -> dict:
    if sha256_file(tablebase) != TIER21_SHA256:
        raise ValueError("tier-21 tablebase SHA mismatch")
    header = tablebase.read_bytes()[:432]
    if (
        len(header) != 432
        or header[:5] != b"KVTB1"
        or header[18] != 21
        or header[23] != 21
    ):
        raise ValueError("invalid tier-21 KVTB1 header")
    return {
        "probe_sha256": sha256_file(probe),
        "tablebase_sha256": TIER21_SHA256,
        "tablebase_tier": 21,
        "tablebase_bytes": tablebase.stat().st_size,
    }


def run(
    rows: list[dict], probe: Path, tablebase: Path, preflight: dict, output: Path
) -> list[dict]:
    completed = {
        row["canonical_state_key"]: row
        for row in (
            json.loads(output.read_text()).get("rows", []) if output.exists() else []
        )
    }
    process = NativeHybridProcess(probe, tablebase)
    try:
        for row in rows:
            key = row["canonical_state_key"]
            if key in completed:
                continue
            started = time.monotonic()
            result: dict[str, Any] = {
                "canonical_state_key": key,
                "radius": row["radius"],
                "provenance": row["provenance"],
                "active_stones": active_stones(row),
                "pr292_status": row["oracle_status"],
                "tier19_status": row["tier19_status"],
                "tier20_status": row["tier20_status"],
                "prior_attempt_history": row.get("oracle_attempt_history", []),
                "timeout_seconds": COMPLETION_RETRY_TIMEOUT_SECONDS,
                "tablebase_tier": 21,
                "probe_sha256": preflight["probe_sha256"],
                "tablebase_sha256": TIER21_SHA256,
            }
            try:
                native = process.request(
                    native_label_payload(row["state"]), COMPLETION_RETRY_TIMEOUT_SECONDS
                )
                game = KalahGame.from_state(row["state"])
                values = {
                    int(action): int(value)
                    for action, value in native["action_values"].items()
                }
                value = validate_native_response(
                    values,
                    native["optimal_actions"],
                    int(native["exact_value"]),
                    game.possible_moves(),
                    game.current_player,
                )
                result |= {
                    "status": "exact_solved",
                    "exact_value": value,
                    "exact_action_values": values,
                    "exact_optimal_actions": sorted(native["optimal_actions"]),
                    "exact_root_value": root_training_value(value, game.current_player),
                    "native_metrics": native["metrics"],
                }
            except TimeoutError as error:
                result |= {"status": "exact_timeout", "failure_reason": str(error)}
                process.close()
                process = NativeHybridProcess(probe, tablebase)
            except Exception as error:
                result |= {"status": "exact_error", "failure_reason": str(error)}
                process.close()
                process = NativeHybridProcess(probe, tablebase)
            result["wall_duration_seconds"] = time.monotonic() - started
            completed[key] = result
            persist(
                output,
                {
                    "rows": [
                        completed[item["canonical_state_key"]]
                        for item in rows
                        if item["canonical_state_key"] in completed
                    ]
                },
            )
    finally:
        process.close()
    return [completed[row["canonical_state_key"]] for row in rows]


def coverage(
    baseline: list[dict], tier19: list[dict], tier20: list[dict], tier21: list[dict]
) -> dict:
    labels = [
        {row["canonical_state_key"]: row for row in rows}
        for rows in (tier19, tier20, tier21)
    ]
    return {
        str(radius): {
            "states": len(subset),
            "exact_solved": sum(
                row["oracle_status"] == "exact_solved"
                or any(
                    group.get(row["canonical_state_key"], {}).get("status")
                    == "exact_solved"
                    for group in labels
                )
                for row in subset
            ),
            "solve_rate": sum(
                row["oracle_status"] == "exact_solved"
                or any(
                    group.get(row["canonical_state_key"], {}).get("status")
                    == "exact_solved"
                    for group in labels
                )
                for row in subset
            )
            / len(subset),
        }
        for radius in range(3)
        for subset in [[row for row in baseline if row["radius"] == radius]]
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-probe", type=Path, required=True)
    parser.add_argument("--tablebase", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    base_doc = json.loads(BASELINE.read_text())
    if base_doc["frozen_cohort_sha256"] != COHORT_SHA256:
        raise ValueError("frozen cohort changed")
    tier19, tier20 = (
        json.loads(TIER19.read_text())["rows"],
        json.loads(TIER20.read_text())["rows"],
    )
    rows, preflight = (
        blockers(base_doc["cohort"], tier19, tier20),
        validate_inputs(args.native_probe, args.tablebase),
    )
    results = run(rows, args.native_probe, args.tablebase, preflight, args.output)
    rates = coverage(base_doc["cohort"], tier19, tier20, results)
    solved = [row for row in results if row["status"] == "exact_solved"]
    metrics = {
        "attempted": 9,
        "solved": len(solved),
        "timeout": sum(row["status"] == "exact_timeout" for row in results),
        "error": sum(row["status"] == "exact_error" for row in results),
        "by_radius": {
            str(radius): dict(
                Counter(row["status"] for row in results if row["radius"] == radius)
            )
            for radius in (0, 1)
        },
        "forward_nodes": sum(
            row.get("native_metrics", {}).get("forward_search_nodes", 0)
            for row in solved
        ),
        "tt_probes": sum(
            row.get("native_metrics", {}).get("tt_probes", 0) for row in solved
        ),
        "tt_hits": sum(
            row.get("native_metrics", {}).get("tt_hits", 0) for row in solved
        ),
        "tablebase_lookups": sum(
            row.get("native_metrics", {}).get("tablebase_lookups", 0) for row in solved
        ),
        "tablebase_hits": sum(
            row.get("native_metrics", {}).get("tablebase_hits", 0) for row in solved
        ),
        "maximum_tablebase_tier_hit": max(
            (
                row.get("native_metrics", {}).get("maximum_tablebase_hit_tier", 0)
                for row in solved
            ),
            default=None,
        ),
    }
    latencies = sorted(row["wall_duration_seconds"] for row in solved)
    metrics["solved_latency_seconds"] = {
        "p50": latencies[(len(latencies) - 1) // 2] if latencies else None,
        "p90": latencies[int((len(latencies) - 1) * 0.9)] if latencies else None,
        "max": max(latencies, default=None),
    }
    passed = original_coverage_gate(rates)
    report = {
        "schema": "azlite_uniform1200_tier21_gate_v1",
        "frozen_cohort_sha256": COHORT_SHA256,
        "tier21_cohort_sha256": cohort_sha(rows),
        "preflight": preflight,
        "timeout_seconds": COMPLETION_RETRY_TIMEOUT_SECONDS,
        "workers": 1,
        "rows": results,
        "metrics": metrics,
        "combined_coverage_by_radius": rates,
        "original_pr291_coverage_gate_passed": passed,
        "classification": "tier21_oracle_gate_unlocked"
        if passed
        else "tier21_oracle_effective_but_gate_unmet"
        if solved
        else "tier21_oracle_not_effective",
    }
    persist(args.output, report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
